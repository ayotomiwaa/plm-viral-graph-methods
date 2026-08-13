#!/usr/bin/env python3
"""Reference-tree edge validation for the seed-42 directional (f_j) filter.

Hypothesis: edges the directional filter removes connect sequences that a
conventional maximum-likelihood phylogeny places further apart than the edges it
retains, and that this holds at comparable embedding distance.

Stages:

1. patristic     - expand the IQ-TREE tree over the panel tips and store the
                   panel-by-panel ML patristic matrix
2. edge-metrics  - one row per baseline edge with both endpoints in the panel,
                   carrying removal status, embedding weight, ML patristic
                   distance, Hamming distance, date delta and lineage agreement
3. analyze       - unmatched, weight-stratified and matched-pair comparisons of
                   removed vs retained edges, plus f_j dose-response
4. summarize     - collect per-graph tables into the summary directory

The matched analyses are the ones that answer the obvious reviewer objection
that removed edges are simply longer embedding edges.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from Bio import Phylo
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.graph_construction.build_panel_spike_reference_tree import (  # noqa: E402
    lca_many,
    parse_accession,
)
from scripts.validation.build_reference_ml_tree_panel import (  # noqa: E402
    DEFAULT_CANDIDATE_LABEL,
    DEFAULT_DIRECTIONAL_ROOT,
    DEFAULT_OUT_ROOT,
    DEFAULT_PANEL_ROOT,
    GRAPH_SPECS,
    canonical_nodes_path,
    decisions_path,
    edge_keys,
    file_signature,
    graph_dir,
    load_canonical_nodes,
    load_decisions,
    load_edges,
    log,
    parse_graph_priority,
    read_json,
    stable_fingerprint,
    write_json,
)

STAGES = ["patristic", "edge-metrics", "analyze", "summarize"]

STATUS_REMOVED = "removed"
STATUS_RETAINED_FOR_CONNECTIVITY = "retained_for_connectivity"
STATUS_RETAINED = "retained"


# ---------------------------------------------------------------------------
# tree handling
# ---------------------------------------------------------------------------


def tree_arrays_iterative(newick_path: Path) -> dict[str, Any]:
    """Preorder tree arrays without Python recursion.

    The recursive walker in build_panel_spike_reference_tree is fine for balanced
    NJ trees but an ML tree on thousands of near-identical spike sequences can be
    deep enough to hit the interpreter recursion limit.
    """
    tree = Phylo.read(str(newick_path), "newick")
    clades: list[Any] = []
    parent_list: list[int] = []
    stack: list[tuple[Any, int]] = [(tree.root, -1)]
    while stack:
        clade, parent_index = stack.pop()
        index = len(clades)
        clades.append(clade)
        parent_list.append(parent_index)
        for child in reversed(clade.clades):
            stack.append((child, index))

    n_nodes = len(clades)
    parent = np.array(parent_list, dtype=np.int32)
    depth = np.zeros(n_nodes, dtype=np.int32)
    root_dist = np.zeros(n_nodes, dtype=np.float64)
    negative_branches = 0
    for index in range(n_nodes):
        parent_index = int(parent[index])
        length = clades[index].branch_length
        length = 0.0 if length is None else float(length)
        if length < 0.0:
            negative_branches += 1
        if parent_index < 0:
            parent[index] = index
        else:
            depth[index] = depth[parent_index] + 1
            root_dist[index] = root_dist[parent_index] + length

    max_log = max(1, math.ceil(math.log2(max(2, n_nodes))) + 1)
    up = np.empty((max_log, n_nodes), dtype=np.int32)
    up[0] = parent
    for level in range(1, max_log):
        up[level] = up[level - 1, up[level - 1]]

    tip_rows = [
        {
            "accession": parse_accession(clade.name),
            "tree_node_index": index,
            "tree_tip_name": "" if clade.name is None else str(clade.name),
        }
        for index, clade in enumerate(clades)
        if clade.is_terminal()
    ]
    return {
        "parent": parent,
        "depth": depth,
        "root_dist": root_dist,
        "up": up,
        "tips": pd.DataFrame(tip_rows),
        "n_nodes": n_nodes,
        "n_negative_branches": int(negative_branches),
    }


def panel_tip_indices(
    arrays: dict[str, Any], panel: pd.DataFrame, duplicate_groups: pd.DataFrame
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map every panel tip to its representative's node index in the ML tree."""
    tips = arrays["tips"]
    tips = tips[tips["accession"] != ""].drop_duplicates("accession", keep="first")
    tip_map = dict(zip(tips["accession"].astype(str), tips["tree_node_index"].astype(int)))
    representative = dict(
        zip(
            duplicate_groups["accession"].astype(str),
            duplicate_groups["representative_accession"].astype(str),
        )
    )
    indices = np.empty(len(panel), dtype=np.int32)
    missing: list[str] = []
    collapsed = 0
    for row, accession in enumerate(panel["accession"].astype(str)):
        rep = representative.get(accession, accession)
        if rep != accession:
            collapsed += 1
        node_index = tip_map.get(rep)
        if node_index is None:
            missing.append(accession)
            continue
        indices[row] = node_index
    if missing:
        raise ValueError(
            f"{len(missing):,} panel tips absent from the ML tree; examples={missing[:5]}"
        )
    qc = {
        "n_tree_tips": int(len(tips)),
        "n_panel_tips": int(len(panel)),
        "n_panel_tips_via_duplicate_representative": int(collapsed),
        "n_tree_nodes": int(arrays["n_nodes"]),
        "n_negative_branches": int(arrays["n_negative_branches"]),
    }
    return indices, qc


def compute_panel_patristic(
    newick_path: Path,
    panel: pd.DataFrame,
    duplicate_groups: pd.DataFrame,
    out_dir: Path,
    block_size: int,
    overwrite: bool,
) -> tuple[Path, dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = out_dir / "D_ml_patristic_float32.npy"
    qc_path = out_dir / "ml_patristic_qc.json"
    if matrix_path.exists() and qc_path.exists() and not overwrite:
        log(f"Using existing ML patristic matrix: {matrix_path}")
        return matrix_path, read_json(qc_path)

    arrays = tree_arrays_iterative(newick_path)
    indices, qc = panel_tip_indices(arrays, panel, duplicate_groups)
    n = len(panel)
    root_dist = arrays["root_dist"]
    up = arrays["up"]
    depth = arrays["depth"]
    D = np.lib.format.open_memmap(matrix_path, mode="w+", dtype=np.float32, shape=(n, n))
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        left = indices[start:stop]
        a = np.repeat(left[:, None], n, axis=1).ravel()
        b = np.repeat(indices[None, :], stop - start, axis=0).ravel()
        lca = lca_many(a, b, up, depth)
        block = root_dist[a] + root_dist[b] - 2.0 * root_dist[lca]
        D[start:stop, :] = block.reshape(stop - start, n).astype(np.float32, copy=False)
        D.flush()
        if (start // block_size) % 10 == 0:
            log(f"ML patristic rows {start:,}-{stop - 1:,}/{n:,}")
    np.fill_diagonal(D, 0.0)
    values = np.asarray(D)
    offdiag = ~np.eye(n, dtype=bool)
    qc.update(
        {
            "computed_at_unix": time.time(),
            "newick_path": str(newick_path),
            "newick_signature": file_signature(newick_path),
            "matrix_path": str(matrix_path),
            "min_offdiagonal": float(values[offdiag].min()) if n > 1 else math.nan,
            "max_distance": float(values.max()) if n else math.nan,
            "n_zero_offdiagonal_pairs": int(np.count_nonzero(values[offdiag] == 0.0)),
            "max_absolute_asymmetry": float(np.max(np.abs(values - values.T))) if n else 0.0,
        }
    )
    if qc["min_offdiagonal"] < 0.0:
        raise ValueError("ML patristic matrix has negative off-diagonal distances")
    del D
    write_json(qc_path, qc)
    log(f"Wrote ML patristic matrix: {matrix_path}")
    return matrix_path, qc


def stage_patristic(args: argparse.Namespace) -> None:
    panel = pd.read_csv(args.out_root / "design/panel_tips.csv")
    duplicate_groups = pd.read_csv(args.out_root / "alignment/duplicate_groups.csv")
    newick_path = args.treefile if args.treefile is not None else args.out_root / "tree/ml.treefile"
    if not newick_path.exists():
        raise FileNotFoundError(
            f"Missing ML tree {newick_path}; run tree/run_iqtree.sh in tmux first"
        )
    compute_panel_patristic(
        newick_path=newick_path,
        panel=panel,
        duplicate_groups=duplicate_groups,
        out_dir=args.out_root / "tree_evaluation",
        block_size=args.patristic_block_size,
        overwrite=args.overwrite,
    )


# ---------------------------------------------------------------------------
# edge table
# ---------------------------------------------------------------------------


def load_candidate_evaluations(args: argparse.Namespace, graph_key: str) -> pd.DataFrame:
    path = (
        args.directional_root
        / "work"
        / args.candidate_label
        / graph_key
        / "local_candidate_evaluations.csv.gz"
    )
    if not path.exists():
        return pd.DataFrame(columns=["source", "target", "f_before", "rank_fraction", "accepted"])
    frame = pd.read_csv(
        path, usecols=["node_id", "neighbor_id", "f_before", "rank_fraction", "accepted"]
    )
    node = frame["node_id"].to_numpy(dtype=np.int64)
    neighbor = frame["neighbor_id"].to_numpy(dtype=np.int64)
    frame = frame.assign(source=np.minimum(node, neighbor), target=np.maximum(node, neighbor))
    return frame


def build_edge_table(args: argparse.Namespace, graph_key: str) -> pd.DataFrame:
    canonical = load_canonical_nodes(args)
    n_nodes = len(canonical)
    panel = pd.read_csv(args.out_root / "design/panel_tips.csv")
    panel_row_of_node = np.full(n_nodes, -1, dtype=np.int64)
    panel_row_of_node[panel["node_id"].to_numpy(dtype=np.int64)] = panel["panel_row"].to_numpy(
        dtype=np.int64
    )
    in_panel = panel_row_of_node >= 0

    edges = load_edges(args, graph_key)
    source = edges["source"].to_numpy(dtype=np.int64)
    target = edges["target"].to_numpy(dtype=np.int64)
    weight = edges["weight"].to_numpy(dtype=np.float64)
    degree = np.bincount(np.concatenate([source, target]), minlength=n_nodes)

    keep = in_panel[source] & in_panel[target]
    source = source[keep]
    target = target[keep]
    weight = weight[keep]
    if source.size == 0:
        raise ValueError(f"{graph_key}: no baseline edge has both endpoints inside the panel")

    keys = edge_keys(source, target, n_nodes)
    decisions = load_decisions(args, graph_key)
    decision_keys = edge_keys(
        decisions["source"].to_numpy(dtype=np.int64),
        decisions["target"].to_numpy(dtype=np.int64),
        n_nodes,
    )
    status = np.full(source.size, STATUS_RETAINED, dtype=object)
    max_f_before = np.full(source.size, np.nan)
    max_rank_fraction = np.full(source.size, np.nan)
    order = np.argsort(keys)
    sorted_keys = keys[order]
    positions = np.searchsorted(sorted_keys, decision_keys)
    valid = (positions < sorted_keys.size) & (
        sorted_keys[np.minimum(positions, sorted_keys.size - 1)] == decision_keys
    )
    matched_rows = order[positions[valid]]
    status[matched_rows] = decisions["global_decision"].to_numpy(dtype=object)[valid]
    max_f_before[matched_rows] = decisions["max_f_before"].to_numpy(dtype=float)[valid]
    max_rank_fraction[matched_rows] = decisions["max_rank_fraction"].to_numpy(dtype=float)[valid]

    candidates = load_candidate_evaluations(args, graph_key)
    candidate_f = np.full(source.size, np.nan)
    candidate_accepted = np.zeros(source.size, dtype=bool)
    is_candidate = np.zeros(source.size, dtype=bool)
    if not candidates.empty:
        candidate_keys = edge_keys(
            candidates["source"].to_numpy(dtype=np.int64),
            candidates["target"].to_numpy(dtype=np.int64),
            n_nodes,
        )
        positions = np.searchsorted(sorted_keys, candidate_keys)
        valid = (positions < sorted_keys.size) & (
            sorted_keys[np.minimum(positions, sorted_keys.size - 1)] == candidate_keys
        )
        candidate_rows = order[positions[valid]]
        f_values = candidates["f_before"].to_numpy(dtype=float)[valid]
        accepted_values = candidates["accepted"].to_numpy()[valid].astype(bool)
        is_candidate[candidate_rows] = True
        # an edge can be evaluated from both endpoints; keep the smaller f_before
        pooled = np.full(source.size, np.inf)
        np.minimum.at(pooled, candidate_rows, f_values)
        candidate_f = np.where(np.isfinite(pooled), pooled, np.nan)
        np.logical_or.at(candidate_accepted, candidate_rows, accepted_values)

    patristic_path = args.out_root / "tree_evaluation/D_ml_patristic_float32.npy"
    if not patristic_path.exists():
        raise FileNotFoundError("Missing ML patristic matrix; run --stages patristic first")
    T = np.load(patristic_path, mmap_mode="r")
    source_panel_row = panel_row_of_node[source]
    target_panel_row = panel_row_of_node[target]
    patristic = np.asarray(T[source_panel_row, target_panel_row], dtype=np.float64)

    hamming_path = (
        args.panel_root
        / "graphs/hamming/pool_n20000/distance_matrices"
        / "hamming_count-gap-state_all_states_uint16.npy"
    )
    hamming = np.full(source.size, np.nan)
    if hamming_path.exists():
        H = np.load(hamming_path, mmap_mode="r")
        if H.shape[0] != n_nodes:
            raise ValueError(f"{hamming_path}: expected {n_nodes} rows, observed {H.shape}")
        hamming = np.asarray(H[source, target], dtype=np.float64)

    dates = pd.to_datetime(canonical["collection_date"], errors="coerce")
    date_values = dates.to_numpy(dtype="datetime64[ns]")
    # pandas keeps NaT -> NaN here; the numpy cast would turn NaT into a sentinel int
    date_delta = np.abs(
        (pd.Series(date_values[source]) - pd.Series(date_values[target]))
        .dt.days.to_numpy(dtype=np.float64)
    )
    lineage = canonical["lineage"].astype(str).to_numpy()
    cohort = canonical["cohort_id"].astype(str).to_numpy()

    designed = pd.read_csv(args.out_root / "design/panel_edge_groups.csv.gz")
    designed = designed[designed["graph"] == graph_key]
    designed_keys = edge_keys(
        designed["source"].to_numpy(dtype=np.int64),
        designed["target"].to_numpy(dtype=np.int64),
        n_nodes,
    )
    designed_role = np.full(source.size, "", dtype=object)
    designed_group = np.full(source.size, "", dtype=object)
    designed_shared_node = np.full(source.size, -1, dtype=np.int64)
    designed_within_caliper = np.zeros(source.size, dtype=bool)
    if designed_keys.size:
        positions = np.searchsorted(sorted_keys, designed_keys)
        valid = (positions < sorted_keys.size) & (
            sorted_keys[np.minimum(positions, sorted_keys.size - 1)] == designed_keys
        )
        rows_designed = order[positions[valid]]
        designed_role[rows_designed] = designed["role"].to_numpy(dtype=object)[valid]
        designed_group[rows_designed] = designed["group_id"].to_numpy(dtype=object)[valid]
        designed_shared_node[rows_designed] = designed["shared_node_id"].to_numpy(dtype=np.int64)[
            valid
        ]
        designed_within_caliper[rows_designed] = designed["within_caliper"].to_numpy()[valid].astype(
            bool
        )

    table = pd.DataFrame(
        {
            "graph": graph_key,
            "source": source,
            "target": target,
            "source_panel_row": source_panel_row,
            "target_panel_row": target_panel_row,
            "status": status.astype(str),
            "is_removed": status == STATUS_REMOVED,
            "is_mutually_rejected": np.isin(
                status.astype(str), [STATUS_REMOVED, STATUS_RETAINED_FOR_CONNECTIVITY]
            ),
            "embedding_weight": weight,
            "ml_patristic": patristic,
            "hamming": hamming,
            "date_delta_days": date_delta,
            "same_lineage": lineage[source] == lineage[target],
            "same_cohort": cohort[source] == cohort[target],
            "source_degree": degree[source],
            "target_degree": degree[target],
            "max_f_before": max_f_before,
            "max_rank_fraction": max_rank_fraction,
            "is_directional_candidate": is_candidate,
            "candidate_f_before": candidate_f,
            "candidate_accepted_locally": candidate_accepted,
            "designed_role": designed_role.astype(str),
            "designed_group_id": designed_group.astype(str),
            "designed_shared_node_id": designed_shared_node,
            "designed_within_caliper": designed_within_caliper,
        }
    )
    table["analysis_set"] = np.where(
        table["is_removed"],
        "removed",
        np.where(table["designed_role"] == "control", "designed_control", "incidental_retained"),
    )
    return table.sort_values(["source", "target"]).reset_index(drop=True)


def stage_edge_metrics(args: argparse.Namespace, graph_keys: list[str]) -> None:
    for graph_key in graph_keys:
        out_dir = args.out_root / "edges" / graph_key
        out_dir.mkdir(parents=True, exist_ok=True)
        table_path = out_dir / "edge_table.csv.gz"
        if table_path.exists() and not args.overwrite:
            log(f"Using existing edge table: {table_path}")
            continue
        log(f"Building edge table: {graph_key}")
        table = build_edge_table(args, graph_key)
        table.to_csv(table_path, index=False)
        counts = table["analysis_set"].value_counts().to_dict()
        write_json(
            out_dir / "edge_table_manifest.json",
            {
                "built_at_unix": time.time(),
                "graph": graph_key,
                "display_name": GRAPH_SPECS[graph_key].display_name,
                "edge_table": str(table_path),
                "n_edges_in_panel": int(len(table)),
                "analysis_set_counts": {str(k): int(v) for k, v in counts.items()},
                "n_removed": int(table["is_removed"].sum()),
                "n_zero_weight_edges": int((table["embedding_weight"] == 0.0).sum()),
                "edges_signature": file_signature(graph_dir(args, graph_key) / "edges.csv"),
                "decisions_signature": file_signature(decisions_path(args, graph_key)),
                "canonical_nodes_signature": file_signature(canonical_nodes_path(args)),
            },
        )
        log(
            f"  {graph_key}: {len(table):,} in-panel edges, "
            f"{int(table['is_removed'].sum()):,} removed, "
            f"{int((table['analysis_set'] == 'designed_control').sum()):,} designed controls"
        )


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def hodges_lehmann(a: np.ndarray, b: np.ndarray, max_pairs: int, seed: int) -> float:
    """Median of all pairwise differences a_i - b_j, subsampled when huge."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return math.nan
    if a.size * b.size <= max_pairs:
        return float(np.median(a[:, None] - b[None, :]))
    rng = np.random.default_rng(seed)
    left = rng.integers(0, a.size, size=max_pairs)
    right = rng.integers(0, b.size, size=max_pairs)
    return float(np.median(a[left] - b[right]))


def bootstrap_ci(
    statistic: Callable[[np.random.Generator], float],
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = np.array([statistic(rng) for _ in range(n_boot)], dtype=np.float64)
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return math.nan, math.nan
    low, high = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def two_group_summary(
    removed: np.ndarray,
    retained: np.ndarray,
    label: str,
    seed: int,
    n_boot: int,
    max_pairs: int,
    bootstrap_pairs: int = 20_000,
) -> dict[str, Any]:
    removed = np.asarray(removed, dtype=np.float64)
    retained = np.asarray(retained, dtype=np.float64)
    removed = removed[np.isfinite(removed)]
    retained = retained[np.isfinite(retained)]
    out: dict[str, Any] = {
        "outcome": label,
        "n_removed": int(removed.size),
        "n_retained": int(retained.size),
        "median_removed": float(np.median(removed)) if removed.size else math.nan,
        "median_retained": float(np.median(retained)) if retained.size else math.nan,
        "mean_removed": float(removed.mean()) if removed.size else math.nan,
        "mean_retained": float(retained.mean()) if retained.size else math.nan,
        "q25_removed": float(np.percentile(removed, 25)) if removed.size else math.nan,
        "q75_removed": float(np.percentile(removed, 75)) if removed.size else math.nan,
        "q25_retained": float(np.percentile(retained, 25)) if retained.size else math.nan,
        "q75_retained": float(np.percentile(retained, 75)) if retained.size else math.nan,
    }
    if removed.size == 0 or retained.size == 0:
        out.update(
            {
                "mann_whitney_u": math.nan,
                "mann_whitney_p": math.nan,
                "rank_biserial": math.nan,
                "hodges_lehmann_shift": math.nan,
                "hodges_lehmann_ci_low": math.nan,
                "hodges_lehmann_ci_high": math.nan,
            }
        )
        return out
    test = stats.mannwhitneyu(removed, retained, alternative="two-sided")
    u = float(test.statistic)
    n1, n2 = removed.size, retained.size
    shift = hodges_lehmann(removed, retained, max_pairs=max_pairs, seed=seed)

    def resample(rng: np.random.Generator) -> float:
        left = removed[rng.integers(0, n1, size=n1)]
        right = retained[rng.integers(0, n2, size=n2)]
        return hodges_lehmann(
            left,
            right,
            max_pairs=min(max_pairs, bootstrap_pairs),
            seed=int(rng.integers(1 << 30)),
        )

    ci_low, ci_high = bootstrap_ci(resample, n_boot=n_boot, seed=seed + 1)
    out.update(
        {
            "mann_whitney_u": u,
            "mann_whitney_p": float(test.pvalue),
            "rank_biserial": float(2.0 * u / (n1 * n2) - 1.0),
            "hodges_lehmann_shift": shift,
            "hodges_lehmann_ci_low": ci_low,
            "hodges_lehmann_ci_high": ci_high,
        }
    )
    return out


def weight_strata(weight: np.ndarray, n_bins: int) -> np.ndarray:
    """Exact-zero stratum plus quantile bins of the positive weights."""
    weight = np.asarray(weight, dtype=np.float64)
    strata = np.full(weight.size, -1, dtype=np.int64)
    zero = weight == 0.0
    strata[zero] = 0
    positive = ~zero
    if positive.any():
        values = weight[positive]
        quantiles = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)))
        edges = quantiles[1:-1] if quantiles.size > 2 else np.array([], dtype=np.float64)
        strata[positive] = 1 + np.searchsorted(edges, values, side="right")
    return strata


def stratum_blocks(
    values: np.ndarray, is_removed: np.ndarray, strata: np.ndarray
) -> list[dict[str, Any]]:
    """Per-stratum centered ranks, reused by both the test and its permutation null.

    Ranks do not change when the removed label is permuted inside a stratum, so
    they are computed once.
    """
    blocks: list[dict[str, Any]] = []
    for stratum in np.unique(strata):
        mask = strata == stratum
        labels = is_removed[mask]
        n = int(labels.size)
        n1 = int(labels.sum())
        n2 = n - n1
        if n1 == 0 or n2 == 0 or n < 2:
            continue
        centered = stats.rankdata(values[mask]) - (n + 1.0) / 2.0
        blocks.append(
            {
                "stratum": int(stratum),
                "centered": centered,
                "n": n,
                "n1": n1,
                "n2": n2,
                "weight": 1.0 / (n + 1.0),
                "observed_sum": float(centered[labels].sum()),
                "sum_squares": float(np.sum(centered**2)),
            }
        )
    return blocks


def van_elteren_z(blocks: list[dict[str, Any]], sums: list[float] | None = None) -> float:
    if not blocks:
        return math.nan
    total = 0.0
    variance = 0.0
    for index, block in enumerate(blocks):
        value = block["observed_sum"] if sums is None else sums[index]
        weight = block["weight"]
        n = float(block["n"])
        total += weight * value
        variance += (weight**2) * (block["n1"] * block["n2"] / (n * (n - 1.0))) * block["sum_squares"]
    if variance <= 0.0:
        return math.nan
    return float(total / math.sqrt(variance))


def van_elteren(values: np.ndarray, is_removed: np.ndarray, strata: np.ndarray) -> dict[str, Any]:
    """Stratified Wilcoxon rank-sum with van Elteren design-free weights."""
    blocks = stratum_blocks(values, is_removed, strata)
    z_score = van_elteren_z(blocks)
    return {
        "van_elteren_z": z_score,
        "van_elteren_p": float(2.0 * stats.norm.sf(abs(z_score))) if np.isfinite(z_score) else math.nan,
        "van_elteren_strata_used": int(len(blocks)),
        "van_elteren_removed_used": int(sum(block["n1"] for block in blocks)),
    }


def stratified_permutation_p(
    values: np.ndarray,
    is_removed: np.ndarray,
    strata: np.ndarray,
    n_permutations: int,
    seed: int,
) -> dict[str, Any]:
    """Permute the removed label inside embedding-weight strata."""
    blocks = stratum_blocks(values, is_removed, strata)
    observed = van_elteren_z(blocks)
    if not np.isfinite(observed) or n_permutations <= 0:
        return {
            "permutation_p_two_sided": math.nan,
            "permutation_observed_z": observed,
            "n_permutations": int(max(0, n_permutations)),
        }
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_permutations):
        sums = [
            float(rng.permutation(block["centered"])[: block["n1"]].sum()) for block in blocks
        ]
        z_score = van_elteren_z(blocks, sums)
        if np.isfinite(z_score) and abs(z_score) >= abs(observed):
            extreme += 1
    return {
        "permutation_p_two_sided": float((extreme + 1) / (n_permutations + 1)),
        "permutation_observed_z": float(observed),
        "n_permutations": int(n_permutations),
    }


def nearest_weight_matching(
    removed_weight: np.ndarray,
    retained_weight: np.ndarray,
    caliper_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy 1:1 nearest-weight matching without replacement.

    Removed edges are processed in a seeded random order so the greedy pass is
    not biased by edge ordering, and matches outside the caliper are dropped.
    A zero-weight removed edge can only match a zero-weight retained edge.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(removed_weight.size)
    sort_index = np.argsort(retained_weight, kind="stable")
    sorted_weights = retained_weight[sort_index]
    taken = np.zeros(sorted_weights.size, dtype=bool)
    removed_index: list[int] = []
    retained_index: list[int] = []
    for position in order:
        target = float(removed_weight[position])
        caliper = caliper_fraction * target if target > 0.0 else 0.0
        left = int(np.searchsorted(sorted_weights, target)) - 1
        right = left + 1
        best = -1
        while True:
            left_delta = abs(sorted_weights[left] - target) if left >= 0 else math.inf
            right_delta = (
                abs(sorted_weights[right] - target) if right < sorted_weights.size else math.inf
            )
            if not math.isfinite(left_delta) and not math.isfinite(right_delta):
                break
            if left_delta <= right_delta:
                if left_delta > caliper:
                    break
                if not taken[left]:
                    best = left
                    break
                left -= 1
            else:
                if right_delta > caliper:
                    break
                if not taken[right]:
                    best = right
                    break
                right += 1
        if best < 0:
            continue
        taken[best] = True
        removed_index.append(int(position))
        retained_index.append(int(sort_index[best]))
    return np.array(removed_index, dtype=np.int64), np.array(retained_index, dtype=np.int64)


def paired_summary(
    differences: np.ndarray, label: str, seed: int, n_boot: int
) -> dict[str, Any]:
    differences = np.asarray(differences, dtype=np.float64)
    differences = differences[np.isfinite(differences)]
    out: dict[str, Any] = {
        "outcome": label,
        "n_pairs": int(differences.size),
        "median_difference": float(np.median(differences)) if differences.size else math.nan,
        "mean_difference": float(differences.mean()) if differences.size else math.nan,
        "n_positive": int(np.count_nonzero(differences > 0)),
        "n_negative": int(np.count_nonzero(differences < 0)),
        "n_zero": int(np.count_nonzero(differences == 0)),
    }
    nonzero = differences[differences != 0]
    if nonzero.size >= 1:
        positives = int(np.count_nonzero(nonzero > 0))
        out["sign_test_p"] = float(
            stats.binomtest(positives, nonzero.size, 0.5).pvalue
        )
        out["sign_test_p_removed_greater"] = float(
            stats.binomtest(positives, nonzero.size, 0.5, alternative="greater").pvalue
        )
    else:
        out["sign_test_p"] = math.nan
        out["sign_test_p_removed_greater"] = math.nan
    if differences.size >= 2 and np.any(differences != 0):
        try:
            wilcoxon = stats.wilcoxon(differences, zero_method="wilcox", alternative="two-sided")
            out["wilcoxon_statistic"] = float(wilcoxon.statistic)
            out["wilcoxon_p"] = float(wilcoxon.pvalue)
            wilcoxon_greater = stats.wilcoxon(
                differences, zero_method="wilcox", alternative="greater"
            )
            out["wilcoxon_p_removed_greater"] = float(wilcoxon_greater.pvalue)
        except ValueError:
            out["wilcoxon_statistic"] = math.nan
            out["wilcoxon_p"] = math.nan
            out["wilcoxon_p_removed_greater"] = math.nan
    else:
        out["wilcoxon_statistic"] = math.nan
        out["wilcoxon_p"] = math.nan
        out["wilcoxon_p_removed_greater"] = math.nan
    if differences.size >= 2:
        def resample(rng: np.random.Generator) -> float:
            draw = differences[rng.integers(0, differences.size, size=differences.size)]
            return float(np.median(draw))

        low, high = bootstrap_ci(resample, n_boot=n_boot, seed=seed)
        out["median_difference_ci_low"] = low
        out["median_difference_ci_high"] = high
    else:
        out["median_difference_ci_low"] = math.nan
        out["median_difference_ci_high"] = math.nan
    return out


def standardized_mean_difference(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return math.nan
    pooled = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0) if min(a.size, b.size) > 1 else math.nan
    if not np.isfinite(pooled) or pooled == 0.0:
        return math.nan
    return float((a.mean() - b.mean()) / pooled)


OUTCOMES = [
    ("ml_patristic", "ML patristic distance"),
    ("hamming", "Hamming distance"),
    ("date_delta_days", "absolute collection-date delta (days)"),
]


def analyze_graph(args: argparse.Namespace, graph_key: str) -> dict[str, pd.DataFrame]:
    table = pd.read_csv(args.out_root / "edges" / graph_key / "edge_table.csv.gz")
    for column in ["designed_role", "designed_group_id", "analysis_set", "status"]:
        table[column] = table[column].fillna("").astype(str)
    removed = table[table["is_removed"]]
    retained = table[~table["is_mutually_rejected"]]
    designed_control = table[table["analysis_set"] == "designed_control"]

    unmatched_rows: list[dict[str, Any]] = []
    for comparison, control_frame in [
        ("all_retained_in_panel", retained),
        ("designed_controls_only", designed_control),
    ]:
        for column, label in OUTCOMES:
            summary = two_group_summary(
                removed[column].to_numpy(dtype=np.float64),
                control_frame[column].to_numpy(dtype=np.float64),
                label=column,
                seed=args.analysis_seed,
                n_boot=args.bootstrap_samples,
                max_pairs=args.max_hodges_lehmann_pairs,
                bootstrap_pairs=args.bootstrap_hodges_lehmann_pairs,
            )
            summary.update(
                {
                    "graph": graph_key,
                    "comparison": comparison,
                    "outcome_label": label,
                    "analysis": "unmatched",
                }
            )
            unmatched_rows.append(summary)
        summary = {
            "graph": graph_key,
            "comparison": comparison,
            "analysis": "unmatched",
            "outcome": "same_lineage",
            "outcome_label": "same-lineage edge fraction",
            "n_removed": int(len(removed)),
            "n_retained": int(len(control_frame)),
            "mean_removed": float(removed["same_lineage"].mean()) if len(removed) else math.nan,
            "mean_retained": float(control_frame["same_lineage"].mean())
            if len(control_frame)
            else math.nan,
        }
        unmatched_rows.append(summary)

    strata = weight_strata(table["embedding_weight"].to_numpy(dtype=np.float64), args.weight_bins)
    table = table.assign(weight_stratum=strata)
    analysis_mask = table["is_removed"].to_numpy() | (~table["is_mutually_rejected"].to_numpy())
    stratified_rows: list[dict[str, Any]] = []
    stratum_detail_rows: list[dict[str, Any]] = []
    for column, label in OUTCOMES:
        values = table[column].to_numpy(dtype=np.float64)
        finite = np.isfinite(values) & analysis_mask
        row = {
            "graph": graph_key,
            "analysis": "weight_stratified",
            "outcome": column,
            "outcome_label": label,
            "n_removed": int(np.count_nonzero(finite & table["is_removed"].to_numpy())),
            "n_retained": int(
                np.count_nonzero(finite & (~table["is_mutually_rejected"].to_numpy()))
            ),
            "n_strata": int(np.unique(strata[finite]).size),
        }
        row.update(
            van_elteren(
                values[finite], table["is_removed"].to_numpy()[finite], strata[finite]
            )
        )
        row.update(
            stratified_permutation_p(
                values[finite],
                table["is_removed"].to_numpy()[finite],
                strata[finite],
                n_permutations=args.permutations,
                seed=args.analysis_seed + 7,
            )
        )
        stratified_rows.append(row)
        for stratum in np.unique(strata[finite]):
            mask = finite & (strata == stratum)
            removed_mask = mask & table["is_removed"].to_numpy()
            retained_mask = mask & (~table["is_mutually_rejected"].to_numpy())
            if not removed_mask.any() or not retained_mask.any():
                continue
            stratum_detail_rows.append(
                {
                    "graph": graph_key,
                    "outcome": column,
                    "weight_stratum": int(stratum),
                    "is_zero_weight_stratum": bool(stratum == 0),
                    "weight_min": float(table["embedding_weight"].to_numpy()[mask].min()),
                    "weight_max": float(table["embedding_weight"].to_numpy()[mask].max()),
                    "n_removed": int(removed_mask.sum()),
                    "n_retained": int(retained_mask.sum()),
                    "median_removed": float(np.median(values[removed_mask])),
                    "median_retained": float(np.median(values[retained_mask])),
                    "hodges_lehmann_shift": hodges_lehmann(
                        values[removed_mask],
                        values[retained_mask],
                        max_pairs=args.max_hodges_lehmann_pairs,
                        seed=args.analysis_seed,
                    ),
                }
            )

    matched_rows: list[dict[str, Any]] = []

    # (a) designed within-node pairs: control edges share an endpoint with the removed edge
    removed_with_group = removed[removed["designed_group_id"] != ""]
    if not designed_control.empty and not removed_with_group.empty:
        removed_by_group = removed_with_group.drop_duplicates("designed_group_id").set_index(
            "designed_group_id"
        )
        pairs = designed_control[designed_control["designed_group_id"].isin(removed_by_group.index)]
        for column, label in OUTCOMES:
            control_values = pairs[column].to_numpy(dtype=np.float64)
            removed_values = removed_by_group.loc[
                pairs["designed_group_id"].to_numpy(), column
            ].to_numpy(dtype=np.float64)
            summary = paired_summary(
                removed_values - control_values,
                label=column,
                seed=args.analysis_seed + 3,
                n_boot=args.bootstrap_samples,
            )
            summary.update(
                {
                    "graph": graph_key,
                    "analysis": "within_node_designed_pairs",
                    "outcome_label": label,
                    "matched_on": "shared endpoint plus nearest embedding weight",
                }
            )
            matched_rows.append(summary)
        removed_weights = removed_by_group.loc[
            pairs["designed_group_id"].to_numpy(), "embedding_weight"
        ].to_numpy(dtype=np.float64)
        control_weights = pairs["embedding_weight"].to_numpy(dtype=np.float64)
        matched_rows.append(
            {
                "graph": graph_key,
                "analysis": "within_node_designed_pairs",
                "outcome": "embedding_weight_balance",
                "outcome_label": "embedding weight balance (should be near zero)",
                "n_pairs": int(len(pairs)),
                "median_difference": float(np.median(removed_weights - control_weights)),
                "mean_difference": float(np.mean(removed_weights - control_weights)),
                "standardized_mean_difference": standardized_mean_difference(
                    removed_weights, control_weights
                ),
                "fraction_within_caliper": float(pairs["designed_within_caliper"].mean()),
            }
        )

    # (b) caliper matching against every retained in-panel edge
    removed_weight = removed["embedding_weight"].to_numpy(dtype=np.float64)
    retained_weight = retained["embedding_weight"].to_numpy(dtype=np.float64)
    left_index, right_index = nearest_weight_matching(
        removed_weight,
        retained_weight,
        caliper_fraction=args.caliper_fraction,
        seed=args.analysis_seed + 11,
    )
    if left_index.size:
        for column, label in OUTCOMES:
            differences = (
                removed[column].to_numpy(dtype=np.float64)[left_index]
                - retained[column].to_numpy(dtype=np.float64)[right_index]
            )
            summary = paired_summary(
                differences,
                label=column,
                seed=args.analysis_seed + 5,
                n_boot=args.bootstrap_samples,
            )
            summary.update(
                {
                    "graph": graph_key,
                    "analysis": "caliper_matched_pairs",
                    "outcome_label": label,
                    "matched_on": f"nearest embedding weight within {args.caliper_fraction:.0%} caliper",
                }
            )
            matched_rows.append(summary)
        matched_rows.append(
            {
                "graph": graph_key,
                "analysis": "caliper_matched_pairs",
                "outcome": "embedding_weight_balance",
                "outcome_label": "embedding weight balance (should be near zero)",
                "n_pairs": int(left_index.size),
                "median_difference": float(
                    np.median(removed_weight[left_index] - retained_weight[right_index])
                ),
                "mean_difference": float(
                    np.mean(removed_weight[left_index] - retained_weight[right_index])
                ),
                "standardized_mean_difference": standardized_mean_difference(
                    removed_weight[left_index], retained_weight[right_index]
                ),
                "matched_fraction_of_removed": float(left_index.size / max(1, removed_weight.size)),
            }
        )

    # (c) f_j dose-response
    dose_rows: list[dict[str, Any]] = []
    rejected = table[table["is_mutually_rejected"]]
    for frame, name, score_column in [
        (rejected, "mutually_rejected_edges", "max_f_before"),
        (table[table["is_directional_candidate"]], "directional_candidate_edges", "candidate_f_before"),
    ]:
        for column, label in OUTCOMES:
            values = frame[column].to_numpy(dtype=np.float64)
            scores = frame[score_column].to_numpy(dtype=np.float64)
            finite = np.isfinite(values) & np.isfinite(scores)
            if np.count_nonzero(finite) < 8:
                continue
            result = stats.spearmanr(scores[finite], values[finite])
            dose_rows.append(
                {
                    "graph": graph_key,
                    "analysis": "f_dose_response",
                    "edge_set": name,
                    "score": score_column,
                    "outcome": column,
                    "outcome_label": label,
                    "n_edges": int(np.count_nonzero(finite)),
                    "spearman_rho": float(result.statistic),
                    "spearman_p": float(result.pvalue),
                }
            )

    return {
        "unmatched": pd.DataFrame(unmatched_rows),
        "weight_stratified": pd.DataFrame(stratified_rows),
        "weight_strata_detail": pd.DataFrame(stratum_detail_rows),
        "matched": pd.DataFrame(matched_rows),
        "dose_response": pd.DataFrame(dose_rows),
    }


def stage_analyze(args: argparse.Namespace, graph_keys: list[str]) -> None:
    collected: dict[str, list[pd.DataFrame]] = {}
    for graph_key in graph_keys:
        log(f"Analyzing edges: {graph_key}")
        results = analyze_graph(args, graph_key)
        out_dir = args.out_root / "edges" / graph_key
        for name, frame in results.items():
            frame.to_csv(out_dir / f"{name}.csv", index=False)
            collected.setdefault(name, []).append(frame)
    summary_dir = args.out_root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    for name, frames in collected.items():
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(summary_dir / f"edge_validation_{name}.csv", index=False)
        log(f"Wrote {summary_dir / f'edge_validation_{name}.csv'}")
    write_json(
        summary_dir / "edge_validation_manifest.json",
        {
            "summarized_at_unix": time.time(),
            "graphs": graph_keys,
            "outcomes": [column for column, _ in OUTCOMES],
            "analysis_seed": int(args.analysis_seed),
            "bootstrap_samples": int(args.bootstrap_samples),
            "permutations": int(args.permutations),
            "weight_bins": int(args.weight_bins),
            "caliper_fraction": float(args.caliper_fraction),
            "hypothesis": (
                "removed edges have larger ML patristic separation than retained edges at "
                "comparable embedding distance"
            ),
            "primary_readouts": [
                "matched/within_node_designed_pairs/ml_patristic/median_difference",
                "matched/caliper_matched_pairs/ml_patristic/median_difference",
                "weight_stratified/ml_patristic/van_elteren_p",
            ],
            "fingerprint": stable_fingerprint(
                {
                    "graphs": graph_keys,
                    "analysis_seed": int(args.analysis_seed),
                    "weight_bins": int(args.weight_bins),
                    "caliper_fraction": float(args.caliper_fraction),
                }
            ),
        },
    )


def stage_summarize(args: argparse.Namespace, graph_keys: list[str]) -> None:
    summary_dir = args.out_root / "summaries"
    headline_rows: list[dict[str, Any]] = []
    matched_path = summary_dir / "edge_validation_matched.csv"
    stratified_path = summary_dir / "edge_validation_weight_stratified.csv"
    if not matched_path.exists() or not stratified_path.exists():
        raise FileNotFoundError("Missing analysis outputs; run --stages analyze first")
    matched = pd.read_csv(matched_path)
    stratified = pd.read_csv(stratified_path)
    for graph_key in graph_keys:
        row: dict[str, Any] = {"graph": graph_key, "display_name": GRAPH_SPECS[graph_key].display_name}
        manifest_path = args.out_root / "edges" / graph_key / "edge_table_manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            row["n_edges_in_panel"] = manifest["n_edges_in_panel"]
            row["n_removed_in_panel"] = manifest["n_removed"]
        for analysis in ["within_node_designed_pairs", "caliper_matched_pairs"]:
            block = matched[
                (matched["graph"] == graph_key)
                & (matched["analysis"] == analysis)
                & (matched["outcome"] == "ml_patristic")
            ]
            if not block.empty:
                record = block.iloc[0]
                row[f"{analysis}_n_pairs"] = record.get("n_pairs")
                row[f"{analysis}_median_patristic_difference"] = record.get("median_difference")
                row[f"{analysis}_ci_low"] = record.get("median_difference_ci_low")
                row[f"{analysis}_ci_high"] = record.get("median_difference_ci_high")
                row[f"{analysis}_wilcoxon_p"] = record.get("wilcoxon_p")
        block = stratified[
            (stratified["graph"] == graph_key) & (stratified["outcome"] == "ml_patristic")
        ]
        if not block.empty:
            record = block.iloc[0]
            row["van_elteren_z"] = record.get("van_elteren_z")
            row["van_elteren_p"] = record.get("van_elteren_p")
            row["permutation_p_two_sided"] = record.get("permutation_p_two_sided")
        headline_rows.append(row)
    headline = pd.DataFrame(headline_rows)
    headline_path = summary_dir / "edge_validation_headline.csv"
    headline.to_csv(headline_path, index=False)
    log(f"Wrote headline table: {headline_path}")
    print(headline.to_string(index=False))


def run_stages(args: argparse.Namespace) -> None:
    graph_keys = parse_graph_priority(args.graphs)
    stages = {stage.strip() for stage in args.stages.split(",") if stage.strip()}
    if "all" in stages:
        stages = set(STAGES)
    unknown = stages.difference(STAGES)
    if unknown:
        raise ValueError(f"Unknown stage(s): {sorted(unknown)}; allowed={STAGES}")
    if "patristic" in stages:
        stage_patristic(args)
    if "edge-metrics" in stages:
        stage_edge_metrics(args, graph_keys)
    if "analyze" in stages:
        stage_analyze(args, graph_keys)
    if "summarize" in stages:
        stage_summarize(args, graph_keys)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel-root", type=Path, default=DEFAULT_PANEL_ROOT)
    ap.add_argument("--directional-root", type=Path, default=DEFAULT_DIRECTIONAL_ROOT)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--candidate-label", default=DEFAULT_CANDIDATE_LABEL)
    ap.add_argument("--treefile", type=Path, default=None)
    ap.add_argument("--stages", default="all")
    ap.add_argument("--graphs", default="rng,knn5,knn50")
    ap.add_argument("--patristic-block-size", type=int, default=128)
    ap.add_argument("--weight-bins", type=int, default=10)
    ap.add_argument("--caliper-fraction", type=float, default=0.25)
    ap.add_argument("--bootstrap-samples", type=int, default=2000)
    ap.add_argument("--permutations", type=int, default=2000)
    ap.add_argument("--max-hodges-lehmann-pairs", type=int, default=5_000_000)
    ap.add_argument("--bootstrap-hodges-lehmann-pairs", type=int, default=20_000)
    ap.add_argument("--analysis-seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    run_stages(args)


if __name__ == "__main__":
    main()
