#!/usr/bin/env python3
"""Reference ML-tree panel construction for seed-42 directional edge validation.

The seed-42 frozen 2k panel contains both endpoints of almost no removed edge
(knn5=10, knn50=30, rng=4 of 1142/4070/726).  A reference tree that can test
removed-vs-retained edges therefore has to be built on a tip set that is chosen
to cover those edges.

This workflow:

1. validate-inputs   - checks graphs, directional decisions, alignment, frozen 2k
2. select-edges      - removed edges plus embedding-distance-matched retained
                       controls that share an endpoint with them
3. build-panel       - packs edge groups into a tip panel under a unique-sequence
                       budget, seeded with the frozen 2k tips
4. write-alignment   - panel alignment, exact duplicate collapsing, unique FASTA
5. emit-command      - the IQ-TREE command line for the unique alignment

The IQ-TREE run itself is deliberately not launched here; it is the only
long-running step and is run manually in tmux.

The unique-sequence budget, not the tip count, is the cost driver: the 20k panel
holds 20,000 tips but only 8,830 distinct aligned spike sequences, and the frozen
2k holds only 1,087.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_PANEL_ROOT = Path(
    "analysis/cohort_validation/07_sampling_design_20k/random_full_dataset_seed42/seed_42"
)
DEFAULT_DIRECTIONAL_ROOT = Path(
    "analysis/cohort_validation/24_seed42_20k_directional_intrinsic_distances/"
    "random_full_dataset_seed42/seed_42"
)
DEFAULT_FROZEN_2K_ROOT = Path(
    "analysis/cohort_validation/25_seed42_2k_paired_tree_geometry/"
    "random_full_dataset_seed42/seed_42"
)
DEFAULT_OUT_ROOT = Path(
    "analysis/cohort_validation/26_seed42_reference_ml_tree_edge_validation/"
    "random_full_dataset_seed42/seed_42"
)
DEFAULT_CANDIDATE_LABEL = "candidate_0p1_delta_0p01"

STAGES = ["validate-inputs", "select-edges", "build-panel", "write-alignment", "emit-command"]


@dataclass(frozen=True)
class GraphSpec:
    key: str
    display_name: str
    graph_dir_name: str


GRAPH_SPECS: dict[str, GraphSpec] = {
    "rng": GraphSpec("rng", "ESM-2 exact RNG", "embedding_rng_exact"),
    "knn5": GraphSpec("knn5", "ESM-2 kNN-5", "embedding_knn_k05"),
    "knn50": GraphSpec("knn50", "ESM-2 kNN-50", "embedding_knn_k50"),
}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def stable_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_graph_priority(value: str) -> list[str]:
    keys = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(keys).difference(GRAPH_SPECS))
    if unknown:
        raise ValueError(f"Unknown graph key(s) {unknown}; allowed={sorted(GRAPH_SPECS)}")
    if not keys:
        raise ValueError("At least one graph key is required")
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate graph key in --graph-priority")
    return keys


# ---------------------------------------------------------------------------
# input loading
# ---------------------------------------------------------------------------


def graph_dir(args: argparse.Namespace, graph_key: str) -> Path:
    spec = GRAPH_SPECS[graph_key]
    return (
        args.panel_root
        / "graphs/esm2_650M/cityblock/pool_n20000"
        / spec.graph_dir_name
    )


def decisions_path(args: argparse.Namespace, graph_key: str) -> Path:
    return (
        args.directional_root
        / "work"
        / args.candidate_label
        / graph_key
        / "mutual_rejection_queue.csv.gz"
    )


def canonical_nodes_path(args: argparse.Namespace) -> Path:
    return args.panel_root / "graphs/esm2_650M/cityblock/pool_n20000/canonical_nodes.csv"


def alignment_path(args: argparse.Namespace) -> Path:
    return args.panel_root / "inputs/pool_n20000/spike_sequences_aligned_mafft.fasta"


def load_canonical_nodes(args: argparse.Namespace) -> pd.DataFrame:
    path = canonical_nodes_path(args)
    nodes = pd.read_csv(
        path,
        usecols=[
            "node_id",
            "accession",
            "collection_date",
            "lineage",
            "cohort_id",
            "cohort_name",
        ],
        low_memory=False,
    )
    nodes = nodes.sort_values("node_id").reset_index(drop=True)
    expected = np.arange(len(nodes), dtype=np.int64)
    if not np.array_equal(nodes["node_id"].to_numpy(dtype=np.int64), expected):
        raise ValueError(f"{path}: node_id is not row-aligned 0..n-1")
    nodes["accession"] = nodes["accession"].astype(str).str.strip()
    if nodes["accession"].duplicated().any():
        raise ValueError(f"{path}: accession values are not unique")
    return nodes


def load_edges(args: argparse.Namespace, graph_key: str) -> pd.DataFrame:
    path = graph_dir(args, graph_key) / "edges.csv"
    edges = pd.read_csv(path, usecols=["source", "target", "weight"])
    source = edges["source"].to_numpy(dtype=np.int64)
    target = edges["target"].to_numpy(dtype=np.int64)
    if np.any(source >= target):
        raise ValueError(f"{path}: expected canonical source < target ordering")
    weight = edges["weight"].to_numpy(dtype=np.float64)
    if not np.isfinite(weight).all() or np.any(weight < 0):
        raise ValueError(f"{path}: non-finite or negative edge weight")
    order = np.lexsort((target, source))
    return pd.DataFrame(
        {"source": source[order], "target": target[order], "weight": weight[order]}
    )


def load_decisions(args: argparse.Namespace, graph_key: str) -> pd.DataFrame:
    path = decisions_path(args, graph_key)
    decisions = pd.read_csv(path)
    required = {"source", "target", "max_f_before", "max_rank_fraction", "global_decision"}
    missing = required.difference(decisions.columns)
    if missing:
        raise ValueError(f"{path}: missing column(s) {sorted(missing)}")
    source = decisions["source"].to_numpy(dtype=np.int64)
    target = decisions["target"].to_numpy(dtype=np.int64)
    lo = np.minimum(source, target)
    hi = np.maximum(source, target)
    decisions = decisions.assign(source=lo, target=hi)
    allowed = {"removed", "retained_for_connectivity"}
    observed = set(decisions["global_decision"].astype(str).unique())
    unexpected = observed.difference(allowed)
    if unexpected:
        raise ValueError(f"{path}: unexpected global_decision value(s) {sorted(unexpected)}")
    order = np.lexsort((decisions["target"].to_numpy(), decisions["source"].to_numpy()))
    return decisions.iloc[order].reset_index(drop=True)


def edge_keys(source: np.ndarray, target: np.ndarray, n_nodes: int) -> np.ndarray:
    return source.astype(np.int64) * np.int64(n_nodes) + target.astype(np.int64)


def read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    header: str | None = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n").rstrip("\r")
            if line.startswith(">"):
                if header is not None:
                    records[header] = "".join(chunks)
                header = line[1:].strip().split()[0]
                chunks = []
            elif line:
                chunks.append(line)
    if header is not None:
        records[header] = "".join(chunks)
    if not records:
        raise ValueError(f"{path}: no FASTA records")
    return records


# ---------------------------------------------------------------------------
# edge selection and matching
# ---------------------------------------------------------------------------


def build_adjacency(
    source: np.ndarray, target: np.ndarray, n_nodes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CSR-style undirected adjacency over directed halves.

    Returns (indptr, neighbor, edge_index) where edge_index points back into the
    undirected edge arrays.
    """
    directed_node = np.concatenate([source, target])
    directed_neighbor = np.concatenate([target, source])
    directed_edge = np.concatenate([np.arange(source.size), np.arange(source.size)])
    order = np.lexsort((directed_neighbor, directed_node))
    directed_node = directed_node[order]
    directed_neighbor = directed_neighbor[order]
    directed_edge = directed_edge[order]
    counts = np.bincount(directed_node, minlength=n_nodes)
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return indptr, directed_neighbor, directed_edge


def caliper_ok(weight_removed: float, weight_control: float, caliper_fraction: float) -> bool:
    if weight_removed == 0.0 or weight_control == 0.0:
        return weight_removed == weight_control
    return abs(weight_control - weight_removed) <= caliper_fraction * weight_removed


def match_control_edges(
    removed_edge_indices: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    mutually_rejected: np.ndarray,
    indptr: np.ndarray,
    neighbor: np.ndarray,
    edge_index: np.ndarray,
    caliper_fraction: float,
    controls_per_endpoint: int,
) -> list[dict[str, Any]]:
    """Pick retained edges that share an endpoint with each removed edge.

    Controls are the retained incident edges whose embedding weight is closest to
    the removed edge's weight.  Every control edge is used at most once so the
    matched-pair analysis stays a genuine paired design.
    """
    used: set[int] = set()
    matches: list[dict[str, Any]] = []
    for removed_index in removed_edge_indices:
        w_removed = float(weight[removed_index])
        for endpoint_role, node in (
            ("source", int(source[removed_index])),
            ("target", int(target[removed_index])),
        ):
            start = int(indptr[node])
            stop = int(indptr[node + 1])
            local_edges = edge_index[start:stop]
            local_neighbors = neighbor[start:stop]
            eligible = ~mutually_rejected[local_edges]
            if not eligible.any():
                continue
            local_edges = local_edges[eligible]
            local_neighbors = local_neighbors[eligible]
            deltas = np.abs(weight[local_edges] - w_removed)
            order = np.lexsort((local_neighbors, deltas))
            taken = 0
            for position in order:
                control_index = int(local_edges[position])
                if control_index in used:
                    continue
                delta = float(deltas[position])
                used.add(control_index)
                matches.append(
                    {
                        "removed_edge_index": int(removed_index),
                        "control_edge_index": control_index,
                        "shared_node_id": node,
                        "shared_endpoint_role": endpoint_role,
                        "control_partner_node_id": int(local_neighbors[position]),
                        "removed_weight": w_removed,
                        "control_weight": float(weight[control_index]),
                        "weight_abs_delta": delta,
                        "within_caliper": bool(
                            caliper_ok(w_removed, float(weight[control_index]), caliper_fraction)
                        ),
                    }
                )
                taken += 1
                if taken >= controls_per_endpoint:
                    break
    return matches


def select_edges(args: argparse.Namespace, graph_keys: list[str]) -> pd.DataFrame:
    canonical = load_canonical_nodes(args)
    n_nodes = len(canonical)
    frames: list[pd.DataFrame] = []
    stats: dict[str, Any] = {}
    for graph_key in graph_keys:
        edges = load_edges(args, graph_key)
        source = edges["source"].to_numpy(dtype=np.int64)
        target = edges["target"].to_numpy(dtype=np.int64)
        weight = edges["weight"].to_numpy(dtype=np.float64)
        if source.max() >= n_nodes or target.max() >= n_nodes:
            raise ValueError(f"{graph_key}: edge endpoint outside canonical node range")
        keys = edge_keys(source, target, n_nodes)
        if np.any(np.diff(keys) <= 0):
            raise ValueError(f"{graph_key}: duplicate or unsorted undirected edges")

        decisions = load_decisions(args, graph_key)
        decision_keys = edge_keys(
            decisions["source"].to_numpy(dtype=np.int64),
            decisions["target"].to_numpy(dtype=np.int64),
            n_nodes,
        )
        positions = np.searchsorted(keys, decision_keys)
        if np.any(positions >= keys.size) or np.any(keys[np.minimum(positions, keys.size - 1)] != decision_keys):
            raise ValueError(f"{graph_key}: a directional decision refers to a non-existent baseline edge")
        mutually_rejected = np.zeros(keys.size, dtype=bool)
        mutually_rejected[positions] = True
        is_removed = np.zeros(keys.size, dtype=bool)
        removed_mask = decisions["global_decision"].to_numpy(dtype=object) == "removed"
        is_removed[positions[removed_mask]] = True

        removed_indices = np.flatnonzero(is_removed)
        rng = np.random.default_rng(
            int.from_bytes(
                hashlib.sha256(f"{args.selection_seed}:{graph_key}".encode("utf-8")).digest()[:8],
                "little",
            )
            % (2**63 - 1)
        )
        removed_indices = removed_indices[rng.permutation(removed_indices.size)]

        indptr, neighbor, edge_index = build_adjacency(source, target, n_nodes)
        matches = match_control_edges(
            removed_edge_indices=removed_indices,
            source=source,
            target=target,
            weight=weight,
            mutually_rejected=mutually_rejected,
            indptr=indptr,
            neighbor=neighbor,
            edge_index=edge_index,
            caliper_fraction=args.caliper_fraction,
            controls_per_endpoint=args.controls_per_endpoint,
        )
        match_frame = pd.DataFrame(matches)

        rows: list[dict[str, Any]] = []
        for group_rank, removed_index in enumerate(removed_indices):
            group_id = f"{graph_key}:{int(source[removed_index])}-{int(target[removed_index])}"
            rows.append(
                {
                    "graph": graph_key,
                    "group_id": group_id,
                    "group_rank": int(group_rank),
                    "role": "removed",
                    "edge_index": int(removed_index),
                    "source": int(source[removed_index]),
                    "target": int(target[removed_index]),
                    "weight": float(weight[removed_index]),
                    "shared_node_id": -1,
                    "shared_endpoint_role": "",
                    "weight_abs_delta": 0.0,
                    "within_caliper": True,
                }
            )
            if match_frame.empty:
                continue
            group_matches = match_frame[match_frame["removed_edge_index"] == int(removed_index)]
            for _, match in group_matches.iterrows():
                control_index = int(match["control_edge_index"])
                rows.append(
                    {
                        "graph": graph_key,
                        "group_id": group_id,
                        "group_rank": int(group_rank),
                        "role": "control",
                        "edge_index": control_index,
                        "source": int(source[control_index]),
                        "target": int(target[control_index]),
                        "weight": float(weight[control_index]),
                        "shared_node_id": int(match["shared_node_id"]),
                        "shared_endpoint_role": str(match["shared_endpoint_role"]),
                        "weight_abs_delta": float(match["weight_abs_delta"]),
                        "within_caliper": bool(match["within_caliper"]),
                    }
                )
        frame = pd.DataFrame(rows)
        frames.append(frame)
        stats[graph_key] = {
            "n_baseline_edges": int(keys.size),
            "n_mutually_rejected": int(mutually_rejected.sum()),
            "n_removed": int(is_removed.sum()),
            "n_control_edges": int(0 if match_frame.empty else len(match_frame)),
            "n_controls_within_caliper": int(
                0 if match_frame.empty else int(match_frame["within_caliper"].sum())
            ),
            "median_control_weight_abs_delta": float(
                match_frame["weight_abs_delta"].median() if not match_frame.empty else math.nan
            ),
            "edges_signature": file_signature(graph_dir(args, graph_key) / "edges.csv"),
            "decisions_signature": file_signature(decisions_path(args, graph_key)),
        }
        log(
            f"{graph_key}: removed={stats[graph_key]['n_removed']:,} "
            f"controls={stats[graph_key]['n_control_edges']:,} "
            f"within_caliper={stats[graph_key]['n_controls_within_caliper']:,}"
        )

    selected = pd.concat(frames, ignore_index=True)
    design_dir = args.out_root / "design"
    design_dir.mkdir(parents=True, exist_ok=True)
    selected_path = design_dir / "selected_edge_groups.csv.gz"
    selected.to_csv(selected_path, index=False)
    write_json(
        design_dir / "edge_selection_manifest.json",
        {
            "selected_at_unix": time.time(),
            "graph_priority": graph_keys,
            "candidate_label": args.candidate_label,
            "selection_seed": int(args.selection_seed),
            "caliper_fraction": float(args.caliper_fraction),
            "controls_per_endpoint": int(args.controls_per_endpoint),
            "control_policy": (
                "nearest embedding weight among retained edges incident to a removed edge "
                "endpoint; each control edge is used at most once"
            ),
            "selected_edge_groups": str(selected_path),
            "graph_stats": stats,
        },
    )
    log(f"Wrote edge selection: {selected_path}")
    return selected


# ---------------------------------------------------------------------------
# panel packing
# ---------------------------------------------------------------------------


def sequence_group_ids(accessions: Sequence[str], records: dict[str, str]) -> np.ndarray:
    """Map every node to an exact aligned-sequence group id."""
    digest_to_group: dict[str, int] = {}
    groups = np.empty(len(accessions), dtype=np.int64)
    for index, accession in enumerate(accessions):
        sequence = records.get(accession)
        if sequence is None:
            raise KeyError(f"accession {accession} missing from panel alignment")
        digest = hashlib.sha256(sequence.upper().encode("utf-8")).hexdigest()
        if digest not in digest_to_group:
            digest_to_group[digest] = len(digest_to_group)
        groups[index] = digest_to_group[digest]
    return groups


def plan_panel(
    group_nodes: list[tuple[str, str, list[int]]],
    sequence_group_of_node: np.ndarray,
    seed_nodes: Iterable[int],
    graph_priority: list[str],
    max_unique_sequences: int,
    max_tips: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Round-robin pack edge groups into a tip panel under a unique-sequence budget.

    `group_nodes` holds `(graph_key, group_id, node_ids)` in per-graph order.  The
    budget counts distinct aligned sequences because that, not the tip count, is
    what an ML tree search actually pays for.
    """
    panel: set[int] = set(int(node) for node in seed_nodes)
    unique: set[int] = {int(sequence_group_of_node[node]) for node in panel}
    per_graph: dict[str, list[tuple[str, list[int]]]] = {key: [] for key in graph_priority}
    for graph_key, group_id, nodes in group_nodes:
        per_graph.setdefault(graph_key, []).append((group_id, nodes))
    cursors = {key: 0 for key in per_graph}
    decisions: list[dict[str, Any]] = []
    active = True
    while active:
        active = False
        for graph_key in graph_priority:
            queue = per_graph.get(graph_key, [])
            cursor = cursors.get(graph_key, 0)
            if cursor >= len(queue):
                continue
            active = True
            group_id, nodes = queue[cursor]
            cursors[graph_key] = cursor + 1
            new_nodes = [int(node) for node in nodes if int(node) not in panel]
            new_unique = {
                int(sequence_group_of_node[node])
                for node in new_nodes
                if int(sequence_group_of_node[node]) not in unique
            }
            accepted = (
                len(unique) + len(new_unique) <= max_unique_sequences
                and len(panel) + len(new_nodes) <= max_tips
            )
            if accepted:
                panel.update(new_nodes)
                unique.update(new_unique)
            decisions.append(
                {
                    "graph": graph_key,
                    "group_id": group_id,
                    "accepted": bool(accepted),
                    "n_new_tips": int(len(new_nodes)),
                    "n_new_unique_sequences": int(len(new_unique)),
                    "panel_tips_after": int(len(panel)),
                    "panel_unique_sequences_after": int(len(unique)),
                }
            )
    return sorted(panel), decisions


def build_panel(args: argparse.Namespace, graph_keys: list[str]) -> pd.DataFrame:
    design_dir = args.out_root / "design"
    selected_path = design_dir / "selected_edge_groups.csv.gz"
    if not selected_path.exists():
        raise FileNotFoundError("Missing selected_edge_groups.csv.gz; run --stages select-edges first")
    selected = pd.read_csv(selected_path)
    canonical = load_canonical_nodes(args)
    records = read_fasta(alignment_path(args))
    sequence_group_of_node = sequence_group_ids(canonical["accession"].tolist(), records)

    frozen_nodes: list[int] = []
    if args.include_frozen_2k:
        frozen_path = args.frozen_2k_root / "design/selected_tips.csv"
        frozen = pd.read_csv(frozen_path, usecols=["node_id", "accession"])
        frozen_nodes = frozen["node_id"].astype(int).tolist()
        canonical_accessions = canonical["accession"].to_numpy()
        if not np.array_equal(
            canonical_accessions[np.array(frozen_nodes, dtype=np.int64)],
            frozen["accession"].astype(str).str.strip().to_numpy(),
        ):
            raise ValueError(f"{frozen_path}: node_id/accession disagree with canonical nodes")

    group_nodes: list[tuple[str, str, list[int]]] = []
    grouped = selected.sort_values(["graph", "group_rank", "role", "source", "target"], kind="stable")
    for (graph_key, _group_rank, group_id), block in grouped.groupby(
        ["graph", "group_rank", "group_id"], sort=False
    ):
        nodes = sorted(
            set(block["source"].astype(int).tolist()) | set(block["target"].astype(int).tolist())
        )
        group_nodes.append((str(graph_key), str(group_id), nodes))

    panel_node_ids, decisions = plan_panel(
        group_nodes=group_nodes,
        sequence_group_of_node=sequence_group_of_node,
        seed_nodes=frozen_nodes,
        graph_priority=graph_keys,
        max_unique_sequences=args.max_unique_sequences,
        max_tips=args.max_tips,
    )

    panel = canonical.iloc[np.array(panel_node_ids, dtype=np.int64)].reset_index(drop=True)
    panel.insert(0, "panel_row", np.arange(len(panel), dtype=int))
    panel["sequence_group_id"] = sequence_group_of_node[np.array(panel_node_ids, dtype=np.int64)]
    decisions_frame = pd.DataFrame(decisions)
    accepted_groups = set(
        decisions_frame.loc[decisions_frame["accepted"], "group_id"].astype(str).tolist()
    )
    core_set = {
        node
        for _graph_key, group_id, nodes in group_nodes
        if group_id in accepted_groups
        for node in nodes
    }
    panel["in_frozen_2k"] = panel["node_id"].isin(set(frozen_nodes))
    panel["in_edge_core"] = panel["node_id"].isin(core_set)

    covered = selected[selected["group_id"].astype(str).isin(accepted_groups)].copy()
    coverage_rows = []
    for graph_key in graph_keys:
        block = covered[covered["graph"] == graph_key]
        all_block = selected[selected["graph"] == graph_key]
        coverage_rows.append(
            {
                "graph": graph_key,
                "n_removed_edges_total": int((all_block["role"] == "removed").sum()),
                "n_removed_edges_in_panel": int((block["role"] == "removed").sum()),
                "n_control_edges_total": int((all_block["role"] == "control").sum()),
                "n_control_edges_in_panel": int((block["role"] == "control").sum()),
                "n_groups_total": int(all_block["group_id"].nunique()),
                "n_groups_in_panel": int(block["group_id"].nunique()),
            }
        )
    coverage = pd.DataFrame(coverage_rows)

    design_dir.mkdir(parents=True, exist_ok=True)
    panel_path = design_dir / "panel_tips.csv"
    panel.to_csv(panel_path, index=False)
    decisions_frame.to_csv(design_dir / "panel_packing_decisions.csv.gz", index=False)
    coverage.to_csv(design_dir / "panel_edge_coverage.csv", index=False)
    covered.to_csv(design_dir / "panel_edge_groups.csv.gz", index=False)

    fingerprint_payload = {
        "panel_node_ids": [int(node) for node in panel_node_ids],
        "graph_priority": graph_keys,
        "selection_seed": int(args.selection_seed),
        "max_unique_sequences": int(args.max_unique_sequences),
        "max_tips": int(args.max_tips),
        "include_frozen_2k": bool(args.include_frozen_2k),
        "caliper_fraction": float(args.caliper_fraction),
        "controls_per_endpoint": int(args.controls_per_endpoint),
    }
    manifest = {
        "prepared_at_unix": time.time(),
        "panel_tips": str(panel_path),
        "n_tips": int(len(panel)),
        "n_unique_sequences": int(panel["sequence_group_id"].nunique()),
        "n_frozen_2k_tips": int(panel["in_frozen_2k"].sum()),
        "n_edge_core_tips": int(panel["in_edge_core"].sum()),
        "include_frozen_2k": bool(args.include_frozen_2k),
        "max_unique_sequences": int(args.max_unique_sequences),
        "max_tips": int(args.max_tips),
        "graph_priority": graph_keys,
        "packing_policy": "round-robin over graphs in priority order under a unique-sequence budget",
        "panel_fingerprint": stable_fingerprint(fingerprint_payload),
        "edge_coverage": coverage.to_dict(orient="records"),
        "alignment_signature": file_signature(alignment_path(args)),
        "canonical_nodes_signature": file_signature(canonical_nodes_path(args)),
    }
    write_json(design_dir / "panel_manifest.json", manifest)
    log(
        f"Panel: {manifest['n_tips']:,} tips / {manifest['n_unique_sequences']:,} unique sequences "
        f"({manifest['n_frozen_2k_tips']:,} frozen-2k tips)"
    )
    for row in coverage.to_dict(orient="records"):
        log(
            f"  {row['graph']}: removed edges in panel "
            f"{row['n_removed_edges_in_panel']:,}/{row['n_removed_edges_total']:,}, "
            f"matched controls {row['n_control_edges_in_panel']:,}"
        )
    return panel


# ---------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------


def dedup_sequences(pairs: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], pd.DataFrame]:
    """Collapse exactly identical aligned sequences.

    The representative is the lexicographically smallest accession so the choice
    does not depend on input order.
    """
    by_sequence: dict[str, list[str]] = {}
    for accession, sequence in pairs:
        by_sequence.setdefault(sequence.upper(), []).append(accession)
    rows: list[dict[str, Any]] = []
    unique: list[tuple[str, str]] = []
    for group_index, sequence in enumerate(sorted(by_sequence)):
        members = sorted(by_sequence[sequence])
        representative = members[0]
        unique.append((representative, sequence))
        for accession in members:
            rows.append(
                {
                    "duplicate_group": group_index,
                    "accession": accession,
                    "representative_accession": representative,
                    "group_size": len(members),
                    "is_representative": accession == representative,
                }
            )
    unique.sort(key=lambda item: item[0])
    return unique, pd.DataFrame(rows).sort_values("accession").reset_index(drop=True)


def alignment_qc(unique: list[tuple[str, str]]) -> dict[str, Any]:
    lengths = {len(sequence) for _, sequence in unique}
    if len(lengths) != 1:
        raise ValueError(f"panel alignment is ragged; observed lengths={sorted(lengths)}")
    width = lengths.pop()
    matrix = np.frombuffer("".join(sequence for _, sequence in unique).encode("ascii"), dtype="S1")
    matrix = matrix.reshape(len(unique), width)
    gap = (matrix == b"-") | (matrix == b".")
    constant = np.zeros(width, dtype=bool)
    for column in range(width):
        values = np.unique(matrix[:, column])
        constant[column] = values.size == 1
    return {
        "n_unique_sequences": int(len(unique)),
        "alignment_width": int(width),
        "gap_fraction": float(gap.mean()),
        "n_constant_columns": int(constant.sum()),
        "n_all_gap_columns": int(np.count_nonzero(gap.all(axis=0))),
        "max_per_sequence_gap_fraction": float(gap.mean(axis=1).max()),
    }


def write_alignment(args: argparse.Namespace) -> dict[str, Any]:
    panel_path = args.out_root / "design/panel_tips.csv"
    if not panel_path.exists():
        raise FileNotFoundError("Missing panel_tips.csv; run --stages build-panel first")
    panel = pd.read_csv(panel_path)
    records = read_fasta(alignment_path(args))
    pairs = [(str(accession), records[str(accession)]) for accession in panel["accession"]]
    unique, duplicate_groups = dedup_sequences(pairs)

    alignment_dir = args.out_root / "alignment"
    alignment_dir.mkdir(parents=True, exist_ok=True)
    unique_path = alignment_dir / "panel_unique.fasta"
    with unique_path.open("w", encoding="utf-8") as handle:
        for accession, sequence in unique:
            handle.write(f">{accession}\n")
            for start in range(0, len(sequence), 60):
                handle.write(sequence[start : start + 60] + "\n")
    duplicate_groups.to_csv(alignment_dir / "duplicate_groups.csv", index=False)

    qc = alignment_qc(unique)
    qc.update(
        {
            "written_at_unix": time.time(),
            "n_panel_tips": int(len(panel)),
            "n_duplicate_groups": int(duplicate_groups["duplicate_group"].nunique()),
            "n_collapsed_tips": int(len(panel) - len(unique)),
            "largest_duplicate_group": int(duplicate_groups["group_size"].max()),
            "unique_fasta": str(unique_path),
            "duplicate_groups": str(alignment_dir / "duplicate_groups.csv"),
            "source_alignment_signature": file_signature(alignment_path(args)),
            "representative_policy": "lexicographically smallest accession in each exact-duplicate group",
        }
    )
    write_json(alignment_dir / "alignment_qc.json", qc)
    log(
        f"Wrote unique alignment: {unique_path} "
        f"({qc['n_unique_sequences']:,} sequences x {qc['alignment_width']:,} columns)"
    )
    return qc


def emit_command(args: argparse.Namespace) -> dict[str, Any]:
    alignment_dir = args.out_root / "alignment"
    qc_path = alignment_dir / "alignment_qc.json"
    if not qc_path.exists():
        raise FileNotFoundError("Missing alignment_qc.json; run --stages write-alignment first")
    qc = read_json(qc_path)
    tree_dir = args.out_root / "tree"
    tree_dir.mkdir(parents=True, exist_ok=True)
    prefix = tree_dir / "ml"
    command = [
        args.iqtree_binary,
        "-s",
        str((alignment_dir / "panel_unique.fasta").resolve()),
        "-m",
        args.iqtree_model,
        "-T",
        str(args.iqtree_threads),
        "--prefix",
        str(prefix.resolve()),
        "--seed",
        str(args.selection_seed),
        "--mem",
        args.iqtree_memory,
    ]
    if args.iqtree_fast:
        command.append("-fast")
    script_path = tree_dir / "run_iqtree.sh"
    script_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        "# Reference ML tree for seed-42 directional edge validation.\n"
        f"# {qc['n_unique_sequences']} unique sequences x {qc['alignment_width']} columns\n"
        "# Run this in tmux; it is the only long-running step of the experiment.\n\n"
        + " \\\n  ".join(command)
        + "\n",
        encoding="utf-8",
    )
    script_path.chmod(0o755)
    payload = {
        "emitted_at_unix": time.time(),
        "command": command,
        "command_string": " ".join(command),
        "script": str(script_path),
        "expected_treefile": str(prefix) + ".treefile",
        "n_unique_sequences": qc["n_unique_sequences"],
        "alignment_width": qc["alignment_width"],
    }
    write_json(tree_dir / "iqtree_command.json", payload)
    log(f"Wrote IQ-TREE command: {script_path}")
    log(f"  {payload['command_string']}")
    return payload


# ---------------------------------------------------------------------------
# validation and driver
# ---------------------------------------------------------------------------


def validate_inputs(args: argparse.Namespace, graph_keys: list[str]) -> dict[str, Any]:
    canonical = load_canonical_nodes(args)
    records = read_fasta(alignment_path(args))
    missing = [
        accession for accession in canonical["accession"].tolist() if accession not in records
    ]
    if missing:
        raise ValueError(
            f"{len(missing):,} canonical accessions missing from the panel alignment; "
            f"examples={missing[:5]}"
        )
    widths = {len(sequence) for sequence in records.values()}
    if len(widths) != 1:
        raise ValueError(f"panel alignment is ragged; observed widths={sorted(widths)}")
    sequence_group_of_node = sequence_group_ids(canonical["accession"].tolist(), records)

    frozen_report: dict[str, Any] = {"include_frozen_2k": bool(args.include_frozen_2k)}
    frozen_nodes: np.ndarray | None = None
    if args.include_frozen_2k:
        frozen_path = args.frozen_2k_root / "design/selected_tips.csv"
        frozen = pd.read_csv(frozen_path, usecols=["node_id", "accession"])
        frozen_nodes = frozen["node_id"].to_numpy(dtype=np.int64)
        frozen_report.update(
            {
                "frozen_2k_tips": str(frozen_path),
                "n_frozen_2k_tips": int(len(frozen)),
                "n_frozen_2k_unique_sequences": int(np.unique(sequence_group_of_node[frozen_nodes]).size),
                "frozen_2k_signature": file_signature(frozen_path),
            }
        )

    graph_reports: dict[str, Any] = {}
    for graph_key in graph_keys:
        edges = load_edges(args, graph_key)
        decisions = load_decisions(args, graph_key)
        removed = decisions[decisions["global_decision"] == "removed"]
        endpoints = np.unique(
            np.concatenate(
                [
                    removed["source"].to_numpy(dtype=np.int64),
                    removed["target"].to_numpy(dtype=np.int64),
                ]
            )
        )
        report = {
            "display_name": GRAPH_SPECS[graph_key].display_name,
            "n_baseline_edges": int(len(edges)),
            "n_zero_weight_edges": int(np.count_nonzero(edges["weight"].to_numpy() == 0.0)),
            "n_mutually_rejected": int(len(decisions)),
            "n_removed": int(len(removed)),
            "n_retained_for_connectivity": int(
                int((decisions["global_decision"] == "retained_for_connectivity").sum())
            ),
            "n_distinct_removed_endpoint_nodes": int(endpoints.size),
            "n_distinct_removed_endpoint_sequences": int(
                np.unique(sequence_group_of_node[endpoints]).size
            ),
            "edges_signature": file_signature(graph_dir(args, graph_key) / "edges.csv"),
            "decisions_signature": file_signature(decisions_path(args, graph_key)),
        }
        if frozen_nodes is not None:
            frozen_set = np.zeros(len(canonical), dtype=bool)
            frozen_set[frozen_nodes] = True
            both = frozen_set[removed["source"].to_numpy(dtype=np.int64)] & frozen_set[
                removed["target"].to_numpy(dtype=np.int64)
            ]
            report["n_removed_edges_inside_frozen_2k"] = int(both.sum())
        graph_reports[graph_key] = report
        log(
            f"{graph_key}: removed={report['n_removed']:,} "
            f"endpoint_nodes={report['n_distinct_removed_endpoint_nodes']:,} "
            f"endpoint_unique_sequences={report['n_distinct_removed_endpoint_sequences']:,} "
            f"inside_frozen_2k={report.get('n_removed_edges_inside_frozen_2k', 'n/a')}"
        )

    payload = {
        "validated_at_unix": time.time(),
        "panel_root": str(args.panel_root),
        "directional_root": str(args.directional_root),
        "candidate_label": args.candidate_label,
        "n_panel_nodes": int(len(canonical)),
        "n_panel_unique_sequences": int(np.unique(sequence_group_of_node).size),
        "alignment_width": int(widths.pop()),
        "alignment_signature": file_signature(alignment_path(args)),
        "graphs": graph_reports,
        **frozen_report,
    }
    write_json(args.out_root / "input_validation.json", payload)
    log(f"Wrote input validation: {args.out_root / 'input_validation.json'}")
    return payload


def run_stages(args: argparse.Namespace) -> None:
    graph_keys = parse_graph_priority(args.graph_priority)
    stages = {stage.strip() for stage in args.stages.split(",") if stage.strip()}
    if "all" in stages:
        stages = set(STAGES)
    unknown = stages.difference(STAGES)
    if unknown:
        raise ValueError(f"Unknown stage(s): {sorted(unknown)}; allowed={STAGES}")
    args.out_root.mkdir(parents=True, exist_ok=True)
    if "validate-inputs" in stages:
        validate_inputs(args, graph_keys)
    if "select-edges" in stages:
        select_edges(args, graph_keys)
    if "build-panel" in stages:
        build_panel(args, graph_keys)
    if "write-alignment" in stages:
        write_alignment(args)
    if "emit-command" in stages:
        emit_command(args)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel-root", type=Path, default=DEFAULT_PANEL_ROOT)
    ap.add_argument("--directional-root", type=Path, default=DEFAULT_DIRECTIONAL_ROOT)
    ap.add_argument("--frozen-2k-root", type=Path, default=DEFAULT_FROZEN_2K_ROOT)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--candidate-label", default=DEFAULT_CANDIDATE_LABEL)
    ap.add_argument("--stages", default="validate-inputs")
    ap.add_argument("--graph-priority", default="rng,knn5,knn50")
    ap.add_argument("--selection-seed", type=int, default=42)
    ap.add_argument("--caliper-fraction", type=float, default=0.25)
    ap.add_argument("--controls-per-endpoint", type=int, default=1)
    ap.add_argument("--max-unique-sequences", type=int, default=3000)
    ap.add_argument("--max-tips", type=int, default=6000)
    ap.add_argument(
        "--include-frozen-2k",
        dest="include_frozen_2k",
        action="store_true",
        default=True,
        help="seed the panel with the frozen 2k tips so exp-25 matrices stay comparable",
    )
    ap.add_argument("--no-include-frozen-2k", dest="include_frozen_2k", action="store_false")
    ap.add_argument("--iqtree-binary", default="iqtree2")
    ap.add_argument("--iqtree-model", default="LG+F+G4")
    ap.add_argument("--iqtree-threads", default="AUTO")
    ap.add_argument("--iqtree-memory", default="16G")
    ap.add_argument("--iqtree-fast", dest="iqtree_fast", action="store_true", default=True)
    ap.add_argument("--no-iqtree-fast", dest="iqtree_fast", action="store_false")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    run_stages(args)


if __name__ == "__main__":
    main()
