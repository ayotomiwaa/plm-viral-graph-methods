#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from protein_embeddings.methods.cc_geodesic import (  # noqa: E402
    build_nj_tree,
    clip_negative_branch_lengths,
    save_newick,
)
from scripts.graph_construction.build_panel_nj_distance_reference_trees import (  # noqa: E402
    count_negative_branch_lengths,
    metric_specs,
    parse_seed_list,
)
from scripts.graph_construction.build_panel_spike_reference_tree import (  # noqa: E402
    compute_patristic_matrix,
    load_panel_accessions,
)
from scripts.validation.nextstrain_spike_tree_validation import (  # noqa: E402
    graph_paths,
    graph_shortest_path_matrix,
    make_pair_indices,
    raw_distance_paths,
    subset_dense_matrix,
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_accessions_from_nodes(path: Path) -> list[str]:
    nodes = pd.read_csv(path, usecols=["accession"], low_memory=False)
    return nodes["accession"].astype(str).str.strip().tolist()


def pair_vectors(
    D: np.ndarray,
    T: np.ndarray,
    pair_mode: str,
    sample_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if D.shape != T.shape:
        raise ValueError(f"Matrix shape mismatch: D={D.shape}, T={T.shape}")
    i, j = make_pair_indices(D.shape[0], pair_mode, sample_size, seed)
    return np.asarray(D[i, j], dtype=np.float64), np.asarray(T[i, j], dtype=np.float64), int(len(i))


def optimal_scale_from_vectors(d: np.ndarray, t: np.ndarray, eps: float) -> float:
    mask = np.isfinite(d) & np.isfinite(t)
    if not np.any(mask):
        return math.nan
    denom = float(np.dot(t[mask], t[mask]))
    if denom <= eps:
        return math.nan
    return float(np.dot(d[mask], t[mask]) / denom)


def distance_tree_metrics(
    D: np.ndarray,
    T: np.ndarray,
    pair_mode: str,
    sample_size: int,
    seed: int,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Finite-aware version of the repo's old RSD and scaled distortion metrics."""
    d, t, raw_pairs = pair_vectors(D, T, pair_mode, sample_size, seed)
    finite = np.isfinite(d) & np.isfinite(t)
    n_finite = int(finite.sum())
    if n_finite:
        d_f = d[finite]
        t_f = t[finite]
        denom = float(np.sum(d_f * d_f))
        rsd = float(np.sqrt(np.sum((d_f - t_f) ** 2) / denom)) if denom > eps else math.nan
    else:
        rsd = math.nan

    s_star = optimal_scale_from_vectors(d, t, eps)
    positive = finite & (d > eps) & (t > eps) & np.isfinite(s_star) & (s_star > eps)
    n_positive = int(positive.sum())
    if n_positive:
        ratio = d[positive] / (s_star * t[positive] + eps)
        distortion = float(np.maximum(ratio, 1.0 / (ratio + eps)).max())
        log_distortion = float(np.abs(np.log(ratio + eps)).max())
    else:
        distortion = math.nan
        log_distortion = math.nan

    return {
        "pair_mode": pair_mode,
        "n_pairs_raw": raw_pairs,
        "n_pairs_used": n_finite,
        "finite_pair_fraction": float(n_finite / raw_pairs) if raw_pairs else math.nan,
        "relative_square_deviation": rsd,
        "tree_dist_scale_s_star": s_star,
        "max_tree_distortion": distortion,
        "tree_distortion_log_delta": log_distortion,
        "tree_distortion_pairs_used": n_positive,
    }


def finite_component_subset(
    D: np.ndarray,
    accessions: list[str],
    mode: str,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    n = D.shape[0]
    finite = np.isfinite(D)
    np.fill_diagonal(finite, False)
    graph = csr_matrix(finite)
    n_components, labels = connected_components(graph, directed=False, return_labels=True)
    sizes = np.bincount(labels, minlength=n_components) if n else np.array([], dtype=int)
    qc: dict[str, Any] = {
        "graph_distance_n_components": int(n_components),
        "graph_distance_giant_component_size": int(sizes.max()) if len(sizes) else 0,
        "graph_component_mode": mode,
    }
    if mode == "all" and n_components > 1:
        raise ValueError(f"Cannot build one NJ tree from disconnected graph distances: n_components={n_components}")
    if mode == "largest" and n_components > 1:
        keep_label = int(np.argmax(sizes))
        idx = np.flatnonzero(labels == keep_label)
        qc["n_nodes_dropped_for_nj"] = int(n - len(idx))
        return np.asarray(D[np.ix_(idx, idx)]), [accessions[i] for i in idx], qc
    qc["n_nodes_dropped_for_nj"] = 0
    return D, accessions, qc


def ensure_nj_patristic(
    D: np.ndarray,
    accessions: list[str],
    out_dir: Path,
    tree_name: str,
    args: argparse.Namespace,
    metadata: dict[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    newick_path = out_dir / tree_name
    manifest_path = out_dir / "nj_tree_manifest.json"

    if newick_path.exists() and manifest_path.exists() and not args.overwrite_tree:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        log(f"Using existing NJ Newick: {newick_path}")
    else:
        log(f"Building NJ tree: {newick_path}")
        tree_kind, tree = build_nj_tree(D.astype(np.float64, copy=False), accessions, prefer=args.prefer_tree_builder)
        neg_before = count_negative_branch_lengths(tree_kind, tree)
        if args.clip_negative_branches:
            clip_negative_branch_lengths(tree_kind, tree)
        neg_after = count_negative_branch_lengths(tree_kind, tree)
        save_newick(tree_kind, tree, str(newick_path))
        manifest = {
            **metadata,
            "tree_builder": "neighbor_joining",
            "tree_builder_backend": tree_kind,
            "n_accessions": int(len(accessions)),
            "newick_path": str(newick_path),
            "clip_negative_branches": bool(args.clip_negative_branches),
            "branch_lengths_before_clip": neg_before,
            "branch_lengths_after_clip": neg_after,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    matrix_path = compute_patristic_matrix(
        newick_path=newick_path,
        panel_accessions=accessions,
        out_dir=out_dir,
        block_size=args.patristic_block_size,
        overwrite=args.overwrite_patristic,
    )
    nodes_path = out_dir / "D_reference_spike_nodes.csv"
    qc_path = out_dir / "D_reference_spike_qc.json"
    pat_qc = json.loads(qc_path.read_text(encoding="utf-8")) if qc_path.exists() else {}
    return matrix_path, nodes_path, {**manifest, **{"matrix_size_gb": pat_qc.get("matrix_size_gb", "")}}


def load_raw_spec_by_name(panel_root: Path, sample_label: str) -> dict[str, dict[str, Any]]:
    return {str(spec["baseline"]): spec for spec in raw_distance_paths(panel_root, sample_label)}


def evaluate_raw(seed_out: Path, panel_root: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    raw_tree_path = seed_out / "nj_self_tree_likeness_correlations.csv"
    if not raw_tree_path.exists():
        raise FileNotFoundError(f"Missing raw NJ tree-likeness file: {raw_tree_path}")
    raw_tree_rows = pd.read_csv(raw_tree_path)
    specs = load_raw_spec_by_name(panel_root, args.sample_label)
    wanted = {item.strip() for item in args.raw_baselines.split(",") if item.strip()}
    rows: list[dict[str, Any]] = []
    for _, row in raw_tree_rows.iterrows():
        baseline = str(row["baseline"])
        if baseline not in wanted:
            continue
        spec = specs[baseline]
        accessions = read_accessions_from_nodes(Path(row["patristic_nodes"]))
        log(f"Scoring raw distortion/RSD: {baseline}, n={len(accessions):,}")
        D = subset_dense_matrix(Path(spec["matrix"]), Path(spec["nodes"]), accessions)
        T = np.load(Path(row["patristic_matrix"]), mmap_mode="r")
        score = distance_tree_metrics(D, T, args.pair_mode, args.pair_sample_size, args.pair_seed + int(row["seed"]))
        rows.append(
            {
                "panel": row["panel"],
                "seed": int(row["seed"]),
                "sample_label": row["sample_label"],
                "comparison_type": "raw_vs_own_nj",
                "baseline": baseline,
                "metric_family": row["metric_family"],
                "metric": row["metric"],
                "n_accessions": int(len(accessions)),
                "newick_path": row["newick_path"],
                "patristic_matrix": row["patristic_matrix"],
                "patristic_nodes": row["patristic_nodes"],
                "tree_builder_backend": row.get("tree_builder_backend", ""),
                "clip_negative_branches": row.get("clip_negative_branches", ""),
                "n_negative_branches_before_clip": row.get("n_negative_branches_before_clip", ""),
                "negative_branch_length_sum_before_clip": row.get("negative_branch_length_sum_before_clip", ""),
                "n_negative_branches_after_clip": row.get("n_negative_branches_after_clip", ""),
                "matrix_size_gb": row.get("matrix_size_gb", ""),
                **score,
            }
        )
        del D, T
    return rows


def evaluate_graphs(seed_out: Path, panel: str, seed: int, panel_root: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    graph_specs = graph_paths(panel_root, args.sample_label)
    wanted = {item.strip() for item in args.graph_names.split(",") if item.strip()}
    if wanted:
        graph_specs = [spec for spec in graph_specs if str(spec["graph_name"]) in wanted]

    raw_baselines = {item.strip() for item in args.raw_baselines.split(",") if item.strip()}
    raw_specs = metric_specs(panel_root, args.sample_label, raw_baselines)
    accessions = load_panel_accessions(panel_root, args.sample_label)
    for raw_spec in raw_specs:
        raw_nodes = pd.read_csv(raw_spec["nodes"], usecols=["accession"], low_memory=False)
        raw_set = set(raw_nodes["accession"].astype(str).str.strip())
        accessions = [acc for acc in accessions if acc in raw_set]

    rows: list[dict[str, Any]] = []
    for spec in graph_specs:
        graph_name = str(spec["graph_name"])
        if not Path(spec["graph_dir"]).exists():
            log(f"Skipping missing graph: {spec['graph_dir']}")
            continue
        log(f"Loading graph geodesic matrix for NJ distortion/RSD: {graph_name}")
        D_all, graph_qc = graph_shortest_path_matrix(Path(spec["graph_dir"]), accessions)
        D, tree_accessions, component_qc = finite_component_subset(D_all, accessions, args.graph_component_mode)
        if len(tree_accessions) < 3:
            log(f"Skipping {graph_name}: fewer than 3 accessions in NJ component")
            continue
        out_dir = seed_out / "nj_graph_trees" / graph_name
        matrix_path, nodes_path, tree_qc = ensure_nj_patristic(
            D=D,
            accessions=tree_accessions,
            out_dir=out_dir,
            tree_name=f"{graph_name}_nj.nwk",
            args=args,
            metadata={
                "panel": panel,
                "seed": int(seed),
                "sample_label": args.sample_label,
                "graph_name": graph_name,
                "metric_family": spec["metric_family"],
                "embedding_metric": spec["embedding_metric"],
                "graph_family": spec["graph_family"],
                "graph_dir": str(spec["graph_dir"]),
            },
        )
        T = np.load(matrix_path, mmap_mode="r")
        score = distance_tree_metrics(D, T, args.pair_mode, args.pair_sample_size, args.pair_seed + seed)
        rows.append(
            {
                "panel": panel,
                "seed": int(seed),
                "sample_label": args.sample_label,
                "comparison_type": "graph_geodesic_vs_own_nj",
                "graph_name": graph_name,
                "metric_family": spec["metric_family"],
                "embedding_metric": spec["embedding_metric"],
                "graph_family": spec["graph_family"],
                "n_accessions": int(len(tree_accessions)),
                "newick_path": str(out_dir / f"{graph_name}_nj.nwk"),
                "patristic_matrix": str(matrix_path),
                "patristic_nodes": str(nodes_path),
                "tree_builder_backend": tree_qc.get("tree_builder_backend", ""),
                "clip_negative_branches": tree_qc.get("clip_negative_branches", ""),
                "n_negative_branches_before_clip": tree_qc.get("branch_lengths_before_clip", {}).get("n_negative_branches", ""),
                "negative_branch_length_sum_before_clip": tree_qc.get("branch_lengths_before_clip", {}).get("negative_branch_length_sum", ""),
                "n_negative_branches_after_clip": tree_qc.get("branch_lengths_after_clip", {}).get("n_negative_branches", ""),
                "matrix_size_gb": tree_qc.get("matrix_size_gb", ""),
                **graph_qc,
                **component_qc,
                **score,
            }
        )
        del D_all, D, T
    return rows


def summarize(
    frame: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    sort_cols: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = pd.to_numeric(group[value_col], errors="coerce").dropna()
        n = int(values.shape[0])
        sd = float(values.std(ddof=1)) if n > 1 else 0.0 if n == 1 else math.nan
        se = float(sd / math.sqrt(n)) if n else math.nan
        out = {col: key for col, key in zip(group_cols, keys)}
        out.update(
            {
                "n_seeds": n,
                f"{value_col}_mean": float(values.mean()) if n else math.nan,
                f"{value_col}_sd": sd,
                f"{value_col}_se": se,
                f"{value_col}_ci95_halfwidth": float(1.96 * se) if n > 1 else 0.0 if n == 1 else math.nan,
                f"{value_col}_min": float(values.min()) if n else math.nan,
                f"{value_col}_max": float(values.max()) if n else math.nan,
            }
        )
        rows.append(out)
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(sort_cols).reset_index(drop=True)
    return result


def aggregate_workspace(workspace: Path) -> None:
    raw_frames = []
    graph_frames = []
    for path in workspace.glob("*/seed_*/raw_tree_distortion_metrics.csv"):
        raw_frames.append(pd.read_csv(path))
    for path in workspace.glob("*/seed_*/graph_tree_distortion_metrics.csv"):
        graph_frames.append(pd.read_csv(path))

    if raw_frames:
        raw = pd.concat(raw_frames, ignore_index=True)
        raw.to_csv(workspace / "all_raw_tree_distortion_metrics.csv", index=False)
        for value_col, out_name in [
            ("relative_square_deviation", "raw_relative_square_deviation_seed_summary.csv"),
            ("max_tree_distortion", "raw_max_tree_distortion_seed_summary.csv"),
        ]:
            summarize(
                raw,
                ["panel", "baseline", "metric_family", "metric"],
                value_col,
                ["panel", f"{value_col}_mean"],
            ).sort_values(["panel", f"{value_col}_mean"], ascending=[True, True]).to_csv(
                workspace / out_name, index=False
            )

    if graph_frames:
        graph = pd.concat(graph_frames, ignore_index=True)
        graph.to_csv(workspace / "all_graph_tree_distortion_metrics.csv", index=False)
        for value_col, out_name in [
            ("relative_square_deviation", "graph_relative_square_deviation_seed_summary.csv"),
            ("max_tree_distortion", "graph_max_tree_distortion_seed_summary.csv"),
        ]:
            summarize(
                graph,
                ["panel", "graph_name", "metric_family", "embedding_metric", "graph_family"],
                value_col,
                ["panel", f"{value_col}_mean"],
            ).sort_values(["panel", f"{value_col}_mean"], ascending=[True, True]).to_csv(
                workspace / out_name, index=False
            )


def infer_workspace_and_panel(args: argparse.Namespace) -> tuple[Path, list[str]]:
    if args.panel_workspace:
        panel_path = args.panel_workspace
        return panel_path.parent, [panel_path.name]
    panels = [item.strip() for item in args.panels.split(",") if item.strip()]
    return args.workspace, panels


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate raw and graph NJ tree-likeness with RSD and max tree distortion."
    )
    ap.add_argument("--workspace", type=Path, default=Path("analysis/cohort_validation/13_random_full_dataset_2k_nj_tree_validation"))
    ap.add_argument("--panel-workspace", type=Path, default=None)
    ap.add_argument("--source-root", type=Path, default=None)
    ap.add_argument("--panels", default="random_full_dataset_2k")
    ap.add_argument("--seeds", default="0-199")
    ap.add_argument("--sample-label", default="pool_n2000")
    ap.add_argument("--raw-baselines", default="raw_hamming,raw_esm2_cityblock,raw_esm2_euclidean")
    ap.add_argument("--graph-names", default="", help="Comma-separated graph names; empty means all graph families.")
    ap.add_argument("--skip-raw", action="store_true")
    ap.add_argument("--skip-graphs", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--pair-mode", choices=["all", "sample"], default="all")
    ap.add_argument("--pair-sample-size", type=int, default=5_000_000)
    ap.add_argument("--pair-seed", type=int, default=12345)
    ap.add_argument("--graph-component-mode", choices=["largest", "all"], default="largest")
    ap.add_argument("--prefer-tree-builder", choices=["auto", "skbio", "biopython"], default="auto")
    ap.add_argument("--clip-negative-branches", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--overwrite-tree", action="store_true")
    ap.add_argument("--overwrite-patristic", action="store_true")
    ap.add_argument("--patristic-block-size", type=int, default=512)
    args = ap.parse_args()

    workspace, panels = infer_workspace_and_panel(args)
    source_root = args.source_root or (workspace / "source")
    seeds = parse_seed_list(args.seeds)

    if not args.aggregate_only:
        for panel in panels:
            for seed in seeds:
                panel_root = source_root / panel / f"seed_{seed}"
                seed_out = workspace / panel / f"seed_{seed}"
                if not panel_root.exists():
                    log(f"Skipping missing panel seed root: {panel_root}")
                    continue
                if not seed_out.exists():
                    log(f"Skipping missing NJ validation seed output: {seed_out}")
                    continue
                if not args.skip_raw:
                    raw_rows = evaluate_raw(seed_out, panel_root, args)
                    pd.DataFrame(raw_rows).to_csv(seed_out / "raw_tree_distortion_metrics.csv", index=False)
                if not args.skip_graphs:
                    graph_rows = evaluate_graphs(seed_out, panel, seed, panel_root, args)
                    pd.DataFrame(graph_rows).to_csv(seed_out / "graph_tree_distortion_metrics.csv", index=False)

    aggregate_workspace(workspace)


if __name__ == "__main__":
    main()
