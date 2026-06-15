#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from scripts.graph_construction.build_panel_nj_distance_reference_trees import (  # noqa: E402
    metric_specs,
    parse_seed_list,
)
from scripts.graph_construction.build_panel_spike_reference_tree import (  # noqa: E402
    load_panel_accessions,
)
from scripts.validation.nextstrain_spike_tree_validation import (  # noqa: E402
    graph_paths,
    graph_shortest_path_matrix,
    raw_distance_paths,
    subset_dense_matrix,
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def upper_tri_values(D: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices_from(D, k=1)
    return np.asarray(D[i, j], dtype=np.float64)


def robust_iqr(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    q25, q75 = np.percentile(finite, [25, 75])
    iqr = float(q75 - q25)
    if iqr == 0.0:
        sd = float(np.std(finite))
        return sd if sd > 0.0 else 1.0
    return iqr


def gromov_delta_hyperbolicity(
    D: np.ndarray,
    n_samples: int,
    seed: int,
    max_attempt_multiplier: int = 20,
) -> dict[str, Any]:
    """
    Finite-aware version of rel_distance/post_metrics.py::gromov_delta_hyperbolicity.

    For each sampled quadruple, compute the three sums:
    d(a,b)+d(c,d), d(a,c)+d(b,d), d(a,d)+d(b,c).
    The quadruple delta is half the difference between the largest and middle sum.
    """
    n = int(D.shape[0])
    if n < 4:
        return {
            "gromov_delta": math.nan,
            "gromov_delta_norm_iqr": math.nan,
            "gromov_samples_requested": int(n_samples),
            "gromov_samples_used": 0,
            "gromov_sample_attempts": 0,
            "distance_iqr": math.nan,
        }

    iqr = robust_iqr(upper_tri_values(D))
    rng = np.random.default_rng(seed)
    delta_max = 0.0
    used = 0
    attempts = 0
    max_attempts = max(int(n_samples) * max_attempt_multiplier, int(n_samples))

    while used < n_samples and attempts < max_attempts:
        attempts += 1
        a, b, c, d = rng.choice(n, size=4, replace=False)
        vals = np.array(
            [
                D[a, b] + D[c, d],
                D[a, c] + D[b, d],
                D[a, d] + D[b, c],
            ],
            dtype=np.float64,
        )
        if not np.isfinite(vals).all():
            continue
        vals.sort()
        delta = float(0.5 * (vals[2] - vals[1]))
        if delta > delta_max:
            delta_max = delta
        used += 1

    return {
        "gromov_delta": float(delta_max) if used else math.nan,
        "gromov_delta_norm_iqr": float(delta_max / iqr) if used and np.isfinite(iqr) and iqr != 0.0 else math.nan,
        "gromov_samples_requested": int(n_samples),
        "gromov_samples_used": int(used),
        "gromov_sample_attempts": int(attempts),
        "distance_iqr": float(iqr) if np.isfinite(iqr) else math.nan,
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
        "distance_n_components": int(n_components),
        "distance_giant_component_size": int(sizes.max()) if len(sizes) else 0,
        "component_mode": mode,
    }
    if mode == "all" and n_components > 1:
        raise ValueError(f"Cannot score disconnected graph distances as one metric space: n_components={n_components}")
    if mode == "largest" and n_components > 1:
        keep_label = int(np.argmax(sizes))
        idx = np.flatnonzero(labels == keep_label)
        qc["n_nodes_dropped"] = int(n - len(idx))
        return np.asarray(D[np.ix_(idx, idx)]), [accessions[i] for i in idx], qc
    qc["n_nodes_dropped"] = 0
    return D, accessions, qc


def raw_spec_by_name(panel_root: Path, sample_label: str) -> dict[str, dict[str, Any]]:
    return {str(spec["baseline"]): spec for spec in raw_distance_paths(panel_root, sample_label)}


def evaluate_raw(panel: str, seed: int, panel_root: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    specs = raw_spec_by_name(panel_root, args.sample_label)
    selected = load_panel_accessions(panel_root, args.sample_label)
    wanted = [item.strip() for item in args.raw_baselines.split(",") if item.strip()]
    rows: list[dict[str, Any]] = []

    for baseline in wanted:
        spec = specs[baseline]
        log(f"Scoring raw Gromov hyperbolicity: {panel}/seed_{seed} {baseline}")
        D = subset_dense_matrix(Path(spec["matrix"]), Path(spec["nodes"]), selected)
        score = gromov_delta_hyperbolicity(D, args.hyper_samples, args.hyper_seed + seed)
        rows.append(
            {
                "panel": panel,
                "seed": int(seed),
                "sample_label": args.sample_label,
                "comparison_type": "raw_distance",
                "baseline": baseline,
                "metric_family": spec["metric_family"],
                "metric": spec["metric"],
                "n_accessions": int(D.shape[0]),
                "matrix_path": str(spec["matrix"]),
                **score,
            }
        )
        del D
    return rows


def graph_scoring_accessions(panel_root: Path, args: argparse.Namespace) -> list[str]:
    raw_baselines = {item.strip() for item in args.raw_baselines.split(",") if item.strip()}
    raw_specs = metric_specs(panel_root, args.sample_label, raw_baselines)
    accessions = load_panel_accessions(panel_root, args.sample_label)
    for raw_spec in raw_specs:
        raw_nodes = pd.read_csv(raw_spec["nodes"], usecols=["accession"], low_memory=False)
        raw_set = set(raw_nodes["accession"].astype(str).str.strip())
        accessions = [acc for acc in accessions if acc in raw_set]
    return accessions


def evaluate_graphs(panel: str, seed: int, panel_root: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    accessions = graph_scoring_accessions(panel_root, args)
    specs = graph_paths(panel_root, args.sample_label)
    wanted = {item.strip() for item in args.graph_names.split(",") if item.strip()}
    if wanted:
        specs = [spec for spec in specs if str(spec["graph_name"]) in wanted]

    rows: list[dict[str, Any]] = []
    for spec in specs:
        graph_name = str(spec["graph_name"])
        graph_dir = Path(spec["graph_dir"])
        if not graph_dir.exists():
            log(f"Skipping missing graph: {graph_dir}")
            continue
        log(f"Scoring graph Gromov hyperbolicity: {panel}/seed_{seed} {graph_name}")
        D_all, graph_qc = graph_shortest_path_matrix(graph_dir, accessions)
        D, component_accessions, component_qc = finite_component_subset(D_all, accessions, args.graph_component_mode)
        if D.shape[0] < 4:
            log(f"Skipping {graph_name}: fewer than 4 accessions in scored component")
            continue
        score = gromov_delta_hyperbolicity(D, args.hyper_samples, args.hyper_seed + seed)
        rows.append(
            {
                "panel": panel,
                "seed": int(seed),
                "sample_label": args.sample_label,
                "comparison_type": "graph_geodesic",
                "graph_name": graph_name,
                "metric_family": spec["metric_family"],
                "embedding_metric": spec["embedding_metric"],
                "graph_family": spec["graph_family"],
                "n_accessions": int(D.shape[0]),
                "graph_dir": str(graph_dir),
                "n_component_accessions": int(len(component_accessions)),
                **graph_qc,
                **component_qc,
                **score,
            }
        )
        del D_all, D
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
    raw_frames = [pd.read_csv(path) for path in workspace.glob("*/seed_*/raw_gromov_hyperbolicity_metrics.csv")]
    graph_frames = [pd.read_csv(path) for path in workspace.glob("*/seed_*/graph_gromov_hyperbolicity_metrics.csv")]

    if raw_frames:
        raw = pd.concat(raw_frames, ignore_index=True)
        raw.to_csv(workspace / "all_raw_gromov_hyperbolicity_metrics.csv", index=False)
        for value_col, out_name in [
            ("gromov_delta", "raw_gromov_delta_seed_summary.csv"),
            ("gromov_delta_norm_iqr", "raw_gromov_delta_norm_iqr_seed_summary.csv"),
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
        graph.to_csv(workspace / "all_graph_gromov_hyperbolicity_metrics.csv", index=False)
        for value_col, out_name in [
            ("gromov_delta", "graph_gromov_delta_seed_summary.csv"),
            ("gromov_delta_norm_iqr", "graph_gromov_delta_norm_iqr_seed_summary.csv"),
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
    ap = argparse.ArgumentParser(description="Evaluate raw and graph Gromov hyperbolicity without NJ tree matrices.")
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
    ap.add_argument("--hyper-samples", type=int, default=50_000)
    ap.add_argument("--hyper-seed", type=int, default=12345)
    ap.add_argument("--graph-component-mode", choices=["largest", "all"], default="largest")
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
                    log(f"Skipping missing validation seed output: {seed_out}")
                    continue
                if not args.skip_raw:
                    raw_rows = evaluate_raw(panel, seed, panel_root, args)
                    pd.DataFrame(raw_rows).to_csv(seed_out / "raw_gromov_hyperbolicity_metrics.csv", index=False)
                if not args.skip_graphs:
                    graph_rows = evaluate_graphs(panel, seed, panel_root, args)
                    pd.DataFrame(graph_rows).to_csv(seed_out / "graph_gromov_hyperbolicity_metrics.csv", index=False)

    aggregate_workspace(workspace)


if __name__ == "__main__":
    main()
