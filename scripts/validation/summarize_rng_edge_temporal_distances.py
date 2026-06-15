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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.graph_construction.build_panel_nj_distance_reference_trees import (  # noqa: E402
    parse_seed_list,
)
from scripts.validation.nextstrain_spike_tree_validation import (  # noqa: E402
    graph_paths,
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def wanted_rng_graphs(panel_root: Path, sample_label: str, graph_names: str) -> list[dict[str, Any]]:
    specs = [spec for spec in graph_paths(panel_root, sample_label) if spec["graph_family"] == "rng_exact"]
    wanted = {item.strip() for item in graph_names.split(",") if item.strip()}
    if wanted:
        specs = [spec for spec in specs if str(spec["graph_name"]) in wanted]
    return specs


def metric_label(spec: dict[str, Any]) -> str:
    if spec["metric_family"] == "hamming":
        return "hamming"
    if spec["embedding_metric"] == "cityblock":
        return "manhattan"
    return str(spec["embedding_metric"])


def node_date_days(nodes_path: Path) -> tuple[np.ndarray, np.ndarray]:
    nodes = pd.read_csv(nodes_path, usecols=["node_id", "collection_date"], low_memory=False)
    node_ids = pd.to_numeric(nodes["node_id"], errors="raise").astype(int).to_numpy()
    max_node_id = int(node_ids.max()) if len(node_ids) else -1
    dates = pd.to_datetime(nodes["collection_date"], errors="coerce")
    date_days = np.zeros(max_node_id + 1, dtype=np.int64)
    valid = np.zeros(max_node_id + 1, dtype=bool)
    valid_mask = dates.notna().to_numpy()
    if valid_mask.any():
        valid_node_ids = node_ids[valid_mask]
        date_days[valid_node_ids] = dates[valid_mask].to_numpy(dtype="datetime64[D]").astype(np.int64)
        valid[valid_node_ids] = True
    return date_days, valid


def update_moments(count: int, mean: float, m2: float, values: np.ndarray) -> tuple[int, float, float]:
    n = int(values.size)
    if n == 0:
        return count, mean, m2
    chunk_mean = float(values.mean())
    chunk_m2 = float(((values - chunk_mean) ** 2).sum())
    if count == 0:
        return n, chunk_mean, chunk_m2
    new_count = count + n
    delta = chunk_mean - mean
    new_mean = mean + delta * n / new_count
    new_m2 = m2 + chunk_m2 + delta * delta * count * n / new_count
    return new_count, new_mean, new_m2


def summarize_graph_edges(
    panel: str,
    seed: int,
    spec: dict[str, Any],
    chunk_size: int,
) -> dict[str, Any]:
    graph_dir = Path(spec["graph_dir"])
    nodes_path = graph_dir / "nodes.csv"
    edges_path = graph_dir / "edges.csv"
    if not nodes_path.exists() or not edges_path.exists():
        raise FileNotFoundError(f"Missing RNG graph nodes/edges in {graph_dir}")

    date_days, valid_dates = node_date_days(nodes_path)
    n_total = 0
    n_kept = 0
    n_valid = 0
    n_missing = 0
    n_same_date = 0
    mean = 0.0
    m2 = 0.0
    min_lag = math.inf
    max_lag = -math.inf

    usecols = ["source", "target", "status"]
    for chunk in pd.read_csv(edges_path, usecols=lambda col: col in usecols, chunksize=chunk_size, low_memory=False):
        n_total += int(chunk.shape[0])
        if "status" in chunk.columns:
            chunk = chunk[chunk["status"].astype(str) == "kept"]
        n_kept += int(chunk.shape[0])
        if chunk.empty:
            continue
        sources = pd.to_numeric(chunk["source"], errors="raise").astype(int).to_numpy()
        targets = pd.to_numeric(chunk["target"], errors="raise").astype(int).to_numpy()
        valid_edge = valid_dates[sources] & valid_dates[targets]
        n_missing += int((~valid_edge).sum())
        if not valid_edge.any():
            continue
        lags = np.abs(date_days[sources[valid_edge]] - date_days[targets[valid_edge]]).astype(np.float64)
        n_valid += int(lags.size)
        n_same_date += int((lags == 0).sum())
        min_lag = min(min_lag, float(lags.min()))
        max_lag = max(max_lag, float(lags.max()))
        n_moments, mean, m2 = update_moments(n_valid - int(lags.size), mean, m2, lags)
        if n_moments != n_valid:
            raise RuntimeError("Internal moment-count mismatch while summarizing temporal edge distances")

    sd = math.sqrt(m2 / (n_valid - 1)) if n_valid > 1 else 0.0 if n_valid == 1 else math.nan
    se = sd / math.sqrt(n_valid) if n_valid else math.nan
    ci95 = 1.96 * se if n_valid > 1 else 0.0 if n_valid == 1 else math.nan
    return {
        "panel": panel,
        "seed": int(seed),
        "graph_name": spec["graph_name"],
        "metric_family": spec["metric_family"],
        "embedding_metric": spec["embedding_metric"],
        "metric_label": metric_label(spec),
        "graph_family": spec["graph_family"],
        "graph_dir": str(graph_dir),
        "edge_temporal_distance_definition": "absolute collection-date difference in days across kept undirected RNG edges",
        "n_edges_total_rows": int(n_total),
        "n_edges_kept": int(n_kept),
        "n_edges_with_valid_dates": int(n_valid),
        "n_edges_missing_dates": int(n_missing),
        "valid_date_edge_fraction": float(n_valid / n_kept) if n_kept else math.nan,
        "n_same_date_edges": int(n_same_date),
        "same_date_edge_fraction": float(n_same_date / n_valid) if n_valid else math.nan,
        "mean_edge_temporal_distance_days": float(mean) if n_valid else math.nan,
        "edge_temporal_distance_days_sd": float(sd) if np.isfinite(sd) else math.nan,
        "edge_temporal_distance_days_se": float(se) if np.isfinite(se) else math.nan,
        "edge_temporal_distance_days_ci95_halfwidth": float(ci95) if np.isfinite(ci95) else math.nan,
        "edge_temporal_distance_days_min": float(min_lag) if n_valid else math.nan,
        "edge_temporal_distance_days_max": float(max_lag) if n_valid else math.nan,
        "nodes_path": str(nodes_path),
        "edges_path": str(edges_path),
    }


def infer_workspace_and_panel(args: argparse.Namespace) -> tuple[Path, list[str]]:
    if args.panel_workspace:
        panel_path = args.panel_workspace
        return panel_path.parent, [panel_path.name]
    panels = [item.strip() for item in args.panels.split(",") if item.strip()]
    return args.workspace, panels


def summarize_across_seeds(frame: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["panel", "graph_name", "metric_family", "embedding_metric", "metric_label", "graph_family"]
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        out = {col: key for col, key in zip(group_cols, keys)}
        values = pd.to_numeric(group["mean_edge_temporal_distance_days"], errors="coerce").dropna()
        n = int(values.shape[0])
        sd = float(values.std(ddof=1)) if n > 1 else 0.0 if n == 1 else math.nan
        se = float(sd / math.sqrt(n)) if n else math.nan
        out.update(
            {
                "n_seeds": n,
                "mean_edge_temporal_distance_days_mean": float(values.mean()) if n else math.nan,
                "mean_edge_temporal_distance_days_sd": sd,
                "mean_edge_temporal_distance_days_se": se,
                "mean_edge_temporal_distance_days_ci95_halfwidth": float(1.96 * se)
                if n > 1
                else 0.0
                if n == 1
                else math.nan,
                "mean_edge_temporal_distance_days_min": float(values.min()) if n else math.nan,
                "mean_edge_temporal_distance_days_max": float(values.max()) if n else math.nan,
                "n_edges_kept_sum": int(pd.to_numeric(group["n_edges_kept"], errors="coerce").sum()),
                "n_edges_kept_mean": float(pd.to_numeric(group["n_edges_kept"], errors="coerce").mean()),
                "n_edges_with_valid_dates_sum": int(
                    pd.to_numeric(group["n_edges_with_valid_dates"], errors="coerce").sum()
                ),
                "n_same_date_edges_sum": int(pd.to_numeric(group["n_same_date_edges"], errors="coerce").sum()),
            }
        )
        rows.append(out)
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["panel", "metric_label"]).reset_index(drop=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize absolute collection-date distances across RNG graph edges.")
    ap.add_argument("--workspace", type=Path, default=Path("analysis/cohort_validation/14_seed42_20k_temporal_rng_dag_validation"))
    ap.add_argument("--panel-workspace", type=Path, default=None)
    ap.add_argument("--source-root", type=Path, default=Path("analysis/cohort_validation/07_sampling_design_20k"))
    ap.add_argument("--panels", default="random_full_dataset_seed42")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--sample-label", default="pool_n20000")
    ap.add_argument("--graph-names", default="", help="Comma-separated RNG graph names; empty means all RNG graphs.")
    ap.add_argument("--chunk-size", type=int, default=500_000)
    args = ap.parse_args()

    workspace, panels = infer_workspace_and_panel(args)
    source_root = args.source_root
    seeds = parse_seed_list(args.seeds)
    all_rows: list[dict[str, Any]] = []

    for panel in panels:
        for seed in seeds:
            panel_root = source_root / panel / f"seed_{seed}"
            seed_out = workspace / panel / f"seed_{seed}"
            if not panel_root.exists():
                log(f"Skipping missing panel seed root: {panel_root}")
                continue
            rows: list[dict[str, Any]] = []
            for spec in wanted_rng_graphs(panel_root, args.sample_label, args.graph_names):
                graph_dir = Path(spec["graph_dir"])
                if not graph_dir.exists():
                    log(f"Skipping missing RNG graph: {graph_dir}")
                    continue
                log(f"Summarizing RNG edge temporal distances: {panel}/seed_{seed} {spec['graph_name']}")
                row = summarize_graph_edges(panel, seed, spec, args.chunk_size)
                rows.append(row)
                all_rows.append(row)
            seed_out.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(seed_out / "rng_edge_temporal_distance_summary.csv", index=False)

    workspace.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(all_rows)
    if not frame.empty:
        frame = frame.sort_values(["panel", "seed", "metric_label"]).reset_index(drop=True)
    frame.to_csv(workspace / "all_rng_edge_temporal_distance_summary.csv", index=False)
    summarize_across_seeds(frame).to_csv(workspace / "rng_edge_temporal_distance_seed_summary.csv", index=False)


if __name__ == "__main__":
    main()
