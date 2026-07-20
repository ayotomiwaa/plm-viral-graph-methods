#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import dijkstra  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.graph_construction.build_panel_nj_distance_reference_trees import (  # noqa: E402
    parse_seed_list,
)
from scripts.validation.summarize_rng_edge_temporal_distances import (  # noqa: E402
    metric_label,
    wanted_rng_graphs,
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_float_list(value: str) -> list[float]:
    out: list[float] = []
    for part in value.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out


def read_nodes(nodes_path: Path) -> pd.DataFrame:
    nodes = pd.read_csv(nodes_path, low_memory=False)
    required = {"node_id", "collection_date"}
    missing = required.difference(nodes.columns)
    if missing:
        raise ValueError(f"{nodes_path} is missing required columns: {sorted(missing)}")
    nodes["node_id"] = pd.to_numeric(nodes["node_id"], errors="raise").astype(int)
    return nodes


def read_kept_edges(edges_path: Path) -> pd.DataFrame:
    edges = pd.read_csv(edges_path, low_memory=False)
    required = {"source", "target", "weight"}
    missing = required.difference(edges.columns)
    if missing:
        raise ValueError(f"{edges_path} is missing required columns: {sorted(missing)}")
    if "status" in edges.columns:
        edges = edges[edges["status"].astype(str) == "kept"].copy()
    edges["source"] = pd.to_numeric(edges["source"], errors="raise").astype(int)
    edges["target"] = pd.to_numeric(edges["target"], errors="raise").astype(int)
    edges["weight"] = pd.to_numeric(edges["weight"], errors="raise").astype(float)
    if (edges["weight"] < 0).any():
        raise ValueError(f"{edges_path} contains negative edge weights")
    return edges


def build_graph(edges: pd.DataFrame, n_nodes: int, unweighted: bool):
    weight = np.ones(edges.shape[0], dtype=np.float64) if unweighted else edges["weight"].to_numpy(dtype=np.float64)
    source = edges["source"].to_numpy(dtype=np.int64)
    target = edges["target"].to_numpy(dtype=np.int64)
    rows = np.concatenate([source, target])
    cols = np.concatenate([target, source])
    data = np.concatenate([weight, weight])
    return coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()


def node_date_arrays(nodes: pd.DataFrame, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    dates = pd.to_datetime(nodes["collection_date"], errors="coerce")
    node_ids = nodes["node_id"].to_numpy(dtype=np.int64)
    date_days = np.zeros(n_nodes, dtype=np.int64)
    valid = np.zeros(n_nodes, dtype=bool)
    valid_mask = dates.notna().to_numpy()
    if valid_mask.any():
        valid_node_ids = node_ids[valid_mask]
        date_days[valid_node_ids] = dates[valid_mask].to_numpy(dtype="datetime64[D]").astype(np.int64)
        valid[valid_node_ids] = True
    return date_days, valid


def node_collection_time_labels(nodes: pd.DataFrame, n_nodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dates = pd.to_datetime(nodes["collection_date"], errors="coerce")
    node_ids = nodes["node_id"].to_numpy(dtype=np.int64)
    date_labels = np.full(n_nodes, "", dtype=object)
    month_labels = np.full(n_nodes, "", dtype=object)
    quarter_labels = np.full(n_nodes, "", dtype=object)
    valid_mask = dates.notna().to_numpy()
    if valid_mask.any():
        valid_node_ids = node_ids[valid_mask]
        valid_dates = dates[valid_mask]
        date_labels[valid_node_ids] = valid_dates.dt.strftime("%Y-%m-%d").to_numpy(dtype=object)
        month_labels[valid_node_ids] = valid_dates.dt.strftime("%Y-%m").to_numpy(dtype=object)
        quarter_labels[valid_node_ids] = valid_dates.dt.to_period("Q").astype(str).to_numpy(dtype=object)
    return date_labels, month_labels, quarter_labels


def mean_pairwise_abs_delta_days(sorted_days: np.ndarray) -> float:
    n = int(sorted_days.size)
    if n <= 1:
        return 0.0 if n == 1 else math.nan
    values = sorted_days.astype(np.float64, copy=False)
    prefix_before = np.concatenate([[0.0], np.cumsum(values[:-1])])
    pair_sum = float(np.sum(values * np.arange(n, dtype=np.float64) - prefix_before))
    return pair_sum / (n * (n - 1) / 2)


def summarize_ball(
    center: int,
    distances: np.ndarray,
    radius: float,
    date_days: np.ndarray,
    valid_dates: np.ndarray,
) -> dict[str, Any]:
    in_ball = np.isfinite(distances) & (distances <= radius)
    ball_node_ids = np.flatnonzero(in_ball)
    valid_ball_node_ids = ball_node_ids[valid_dates[ball_node_ids]]
    dates = np.sort(date_days[valid_ball_node_ids])
    n_valid = int(dates.size)
    n_pairs = int(n_valid * (n_valid - 1) // 2)
    max_delta = float(dates[-1] - dates[0]) if n_valid >= 1 else math.nan
    mean_delta = mean_pairwise_abs_delta_days(dates)
    return {
        "center_node_id": int(center),
        "radius": float(radius),
        "n_ball_nodes": int(ball_node_ids.size),
        "n_ball_nodes_with_valid_dates": n_valid,
        "n_date_pairs": n_pairs,
        "max_pairwise_delta_days": max_delta,
        "mean_pairwise_delta_days": mean_delta,
    }


def choose_centers(
    node_ids: np.ndarray,
    max_centers: int | None,
    center_mode: str,
    seed: int,
    explicit_centers: str,
) -> np.ndarray:
    if explicit_centers:
        wanted = np.array([int(x) for x in explicit_centers.split(",") if x.strip()], dtype=np.int64)
        missing = sorted(set(wanted.tolist()).difference(set(node_ids.tolist())))
        if missing:
            raise ValueError(f"Requested center node IDs not present in graph: {missing[:10]}")
        return wanted
    if max_centers is None or max_centers >= int(node_ids.size):
        return node_ids
    if center_mode == "first":
        return node_ids[:max_centers]
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(node_ids, size=max_centers, replace=False))


def calibrate_radii(
    graph,
    centers: np.ndarray,
    quantiles: list[float],
    include_zero: bool,
    calibration_centers: int,
) -> list[float]:
    if not quantiles:
        raise ValueError("At least one radius quantile is required when --radii is not provided")
    sample_centers = centers[: min(calibration_centers, int(centers.size))]
    values: list[np.ndarray] = []
    for center in sample_centers:
        distances = dijkstra(graph, directed=False, indices=int(center), unweighted=False)
        finite = distances[np.isfinite(distances) & (distances > 0)]
        if finite.size:
            values.append(finite.astype(np.float64, copy=False))
    if not values:
        raise ValueError("Could not calibrate radii because no positive finite graph distances were found")
    pooled = np.concatenate(values)
    radii = [float(x) for x in np.quantile(pooled, quantiles)]
    if include_zero:
        radii.insert(0, 0.0)
    return sorted(dict.fromkeys(radii))


def boxplot_stats(values: pd.Series) -> dict[str, float | int]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    n = int(clean.shape[0])
    if n == 0:
        return {"n_balls": 0, "min": math.nan, "q1": math.nan, "median": math.nan, "q3": math.nan, "max": math.nan, "mean": math.nan}
    return {
        "n_balls": n,
        "min": float(clean.min()),
        "q1": float(clean.quantile(0.25)),
        "median": float(clean.quantile(0.50)),
        "q3": float(clean.quantile(0.75)),
        "max": float(clean.max()),
        "mean": float(clean.mean()),
    }


def summarize_by_radius(ball_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["panel", "seed", "graph_name", "metric_label", "radius"]
    for keys, group in ball_frame.groupby(group_cols, dropna=False):
        out = {col: key for col, key in zip(group_cols, keys)}
        out.update({f"max_delta_days_{key}": value for key, value in boxplot_stats(group["max_pairwise_delta_days"]).items()})
        out.update({f"mean_delta_days_{key}": value for key, value in boxplot_stats(group["mean_pairwise_delta_days"]).items()})
        out["n_ball_nodes_median"] = float(pd.to_numeric(group["n_ball_nodes"], errors="coerce").median())
        out["n_ball_nodes_max"] = int(pd.to_numeric(group["n_ball_nodes"], errors="coerce").max())
        rows.append(out)
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["panel", "seed", "metric_label", "radius"]).reset_index(drop=True)
    return result


def load_pyplot(out_path: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        log(f"Skipping plot {out_path}: missing Python package {exc.name!r}")
        return None
    return plt


def safe_radius_label(radius: float) -> str:
    label = f"{radius:.6g}"
    return label.replace("-", "m").replace(".", "p")


def write_boxplot(ball_frame: pd.DataFrame, value_col: str, ylabel: str, out_path: Path) -> None:
    plt = load_pyplot(out_path)
    if plt is None:
        return

    radii = sorted(pd.to_numeric(ball_frame["radius"], errors="coerce").dropna().unique())
    data = [
        pd.to_numeric(ball_frame.loc[ball_frame["radius"] == radius, value_col], errors="coerce").dropna().to_numpy()
        for radius in radii
    ]
    fig_width = max(8.0, min(18.0, 0.55 * len(radii) + 4.0))
    fig, ax = plt.subplots(figsize=(fig_width, 5.5), constrained_layout=True)
    ax.boxplot(data, tick_labels=[f"{radius:.4g}" for radius in radii], showfliers=False)
    ax.set_xlabel("RNG graph-distance radius")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_time_bin_radius_grid_boxplot(
    ball_frame: pd.DataFrame,
    value_col: str,
    ylabel: str,
    time_bin_col: str,
    time_bin_label: str,
    min_bin_count: int,
    out_path: Path,
) -> None:
    plt = load_pyplot(out_path)
    if plt is None:
        return
    if time_bin_col not in ball_frame.columns:
        log(f"Skipping plot {out_path}: missing column {time_bin_col!r}")
        return

    frame = ball_frame.copy()
    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
    frame["radius"] = pd.to_numeric(frame["radius"], errors="coerce")
    frame[time_bin_col] = frame[time_bin_col].astype(str)
    frame = frame[(frame[time_bin_col] != "") & frame[value_col].notna() & frame["radius"].notna()]
    if frame.empty:
        log(f"Skipping plot {out_path}: no valid rows")
        return

    radii = sorted(frame["radius"].unique())
    n_cols = min(3, len(radii))
    n_rows = int(math.ceil(len(radii) / n_cols))
    fig_width = max(11.0, min(24.0, 7.0 * n_cols))
    fig_height = max(4.5, 3.8 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), sharey=True, constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, radius in zip(axes_flat, radii):
        subset = frame[frame["radius"] == radius]
        counts = subset.groupby(time_bin_col)[value_col].count()
        bins = [str(idx) for idx, count in counts.items() if int(count) >= min_bin_count]
        bins = sorted(bins)
        if not bins:
            ax.set_axis_off()
            ax.set_title(f"R = {radius:.4g}; no bins >= {min_bin_count}")
            continue
        data = [subset.loc[subset[time_bin_col] == bin_label, value_col].dropna().to_numpy() for bin_label in bins]
        ax.boxplot(data, tick_labels=bins, showfliers=False)
        ax.set_title(f"R = {radius:.4g}")
        ax.tick_params(axis="x", labelrotation=60, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
    for ax in axes_flat[len(radii) :]:
        ax.set_axis_off()
    for ax in axes_flat[::n_cols]:
        ax.set_ylabel(ylabel)
    for ax in axes_flat[-n_cols:]:
        ax.set_xlabel(time_bin_label)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def output_prefix(panel: str, seed: int, spec: dict[str, Any]) -> str:
    return f"{panel}_seed_{seed}_{spec['graph_name']}"


def summarize_graph_balls(
    panel: str,
    seed: int,
    spec: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    graph_dir = Path(spec["graph_dir"])
    nodes_path = graph_dir / "nodes.csv"
    edges_path = graph_dir / "edges.csv"
    if not nodes_path.exists() or not edges_path.exists():
        raise FileNotFoundError(f"Missing RNG graph nodes/edges in {graph_dir}")

    nodes = read_nodes(nodes_path)
    edges = read_kept_edges(edges_path)
    max_node_id = int(max(nodes["node_id"].max(), edges["source"].max(), edges["target"].max()))
    n_nodes = max_node_id + 1
    graph = build_graph(edges, n_nodes=n_nodes, unweighted=args.unweighted)
    date_days, valid_dates = node_date_arrays(nodes, n_nodes=n_nodes)
    center_dates, center_months, center_quarters = node_collection_time_labels(nodes, n_nodes=n_nodes)
    node_ids = np.sort(nodes["node_id"].to_numpy(dtype=np.int64))
    centers = choose_centers(
        node_ids=node_ids,
        max_centers=args.max_centers,
        center_mode=args.center_mode,
        seed=args.center_seed,
        explicit_centers=args.centers,
    )
    if args.radii:
        radii = sorted(dict.fromkeys(parse_float_list(args.radii)))
    else:
        radii = calibrate_radii(
            graph,
            centers=centers,
            quantiles=parse_float_list(args.radius_quantiles),
            include_zero=not args.exclude_zero_radius,
            calibration_centers=args.calibration_centers,
        )

    log(
        "Summarizing RNG balls: "
        f"{panel}/seed_{seed} {spec['graph_name']} centers={len(centers):,} radii={len(radii):,}"
    )
    rows: list[dict[str, Any]] = []
    for index, center in enumerate(centers, start=1):
        if args.progress_every and (index == 1 or index % args.progress_every == 0 or index == len(centers)):
            log(f"  Dijkstra center {index:,}/{len(centers):,}: node_id={int(center)}")
        distances = dijkstra(graph, directed=False, indices=int(center), unweighted=args.unweighted)
        reachable = int(np.isfinite(distances).sum())
        for radius in radii:
            row = summarize_ball(int(center), distances, radius, date_days, valid_dates)
            row.update(
                {
                    "panel": panel,
                    "seed": int(seed),
                    "graph_name": spec["graph_name"],
                    "metric_family": spec["metric_family"],
                    "embedding_metric": spec["embedding_metric"],
                    "metric_label": metric_label(spec),
                    "graph_family": spec["graph_family"],
                    "graph_distance_mode": "unweighted_hops" if args.unweighted else "weighted_edge_distance",
                    "center_collection_date": center_dates[int(center)],
                    "center_collection_month": center_months[int(center)],
                    "center_collection_quarter": center_quarters[int(center)],
                    "n_reachable_nodes_from_center": reachable,
                    "graph_dir": str(graph_dir),
                    "nodes_path": str(nodes_path),
                    "edges_path": str(edges_path),
                }
            )
            rows.append(row)
    ball_frame = pd.DataFrame(rows)
    summary = summarize_by_radius(ball_frame)
    return ball_frame, summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run Dijkstra from RNG graph centers, summarize collection-date spread "
            "inside graph-distance balls, and plot boxplot distributions across radii."
        )
    )
    ap.add_argument(
        "--workspace",
        type=Path,
        default=Path("analysis/cohort_validation/15_random_full_dataset_2k_rng_ball_temporal_spread"),
    )
    ap.add_argument("--source-root", type=Path, default=Path("analysis/cohort_validation/08_sampling_design_2k"))
    ap.add_argument("--panels", default="random_full_dataset")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--sample-label", default="pool_n2000")
    ap.add_argument(
        "--graph-names",
        default="embedding_cityblock_rng_exact",
        help="Comma-separated RNG graph names; defaults to the cityblock embedding RNG.",
    )
    ap.add_argument("--radii", default="", help="Comma-separated graph-distance radii. If empty, radii are calibrated.")
    ap.add_argument(
        "--radius-quantiles",
        default="0.01,0.02,0.05,0.1,0.2,0.35,0.5,0.75,0.9,1.0",
        help="Distance quantiles used to calibrate radii when --radii is empty.",
    )
    ap.add_argument("--exclude-zero-radius", action="store_true", help="Do not prepend radius 0 when calibrating radii.")
    ap.add_argument("--calibration-centers", type=int, default=64)
    ap.add_argument("--centers", default="", help="Comma-separated center node IDs. Empty means use graph nodes.")
    ap.add_argument("--max-centers", type=int, default=None, help="Optional cap for smoke tests or sampled runs.")
    ap.add_argument("--center-mode", choices=["first", "random"], default="first")
    ap.add_argument("--center-seed", type=int, default=42)
    ap.add_argument("--unweighted", action="store_true", help="Use hop-count balls instead of weighted RNG edge distances.")
    ap.add_argument("--time-bin", choices=["month", "quarter", "year"], default="quarter")
    ap.add_argument("--time-plot-min-bin-count", type=int, default=10)
    ap.add_argument("--progress-every", type=int, default=250)
    args = ap.parse_args()

    panels = [item.strip() for item in args.panels.split(",") if item.strip()]
    seeds = parse_seed_list(args.seeds)
    args.workspace.mkdir(parents=True, exist_ok=True)

    all_ball_frames: list[pd.DataFrame] = []
    all_summary_frames: list[pd.DataFrame] = []
    for panel in panels:
        for seed in seeds:
            panel_root = args.source_root / panel / f"seed_{seed}"
            if not panel_root.exists():
                log(f"Skipping missing panel seed root: {panel_root}")
                continue
            seed_out = args.workspace / panel / f"seed_{seed}"
            seed_out.mkdir(parents=True, exist_ok=True)
            for spec in wanted_rng_graphs(panel_root, args.sample_label, args.graph_names):
                graph_dir = Path(spec["graph_dir"])
                if not graph_dir.exists():
                    log(f"Skipping missing RNG graph: {graph_dir}")
                    continue
                ball_frame, summary = summarize_graph_balls(panel, seed, spec, args)
                prefix = output_prefix(panel, seed, spec)
                ball_path = seed_out / f"{prefix}_rng_ball_temporal_spread.csv"
                summary_path = seed_out / f"{prefix}_rng_ball_temporal_spread_radius_summary.csv"
                ball_frame.to_csv(ball_path, index=False)
                summary.to_csv(summary_path, index=False)
                write_boxplot(
                    ball_frame,
                    value_col="max_pairwise_delta_days",
                    ylabel="Ball max pairwise collection-date delta (days)",
                    out_path=seed_out / f"{prefix}_max_delta_days_boxplot.png",
                )
                write_boxplot(
                    ball_frame,
                    value_col="mean_pairwise_delta_days",
                    ylabel="Ball mean pairwise collection-date delta (days)",
                    out_path=seed_out / f"{prefix}_mean_delta_days_boxplot.png",
                )
                write_boxplot(
                    ball_frame,
                    value_col="mean_pairwise_delta_days",
                    ylabel="Mean pairwise sequence collection-time difference (days)",
                    out_path=seed_out / f"{prefix}_sequence_time_difference_by_radius_boxplot.png",
                )
                time_bin_col = f"center_collection_{args.time_bin}"
                write_time_bin_radius_grid_boxplot(
                    ball_frame,
                    value_col="mean_pairwise_delta_days",
                    ylabel="Ball mean pairwise collection-date delta (days)",
                    time_bin_col=time_bin_col,
                    time_bin_label=f"Center collection {args.time_bin}",
                    min_bin_count=args.time_plot_min_bin_count,
                    out_path=seed_out / f"{prefix}_mean_delta_days_by_center_{args.time_bin}_boxplot.png",
                )
                all_ball_frames.append(ball_frame)
                all_summary_frames.append(summary)
                log(f"Wrote {ball_path}")
                log(f"Wrote {summary_path}")

    if all_ball_frames:
        all_balls = pd.concat(all_ball_frames, ignore_index=True)
        all_balls.to_csv(args.workspace / "all_rng_ball_temporal_spread.csv", index=False)
    if all_summary_frames:
        all_summaries = pd.concat(all_summary_frames, ignore_index=True)
        all_summaries.to_csv(args.workspace / "all_rng_ball_temporal_spread_radius_summary.csv", index=False)


if __name__ == "__main__":
    main()
