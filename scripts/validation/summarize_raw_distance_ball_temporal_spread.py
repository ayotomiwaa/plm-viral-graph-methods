#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.graph_construction.build_panel_nj_distance_reference_trees import parse_seed_list  # noqa: E402
from scripts.validation.summarize_rng_ball_temporal_spread import (  # noqa: E402
    choose_centers,
    collection_time_labels_from_date_arrays,
    locally_shuffled_date_arrays,
    log,
    node_date_arrays,
    parse_float_list,
    powers_of_two_radii,
    read_nodes,
    stable_date_shuffle_seed,
    summarize_ball,
    summarize_by_radius,
    summarize_center_time_correlation_by_radius,
    shuffled_date_arrays,
    write_boxplot,
    write_time_bin_radius_grid_boxplot,
    write_time_correlation_by_radius_plot,
)


DEFAULT_WORKSPACE = Path(
    "analysis/cohort_validation/15_seed42_20k/raw_distance_ball_temporal_spread/"
    "pow2_local_shuffle"
)


def raw_metric_specs(panel_root: Path, sample_label: str, names: str) -> list[dict[str, Any]]:
    available = {
        "raw_hamming": {
            "metric_name": "raw_hamming",
            "metric_family": "hamming",
            "embedding_metric": "",
            "metric_label": "hamming",
            "matrix_path": panel_root
            / f"graphs/hamming/{sample_label}/distance_matrices/"
            "hamming_count-gap-state_all_states_uint16.npy",
            "nodes_path": panel_root / f"graphs/hamming/{sample_label}/hamming_rng_exact/nodes.csv",
        },
        "raw_embedding_cityblock": {
            "metric_name": "raw_embedding_cityblock",
            "metric_family": "embedding",
            "embedding_metric": "cityblock",
            "metric_label": "manhattan",
            "matrix_path": panel_root
            / f"graphs/esm2_650M/cityblock/{sample_label}/distance_matrices/"
            "embedding_cityblock_float32.npy",
            "nodes_path": panel_root
            / f"graphs/esm2_650M/cityblock/{sample_label}/embedding_rng_exact/nodes.csv",
        },
    }
    requested = [part.strip() for part in names.split(",") if part.strip()]
    unknown = sorted(set(requested).difference(available))
    if unknown:
        raise ValueError(f"Unknown raw metric names: {unknown}; available={sorted(available)}")
    return [available[name] for name in requested]


def validate_nodes_and_matrix(nodes: pd.DataFrame, matrix: np.ndarray, spec: dict[str, Any]) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{spec['matrix_path']} is not square: shape={matrix.shape}")
    n_nodes = int(matrix.shape[0])
    node_ids = np.sort(nodes["node_id"].to_numpy(dtype=np.int64))
    expected = np.arange(n_nodes, dtype=np.int64)
    if node_ids.shape != expected.shape or not np.array_equal(node_ids, expected):
        raise ValueError(
            f"{spec['nodes_path']} node IDs do not match matrix rows 0..{n_nodes - 1}"
        )


def distance_row(matrix: np.ndarray, center: int) -> np.ndarray:
    row = np.asarray(matrix[int(center)], dtype=np.float64).copy()
    row[int(center)] = 0.0
    off_diagonal = np.ones(row.shape[0], dtype=bool)
    off_diagonal[int(center)] = False
    if not np.isfinite(row[off_diagonal]).all():
        bad = np.flatnonzero(off_diagonal & ~np.isfinite(row))[:5]
        raise ValueError(f"Distance row {center} has non-finite off-diagonal values at {bad.tolist()}")
    if (row[off_diagonal] < 0).any():
        bad = np.flatnonzero(off_diagonal & (row < 0))[:5]
        raise ValueError(f"Distance row {center} has negative values at {bad.tolist()}")
    return row


def calibrate_raw_radii(
    matrix: np.ndarray,
    centers: np.ndarray,
    calibration_centers: int,
    radius_mode: str,
    quantiles: list[float],
    include_zero: bool,
) -> list[float]:
    sampled: list[np.ndarray] = []
    for center in centers[: min(calibration_centers, int(centers.size))]:
        row = distance_row(matrix, int(center))
        positive = row[np.isfinite(row) & (row > 0)]
        if positive.size:
            sampled.append(positive)
    if not sampled:
        raise ValueError("Could not calibrate raw-distance radii: no positive finite distances")
    pooled = np.concatenate(sampled)
    if radius_mode == "powers-of-two":
        return powers_of_two_radii(float(np.max(pooled)), include_zero=include_zero)
    if not quantiles:
        raise ValueError("At least one radius quantile is required for --radius-mode quantile")
    radii = [float(value) for value in np.quantile(pooled, quantiles)]
    if include_zero:
        radii.insert(0, 0.0)
    return sorted(dict.fromkeys(radii))


def date_assignments(
    panel: str,
    seed: int,
    metric_name: str,
    date_days: np.ndarray,
    valid_dates: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    assignments: list[dict[str, Any]] = [
        {
            "date_assignment": "observed",
            "date_shuffle_index": -1,
            "date_days": date_days,
            "valid_dates": valid_dates,
            "labels": collection_time_labels_from_date_arrays(date_days, valid_dates),
            "shuffle_stats": {
                "date_shuffle_window_days": math.nan,
                "date_shuffle_attempts_per_node": 0,
                "date_shuffle_proposed_swaps": 0,
                "date_shuffle_accepted_swaps": 0,
                "date_shuffle_rejected_swaps": 0,
                "date_shuffle_moved_nodes": 0,
                "date_shuffle_max_displacement_days": 0,
            },
        }
    ]
    for shuffle_index in range(1, args.date_shuffle_count + 1):
        shuffle_seed = stable_date_shuffle_seed(
            args.date_shuffle_seed,
            panel=panel,
            seed=seed,
            graph_name=metric_name,
            shuffle_index=shuffle_index,
        )
        if args.date_shuffle_max_window_days >= 0:
            shuffled_days, shuffled_valid, shuffle_stats = locally_shuffled_date_arrays(
                date_days,
                valid_dates,
                shuffle_seed=shuffle_seed,
                max_window_days=args.date_shuffle_max_window_days,
                attempts_per_node=args.date_shuffle_attempts_per_node,
            )
        else:
            shuffled_days, shuffled_valid = shuffled_date_arrays(
                date_days, valid_dates, shuffle_seed=shuffle_seed
            )
            displacement = np.abs(shuffled_days[valid_dates] - date_days[valid_dates])
            moved = int(np.count_nonzero(displacement))
            shuffle_stats = {
                "date_shuffle_window_days": -1,
                "date_shuffle_attempts_per_node": 0,
                "date_shuffle_proposed_swaps": 0,
                "date_shuffle_accepted_swaps": moved,
                "date_shuffle_rejected_swaps": 0,
                "date_shuffle_moved_nodes": moved,
                "date_shuffle_max_displacement_days": int(displacement.max(initial=0)),
            }
        assignments.append(
            {
                "date_assignment": f"date_shuffle_{shuffle_index:03d}",
                "date_shuffle_index": shuffle_index,
                "date_days": shuffled_days,
                "valid_dates": shuffled_valid,
                "labels": collection_time_labels_from_date_arrays(shuffled_days, shuffled_valid),
                "shuffle_stats": shuffle_stats,
            }
        )
    return assignments


def summarize_raw_metric_balls(
    panel: str,
    seed: int,
    spec: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrix_path = Path(spec["matrix_path"])
    nodes_path = Path(spec["nodes_path"])
    if not matrix_path.exists() or not nodes_path.exists():
        raise FileNotFoundError(f"Missing raw metric inputs: matrix={matrix_path}, nodes={nodes_path}")

    matrix = np.load(matrix_path, mmap_mode="r")
    nodes = read_nodes(nodes_path)
    validate_nodes_and_matrix(nodes, matrix, spec)
    n_nodes = int(matrix.shape[0])
    date_days, valid_dates = node_date_arrays(nodes, n_nodes=n_nodes)
    centers = choose_centers(
        node_ids=np.arange(n_nodes, dtype=np.int64),
        max_centers=args.max_centers,
        center_mode=args.center_mode,
        seed=args.center_seed,
        explicit_centers=args.centers,
    )
    if args.radii:
        radii = sorted(dict.fromkeys(parse_float_list(args.radii)))
        radius_selection = "explicit"
    else:
        radii = calibrate_raw_radii(
            matrix,
            centers=centers,
            calibration_centers=args.calibration_centers,
            radius_mode=args.radius_mode,
            quantiles=parse_float_list(args.radius_quantiles),
            include_zero=not args.exclude_zero_radius,
        )
        radius_selection = args.radius_mode

    assignments = date_assignments(
        panel,
        seed,
        metric_name=str(spec["metric_name"]),
        date_days=date_days,
        valid_dates=valid_dates,
        args=args,
    )
    rows_by_assignment: dict[str, list[dict[str, Any]]] = {
        str(item["date_assignment"]): [] for item in assignments
    }
    log(
        f"Summarizing raw-distance balls: {panel}/seed_{seed} {spec['metric_name']} "
        f"centers={len(centers):,} radii={len(radii):,} date_shuffles={args.date_shuffle_count:,}"
    )
    for index, center in enumerate(centers, start=1):
        if args.progress_every and (index == 1 or index % args.progress_every == 0 or index == len(centers)):
            log(f"  Raw-distance center {index:,}/{len(centers):,}: node_id={int(center)}")
        distances = distance_row(matrix, int(center))
        finite_count = int(np.isfinite(distances).sum())
        for radius in radii:
            for assignment in assignments:
                center_dates, center_months, center_quarters = assignment["labels"]
                row = summarize_ball(
                    int(center),
                    distances,
                    radius,
                    assignment["date_days"],
                    assignment["valid_dates"],
                )
                row.update(
                    {
                        "panel": panel,
                        "seed": int(seed),
                        "graph_name": spec["metric_name"],
                        "metric_family": spec["metric_family"],
                        "embedding_metric": spec["embedding_metric"],
                        "metric_label": spec["metric_label"],
                        "graph_family": "raw_distance",
                        "graph_distance_mode": "raw_pairwise_distance",
                        "radius_selection": radius_selection,
                        "date_assignment": assignment["date_assignment"],
                        "date_shuffle_index": int(assignment["date_shuffle_index"]),
                        **assignment["shuffle_stats"],
                        "center_collection_date": center_dates[int(center)],
                        "center_collection_month": center_months[int(center)],
                        "center_collection_quarter": center_quarters[int(center)],
                        "n_finite_distances_from_center": finite_count,
                        "distance_matrix_path": str(matrix_path),
                        "nodes_path": str(nodes_path),
                    }
                )
                rows_by_assignment[str(assignment["date_assignment"])].append(row)

    observed = pd.DataFrame(rows_by_assignment["observed"])
    observed_summary = summarize_by_radius(observed)
    shuffled_frames = [
        pd.DataFrame(rows)
        for label, rows in rows_by_assignment.items()
        if label != "observed" and rows
    ]
    shuffled = pd.concat(shuffled_frames, ignore_index=True) if shuffled_frames else pd.DataFrame()
    shuffled_summary = summarize_by_radius(shuffled) if not shuffled.empty else pd.DataFrame()
    return observed, observed_summary, shuffled, shuffled_summary


def write_metric_outputs(
    seed_out: Path,
    panel: str,
    seed: int,
    spec: dict[str, Any],
    observed: pd.DataFrame,
    observed_summary: pd.DataFrame,
    shuffled: pd.DataFrame,
    shuffled_summary: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prefix = f"{panel}_seed_{seed}_{spec['metric_name']}"
    observed_path = seed_out / f"{prefix}_raw_distance_ball_temporal_spread.csv"
    summary_path = seed_out / f"{prefix}_raw_distance_ball_temporal_spread_radius_summary.csv"
    observed.to_csv(observed_path, index=False)
    observed_summary.to_csv(summary_path, index=False)
    xlabel = "Raw pairwise-distance radius"
    write_boxplot(
        observed,
        "max_pairwise_delta_days",
        "Ball max pairwise collection-date delta (days)",
        seed_out / f"{prefix}_max_delta_days_boxplot.png",
        xlabel=xlabel,
    )
    write_boxplot(
        observed,
        "mean_pairwise_delta_days",
        "Ball mean pairwise collection-date delta (days)",
        seed_out / f"{prefix}_mean_delta_days_boxplot.png",
        xlabel=xlabel,
    )
    write_boxplot(
        observed,
        "mean_pairwise_delta_days",
        "Mean pairwise sequence collection-time difference (days)",
        seed_out / f"{prefix}_sequence_time_difference_by_radius_boxplot.png",
        xlabel=xlabel,
    )
    time_bin_col = f"center_collection_{args.time_bin}"
    write_time_bin_radius_grid_boxplot(
        observed,
        "mean_pairwise_delta_days",
        "Ball mean pairwise collection-date delta (days)",
        time_bin_col,
        f"Center collection {args.time_bin}",
        args.time_plot_min_bin_count,
        seed_out / f"{prefix}_mean_delta_days_by_center_{args.time_bin}_boxplot.png",
    )
    observed_corr = summarize_center_time_correlation_by_radius(
        observed, value_col="mean_pairwise_delta_days"
    )
    observed_corr_path = seed_out / f"{prefix}_mean_delta_days_center_time_correlation_by_radius.csv"
    observed_corr.to_csv(observed_corr_path, index=False)
    write_time_correlation_by_radius_plot(
        observed_corr,
        "pearson_r",
        "Pearson r: center collection date vs ball mean delta",
        seed_out / f"{prefix}_mean_delta_days_center_time_correlation_by_radius.png",
        xlabel=xlabel,
    )

    shuffled_corr = pd.DataFrame()
    if not shuffled.empty:
        shuffled_path = seed_out / f"{prefix}_raw_distance_ball_temporal_spread_shuffled_dates.csv"
        shuffled_summary_path = (
            seed_out / f"{prefix}_raw_distance_ball_temporal_spread_shuffled_dates_radius_summary.csv"
        )
        shuffled.to_csv(shuffled_path, index=False)
        shuffled_summary.to_csv(shuffled_summary_path, index=False)
        write_boxplot(
            shuffled,
            "max_pairwise_delta_days",
            "Shuffled-date ball max pairwise collection-date delta (days)",
            seed_out / f"{prefix}_shuffled_dates_max_delta_days_boxplot.png",
            xlabel=xlabel,
        )
        write_boxplot(
            shuffled,
            "mean_pairwise_delta_days",
            "Shuffled-date ball mean pairwise collection-date delta (days)",
            seed_out / f"{prefix}_shuffled_dates_mean_delta_days_boxplot.png",
            xlabel=xlabel,
        )
        write_boxplot(
            shuffled,
            "mean_pairwise_delta_days",
            "Shuffled-date mean pairwise sequence collection-time difference (days)",
            seed_out / f"{prefix}_shuffled_dates_sequence_time_difference_by_radius_boxplot.png",
            xlabel=xlabel,
        )
        write_time_bin_radius_grid_boxplot(
            shuffled,
            "mean_pairwise_delta_days",
            "Shuffled-date ball mean pairwise collection-date delta (days)",
            time_bin_col,
            f"Shuffled center collection {args.time_bin}",
            args.time_plot_min_bin_count,
            seed_out / f"{prefix}_shuffled_dates_mean_delta_days_by_center_{args.time_bin}_boxplot.png",
        )
        shuffled_corr = summarize_center_time_correlation_by_radius(
            shuffled, value_col="mean_pairwise_delta_days"
        )
        shuffled_corr_path = (
            seed_out / f"{prefix}_shuffled_dates_mean_delta_days_center_time_correlation_by_radius.csv"
        )
        shuffled_corr.to_csv(shuffled_corr_path, index=False)
        write_time_correlation_by_radius_plot(
            shuffled_corr,
            "pearson_r",
            "Shuffled-date Pearson r: center collection date vs ball mean delta",
            seed_out / f"{prefix}_shuffled_dates_mean_delta_days_center_time_correlation_by_radius.png",
            xlabel=xlabel,
        )
    log(f"Wrote {observed_path}")
    log(f"Wrote {summary_path}")
    return observed_corr, shuffled_corr


def write_combined_and_metric_summaries(
    frames: list[pd.DataFrame], workspace: Path, stem: str
) -> None:
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(workspace / f"all_{stem}.csv", index=False)
    for metric_name, metric_frame in combined.groupby("graph_name", sort=False):
        metric_frame.to_csv(workspace / f"all_{metric_name}_{stem}.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Summarize collection-date spread inside balls defined directly by raw Hamming or "
            "embedding-cityblock pairwise distance matrices."
        )
    )
    ap.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    ap.add_argument(
        "--source-root",
        type=Path,
        default=Path("analysis/cohort_validation/07_sampling_design_20k"),
    )
    ap.add_argument("--panels", default="random_full_dataset_seed42")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--sample-label", default="pool_n20000")
    ap.add_argument("--metric-names", default="raw_hamming,raw_embedding_cityblock")
    ap.add_argument("--radii", default="", help="Comma-separated raw-distance radii; overrides --radius-mode.")
    ap.add_argument("--radius-mode", choices=["quantile", "powers-of-two"], default="powers-of-two")
    ap.add_argument(
        "--radius-quantiles",
        default="0.01,0.02,0.05,0.1,0.2,0.35,0.5,0.75,0.9,1.0",
    )
    ap.add_argument("--exclude-zero-radius", action="store_true")
    ap.add_argument("--calibration-centers", type=int, default=64)
    ap.add_argument("--centers", default="")
    ap.add_argument("--max-centers", type=int, default=None)
    ap.add_argument("--center-mode", choices=["first", "random"], default="first")
    ap.add_argument("--center-seed", type=int, default=42)
    ap.add_argument("--time-bin", choices=["month", "quarter", "year"], default="quarter")
    ap.add_argument("--time-plot-min-bin-count", type=int, default=10)
    ap.add_argument("--date-shuffle-count", type=int, default=1)
    ap.add_argument("--date-shuffle-seed", type=int, default=42)
    ap.add_argument("--date-shuffle-max-window-days", type=int, default=62)
    ap.add_argument("--date-shuffle-attempts-per-node", type=int, default=20)
    ap.add_argument("--progress-every", type=int, default=250)
    args = ap.parse_args()
    if args.date_shuffle_count < 0:
        raise ValueError("--date-shuffle-count must be non-negative")
    if args.date_shuffle_attempts_per_node < 0:
        raise ValueError("--date-shuffle-attempts-per-node must be non-negative")

    args.workspace.mkdir(parents=True, exist_ok=True)
    observed_frames: list[pd.DataFrame] = []
    observed_summary_frames: list[pd.DataFrame] = []
    observed_corr_frames: list[pd.DataFrame] = []
    shuffled_frames: list[pd.DataFrame] = []
    shuffled_summary_frames: list[pd.DataFrame] = []
    shuffled_corr_frames: list[pd.DataFrame] = []
    panels = [part.strip() for part in args.panels.split(",") if part.strip()]
    for panel in panels:
        for seed in parse_seed_list(args.seeds):
            panel_root = args.source_root / panel / f"seed_{seed}"
            if not panel_root.exists():
                log(f"Skipping missing panel seed root: {panel_root}")
                continue
            seed_out = args.workspace / panel / f"seed_{seed}"
            seed_out.mkdir(parents=True, exist_ok=True)
            for spec in raw_metric_specs(panel_root, args.sample_label, args.metric_names):
                observed, observed_summary, shuffled, shuffled_summary = summarize_raw_metric_balls(
                    panel, seed, spec, args
                )
                observed_corr, shuffled_corr = write_metric_outputs(
                    seed_out,
                    panel,
                    seed,
                    spec,
                    observed,
                    observed_summary,
                    shuffled,
                    shuffled_summary,
                    args,
                )
                observed_frames.append(observed)
                observed_summary_frames.append(observed_summary)
                observed_corr_frames.append(observed_corr)
                if not shuffled.empty:
                    shuffled_frames.append(shuffled)
                    shuffled_summary_frames.append(shuffled_summary)
                    shuffled_corr_frames.append(shuffled_corr)

    write_combined_and_metric_summaries(
        observed_frames, args.workspace, "raw_distance_ball_temporal_spread"
    )
    write_combined_and_metric_summaries(
        observed_summary_frames,
        args.workspace,
        "raw_distance_ball_temporal_spread_radius_summary",
    )
    write_combined_and_metric_summaries(
        observed_corr_frames,
        args.workspace,
        "raw_distance_ball_temporal_spread_center_time_correlation_by_radius",
    )
    write_combined_and_metric_summaries(
        shuffled_frames,
        args.workspace,
        "raw_distance_ball_temporal_spread_shuffled_dates",
    )
    write_combined_and_metric_summaries(
        shuffled_summary_frames,
        args.workspace,
        "raw_distance_ball_temporal_spread_shuffled_dates_radius_summary",
    )
    write_combined_and_metric_summaries(
        shuffled_corr_frames,
        args.workspace,
        "raw_distance_ball_temporal_spread_shuffled_dates_center_time_correlation_by_radius",
    )


if __name__ == "__main__":
    main()
