#!/usr/bin/env python3
"""Audit timestamp ranges and temporal directionability of the seed-42 RNG."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PANEL_ROOT = Path(
    "analysis/cohort_validation/07_sampling_design_20k/"
    "random_full_dataset_seed42/seed_42"
)
DEFAULT_OUT_ROOT = Path(
    "analysis/cohort_validation/31_seed42_20k_rng_timestamp_range_directionality/"
    "random_full_dataset_seed42/seed_42"
)
ALGORITHM_VERSION = 1


def load_module(name: str, relative_path: str):
    path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CALIBRATION = load_module(
    "rng_timestamp_calibration",
    "scripts/validation/evaluate_seed42_rng_degree_dimension_calibration.py",
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def day_to_iso(value: int | float) -> str:
    if not np.isfinite(value):
        return ""
    return str(np.datetime64(int(value), "D"))


def degree_vector(n_nodes: int, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.bincount(
        np.concatenate([source, target]), minlength=n_nodes
    ).astype(np.int64, copy=False)


def degree_bin(value: int) -> str:
    if value <= 5:
        return str(value)
    if value <= 9:
        return "6-9"
    return "10+"


def multiplicity_bin(value: int) -> str:
    if value <= 5:
        return str(value)
    if value <= 9:
        return "6-9"
    if value <= 24:
        return "10-24"
    if value <= 49:
        return "25-49"
    if value <= 99:
        return "50-99"
    return "100+"


def build_date_profiles(
    inverse: np.ndarray,
    collection_dates: pd.Series,
    n_unique: int,
    multiplicity: np.ndarray,
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    parsed = pd.to_datetime(collection_dates, errors="coerce")
    valid = parsed.notna().to_numpy()
    day_values = np.zeros(len(parsed), dtype=np.int64)
    if valid.any():
        day_values[valid] = parsed[valid].to_numpy(dtype="datetime64[D]").astype(np.int64)

    grouped: list[list[int]] = [[] for _ in range(n_unique)]
    for node, day, is_valid in zip(inverse, day_values, valid):
        if is_valid:
            grouped[int(node)].append(int(day))

    arrays: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for node_id, values in enumerate(grouped):
        dates = np.sort(np.asarray(values, dtype=np.int64))
        arrays.append(dates)
        if dates.size:
            first = int(dates[0])
            last = int(dates[-1])
            q25, median, q75 = np.quantile(
                dates, [0.25, 0.5, 0.75], method="nearest"
            ).astype(np.int64)
            span = last - first
        else:
            first = last = q25 = median = q75 = -1
            span = -1
        rows.append(
            {
                "unique_node_id": node_id,
                "coordinate_multiplicity": int(multiplicity[node_id]),
                "n_valid_collection_dates": int(dates.size),
                "n_missing_collection_dates": int(multiplicity[node_id] - dates.size),
                "first_collection_date": day_to_iso(first) if first >= 0 else "",
                "q25_collection_date": day_to_iso(q25) if q25 >= 0 else "",
                "median_collection_date": day_to_iso(median) if median >= 0 else "",
                "q75_collection_date": day_to_iso(q75) if q75 >= 0 else "",
                "last_collection_date": day_to_iso(last) if last >= 0 else "",
                "observed_date_span_days": span if span >= 0 else np.nan,
                "first_day": first if first >= 0 else np.nan,
                "median_day": int(median) if median >= 0 else np.nan,
                "last_day": last if last >= 0 else np.nan,
            }
        )
    return pd.DataFrame(rows), arrays


def pairwise_date_order_probabilities(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float]:
    if left.size == 0 or right.size == 0:
        return math.nan, math.nan, math.nan
    total = int(left.size) * int(right.size)
    left_before = int(np.sum(right.size - np.searchsorted(right, left, side="right")))
    left_after = int(np.sum(np.searchsorted(right, left, side="left")))
    equal = total - left_before - left_after
    return left_before / total, left_after / total, equal / total


def edge_timestamp_table(
    graph_state: str,
    source: np.ndarray,
    target: np.ndarray,
    degree: np.ndarray,
    node_profiles: pd.DataFrame,
    date_arrays: list[np.ndarray],
    retained_keys: set[int],
    n_nodes: int,
    confidence_threshold: float,
) -> pd.DataFrame:
    first = node_profiles["first_day"].to_numpy(dtype=float)
    median = node_profiles["median_day"].to_numpy(dtype=float)
    last = node_profiles["last_day"].to_numpy(dtype=float)
    multiplicity = node_profiles["coordinate_multiplicity"].to_numpy(dtype=np.int64)

    rows: list[dict[str, Any]] = []
    for left, right in zip(source.astype(int), target.astype(int)):
        valid = np.isfinite(first[left]) and np.isfinite(first[right])
        left_before_prob, left_after_prob, equal_prob = pairwise_date_order_probabilities(
            date_arrays[left], date_arrays[right]
        )
        if valid and last[left] < first[right]:
            relation = "source_strictly_before_target"
            earlier, later = left, right
            gap = int(first[right] - last[left])
            overlap = 0
        elif valid and last[right] < first[left]:
            relation = "target_strictly_before_source"
            earlier, later = right, left
            gap = int(first[left] - last[right])
            overlap = 0
        elif valid:
            relation = "intervals_overlap"
            earlier = later = -1
            gap = 0
            overlap = int(min(last[left], last[right]) - max(first[left], first[right]) + 1)
        else:
            relation = "missing_date"
            earlier = later = -1
            gap = -1
            overlap = -1

        if valid and first[left] < first[right]:
            emergence_relation = "source_first_seen_earlier"
        elif valid and first[right] < first[left]:
            emergence_relation = "target_first_seen_earlier"
        elif valid:
            emergence_relation = "equal_first_seen"
        else:
            emergence_relation = "missing_date"

        confidence = max(left_before_prob, left_after_prob) if np.isfinite(left_before_prob) else math.nan
        if np.isfinite(confidence) and confidence >= confidence_threshold:
            probable_earlier = left if left_before_prob > left_after_prob else right
            probable_later = right if left_before_prob > left_after_prob else left
        else:
            probable_earlier = probable_later = -1

        key = left * n_nodes + right
        rows.append(
            {
                "graph_state": graph_state,
                "source_unique_node_id": left,
                "target_unique_node_id": right,
                "refinement_status": "retained" if key in retained_keys else "removed",
                "source_degree": int(degree[left]),
                "target_degree": int(degree[right]),
                "source_coordinate_multiplicity": int(multiplicity[left]),
                "target_coordinate_multiplicity": int(multiplicity[right]),
                "source_first_collection_date": day_to_iso(first[left]),
                "source_last_collection_date": day_to_iso(last[left]),
                "target_first_collection_date": day_to_iso(first[right]),
                "target_last_collection_date": day_to_iso(last[right]),
                "interval_relation": relation,
                "strict_interval_earlier_node_id": earlier if earlier >= 0 else np.nan,
                "strict_interval_later_node_id": later if later >= 0 else np.nan,
                "strict_interval_gap_days": gap if gap >= 0 else np.nan,
                "interval_overlap_days_inclusive": overlap if overlap >= 0 else np.nan,
                "first_seen_relation": emergence_relation,
                "absolute_first_seen_gap_days": abs(first[left] - first[right]) if valid else np.nan,
                "absolute_median_date_gap_days": abs(median[left] - median[right]) if valid else np.nan,
                "p_source_date_before_target_date": left_before_prob,
                "p_source_date_after_target_date": left_after_prob,
                "p_equal_record_dates": equal_prob,
                "pairwise_direction_confidence": confidence,
                "probable_earlier_node_id": probable_earlier if probable_earlier >= 0 else np.nan,
                "probable_later_node_id": probable_later if probable_later >= 0 else np.nan,
                "probabilistically_directionable": bool(
                    np.isfinite(confidence) and confidence >= confidence_threshold
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_edges(frame: pd.DataFrame, confidence_threshold: float) -> dict[str, Any]:
    valid = frame[frame["interval_relation"] != "missing_date"]
    strict = valid["interval_relation"].isin(
        ["source_strictly_before_target", "target_strictly_before_source"]
    )
    return {
        "graph_state": str(frame["graph_state"].iloc[0]),
        "n_edges": int(len(frame)),
        "n_edges_with_complete_endpoint_ranges": int(len(valid)),
        "strictly_ordered_nonoverlapping_edges": int(strict.sum()),
        "strictly_ordered_nonoverlapping_fraction": float(strict.mean()) if len(valid) else math.nan,
        "overlapping_interval_edges": int((valid["interval_relation"] == "intervals_overlap").sum()),
        "overlapping_interval_fraction": float((valid["interval_relation"] == "intervals_overlap").mean()) if len(valid) else math.nan,
        "equal_first_seen_edges": int((valid["first_seen_relation"] == "equal_first_seen").sum()),
        "equal_first_seen_fraction": float((valid["first_seen_relation"] == "equal_first_seen").mean()) if len(valid) else math.nan,
        "median_absolute_first_seen_gap_days": float(valid["absolute_first_seen_gap_days"].median()),
        "mean_absolute_first_seen_gap_days": float(valid["absolute_first_seen_gap_days"].mean()),
        "median_absolute_median_date_gap_days": float(valid["absolute_median_date_gap_days"].median()),
        "median_strict_interval_gap_days": float(valid.loc[strict, "strict_interval_gap_days"].median()) if strict.any() else math.nan,
        "pairwise_direction_confidence_threshold": confidence_threshold,
        "probabilistically_directionable_edges": int(valid["probabilistically_directionable"].sum()),
        "probabilistically_directionable_fraction": float(valid["probabilistically_directionable"].mean()) if len(valid) else math.nan,
        "median_pairwise_direction_confidence": float(valid["pairwise_direction_confidence"].median()),
        "strict_orientation_is_dag_by_construction": True,
    }


def leaf_edge_table(edge_frame: pd.DataFrame, degree: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in edge_frame.itertuples(index=False):
        left = int(row.source_unique_node_id)
        right = int(row.target_unique_node_id)
        left_leaf = degree[left] == 1
        right_leaf = degree[right] == 1
        if not (left_leaf or right_leaf):
            continue
        if left_leaf and right_leaf:
            leaf, neighbor = left, right
            leaf_is_source = True
            double_leaf = True
        elif left_leaf:
            leaf, neighbor = left, right
            leaf_is_source = True
            double_leaf = False
        else:
            leaf, neighbor = right, left
            leaf_is_source = False
            double_leaf = False

        leaf_first = pd.Timestamp(
            row.source_first_collection_date if leaf_is_source else row.target_first_collection_date
        )
        leaf_last = pd.Timestamp(
            row.source_last_collection_date if leaf_is_source else row.target_last_collection_date
        )
        neighbor_first = pd.Timestamp(
            row.target_first_collection_date if leaf_is_source else row.source_first_collection_date
        )
        neighbor_last = pd.Timestamp(
            row.target_last_collection_date if leaf_is_source else row.source_last_collection_date
        )
        if leaf_first > neighbor_first:
            first_relation = "leaf_first_seen_later"
        elif leaf_first < neighbor_first:
            first_relation = "leaf_first_seen_earlier"
        else:
            first_relation = "equal_first_seen"
        if leaf_first > neighbor_last:
            interval_relation = "leaf_strictly_after_neighbor"
        elif leaf_last < neighbor_first:
            interval_relation = "leaf_strictly_before_neighbor"
        else:
            interval_relation = "intervals_overlap"
        rows.append(
            {
                "graph_state": row.graph_state,
                "leaf_unique_node_id": leaf,
                "neighbor_unique_node_id": neighbor,
                "double_leaf_edge": double_leaf,
                "leaf_degree": int(degree[leaf]),
                "neighbor_degree": int(degree[neighbor]),
                "leaf_first_collection_date": str(leaf_first.date()),
                "leaf_last_collection_date": str(leaf_last.date()),
                "neighbor_first_collection_date": str(neighbor_first.date()),
                "neighbor_last_collection_date": str(neighbor_last.date()),
                "leaf_first_seen_minus_neighbor_first_seen_days": int((leaf_first - neighbor_first).days),
                "leaf_first_seen_relation": first_relation,
                "leaf_neighbor_interval_relation": interval_relation,
                "refinement_status": row.refinement_status,
            }
        )
    return pd.DataFrame(rows)


def summarize_leaf_edges(frame: pd.DataFrame) -> dict[str, Any]:
    n = len(frame)
    return {
        "graph_state": str(frame["graph_state"].iloc[0]) if n else "",
        "n_leaf_edges": n,
        "leaf_first_seen_later_count": int((frame["leaf_first_seen_relation"] == "leaf_first_seen_later").sum()) if n else 0,
        "leaf_first_seen_later_fraction": float((frame["leaf_first_seen_relation"] == "leaf_first_seen_later").mean()) if n else math.nan,
        "leaf_first_seen_earlier_count": int((frame["leaf_first_seen_relation"] == "leaf_first_seen_earlier").sum()) if n else 0,
        "leaf_first_seen_earlier_fraction": float((frame["leaf_first_seen_relation"] == "leaf_first_seen_earlier").mean()) if n else math.nan,
        "equal_first_seen_fraction": float((frame["leaf_first_seen_relation"] == "equal_first_seen").mean()) if n else math.nan,
        "leaf_strictly_after_neighbor_fraction": float((frame["leaf_neighbor_interval_relation"] == "leaf_strictly_after_neighbor").mean()) if n else math.nan,
        "leaf_strictly_before_neighbor_fraction": float((frame["leaf_neighbor_interval_relation"] == "leaf_strictly_before_neighbor").mean()) if n else math.nan,
        "overlapping_interval_fraction": float((frame["leaf_neighbor_interval_relation"] == "intervals_overlap").mean()) if n else math.nan,
        "median_leaf_first_seen_minus_neighbor_days": float(frame["leaf_first_seen_minus_neighbor_first_seen_days"].median()) if n else math.nan,
    }


def null_metrics(
    source: np.ndarray,
    target: np.ndarray,
    degree: np.ndarray,
    first: np.ndarray,
    last: np.ndarray,
) -> dict[str, float]:
    valid = np.isfinite(first[source]) & np.isfinite(first[target])
    left = source[valid]
    right = target[valid]
    strict = (last[left] < first[right]) | (last[right] < first[left])
    overlap = ~strict
    leaf_left = degree[left] == 1
    leaf_right = degree[right] == 1
    leaf_mask = leaf_left | leaf_right
    leaf = np.where(leaf_left[leaf_mask], left[leaf_mask], right[leaf_mask])
    neighbor = np.where(leaf_left[leaf_mask], right[leaf_mask], left[leaf_mask])
    return {
        "strictly_ordered_nonoverlapping_fraction": float(strict.mean()) if len(strict) else math.nan,
        "overlapping_interval_fraction": float(overlap.mean()) if len(overlap) else math.nan,
        "mean_absolute_first_seen_gap_days": float(np.mean(np.abs(first[left] - first[right]))) if len(left) else math.nan,
        "leaf_first_seen_later_fraction": float(np.mean(first[leaf] > first[neighbor])) if len(leaf) else math.nan,
    }


def matched_profile_permutation_null(
    graph_state: str,
    source: np.ndarray,
    target: np.ndarray,
    degree: np.ndarray,
    node_profiles: pd.DataFrame,
    permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    first = node_profiles["first_day"].to_numpy(dtype=float)
    last = node_profiles["last_day"].to_numpy(dtype=float)
    multiplicity = node_profiles["coordinate_multiplicity"].to_numpy(dtype=int)
    strata: dict[tuple[str, str], list[int]] = {}
    for node in range(len(degree)):
        key = (degree_bin(int(degree[node])), multiplicity_bin(int(multiplicity[node])))
        strata.setdefault(key, []).append(node)
    movable = sum(len(nodes) for nodes in strata.values() if len(nodes) > 1)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for permutation in range(permutations):
        profile_index = np.arange(len(degree))
        for nodes in strata.values():
            if len(nodes) > 1:
                index = np.asarray(nodes, dtype=np.int64)
                profile_index[index] = rng.permutation(index)
        metrics = null_metrics(source, target, degree, first[profile_index], last[profile_index])
        rows.append({"graph_state": graph_state, "permutation": permutation, **metrics})
    qc = {
        "graph_state": graph_state,
        "permutations": permutations,
        "seed": seed,
        "stratification": "graph-state degree bin x coordinate-multiplicity bin",
        "n_strata": len(strata),
        "n_movable_nodes": movable,
        "movable_node_fraction": movable / len(degree),
    }
    return pd.DataFrame(rows), qc


def empirical_summary(
    graph_state: str,
    observed: dict[str, float],
    null_frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric, observed_value in observed.items():
        values = pd.to_numeric(null_frame[metric], errors="coerce").dropna().to_numpy()
        mean = float(values.mean())
        sd = float(values.std(ddof=1)) if len(values) > 1 else math.nan
        z = (observed_value - mean) / sd if sd > 0 else math.nan
        rows.append(
            {
                "graph_state": graph_state,
                "metric": metric,
                "observed": observed_value,
                "null_mean": mean,
                "null_sd": sd,
                "observed_minus_null_mean": observed_value - mean,
                "z_score": z,
                "empirical_p_ge": (1 + int(np.sum(values >= observed_value))) / (len(values) + 1),
                "empirical_p_le": (1 + int(np.sum(values <= observed_value))) / (len(values) + 1),
                "empirical_p_two_sided": (
                    1 + int(np.sum(np.abs(values - mean) >= abs(observed_value - mean)))
                ) / (len(values) + 1),
            }
        )
    return pd.DataFrame(rows)


def refinement_comparison(edge_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    original = edge_frame[edge_frame["graph_state"] == "original"].copy()
    summaries: list[dict[str, Any]] = []
    for status, group in original.groupby("refinement_status"):
        strict = group["interval_relation"] != "intervals_overlap"
        summaries.append(
            {
                "refinement_status": status,
                "n_edges": int(len(group)),
                "strictly_ordered_nonoverlapping_fraction": float(strict.mean()),
                "median_absolute_first_seen_gap_days": float(group["absolute_first_seen_gap_days"].median()),
                "median_pairwise_direction_confidence": float(group["pairwise_direction_confidence"].median()),
                "probabilistically_directionable_fraction": float(group["probabilistically_directionable"].mean()),
            }
        )
    removed = original[original["refinement_status"] == "removed"]
    retained = original[original["refinement_status"] == "retained"]
    tests: list[dict[str, Any]] = []
    for metric, alternative in [
        ("absolute_first_seen_gap_days", "two-sided"),
        ("pairwise_direction_confidence", "two-sided"),
    ]:
        result = mannwhitneyu(
            removed[metric].dropna(), retained[metric].dropna(), alternative=alternative
        )
        tests.append(
            {
                "test": "Mann-Whitney U",
                "metric": metric,
                "alternative": alternative,
                "statistic": float(result.statistic),
                "p_value": float(result.pvalue),
            }
        )
    removed_strict = removed["interval_relation"] != "intervals_overlap"
    retained_strict = retained["interval_relation"] != "intervals_overlap"
    table = np.array(
        [
            [removed_strict.sum(), (~removed_strict).sum()],
            [retained_strict.sum(), (~retained_strict).sum()],
        ]
    )
    odds, p_value = fisher_exact(table, alternative="two-sided")
    tests.append(
        {
            "test": "Fisher exact",
            "metric": "strictly_ordered_nonoverlapping",
            "alternative": "two-sided",
            "statistic": float(odds),
            "p_value": float(p_value),
        }
    )
    return pd.DataFrame(summaries), pd.DataFrame(tests)


def node_degree_summary(node_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for graph_state in ["original", "refined"]:
        degree_col = f"{graph_state}_degree"
        work = node_frame.copy()
        work["degree_bin"] = work[degree_col].map(lambda x: degree_bin(int(x)))
        for label, group in work.groupby("degree_bin", sort=False):
            span = pd.to_numeric(group["observed_date_span_days"], errors="coerce").dropna()
            rows.append(
                {
                    "graph_state": graph_state,
                    "degree_bin": label,
                    "n_nodes": int(len(group)),
                    "median_degree": float(group[degree_col].median()),
                    "median_coordinate_multiplicity": float(group["coordinate_multiplicity"].median()),
                    "median_observed_date_span_days": float(span.median()),
                    "q25_observed_date_span_days": float(span.quantile(0.25)),
                    "q75_observed_date_span_days": float(span.quantile(0.75)),
                }
            )
    return pd.DataFrame(rows)


def plot_results(
    edge_summary: pd.DataFrame,
    leaf_summary: pd.DataFrame,
    degree_summary: pd.DataFrame,
    refinement_summary: pd.DataFrame,
    out_root: Path,
) -> None:
    figure_root = out_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2), constrained_layout=True)
    states = ["original", "refined"]
    x = np.arange(2)
    strict = edge_summary.set_index("graph_state").loc[states, "strictly_ordered_nonoverlapping_fraction"]
    overlap = edge_summary.set_index("graph_state").loc[states, "overlapping_interval_fraction"]
    axes[0].bar(x, strict, color="#2563eb", label="Strictly ordered")
    axes[0].bar(x, overlap, bottom=strict, color="#9ca3af", label="Overlapping")
    axes[0].set_xticks(x, ["Original", "Refined"])
    axes[0].set_ylabel("Fraction of RNG edges")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Endpoint timestamp ranges")
    axes[0].legend(frameon=False, fontsize=8)

    leaf = leaf_summary.set_index("graph_state").loc[states]
    later = leaf["leaf_first_seen_later_fraction"]
    earlier = leaf["leaf_first_seen_earlier_fraction"]
    equal = leaf["equal_first_seen_fraction"]
    axes[1].bar(x, later, color="#059669", label="Leaf later")
    axes[1].bar(x, earlier, bottom=later, color="#dc2626", label="Leaf earlier")
    axes[1].bar(x, equal, bottom=later + earlier, color="#9ca3af", label="Equal")
    axes[1].set_xticks(x, ["Original", "Refined"])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Leaf vs adjacent non-leaf first seen")
    axes[1].legend(frameon=False, fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(figure_root / f"timestamp_directionality_summary.{suffix}", dpi=220)
    plt.close(fig)

    labels = ["1", "2", "3", "4", "5", "6-9", "10+"]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=True, constrained_layout=True)
    for axis, state in zip(axes, states):
        frame = degree_summary[degree_summary["graph_state"] == state].set_index("degree_bin").reindex(labels)
        positions = np.arange(len(labels))
        axis.errorbar(
            positions,
            frame["median_observed_date_span_days"],
            yerr=np.vstack(
                [
                    frame["median_observed_date_span_days"] - frame["q25_observed_date_span_days"],
                    frame["q75_observed_date_span_days"] - frame["median_observed_date_span_days"],
                ]
            ),
            color="#111827",
            marker="o",
            capsize=3,
        )
        axis.set_xticks(positions, labels)
        axis.set_xlabel("RNG degree")
        axis.set_title(state.capitalize())
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Vertex timestamp span (days), median and IQR")
    for suffix in ("png", "pdf"):
        fig.savefig(figure_root / f"node_timestamp_span_by_degree.{suffix}", dpi=220)
    plt.close(fig)

    if not refinement_summary.empty:
        fig, axis = plt.subplots(figsize=(5.8, 4.2), constrained_layout=True)
        order = [item for item in ["retained", "removed"] if item in set(refinement_summary["refinement_status"])]
        frame = refinement_summary.set_index("refinement_status").loc[order]
        axis.bar(order, frame["median_absolute_first_seen_gap_days"], color=["#2563eb", "#dc2626"][: len(order)])
        axis.set_ylabel("Median absolute first-seen gap (days)")
        axis.set_title("Original RNG edges by refinement outcome")
        axis.grid(axis="y", alpha=0.2)
        for suffix in ("png", "pdf"):
            fig.savefig(figure_root / f"retained_removed_temporal_gap.{suffix}", dpi=220)
        plt.close(fig)


def write_readme(
    out_root: Path,
    args: argparse.Namespace,
    edge_summary: pd.DataFrame,
    leaf_summary: pd.DataFrame,
    null_summary: pd.DataFrame,
    refinement_summary: pd.DataFrame,
    upper_tail: pd.DataFrame,
) -> None:
    def markdown_table(frame: pd.DataFrame, float_digits: int = 4) -> str:
        columns = list(frame.columns)
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        for row in frame.itertuples(index=False, name=None):
            rendered = []
            for value in row:
                if isinstance(value, (float, np.floating)):
                    rendered.append(f"{float(value):.{float_digits}g}")
                else:
                    rendered.append(str(value))
            lines.append("| " + " | ".join(rendered) + " |")
        return "\n".join(lines)

    edge_lines = markdown_table(edge_summary[
        [
            "graph_state",
            "n_edges",
            "strictly_ordered_nonoverlapping_fraction",
            "overlapping_interval_fraction",
            "median_absolute_first_seen_gap_days",
            "probabilistically_directionable_fraction",
        ]
    ])
    leaf_lines = markdown_table(leaf_summary[
        [
            "graph_state",
            "n_leaf_edges",
            "leaf_first_seen_later_fraction",
            "leaf_first_seen_earlier_fraction",
            "overlapping_interval_fraction",
        ]
    ])
    null_lines = markdown_table(null_summary[
        ["graph_state", "metric", "observed", "null_mean", "z_score", "empirical_p_two_sided"]
    ])
    original = edge_summary.set_index("graph_state").loc["original"]
    refined = edge_summary.set_index("graph_state").loc["refined"]
    original_null = null_summary[null_summary["graph_state"] == "original"].set_index("metric")
    leaf_original = leaf_summary.set_index("graph_state").loc["original"]
    top = upper_tail[
        (upper_tail["graph_state"] == "original") & (upper_tail["degree_rank"] == 1)
    ].iloc[0]
    refinement = refinement_summary.set_index("refinement_status")
    text = f"""# RNG timestamp-range directionality audit

## Question

Do edges in the seed-42 embedding RNG connect unique-coordinate vertices whose
observed collection-time profiles are consistent with a temporal progression?

Each of the 20,000 sampled records retains its collection date. Records with an
exactly identical stored embedding coordinate are collapsed to one of 8,921 RNG
vertices, and the vertex receives the complete observed range from its records.

## Direction rules

- A strict interval direction is assigned only when one endpoint's last observed
  date is earlier than the other endpoint's first observed date.
- Overlapping intervals are explicitly ambiguous and receive no strict direction.
- First-seen ordering is reported separately because a persistent earlier vertex
  can overlap a later-emerging neighbor.
- Pairwise direction confidence is the larger of `P(date_u < date_v)` and
  `P(date_v < date_u)` over all record-date pairs. The reporting threshold is
  `{args.confidence_threshold:.2f}`.
- Strict date ordering and strict first-seen ordering are DAGs by construction;
  this does not establish direct ancestry.

## Graph comparison

{edge_lines}

## Main findings

- `{100 * original['strictly_ordered_nonoverlapping_fraction']:.1f}%` of original
  RNG edges and `{100 * refined['strictly_ordered_nonoverlapping_fraction']:.1f}%`
  of refined edges have non-overlapping endpoint ranges and therefore admit a
  conservative strict temporal direction.
- RNG adjacency is much more temporally local than matched random date-profile
  placement: the original mean absolute first-seen gap is
  `{original_null.loc['mean_absolute_first_seen_gap_days', 'observed']:.1f}` days,
  versus a matched-null mean of
  `{original_null.loc['mean_absolute_first_seen_gap_days', 'null_mean']:.1f}` days.
  The correspondingly higher observed interval-overlap fraction is a locality
  signal, not a failure of directionality.
- Leaves are first observed later than their adjacent neighbor on
  `{100 * leaf_original['leaf_first_seen_later_fraction']:.1f}%` of leaf edges,
  compared with `{100 * original_null.loc['leaf_first_seen_later_fraction', 'null_mean']:.1f}%`
  under the degree/multiplicity-matched null.
- Original and refined leaf summaries are identical because connectivity
  protection retains every edge incident to an original degree-1 vertex; deleting
  such an edge would isolate that leaf.
- The `f_j` refinement does not increase the fraction of strictly directionable
  edges. It removes temporally farther edges: removed edges have a median
  first-seen gap of `{refinement.loc['removed', 'median_absolute_first_seen_gap_days']:.0f}`
  days versus `{refinement.loc['retained', 'median_absolute_first_seen_gap_days']:.0f}`
  days for retained edges.
- The maximum-degree original vertex has degree `{int(top['graph_degree'])}`,
  coordinate multiplicity `{int(top['coordinate_multiplicity'])}`, and an
  observed date range from `{top['first_collection_date']}` to
  `{top['last_collection_date']}` (`{int(top['observed_date_span_days'])}` days).
  The association between hub degree and span remains confounded by multiplicity
  and sampling frequency.

## Leaf comparison

The word `neighbor` is used instead of `ancestor`: topology and dates alone do
not prove that the adjacent non-leaf generated the leaf.

{leaf_lines}

## Matched permutation control

Whole vertex date profiles are permuted within graph-state degree-bin by
coordinate-multiplicity-bin strata. This preserves the observed profile shapes
and controls major leaf/hub and multiplicity differences while breaking their
specific placement on the graph. `{args.permutations}` deterministic
permutations use base seed `{args.seed}`.

{null_lines}

## Outputs

- `node_timestamp_ranges.csv`: one row per unique-coordinate vertex.
- `edge_timestamp_range_audit.csv`: original and refined edge-level interval and
  record-pair direction metrics.
- `graph_timestamp_directionality_summary.csv`: graph-level summaries.
- `leaf_neighbor_timestamp_audit.csv` and `leaf_neighbor_summary.csv`.
- `degree_bin_timestamp_summary.csv`: node range summaries by RNG degree.
- `upper_tail_node_timestamp_ranges.csv`: timestamp profiles for the highest-degree
  vertices in each graph state.
- `refinement_temporal_summary.csv` and `refinement_temporal_tests.csv`.
- `matched_profile_permutation_metrics.csv` and
  `matched_profile_permutation_summary.csv`.
- `run_manifest.json`: input signatures, graph/refinement QC, alignment checks,
  definitions, and output inventory.

## Interpretation boundary

This audit can support **temporal consistency of RNG adjacency**. It cannot show
that one sampled sequence is the direct ancestor of another: sampling is sparse,
collection date is later than infection time, identical coordinates can recur,
and unsampled intermediates are expected. Use `earlier-compatible neighbor` or
`temporally ordered edge`, not `parent-child edge`, unless an independent
phylogenetic analysis supplies that evidence.
"""
    (out_root / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", type=Path, default=DEFAULT_PANEL_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--candidate-fraction", type=float, default=0.10)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--norm-epsilon", type=float, default=1e-12)
    parser.add_argument("--confidence-threshold", type=float, default=0.80)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upper-tail-nodes", type=int, default=100)
    args = parser.parse_args()

    if not 0.5 <= args.confidence_threshold <= 1.0:
        raise ValueError("--confidence-threshold must be between 0.5 and 1")
    if args.permutations < 1:
        raise ValueError("--permutations must be positive")
    args.out_root.mkdir(parents=True, exist_ok=True)

    embedding_path = CALIBRATION.embedding_path(args.panel_root)
    adjacency_path = CALIBRATION.original_rng_path(args.panel_root)
    metadata_path = args.panel_root / "inputs/pool_n20000/metadata.csv"
    log("Loading embeddings, metadata, and original cityblock RNG")
    coordinates = np.load(embedding_path, mmap_mode="r")
    metadata = pd.read_csv(
        metadata_path, usecols=["pool_node_id", "collection_date"], low_memory=False
    )
    if len(metadata) != len(coordinates):
        raise ValueError("metadata and embeddings have different row counts")
    pool_node_id = pd.to_numeric(metadata["pool_node_id"], errors="raise").to_numpy(dtype=int)
    if not np.array_equal(pool_node_id, np.arange(len(metadata))):
        raise ValueError("metadata pool_node_id is not aligned to embedding row order")

    adjacency = CALIBRATION.validate_adjacency(adjacency_path, len(coordinates))
    unique_coordinates, inverse, multiplicity = np.unique(
        np.asarray(coordinates), axis=0, return_inverse=True, return_counts=True
    )
    n_unique = len(unique_coordinates)
    source, target, weight, collapse_qc = CALIBRATION.collapse_graph_to_unique_coordinates(
        adjacency, inverse, n_unique
    )
    refined_source, refined_target, _, refinement_qc = CALIBRATION.refine_rng(
        unique_coordinates,
        source,
        target,
        weight,
        candidate_fraction=args.candidate_fraction,
        delta=args.delta,
        norm_epsilon=args.norm_epsilon,
    )
    if refinement_qc.get("endpoint_rule") != "both_endpoints_AND":
        raise RuntimeError("refinement did not use the mutual endpoint-AND rule")

    log(f"Building date profiles for {n_unique:,} unique-coordinate vertices")
    nodes, date_arrays = build_date_profiles(
        inverse, metadata["collection_date"], n_unique, multiplicity
    )
    original_degree = degree_vector(n_unique, source, target)
    refined_degree = degree_vector(n_unique, refined_source, refined_target)
    nodes["original_degree"] = original_degree
    nodes["refined_degree"] = refined_degree
    nodes["original_is_leaf"] = original_degree == 1
    nodes["refined_is_leaf"] = refined_degree == 1
    nodes.drop(columns=["first_day", "median_day", "last_day"]).to_csv(
        args.out_root / "node_timestamp_ranges.csv", index=False
    )

    refined_keys = set((refined_source * np.int64(n_unique) + refined_target).tolist())
    original_edges = edge_timestamp_table(
        "original",
        source,
        target,
        original_degree,
        nodes,
        date_arrays,
        refined_keys,
        n_unique,
        args.confidence_threshold,
    )
    refined_edges = edge_timestamp_table(
        "refined",
        refined_source,
        refined_target,
        refined_degree,
        nodes,
        date_arrays,
        refined_keys,
        n_unique,
        args.confidence_threshold,
    )
    edge_frame = pd.concat([original_edges, refined_edges], ignore_index=True)
    edge_frame.to_csv(args.out_root / "edge_timestamp_range_audit.csv", index=False)
    edge_summary = pd.DataFrame(
        [
            summarize_edges(original_edges, args.confidence_threshold),
            summarize_edges(refined_edges, args.confidence_threshold),
        ]
    )
    edge_summary.to_csv(
        args.out_root / "graph_timestamp_directionality_summary.csv", index=False
    )

    original_leaf = leaf_edge_table(original_edges, original_degree)
    refined_leaf = leaf_edge_table(refined_edges, refined_degree)
    leaf_frame = pd.concat([original_leaf, refined_leaf], ignore_index=True)
    leaf_frame.to_csv(args.out_root / "leaf_neighbor_timestamp_audit.csv", index=False)
    leaf_summary = pd.DataFrame(
        [summarize_leaf_edges(original_leaf), summarize_leaf_edges(refined_leaf)]
    )
    leaf_summary.to_csv(args.out_root / "leaf_neighbor_summary.csv", index=False)

    degree_summary = node_degree_summary(nodes)
    degree_summary.to_csv(args.out_root / "degree_bin_timestamp_summary.csv", index=False)
    upper_tail_frames: list[pd.DataFrame] = []
    for state in ["original", "refined"]:
        degree_col = f"{state}_degree"
        selected = nodes.sort_values(
            [degree_col, "coordinate_multiplicity", "unique_node_id"],
            ascending=[False, False, True],
        ).head(args.upper_tail_nodes)
        selected = selected[
            [
                "unique_node_id",
                "coordinate_multiplicity",
                "n_valid_collection_dates",
                "first_collection_date",
                "median_collection_date",
                "last_collection_date",
                "observed_date_span_days",
                degree_col,
            ]
        ].copy()
        selected.insert(0, "degree_rank", np.arange(1, len(selected) + 1))
        selected.insert(0, "graph_state", state)
        selected = selected.rename(columns={degree_col: "graph_degree"})
        upper_tail_frames.append(selected)
    upper_tail = pd.concat(upper_tail_frames, ignore_index=True)
    upper_tail.to_csv(args.out_root / "upper_tail_node_timestamp_ranges.csv", index=False)
    refinement_summary, refinement_tests = refinement_comparison(edge_frame)
    refinement_summary.to_csv(args.out_root / "refinement_temporal_summary.csv", index=False)
    refinement_tests.to_csv(args.out_root / "refinement_temporal_tests.csv", index=False)

    log(f"Running {args.permutations} degree/multiplicity-matched date-profile permutations")
    null_frames: list[pd.DataFrame] = []
    null_qc: list[dict[str, Any]] = []
    null_summaries: list[pd.DataFrame] = []
    for index, (state, graph_source, graph_target, degree, leaf) in enumerate(
        [
            ("original", source, target, original_degree, original_leaf),
            ("refined", refined_source, refined_target, refined_degree, refined_leaf),
        ]
    ):
        frame, qc = matched_profile_permutation_null(
            state,
            graph_source,
            graph_target,
            degree,
            nodes,
            args.permutations,
            args.seed + index * 100_003,
        )
        null_frames.append(frame)
        null_qc.append(qc)
        observed = null_metrics(
            graph_source,
            graph_target,
            degree,
            nodes["first_day"].to_numpy(dtype=float),
            nodes["last_day"].to_numpy(dtype=float),
        )
        null_summaries.append(empirical_summary(state, observed, frame))
    null_frame = pd.concat(null_frames, ignore_index=True)
    null_summary = pd.concat(null_summaries, ignore_index=True)
    null_frame.to_csv(args.out_root / "matched_profile_permutation_metrics.csv", index=False)
    null_summary.to_csv(args.out_root / "matched_profile_permutation_summary.csv", index=False)

    plot_results(edge_summary, leaf_summary, degree_summary, refinement_summary, args.out_root)
    write_readme(
        args.out_root,
        args,
        edge_summary,
        leaf_summary,
        null_summary,
        refinement_summary,
        upper_tail,
    )

    output_files = sorted(
        str(path) for path in args.out_root.rglob("*") if path.is_file() and path.name != "run_manifest.json"
    )
    manifest = {
        "algorithm_version": ALGORITHM_VERSION,
        "completed_at_unix": time.time(),
        "panel_root": str(args.panel_root.resolve()),
        "n_records": int(len(coordinates)),
        "n_unique_coordinates": int(n_unique),
        "metadata_embedding_alignment": "pool_node_id equals embedding row index 0..N-1",
        "valid_collection_dates": int(pd.to_datetime(metadata["collection_date"], errors="coerce").notna().sum()),
        "collection_date_min": str(pd.to_datetime(metadata["collection_date"]).min().date()),
        "collection_date_max": str(pd.to_datetime(metadata["collection_date"]).max().date()),
        "graph_edges": {"original": int(len(source)), "refined": int(len(refined_source))},
        "collapse_qc": collapse_qc,
        "refinement_qc": refinement_qc,
        "refinement_parameters": {
            "candidate_fraction": args.candidate_fraction,
            "delta": args.delta,
            "norm_epsilon": args.norm_epsilon,
        },
        "confidence_threshold": args.confidence_threshold,
        "permutation_qc": null_qc,
        "input_signatures": {
            "embeddings": file_signature(embedding_path),
            "original_rng": file_signature(adjacency_path),
            "metadata": file_signature(metadata_path),
        },
        "sequence_content_written": False,
        "coordinate_content_written": False,
        "outputs": output_files,
        "fingerprint": hashlib.sha256(
            json.dumps(
                {
                    "algorithm_version": ALGORITHM_VERSION,
                    "inputs": [file_signature(embedding_path), file_signature(adjacency_path), file_signature(metadata_path)],
                    "parameters": vars(args),
                },
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest(),
    }
    write_json(args.out_root / "run_manifest.json", manifest)
    log(f"Completed timestamp-range audit: {args.out_root}")


if __name__ == "__main__":
    main()
