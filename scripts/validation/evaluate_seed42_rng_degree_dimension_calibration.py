#!/usr/bin/env python3
"""Calibrate seed-42 RNG degree against known-dimensional L1 point clouds.

The biological panel contains repeated ESM-2 coordinate rows. Under the strict
RNG witness convention used by this repository, tied zero-distance pairs are
retained and strongly inflate record-level degree. This workflow therefore
reports both the requested 20,000-record graph and a method-matched analysis on
unique coordinate vectors. Direct dimension estimates are emitted only for the
unique-coordinate comparison.
"""

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

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PANEL_ROOT = Path(
    "analysis/cohort_validation/07_sampling_design_20k/"
    "random_full_dataset_seed42/seed_42"
)
DEFAULT_DIRECTIONAL_ROOT = Path(
    "analysis/cohort_validation/24_seed42_20k_directional_intrinsic_distances/"
    "random_full_dataset_seed42/seed_42"
)
DEFAULT_OUT_ROOT = Path(
    "analysis/cohort_validation/29_seed42_20k_rng_degree_dimension_calibration/"
    "random_full_dataset_seed42/seed_42"
)
DEFAULT_CANDIDATE_LABEL = "candidate_0p1_delta_0p01"
ALGORITHM_VERSION = 1


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GRAPH_MOD = load_script_module(
    "rng_degree_graph_builder",
    PROJECT_ROOT / "scripts/graph_construction/build_cohort_embedding_graphs.py",
)
DIRECTIONAL_MOD = load_script_module(
    "rng_degree_directional_filter",
    PROJECT_ROOT / "scripts/validation/build_directional_intrinsic_distances.py",
)


def parse_int_csv(value: str) -> list[int]:
    values = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


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
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def embedding_path(panel_root: Path) -> Path:
    return panel_root / "embeddings/esm2_650M/pool_n20000/embeddings.npy"


def original_rng_path(panel_root: Path) -> Path:
    return panel_root / "graphs/esm2_650M/cityblock/pool_n20000/embedding_rng_exact/adj.npz"


def refined_rng_path(directional_root: Path, candidate_label: str) -> Path:
    return directional_root / "refined_graphs" / candidate_label / "rng/adj.npz"


def validate_adjacency(path: Path, n_nodes: int) -> csr_matrix:
    adj = load_npz(path).tocsr()
    adj.sort_indices()
    if adj.shape != (n_nodes, n_nodes):
        raise ValueError(f"{path}: expected {(n_nodes, n_nodes)}, observed {adj.shape}")
    if np.any(adj.diagonal() != 0):
        raise ValueError(f"{path}: adjacency contains self-loops")
    delta = (adj - adj.T).tocsr()
    if delta.nnz and np.max(np.abs(delta.data)) > 1e-6:
        raise ValueError(f"{path}: adjacency is not symmetric")
    return adj


def upper_edges(adj: csr_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coo = adj.tocoo(copy=False)
    keep = coo.row < coo.col
    source = coo.row[keep].astype(np.int64, copy=False)
    target = coo.col[keep].astype(np.int64, copy=False)
    weight = coo.data[keep].astype(np.float32, copy=False)
    order = np.lexsort((target, source))
    return source[order], target[order], weight[order]


def degree_summary(
    n_nodes: int,
    sources: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    degree = degree_vector(n_nodes, sources, targets)
    if int(degree.sum()) != 2 * int(len(sources)):
        raise AssertionError("degree sum does not equal twice the undirected edge count")
    if degree.size and int(degree.max()) > n_nodes - 1:
        raise AssertionError("maximum degree exceeds the simple-graph bound n_nodes - 1")
    summary: dict[str, Any] = {
        "n_nodes": int(n_nodes),
        "n_edges": int(len(sources)),
        "mean_degree": float(degree.mean()),
        "median_degree": float(np.median(degree)),
        "q25_degree": float(np.quantile(degree, 0.25)),
        "q75_degree": float(np.quantile(degree, 0.75)),
        "min_degree": int(degree.min()) if degree.size else 0,
        "max_degree": int(degree.max()) if degree.size else 0,
        "isolated_nodes": int(np.count_nonzero(degree == 0)),
    }
    if weights is not None:
        summary["zero_weight_edges"] = int(np.count_nonzero(weights == 0))
        summary["nonzero_weight_edges"] = int(np.count_nonzero(weights > 0))
    return summary


def degree_vector(
    n_nodes: int,
    sources: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    return np.bincount(
        np.concatenate([sources, targets]).astype(np.int64, copy=False),
        minlength=n_nodes,
    )


def degree_histogram(
    n_nodes: int,
    sources: np.ndarray,
    targets: np.ndarray,
) -> list[dict[str, int]]:
    degree = degree_vector(n_nodes, sources, targets)
    counts = np.bincount(degree)
    return [
        {"degree": int(value), "n_nodes_at_degree": int(count)}
        for value, count in enumerate(counts)
    ]


def histogram_frame(
    histogram: list[dict[str, int]],
    n_nodes: int,
    **labels: Any,
) -> pd.DataFrame:
    frame = pd.DataFrame(histogram)
    frame["fraction_nodes"] = frame["n_nodes_at_degree"] / int(n_nodes)
    for key, value in labels.items():
        frame[key] = value
    leading = list(labels)
    return frame[leading + ["degree", "n_nodes_at_degree", "fraction_nodes"]]


def edges_to_csr(
    n_nodes: int,
    sources: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
) -> csr_matrix:
    return GRAPH_MOD.edges_to_csr(
        n_nodes,
        sources.astype(np.int32, copy=False),
        targets.astype(np.int32, copy=False),
        weights.astype(np.float32, copy=False),
    )


def refine_rng(
    coordinates: np.ndarray,
    sources: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    candidate_fraction: float,
    delta: float,
    norm_epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    n_nodes = int(coordinates.shape[0])
    adj = edges_to_csr(n_nodes, sources, targets, weights)
    embedding_rows = np.arange(n_nodes, dtype=np.int64)
    evaluations: list[dict[str, Any]] = []
    local_rejections = 0
    for node_id in range(n_nodes):
        neighbors = adj.indices[adj.indptr[node_id] : adj.indptr[node_id + 1]]
        _, node_evaluations = DIRECTIONAL_MOD.directional_decisions_for_node(
            embeddings=coordinates,
            embedding_rows=embedding_rows,
            neighbors=neighbors,
            node_id=node_id,
            candidate_fraction=candidate_fraction,
            delta=delta,
            norm_epsilon=norm_epsilon,
        )
        evaluations.extend(node_evaluations)
        local_rejections += sum(int(item["accepted"]) for item in node_evaluations)

    evaluations_frame = pd.DataFrame(evaluations)
    mutual = DIRECTIONAL_MOD.mutual_rejection_queue(evaluations_frame)
    edge_keys = sources.astype(np.int64) * np.int64(n_nodes) + targets.astype(np.int64)
    mutual_keys = (
        mutual["source"].to_numpy(dtype=np.int64) * np.int64(n_nodes)
        + mutual["target"].to_numpy(dtype=np.int64)
    )
    queue_indices = np.searchsorted(edge_keys, mutual_keys).astype(np.int64)
    if mutual_keys.size and (
        np.any(queue_indices >= edge_keys.size)
        or not np.array_equal(edge_keys[queue_indices], mutual_keys)
    ):
        raise ValueError("mutually rejected synthetic edge is absent from the RNG")
    delete_mask = DIRECTIONAL_MOD.connectivity_safe_delete_mask(
        n_nodes,
        sources.astype(np.int64, copy=False),
        targets.astype(np.int64, copy=False),
        queue_indices,
    )
    queue_deleted = delete_mask[queue_indices] if queue_indices.size else np.array([], dtype=bool)
    keep = ~delete_mask
    refined = edges_to_csr(n_nodes, sources[keep], targets[keep], weights[keep])
    before_components = connected_components(adj, directed=False, return_labels=False)
    after_components = connected_components(refined, directed=False, return_labels=False)
    if int(before_components) != int(after_components):
        raise RuntimeError("synthetic f_j connectivity safeguard failed")
    qc = {
        "endpoint_rule": "both_endpoints_AND",
        "connectivity_rule": "reverse_delete_preserving_original_component_count",
        "local_endpoint_rejections": int(local_rejections),
        "mutually_rejected_edges": int(len(mutual)),
        "mutually_rejected_edges_removed": int(queue_deleted.sum()),
        "mutually_rejected_edges_retained_for_connectivity": int((~queue_deleted).sum()),
        "components_before": int(before_components),
        "components_after": int(after_components),
    }
    return sources[keep], targets[keep], weights[keep], qc


def collapse_graph_to_unique_coordinates(
    adj: csr_matrix,
    inverse: np.ndarray,
    n_unique: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    source, target, weight = upper_edges(adj)
    left = inverse[source]
    right = inverse[target]
    group_source = np.minimum(left, right).astype(np.int64, copy=False)
    group_target = np.maximum(left, right).astype(np.int64, copy=False)
    cross = group_source != group_target
    keys = group_source[cross] * np.int64(n_unique) + group_target[cross]
    if keys.size == 0:
        empty_i = np.array([], dtype=np.int64)
        empty_w = np.array([], dtype=np.float32)
        return (
            empty_i,
            empty_i.copy(),
            empty_w,
            {
                "record_edges": int(len(source)),
                "within_coordinate_edges": int(np.count_nonzero(~cross)),
                "cross_coordinate_record_edges": 0,
                "collapsed_cross_coordinate_edges": 0,
            },
        )
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    first = np.r_[True, sorted_keys[1:] != sorted_keys[:-1]]
    chosen = order[first]
    collapsed_source = group_source[cross][chosen]
    collapsed_target = group_target[cross][chosen]
    collapsed_weight = weight[cross][chosen]
    return (
        collapsed_source,
        collapsed_target,
        collapsed_weight,
        {
            "record_edges": int(len(source)),
            "within_coordinate_edges": int(np.count_nonzero(~cross)),
            "cross_coordinate_record_edges": int(np.count_nonzero(cross)),
            "collapsed_cross_coordinate_edges": int(len(collapsed_source)),
        },
    )


def biological_summaries(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, int]:
    coordinates = np.load(embedding_path(args.panel_root), mmap_mode="r")
    n_records = int(coordinates.shape[0])
    original = validate_adjacency(original_rng_path(args.panel_root), n_records)
    refined = validate_adjacency(
        refined_rng_path(args.directional_root, args.candidate_label), n_records
    )
    unique_coordinates, inverse, multiplicities = np.unique(
        np.asarray(coordinates), axis=0, return_inverse=True, return_counts=True
    )
    n_unique = int(len(unique_coordinates))

    rows: list[dict[str, Any]] = []
    distribution_frames: list[pd.DataFrame] = []
    collapsed: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for graph_state, adj in [("original", original), ("refined", refined)]:
        source, target, weight = upper_edges(adj)
        rows.append(
            {
                "representation": f"biological_record_{graph_state}",
                "graph_state": graph_state,
                "coordinate_scope": "20k_records_with_ties",
                "dimension_estimate_allowed": False,
                "comparison_note": "not comparable to continuous uniform points because repeated coordinates inflate strict-tie RNG degree",
                **degree_summary(n_records, source, target, weight),
            }
        )
        distribution_frames.append(
            histogram_frame(
                degree_histogram(n_records, source, target),
                n_records,
                representation=f"biological_record_{graph_state}",
                graph_state=graph_state,
                coordinate_scope="20k_records_with_ties",
            )
        )
        c_source, c_target, c_weight, collapse_qc = collapse_graph_to_unique_coordinates(
            adj, inverse, n_unique
        )
        collapsed[graph_state] = (c_source, c_target, c_weight)
        rows.append(
            {
                "representation": f"biological_unique_collapsed_{graph_state}",
                "graph_state": graph_state,
                "coordinate_scope": "unique_coordinates",
                "dimension_estimate_allowed": graph_state == "original",
                "comparison_note": (
                    "method-matched raw unique-coordinate RNG"
                    if graph_state == "original"
                    else "descriptive collapse after record-level f_j; refinement was not performed on collapsed graph"
                ),
                **degree_summary(n_unique, c_source, c_target, c_weight),
                **collapse_qc,
            }
        )
        distribution_frames.append(
            histogram_frame(
                degree_histogram(n_unique, c_source, c_target),
                n_unique,
                representation=f"biological_unique_collapsed_{graph_state}",
                graph_state=graph_state,
                coordinate_scope="unique_coordinates",
            )
        )

    unique_source, unique_target, unique_weight = collapsed["original"]
    direct_source, direct_target, direct_weight, direct_qc = refine_rng(
        unique_coordinates,
        unique_source,
        unique_target,
        unique_weight,
        candidate_fraction=args.candidate_fraction,
        delta=args.delta,
        norm_epsilon=args.norm_epsilon,
    )
    rows.append(
        {
            "representation": "biological_unique_refined_direct",
            "graph_state": "refined",
            "coordinate_scope": "unique_coordinates",
            "dimension_estimate_allowed": True,
            "comparison_note": "method-matched unique-coordinate RNG with the same mutual f_j refinement",
            **degree_summary(n_unique, direct_source, direct_target, direct_weight),
            **direct_qc,
        }
    )
    distribution_frames.append(
        histogram_frame(
            degree_histogram(n_unique, direct_source, direct_target),
            n_unique,
            representation="biological_unique_refined_direct",
            graph_state="refined",
            coordinate_scope="unique_coordinates",
        )
    )

    multiplicity_values, group_counts = np.unique(multiplicities, return_counts=True)
    multiplicity_frame = pd.DataFrame(
        {
            "coordinate_multiplicity": multiplicity_values.astype(np.int64),
            "n_coordinate_groups": group_counts.astype(np.int64),
            "n_records": (multiplicity_values * group_counts).astype(np.int64),
        }
    )
    return (
        pd.DataFrame(rows),
        multiplicity_frame,
        pd.concat(distribution_frames, ignore_index=True),
        unique_coordinates,
        n_records,
    )


def stable_condition_seed(base_seed: int, n_points: int, dimension: int, replicate: int) -> int:
    payload = f"{base_seed}|{n_points}|{dimension}|{replicate}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def condition_fingerprint(args: argparse.Namespace, n_points: int, dimension: int, replicate: int) -> str:
    payload = {
        "algorithm_version": ALGORITHM_VERSION,
        "n_points": n_points,
        "dimension": dimension,
        "replicate": replicate,
        "base_seed": args.seed,
        "metric": "cityblock",
        "rng_rule": "strict_open_lune",
        "candidate_fraction": args.candidate_fraction,
        "delta": args.delta,
        "norm_epsilon": args.norm_epsilon,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def run_synthetic_condition(
    args: argparse.Namespace,
    n_points: int,
    dimension: int,
    replicate: int,
) -> dict[str, Any]:
    fingerprint = condition_fingerprint(args, n_points, dimension, replicate)
    condition_dir = args.out_root / "work/conditions"
    result_path = condition_dir / f"n{n_points:06d}_d{dimension:02d}_rep{replicate:03d}.json"
    if result_path.exists() and not args.force:
        cached = json.loads(result_path.read_text())
        if cached.get("fingerprint") == fingerprint and "degree_histograms" in cached:
            log(f"Reusing {result_path.name}")
            return cached
        if cached.get("fingerprint") == fingerprint:
            log(f"Recomputing {result_path.name}: legacy checkpoint lacks degree histograms")

    condition_seed = stable_condition_seed(args.seed, n_points, dimension, replicate)
    log(
        f"Generating uniform L1 condition n={n_points:,}, d={dimension}, "
        f"replicate={replicate}, seed={condition_seed}"
    )
    rng = np.random.default_rng(condition_seed)
    coordinates = rng.random((n_points, dimension), dtype=np.float32)
    distance = cdist(coordinates, coordinates, metric="cityblock").astype(np.float32)
    np.fill_diagonal(distance, np.inf)
    sources, targets, weights, checked, pruned = GRAPH_MOD.exact_rng_edges_blockwise_order(
        distance,
        row_block_size=args.rng_row_block_size,
        max_block_edges=args.max_block_edges,
    )
    del distance
    sources = sources.astype(np.int64, copy=False)
    targets = targets.astype(np.int64, copy=False)
    weights = weights.astype(np.float32, copy=False)
    raw_summary = degree_summary(n_points, sources, targets, weights)
    refined_source, refined_target, refined_weight, refinement_qc = refine_rng(
        coordinates,
        sources,
        targets,
        weights,
        candidate_fraction=args.candidate_fraction,
        delta=args.delta,
        norm_epsilon=args.norm_epsilon,
    )
    refined_summary = degree_summary(
        n_points, refined_source, refined_target, refined_weight
    )
    payload = {
        "fingerprint": fingerprint,
        "algorithm_version": ALGORITHM_VERSION,
        "completed_at_unix": time.time(),
        "n_points": int(n_points),
        "dimension": int(dimension),
        "replicate": int(replicate),
        "condition_seed": int(condition_seed),
        "distribution": "uniform_unit_hypercube",
        "metric": "cityblock",
        "rng_rule": "retain edge (i,j) unless a witness k has both distances strictly less than d(i,j)",
        "candidate_edges_checked": int(checked),
        "candidate_edges_pruned": int(pruned),
        "raw": raw_summary,
        "refined": refined_summary,
        "degree_histograms": {
            "raw": degree_histogram(n_points, sources, targets),
            "refined": degree_histogram(
                n_points, refined_source, refined_target
            ),
        },
        "refinement_qc": refinement_qc,
        "coordinate_content_written": False,
    }
    write_json(result_path, payload)
    return payload


def flatten_condition(payload: dict[str, Any]) -> list[dict[str, Any]]:
    common = {
        "n_points": payload["n_points"],
        "dimension": payload["dimension"],
        "replicate": payload["replicate"],
        "condition_seed": payload["condition_seed"],
        "candidate_edges_checked": payload["candidate_edges_checked"],
    }
    rows = []
    for graph_state in ["raw", "refined"]:
        rows.append({**common, "graph_state": graph_state, **payload[graph_state]})
    return rows


def flatten_condition_histograms(payload: dict[str, Any]) -> pd.DataFrame:
    frames = []
    for graph_state in ["raw", "refined"]:
        frames.append(
            histogram_frame(
                payload["degree_histograms"][graph_state],
                int(payload["n_points"]),
                n_points=int(payload["n_points"]),
                dimension=int(payload["dimension"]),
                replicate=int(payload["replicate"]),
                condition_seed=int(payload["condition_seed"]),
                graph_state=graph_state,
            )
        )
    return pd.concat(frames, ignore_index=True)


def summarize_degree_distributions(run_histograms: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for (n_points, dimension, graph_state), group in run_histograms.groupby(
        ["n_points", "dimension", "graph_state"], sort=True
    ):
        replicates = np.sort(group["replicate"].unique())
        max_degree = int(group["degree"].max())
        complete_index = pd.MultiIndex.from_product(
            [replicates, np.arange(max_degree + 1, dtype=np.int64)],
            names=["replicate", "degree"],
        )
        complete = (
            group.set_index(["replicate", "degree"])[["n_nodes_at_degree", "fraction_nodes"]]
            .reindex(complete_index, fill_value=0)
            .reset_index()
        )
        summary = (
            complete.groupby("degree", as_index=False)
            .agg(
                replicates=("replicate", "nunique"),
                mean_n_nodes=("n_nodes_at_degree", "mean"),
                sd_n_nodes=("n_nodes_at_degree", "std"),
                mean_fraction_nodes=("fraction_nodes", "mean"),
                sd_fraction_nodes=("fraction_nodes", "std"),
                min_fraction_nodes=("fraction_nodes", "min"),
                max_fraction_nodes=("fraction_nodes", "max"),
            )
        )
        summary.insert(0, "graph_state", graph_state)
        summary.insert(0, "dimension", int(dimension))
        summary.insert(0, "n_points", int(n_points))
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def summarize_calibration(run_frame: pd.DataFrame) -> pd.DataFrame:
    return (
        run_frame.groupby(["n_points", "dimension", "graph_state"], as_index=False)
        .agg(
            replicates=("replicate", "nunique"),
            mean_degree=("mean_degree", "mean"),
            mean_degree_sd=("mean_degree", "std"),
            median_degree=("median_degree", "median"),
            median_degree_min=("median_degree", "min"),
            median_degree_max=("median_degree", "max"),
            mean_edges=("n_edges", "mean"),
            mean_max_degree=("max_degree", "mean"),
            min_max_degree=("max_degree", "min"),
            max_max_degree=("max_degree", "max"),
        )
        .sort_values(["n_points", "graph_state", "dimension"])
        .reset_index(drop=True)
    )


def interpolate_dimension(dimensions: np.ndarray, values: np.ndarray, target: float) -> dict[str, Any]:
    order = np.argsort(dimensions)
    dimensions = np.asarray(dimensions, dtype=float)[order]
    values = np.asarray(values, dtype=float)[order]
    nearest_index = int(np.argmin(np.abs(values - target)))
    result: dict[str, Any] = {
        "nearest_dimension": float(dimensions[nearest_index]),
        "nearest_calibration_value": float(values[nearest_index]),
        "interpolated_dimension": math.nan,
        "interpolation_status": "not_bracketed",
        "calibration_monotone_nondecreasing": bool(np.all(np.diff(values) >= -1e-12)),
    }
    if not result["calibration_monotone_nondecreasing"]:
        result["interpolation_status"] = "nonmonotone_calibration"
        return result
    if target < values[0] or target > values[-1]:
        result["interpolation_status"] = "outside_calibration_range"
        return result
    exact = np.flatnonzero(np.isclose(values, target, rtol=0.0, atol=1e-12))
    if exact.size:
        result["interpolated_dimension"] = float(dimensions[int(exact[0])])
        result["interpolation_status"] = "exact"
        return result
    upper = int(np.searchsorted(values, target, side="right"))
    lower = upper - 1
    if values[upper] == values[lower]:
        result["interpolation_status"] = "flat_bracket"
        return result
    fraction = (target - values[lower]) / (values[upper] - values[lower])
    result["interpolated_dimension"] = float(
        dimensions[lower] + fraction * (dimensions[upper] - dimensions[lower])
    )
    result["interpolation_status"] = "interpolated"
    return result


def build_dimension_estimates(
    biological: pd.DataFrame,
    calibration: pd.DataFrame,
    unique_n: int,
) -> pd.DataFrame:
    targets = [
        ("biological_unique_collapsed_original", "raw"),
        ("biological_unique_refined_direct", "refined"),
    ]
    rows: list[dict[str, Any]] = []
    for representation, graph_state in targets:
        bio = biological.loc[biological["representation"] == representation]
        cal = calibration.loc[
            (calibration["n_points"] == unique_n)
            & (calibration["graph_state"] == graph_state)
        ].sort_values("dimension")
        if bio.empty or cal.empty:
            continue
        for statistic in ["mean_degree", "median_degree"]:
            target = float(bio.iloc[0][statistic])
            estimate = interpolate_dimension(
                cal["dimension"].to_numpy(), cal[statistic].to_numpy(), target
            )
            rows.append(
                {
                    "biological_representation": representation,
                    "synthetic_graph_state": graph_state,
                    "n_points": int(unique_n),
                    "statistic": statistic,
                    "biological_value": target,
                    **estimate,
                    "interpretation_scope": "descriptive L1 uniform-hypercube RNG degree calibration, not an intrinsic-dimension estimator with a general consistency guarantee",
                }
            )
    return pd.DataFrame(rows)


def build_combined_calibration_table(
    biological: pd.DataFrame,
    calibration: pd.DataFrame,
) -> pd.DataFrame:
    synthetic = calibration.copy()
    synthetic.insert(0, "row_type", "synthetic_calibration")
    synthetic.insert(
        1,
        "representation",
        synthetic["dimension"].map(lambda value: f"uniform_L1_d{int(value)}"),
    )
    synthetic["dimension_estimate_allowed"] = True
    synthetic["comparison_note"] = "known-dimensional uniform unit hypercube"
    synthetic = synthetic[
        [
            "row_type",
            "representation",
            "n_points",
            "dimension",
            "graph_state",
            "replicates",
            "mean_degree",
            "mean_degree_sd",
            "median_degree",
            "median_degree_min",
            "median_degree_max",
            "min_max_degree",
            "mean_max_degree",
            "max_max_degree",
            "dimension_estimate_allowed",
            "comparison_note",
        ]
    ]

    biological_rows = biological.copy()
    biological_rows.insert(0, "row_type", "biological_observation")
    biological_rows["n_points"] = biological_rows["n_nodes"]
    biological_rows["dimension"] = np.nan
    biological_rows["replicates"] = 1
    biological_rows["mean_degree_sd"] = np.nan
    biological_rows["median_degree_min"] = biological_rows["median_degree"]
    biological_rows["median_degree_max"] = biological_rows["median_degree"]
    biological_rows["min_max_degree"] = biological_rows["max_degree"]
    biological_rows["mean_max_degree"] = biological_rows["max_degree"]
    biological_rows["max_max_degree"] = biological_rows["max_degree"]
    biological_rows = biological_rows[synthetic.columns]
    return pd.concat([synthetic, biological_rows], ignore_index=True)


def run(args: argparse.Namespace) -> None:
    if not 0 < args.candidate_fraction <= 1:
        raise ValueError("--candidate-fraction must be in (0, 1]")
    if args.delta < 0:
        raise ValueError("--delta must be non-negative")
    if args.replicates < 1:
        raise ValueError("--replicates must be at least 1")
    args.out_root.mkdir(parents=True, exist_ok=True)

    log("Auditing biological record-level and unique-coordinate RNG degrees")
    biological, multiplicity, biological_distribution, unique_coordinates, n_records = biological_summaries(args)
    unique_n = int(len(unique_coordinates))
    biological.to_csv(args.out_root / "biological_rng_degree_summary.csv", index=False)
    biological_distribution.to_csv(
        args.out_root / "biological_rng_degree_distribution.csv", index=False
    )
    multiplicity.to_csv(args.out_root / "coordinate_multiplicity_summary.csv", index=False)

    sample_sizes = set(args.sample_sizes)
    if args.add_biological_unique_size:
        sample_sizes.add(unique_n)
    condition_payloads = []
    for n_points in sorted(sample_sizes):
        for dimension in args.dimensions:
            for replicate in range(args.replicates):
                condition_payloads.append(
                    run_synthetic_condition(args, n_points, dimension, replicate)
                )

    run_frame = pd.DataFrame(
        [row for payload in condition_payloads for row in flatten_condition(payload)]
    )
    run_frame.to_csv(args.out_root / "synthetic_rng_degree_runs.csv", index=False)
    run_histograms = pd.concat(
        [flatten_condition_histograms(payload) for payload in condition_payloads],
        ignore_index=True,
    )
    run_histograms.to_csv(
        args.out_root / "synthetic_rng_degree_distribution_runs.csv", index=False
    )
    distribution_summary = summarize_degree_distributions(run_histograms)
    distribution_summary.to_csv(
        args.out_root / "synthetic_rng_degree_distribution_summary.csv", index=False
    )
    calibration = summarize_calibration(run_frame)
    calibration.to_csv(args.out_root / "synthetic_rng_degree_calibration.csv", index=False)
    combined = build_combined_calibration_table(biological, calibration)
    combined.to_csv(args.out_root / "global_rng_degree_calibration_table.csv", index=False)
    estimates = build_dimension_estimates(biological, calibration, unique_n)
    estimates.to_csv(args.out_root / "dimension_estimates.csv", index=False)

    write_json(
        args.out_root / "calibration_manifest.json",
        {
            "algorithm_version": ALGORITHM_VERSION,
            "completed_at_unix": time.time(),
            "sequence_content_written": False,
            "coordinate_content_written": False,
            "panel_root": args.panel_root,
            "directional_root": args.directional_root,
            "candidate_label": args.candidate_label,
            "dimensions": args.dimensions,
            "requested_sample_sizes": args.sample_sizes,
            "executed_sample_sizes": sorted(sample_sizes),
            "replicates": args.replicates,
            "n_biological_records": n_records,
            "n_unique_biological_coordinates": unique_n,
            "n_duplicate_record_rows": int(n_records - unique_n),
            "metric": "cityblock",
            "point_distribution": "uniform_unit_hypercube",
            "rng_rule": "strict_open_lune; tied boundary points do not witness removal",
            "synthetic_refinement": {
                "candidate_fraction": args.candidate_fraction,
                "delta": args.delta,
                "endpoint_rule": "both_endpoints_AND",
                "connectivity_protection": True,
            },
            "interpretation_boundary": (
                "Do not infer dimension for the 20k biological record graph from continuous uniform points. "
                "Repeated coordinate rows inflate strict-tie RNG degree. Dimension estimates are limited "
                "to the method-matched unique-coordinate comparisons."
            ),
            "inputs": {
                "embeddings": file_signature(embedding_path(args.panel_root)),
                "original_rng": file_signature(original_rng_path(args.panel_root)),
                "refined_rng": file_signature(
                    refined_rng_path(args.directional_root, args.candidate_label)
                ),
            },
            "outputs": {
                "biological_summary": args.out_root / "biological_rng_degree_summary.csv",
                "biological_degree_distribution": args.out_root
                / "biological_rng_degree_distribution.csv",
                "multiplicity_summary": args.out_root / "coordinate_multiplicity_summary.csv",
                "synthetic_runs": args.out_root / "synthetic_rng_degree_runs.csv",
                "synthetic_degree_distribution_runs": args.out_root
                / "synthetic_rng_degree_distribution_runs.csv",
                "synthetic_degree_distribution_summary": args.out_root
                / "synthetic_rng_degree_distribution_summary.csv",
                "calibration": args.out_root / "synthetic_rng_degree_calibration.csv",
                "combined_calibration_table": args.out_root
                / "global_rng_degree_calibration_table.csv",
                "dimension_estimates": args.out_root / "dimension_estimates.csv",
            },
        },
    )
    log(f"Wrote calibration outputs under {args.out_root}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", type=Path, default=DEFAULT_PANEL_ROOT)
    parser.add_argument("--directional-root", type=Path, default=DEFAULT_DIRECTIONAL_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--candidate-label", default=DEFAULT_CANDIDATE_LABEL)
    parser.add_argument("--dimensions", type=parse_int_csv, default=parse_int_csv("2,3,4,5,6,7"))
    parser.add_argument("--sample-sizes", type=parse_int_csv, default=parse_int_csv("20000"))
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-fraction", type=float, default=0.10)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--norm-epsilon", type=float, default=1e-12)
    parser.add_argument("--rng-row-block-size", type=int, default=100)
    parser.add_argument("--max-block-edges", type=int, default=2_000_000)
    parser.add_argument("--add-biological-unique-size", action="store_true", default=True)
    parser.add_argument(
        "--no-add-biological-unique-size",
        dest="add_biological_unique_size",
        action="store_false",
    )
    parser.add_argument("--force", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
