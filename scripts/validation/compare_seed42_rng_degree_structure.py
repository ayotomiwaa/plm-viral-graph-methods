#!/usr/bin/env python3
"""Compare biological and matched synthetic RNG structure before/after f_j refinement."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PANEL_ROOT = Path(
    "analysis/cohort_validation/07_sampling_design_20k/"
    "random_full_dataset_seed42/seed_42"
)
DEFAULT_OUT_ROOT = Path(
    "analysis/cohort_validation/29_seed42_20k_rng_degree_dimension_calibration/"
    "random_full_dataset_seed42/seed_42/degree_structure_comparison"
)
ALGORITHM_VERSION = 1
STAGE_ORDER = [
    "original_graph",
    "one_round_degree_1_removed",
    "recursive_2_core",
    "charikar_best",
    "goldberg_exact",
]
STAGE_LABELS = {
    "original_graph": "Whole graph",
    "one_round_degree_1_removed": "One leaf round",
    "recursive_2_core": "Recursive 2-core",
    "charikar_best": "Charikar best",
    "goldberg_exact": "Goldberg exact",
}


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
    "rng_structure_calibration",
    "scripts/validation/evaluate_seed42_rng_degree_dimension_calibration.py",
)
STRUCTURE = load_module(
    "rng_structure_audit",
    "scripts/validation/analyze_seed42_rng_degree_structure.py",
)


@dataclass
class GraphCondition:
    graph_id: str
    data_source: str
    graph_state: str
    n_nodes: int
    source: np.ndarray
    target: np.ndarray
    replicate: int | None = None
    condition_seed: int | None = None
    multiplicity: np.ndarray | None = None
    refinement_qc: dict[str, Any] | None = None


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def graph_fingerprint(condition: GraphCondition, exact_flow: bool) -> str:
    digest = hashlib.sha256()
    digest.update(str(ALGORITHM_VERSION).encode("ascii"))
    digest.update(condition.graph_id.encode("utf-8"))
    digest.update(np.asarray(condition.source, dtype=np.int64).tobytes())
    digest.update(np.asarray(condition.target, dtype=np.int64).tobytes())
    digest.update(str(bool(exact_flow)).encode("ascii"))
    return digest.hexdigest()


def common_labels(condition: GraphCondition) -> dict[str, Any]:
    return {
        "graph_id": condition.graph_id,
        "data_source": condition.data_source,
        "graph_state": condition.graph_state,
        "replicate": condition.replicate,
        "condition_seed": condition.condition_seed,
    }


def add_simple_density(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    denominator = frame["n_nodes"] * (frame["n_nodes"] - 1)
    frame["simple_graph_density"] = np.where(
        denominator > 0, 2.0 * frame["n_edges"] / denominator, np.nan
    )
    return frame


def upper_tail_rows(
    condition: GraphCondition,
    degree: np.ndarray,
    neighbors: list[set[int]],
    one_round: np.ndarray,
    core: np.ndarray,
    n_nodes: int,
) -> pd.DataFrame:
    selected = np.argsort(-degree, kind="stable")[:n_nodes]
    rows: list[dict[str, Any]] = []
    for node in selected:
        adjacent = neighbors[int(node)]
        row = {
            **common_labels(condition),
            "node_id": int(node),
            "original_degree": int(degree[node]),
            "initial_degree_1_neighbors": int(sum(degree[item] == 1 for item in adjacent)),
            "degree_after_one_round_leaf_removal": int(sum(one_round[item] for item in adjacent)),
            "present_in_recursive_2_core": bool(core[node]),
            "degree_within_recursive_2_core": int(sum(core[item] for item in adjacent)),
        }
        if condition.multiplicity is not None:
            row["coordinate_multiplicity"] = int(condition.multiplicity[node])
            row["additional_twins"] = int(condition.multiplicity[node] - 1)
        else:
            row["coordinate_multiplicity"] = np.nan
            row["additional_twins"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def audit_graph(
    condition: GraphCondition,
    work_root: Path,
    upper_tail_nodes: int,
    exact_flow: bool,
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    audit_root = work_root / "audits" / condition.graph_id
    manifest_path = audit_root / "manifest.json"
    fingerprint = graph_fingerprint(condition, exact_flow)
    output_paths = {
        "stages": audit_root / "stages.csv",
        "trace": audit_root / "charikar_trace.csv",
        "degree": audit_root / "degree_distribution.csv",
        "upper_tail": audit_root / "upper_tail.csv",
    }
    if not force and manifest_path.exists() and all(path.exists() for path in output_paths.values()):
        cached = json.loads(manifest_path.read_text())
        if cached.get("fingerprint") == fingerprint:
            log(f"Reusing structural audit for {condition.graph_id}")
            return (
                pd.read_csv(output_paths["stages"]),
                pd.read_csv(output_paths["trace"]),
                pd.read_csv(output_paths["degree"]),
                pd.read_csv(output_paths["upper_tail"]),
                cached,
            )

    log(f"Auditing {condition.graph_id}")
    source = condition.source.astype(np.int64, copy=False)
    target = condition.target.astype(np.int64, copy=False)
    neighbors = STRUCTURE.neighbors_from_edges(condition.n_nodes, source, target)
    degree = CALIBRATION.degree_vector(condition.n_nodes, source, target)
    if degree.min() < 1:
        raise ValueError(f"{condition.graph_id}: graph contains isolated vertices")
    whole = np.ones(condition.n_nodes, dtype=bool)
    one_round = STRUCTURE.one_round_leaf_removal(condition.n_nodes, source, target)
    core = STRUCTURE.k_core_mask(neighbors, minimum_degree=2)
    charikar, trace = STRUCTURE.charikar_peeling(neighbors, source, target)
    stages = [
        STRUCTURE.graph_stage_row("original_graph", whole, source, target),
        STRUCTURE.graph_stage_row("one_round_degree_1_removed", one_round, source, target),
        STRUCTURE.graph_stage_row("recursive_2_core", core, source, target),
        STRUCTURE.graph_stage_row("charikar_best", charikar, source, target),
    ]
    exact_qc = None
    if exact_flow:
        exact, exact_qc = STRUCTURE.goldberg_exact_densest_subgraph(
            condition.n_nodes, source, target
        )
        stages.append(STRUCTURE.graph_stage_row("goldberg_exact", exact, source, target))
    stages_frame = add_simple_density(pd.DataFrame(stages))
    for key, value in reversed(list(common_labels(condition).items())):
        stages_frame.insert(0, key, value)

    trace = trace.copy()
    trace["fraction_nodes_retained"] = (
        trace["n_nodes_before_removal"] / condition.n_nodes
    )
    for key, value in reversed(list(common_labels(condition).items())):
        trace.insert(0, key, value)

    counts = np.bincount(degree)
    distribution = pd.DataFrame(
        {
            "degree": np.arange(len(counts), dtype=np.int64),
            "n_nodes_at_degree": counts,
            "fraction_nodes": counts / condition.n_nodes,
        }
    )
    for key, value in reversed(list(common_labels(condition).items())):
        distribution.insert(0, key, value)
    upper_tail = upper_tail_rows(
        condition, degree, neighbors, one_round, core, upper_tail_nodes
    )

    audit_root.mkdir(parents=True, exist_ok=True)
    stages_frame.to_csv(output_paths["stages"], index=False)
    trace.to_csv(output_paths["trace"], index=False)
    distribution.to_csv(output_paths["degree"], index=False)
    upper_tail.to_csv(output_paths["upper_tail"], index=False)
    manifest = {
        "fingerprint": fingerprint,
        "graph_id": condition.graph_id,
        "n_nodes": condition.n_nodes,
        "n_edges": int(len(source)),
        "exact_flow_enabled": exact_flow,
        "exact_flow": exact_qc,
        "completed_at_unix": time.time(),
        "outputs": {key: str(value) for key, value in output_paths.items()},
    }
    write_json(manifest_path, manifest)
    return stages_frame, trace, distribution, upper_tail, manifest


def biological_conditions(args: argparse.Namespace) -> tuple[list[GraphCondition], dict[str, Any]]:
    embedding_file = CALIBRATION.embedding_path(args.panel_root)
    adjacency_file = CALIBRATION.original_rng_path(args.panel_root)
    coordinates = np.load(embedding_file, mmap_mode="r")
    adjacency = CALIBRATION.validate_adjacency(adjacency_file, int(coordinates.shape[0]))
    unique_coordinates, inverse, multiplicity = np.unique(
        np.asarray(coordinates), axis=0, return_inverse=True, return_counts=True
    )
    source, target, weight, collapse_qc = CALIBRATION.collapse_graph_to_unique_coordinates(
        adjacency, inverse, len(unique_coordinates)
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
    conditions = [
        GraphCondition(
            "biological_original",
            "biological_unique_coordinates",
            "original",
            len(unique_coordinates),
            source,
            target,
            multiplicity=multiplicity,
        ),
        GraphCondition(
            "biological_refined",
            "biological_unique_coordinates",
            "refined",
            len(unique_coordinates),
            refined_source,
            refined_target,
            multiplicity=multiplicity,
            refinement_qc=refinement_qc,
        ),
    ]
    qc = {
        "n_records": int(len(inverse)),
        "n_unique_coordinates": int(len(unique_coordinates)),
        "collapse_qc": collapse_qc,
        "refinement_qc": refinement_qc,
        "inputs": {
            "embeddings": file_signature(embedding_file),
            "original_rng": file_signature(adjacency_file),
        },
    }
    return conditions, qc


def synthetic_cache_fingerprint(args: argparse.Namespace, n_points: int, replicate: int) -> str:
    payload = {
        "algorithm_version": CALIBRATION.ALGORITHM_VERSION,
        "n_points": n_points,
        "dimension": args.dimension,
        "replicate": replicate,
        "base_seed": args.seed,
        "metric": "cityblock",
        "rng_rule": "strict_open_lune",
        "candidate_fraction": args.candidate_fraction,
        "delta": args.delta,
        "norm_epsilon": args.norm_epsilon,
        "rng_row_block_size": args.rng_row_block_size,
        "max_block_edges": args.max_block_edges,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def synthetic_conditions(
    args: argparse.Namespace, n_points: int, replicate: int
) -> tuple[list[GraphCondition], dict[str, Any]]:
    cache_root = args.out_root / "work" / "synthetic_edges"
    stem = f"n{n_points:06d}_d{args.dimension:02d}_rep{replicate:03d}"
    edge_path = cache_root / f"{stem}.npz"
    metadata_path = cache_root / f"{stem}.json"
    fingerprint = synthetic_cache_fingerprint(args, n_points, replicate)
    payload: dict[str, Any] | None = None
    if not args.force_graph_rebuild and edge_path.exists() and metadata_path.exists():
        candidate = json.loads(metadata_path.read_text())
        if candidate.get("fingerprint") == fingerprint:
            payload = candidate
            log(f"Reusing synthetic edge checkpoint {edge_path.name}")

    if payload is None:
        condition_seed = CALIBRATION.stable_condition_seed(
            args.seed, n_points, args.dimension, replicate
        )
        log(
            f"Building uniform L1 RNG n={n_points:,}, d={args.dimension}, "
            f"replicate={replicate}, seed={condition_seed}"
        )
        rng = np.random.default_rng(condition_seed)
        coordinates = rng.random((n_points, args.dimension), dtype=np.float32)
        distance = cdist(coordinates, coordinates, metric="cityblock").astype(np.float32)
        np.fill_diagonal(distance, np.inf)
        source, target, weight, checked, pruned = (
            CALIBRATION.GRAPH_MOD.exact_rng_edges_blockwise_order(
                distance,
                row_block_size=args.rng_row_block_size,
                max_block_edges=args.max_block_edges,
            )
        )
        del distance
        source = source.astype(np.int64, copy=False)
        target = target.astype(np.int64, copy=False)
        weight = weight.astype(np.float32, copy=False)
        refined_source, refined_target, _, refinement_qc = CALIBRATION.refine_rng(
            coordinates,
            source,
            target,
            weight,
            candidate_fraction=args.candidate_fraction,
            delta=args.delta,
            norm_epsilon=args.norm_epsilon,
        )
        save_npz_atomic(
            edge_path,
            original_source=source,
            original_target=target,
            refined_source=refined_source,
            refined_target=refined_target,
        )
        payload = {
            "fingerprint": fingerprint,
            "n_points": n_points,
            "dimension": args.dimension,
            "replicate": replicate,
            "condition_seed": condition_seed,
            "distribution": "uniform_unit_hypercube",
            "metric": "cityblock",
            "candidate_edges_checked": int(checked),
            "candidate_edges_pruned": int(pruned),
            "refinement_qc": refinement_qc,
            "coordinate_content_written": False,
            "completed_at_unix": time.time(),
        }
        write_json(metadata_path, payload)

    with np.load(edge_path) as arrays:
        original_source = arrays["original_source"].astype(np.int64, copy=True)
        original_target = arrays["original_target"].astype(np.int64, copy=True)
        refined_source = arrays["refined_source"].astype(np.int64, copy=True)
        refined_target = arrays["refined_target"].astype(np.int64, copy=True)
    prefix = f"synthetic_d{args.dimension}_rep{replicate}"
    conditions = [
        GraphCondition(
            f"{prefix}_original",
            f"synthetic_uniform_L1_d{args.dimension}",
            "original",
            n_points,
            original_source,
            original_target,
            replicate=replicate,
            condition_seed=int(payload["condition_seed"]),
        ),
        GraphCondition(
            f"{prefix}_refined",
            f"synthetic_uniform_L1_d{args.dimension}",
            "refined",
            n_points,
            refined_source,
            refined_target,
            replicate=replicate,
            condition_seed=int(payload["condition_seed"]),
            refinement_qc=payload["refinement_qc"],
        ),
    ]
    return conditions, payload


def edge_keys(condition: GraphCondition) -> np.ndarray:
    return condition.source.astype(np.int64) * np.int64(condition.n_nodes) + condition.target


def refinement_transition(
    original: GraphCondition, refined: GraphCondition
) -> tuple[dict[str, Any], pd.DataFrame]:
    if original.n_nodes != refined.n_nodes:
        raise ValueError("paired original/refined graphs have different node counts")
    original_keys = edge_keys(original)
    refined_keys = edge_keys(refined)
    retained = np.isin(original_keys, refined_keys, assume_unique=True)
    if not np.all(np.isin(refined_keys, original_keys, assume_unique=True)):
        raise ValueError(f"{refined.graph_id}: refinement introduced an edge")
    degree_original = CALIBRATION.degree_vector(
        original.n_nodes, original.source, original.target
    )
    degree_refined = CALIBRATION.degree_vector(
        refined.n_nodes, refined.source, refined.target
    )
    change = degree_refined - degree_original
    values, counts = np.unique(change, return_counts=True)
    distribution = pd.DataFrame(
        {
            **{key: [value] * len(values) for key, value in common_labels(refined).items()},
            "degree_change_refined_minus_original": values,
            "n_nodes": counts,
            "fraction_nodes": counts / original.n_nodes,
        }
    )
    qc = refined.refinement_qc or {}
    summary = {
        "data_source": original.data_source,
        "replicate": original.replicate,
        "condition_seed": original.condition_seed,
        "n_nodes": original.n_nodes,
        "original_edges": int(len(original.source)),
        "refined_edges": int(len(refined.source)),
        "removed_edges": int(np.count_nonzero(~retained)),
        "fraction_edges_removed": float(np.mean(~retained)),
        "nodes_with_degree_reduction": int(np.count_nonzero(change < 0)),
        "unchanged_degree_nodes": int(np.count_nonzero(change == 0)),
        "original_degree_1_nodes": int(np.count_nonzero(degree_original == 1)),
        "refined_degree_1_nodes": int(np.count_nonzero(degree_refined == 1)),
        "endpoint_rule": qc.get("endpoint_rule"),
        "connectivity_rule": qc.get("connectivity_rule"),
        "mutually_rejected_edges": qc.get("mutually_rejected_edges"),
        "mutually_rejected_edges_removed": qc.get("mutually_rejected_edges_removed"),
        "mutually_rejected_edges_retained_for_connectivity": qc.get(
            "mutually_rejected_edges_retained_for_connectivity"
        ),
        "components_before": qc.get("components_before"),
        "components_after": qc.get("components_after"),
    }
    return summary, distribution


def aggregate_synthetic_stages(stages: pd.DataFrame) -> pd.DataFrame:
    synthetic = stages[stages["data_source"].str.startswith("synthetic_")]
    metrics = [
        "n_nodes",
        "n_edges",
        "density_m_over_n",
        "mean_degree_2m_over_n",
        "simple_graph_density",
        "median_degree",
        "min_degree",
        "max_degree",
    ]
    rows = []
    for (data_source, graph_state, stage), group in synthetic.groupby(
        ["data_source", "graph_state", "stage"], sort=False
    ):
        row: dict[str, Any] = {
            "data_source": data_source,
            "graph_state": graph_state,
            "stage": stage,
            "replicates": int(group["replicate"].nunique()),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
            row[f"{metric}_min"] = float(group[metric].min())
            row[f"{metric}_max"] = float(group[metric].max())
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_synthetic_degree(distribution: pd.DataFrame) -> pd.DataFrame:
    synthetic = distribution[distribution["data_source"].str.startswith("synthetic_")]
    rows = []
    for (data_source, graph_state), group in synthetic.groupby(
        ["data_source", "graph_state"], sort=False
    ):
        replicates = sorted(group["replicate"].dropna().astype(int).unique())
        max_degree = int(group["degree"].max())
        index = pd.MultiIndex.from_product(
            [replicates, range(max_degree + 1)], names=["replicate", "degree"]
        )
        complete = (
            group.set_index(["replicate", "degree"])["fraction_nodes"]
            .reindex(index, fill_value=0.0)
            .reset_index()
        )
        summary = complete.groupby("degree", as_index=False).agg(
            replicates=("replicate", "nunique"),
            mean_fraction_nodes=("fraction_nodes", "mean"),
            sd_fraction_nodes=("fraction_nodes", "std"),
            min_fraction_nodes=("fraction_nodes", "min"),
            max_fraction_nodes=("fraction_nodes", "max"),
        )
        summary.insert(0, "graph_state", graph_state)
        summary.insert(0, "data_source", data_source)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def plot_degree_distribution(
    distribution: pd.DataFrame, synthetic_summary: pd.DataFrame, out_root: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), sharey=True, constrained_layout=True)
    for axis, state in zip(axes, ["original", "refined"]):
        biological = distribution[
            (distribution["data_source"] == "biological_unique_coordinates")
            & (distribution["graph_state"] == state)
        ]
        synthetic = synthetic_summary[synthetic_summary["graph_state"] == state]
        axis.plot(
            biological["degree"],
            biological["fraction_nodes"],
            color="#111827",
            marker="o",
            markersize=3,
            linewidth=1.5,
            label="Biological unique-coordinate RNG",
        )
        axis.plot(
            synthetic["degree"],
            synthetic["mean_fraction_nodes"],
            color="#059669",
            marker="s",
            markersize=3,
            linewidth=1.4,
            label="Uniform L1 d=2 mean",
        )
        axis.fill_between(
            synthetic["degree"],
            synthetic["min_fraction_nodes"],
            synthetic["max_fraction_nodes"],
            color="#059669",
            alpha=0.18,
            label="Synthetic replicate range",
        )
        axis.set_yscale("log")
        axis.set_ylim(bottom=5e-5)
        axis.set_xlabel("Vertex degree")
        axis.set_title(state.capitalize())
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Fraction of vertices (log scale)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("RNG degree distributions at N=8,921")
    for suffix in ("png", "pdf"):
        fig.savefig(out_root / f"degree_distribution_comparison.{suffix}", dpi=220)
    plt.close(fig)


def plot_stage_comparison(
    stages: pd.DataFrame, synthetic_summary: pd.DataFrame, out_root: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.7), sharey=True, constrained_layout=True)
    positions = np.arange(len(STAGE_ORDER))
    for axis, state in zip(axes, ["original", "refined"]):
        biological = (
            stages[
                (stages["data_source"] == "biological_unique_coordinates")
                & (stages["graph_state"] == state)
            ]
            .set_index("stage")
            .reindex(STAGE_ORDER)
        )
        synthetic = (
            synthetic_summary[synthetic_summary["graph_state"] == state]
            .set_index("stage")
            .reindex(STAGE_ORDER)
        )
        axis.plot(
            positions,
            biological["mean_degree_2m_over_n"],
            color="#111827",
            marker="o",
            linewidth=1.7,
            label="Biological",
        )
        axis.errorbar(
            positions,
            synthetic["mean_degree_2m_over_n_mean"],
            yerr=np.vstack(
                [
                    synthetic["mean_degree_2m_over_n_mean"]
                    - synthetic["mean_degree_2m_over_n_min"],
                    synthetic["mean_degree_2m_over_n_max"]
                    - synthetic["mean_degree_2m_over_n_mean"],
                ]
            ),
            color="#059669",
            marker="s",
            linewidth=1.5,
            capsize=3,
            label="Uniform L1 d=2, mean and range",
        )
        axis.set_xticks(positions, [STAGE_LABELS[item] for item in STAGE_ORDER], rotation=25, ha="right")
        axis.set_title(state.capitalize())
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Mean induced degree (2m/n)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Leaf removal, 2-core, and densest-subgraph audit")
    for suffix in ("png", "pdf"):
        fig.savefig(out_root / f"graph_stage_comparison.{suffix}", dpi=220)
    plt.close(fig)


def plot_charikar_comparison(trace: pd.DataFrame, out_root: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), sharey=True, constrained_layout=True)
    for axis, state in zip(axes, ["original", "refined"]):
        selected = trace[trace["graph_state"] == state]
        biological = selected[selected["data_source"] == "biological_unique_coordinates"]
        axis.plot(
            biological["fraction_nodes_retained"],
            biological["mean_degree_2m_over_n"],
            color="#111827",
            linewidth=1.8,
            label="Biological",
        )
        synthetic = selected[selected["data_source"].str.startswith("synthetic_")]
        for index, (_, group) in enumerate(synthetic.groupby("replicate")):
            axis.plot(
                group["fraction_nodes_retained"],
                group["mean_degree_2m_over_n"],
                color="#059669",
                alpha=0.45,
                linewidth=1.0,
                label="Uniform L1 d=2 replicates" if index == 0 else None,
            )
        axis.set_xlim(1.0, 0.0)
        axis.set_xlabel("Fraction of vertices retained")
        axis.set_title(state.capitalize())
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Mean induced degree (2m/n)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Minimum-degree peeling trajectories")
    for suffix in ("png", "pdf"):
        fig.savefig(out_root / f"charikar_peeling_comparison.{suffix}", dpi=220)
    plt.close(fig)


def write_readme(args: argparse.Namespace, n_points: int) -> None:
    text = f"""# Biological versus matched synthetic RNG structure

This directory compares four method-matched graph classes at `N={n_points:,}`:

1. original biological unique-coordinate RNG;
2. directly mutual-`f_j`-refined biological unique-coordinate RNG;
3. original RNGs from uniform random points in `[0,1]^{args.dimension}` under L1 distance;
4. directly mutual-`f_j`-refined versions of those same synthetic RNGs.

The synthetic comparison uses {args.replicates} deterministic replicate point clouds. Within
each replicate, original and refined graphs use exactly the same points and original RNG.
Dimension {args.dimension} is the default matched control because the completed degree calibration
places both biological graph states nearest to the two-dimensional uniform L1 condition. It is a
known-geometry calibration control, not a generative null model for biological evolution.

## Structural stages

`graph_stage_summary_all_runs.csv` reports the whole graph, one-round deletion of
original degree-1 vertices, recursive 2-core, Charikar minimum-degree peeling optimum,
and Goldberg exact densest subgraph. `density_m_over_n` follows the meeting convention;
`mean_degree_2m_over_n` is twice that value; `simple_graph_density` is `2m/[n(n-1)]`.

## Refinement rule

An edge enters the deletion queue only when both endpoints independently reject it.
Queued deletions are then reverse-deleted with connectivity protection. The paired audit
is in `refinement_transition_summary.csv` and `refinement_degree_change_distribution.csv`.

## Other outputs

- `degree_distribution_all_runs.csv`: complete integer degree PMFs.
- `degree_distribution_synthetic_aggregate.csv`: synthetic mean, SD, and range.
- `upper_tail_vertex_audit.csv`: sequence-free high-degree structural audit.
- `biological_duplicate_multiplicity_*.csv`: exact-coordinate multiplicity by graph state.
- `figures/`: presentation plots for degree, structural stages, and peeling trajectories.
- `comparison_manifest.json`: parameters, input signatures, refinement QC, and checkpoints.

Synthetic point coordinates and biological embedding values are not written to this directory.
"""
    (args.out_root / "README.md").write_text(text)


def run(args: argparse.Namespace) -> None:
    if args.replicates < 1:
        raise ValueError("--replicates must be positive")
    if args.dimension < 1:
        raise ValueError("--dimension must be positive")
    args.out_root.mkdir(parents=True, exist_ok=True)
    biological, biological_qc = biological_conditions(args)
    n_points = biological[0].n_nodes
    synthetic: list[GraphCondition] = []
    synthetic_qc: list[dict[str, Any]] = []
    for replicate in range(args.replicates):
        conditions, qc = synthetic_conditions(args, n_points, replicate)
        synthetic.extend(conditions)
        synthetic_qc.append(qc)
    conditions = biological + synthetic

    stage_frames = []
    trace_frames = []
    distribution_frames = []
    upper_tail_frames = []
    audit_manifests = []
    for condition in conditions:
        stages, trace, distribution, upper_tail, manifest = audit_graph(
            condition,
            args.out_root / "work",
            args.upper_tail_nodes,
            args.exact_flow,
            args.force_audit,
        )
        stage_frames.append(stages)
        trace_frames.append(trace)
        distribution_frames.append(distribution)
        upper_tail_frames.append(upper_tail)
        audit_manifests.append(manifest)
    stages = pd.concat(stage_frames, ignore_index=True)
    trace = pd.concat(trace_frames, ignore_index=True)
    distribution = pd.concat(distribution_frames, ignore_index=True)
    upper_tail = pd.concat(upper_tail_frames, ignore_index=True)

    transition_rows = []
    change_frames = []
    paired_conditions = [(biological[0], biological[1])]
    paired_conditions.extend(
        (synthetic[index], synthetic[index + 1]) for index in range(0, len(synthetic), 2)
    )
    for original, refined in paired_conditions:
        summary, changes = refinement_transition(original, refined)
        transition_rows.append(summary)
        change_frames.append(changes)
    transitions = pd.DataFrame(transition_rows)
    changes = pd.concat(change_frames, ignore_index=True)

    duplicate_summaries = []
    duplicate_by_degree = []
    duplicate_tests: dict[str, Any] = {}
    for condition in biological:
        degree = CALIBRATION.degree_vector(
            condition.n_nodes, condition.source, condition.target
        )
        summary, by_degree, tests = STRUCTURE.duplicate_audit(
            degree, condition.multiplicity
        )
        summary.insert(0, "graph_state", condition.graph_state)
        by_degree.insert(0, "graph_state", condition.graph_state)
        duplicate_summaries.append(summary)
        duplicate_by_degree.append(by_degree)
        duplicate_tests[condition.graph_state] = tests

    synthetic_stages = aggregate_synthetic_stages(stages)
    synthetic_degree = aggregate_synthetic_degree(distribution)
    stages.to_csv(args.out_root / "graph_stage_summary_all_runs.csv", index=False)
    synthetic_stages.to_csv(
        args.out_root / "graph_stage_summary_synthetic_aggregate.csv", index=False
    )
    trace.to_csv(args.out_root / "charikar_density_trace_all_runs.csv", index=False)
    distribution.to_csv(args.out_root / "degree_distribution_all_runs.csv", index=False)
    synthetic_degree.to_csv(
        args.out_root / "degree_distribution_synthetic_aggregate.csv", index=False
    )
    upper_tail.to_csv(args.out_root / "upper_tail_vertex_audit.csv", index=False)
    transitions.to_csv(args.out_root / "refinement_transition_summary.csv", index=False)
    changes.to_csv(
        args.out_root / "refinement_degree_change_distribution.csv", index=False
    )
    pd.concat(duplicate_summaries, ignore_index=True).to_csv(
        args.out_root / "biological_duplicate_multiplicity_group_summary.csv", index=False
    )
    pd.concat(duplicate_by_degree, ignore_index=True).to_csv(
        args.out_root / "biological_duplicate_multiplicity_by_degree.csv", index=False
    )

    figures = args.out_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_degree_distribution(distribution, synthetic_degree, figures)
    plot_stage_comparison(stages, synthetic_stages, figures)
    plot_charikar_comparison(trace, figures)
    write_readme(args, n_points)

    manifest = {
        "algorithm_version": ALGORITHM_VERSION,
        "completed_at_unix": time.time(),
        "n_points_per_graph": n_points,
        "comparison_design": {
            "biological_states": ["original_unique_coordinate", "direct_fj_refined_unique_coordinate"],
            "synthetic_distribution": "uniform_unit_hypercube",
            "synthetic_dimension": args.dimension,
            "synthetic_replicates": args.replicates,
            "metric": "cityblock",
            "same_points_within_original_refined_pair": True,
        },
        "refinement": {
            "candidate_fraction": args.candidate_fraction,
            "delta": args.delta,
            "norm_epsilon": args.norm_epsilon,
            "endpoint_rule": "both_endpoints_AND",
            "connectivity_rule": "reverse_delete_preserving_original_component_count",
        },
        "exact_flow_enabled": args.exact_flow,
        "biological_qc": biological_qc,
        "synthetic_qc": synthetic_qc,
        "duplicate_tests": duplicate_tests,
        "audit_manifests": audit_manifests,
        "sequence_content_written": False,
        "coordinate_content_written": False,
        "outputs": sorted(
            str(path) for path in args.out_root.rglob("*") if path.is_file()
        ),
    }
    write_json(args.out_root / "comparison_manifest.json", manifest)
    log("Comparison complete")
    print(
        stages[stages["stage"].isin(["original_graph", "recursive_2_core", "goldberg_exact"])]
        .to_string(index=False)
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", type=Path, default=DEFAULT_PANEL_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-fraction", type=float, default=0.10)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--norm-epsilon", type=float, default=1e-12)
    parser.add_argument("--rng-row-block-size", type=int, default=100)
    parser.add_argument("--max-block-edges", type=int, default=2_000_000)
    parser.add_argument("--upper-tail-nodes", type=int, default=4)
    parser.add_argument("--exact-flow", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-graph-rebuild", action="store_true")
    parser.add_argument("--force-audit", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
