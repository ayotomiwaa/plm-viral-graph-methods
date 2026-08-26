#!/usr/bin/env python3
"""Audit leaf multiplicity, upper-tail nodes, and dense structure of the seed-42 RNG."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import importlib.util
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import fisher_exact, mannwhitneyu


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PANEL_ROOT = Path(
    "analysis/cohort_validation/07_sampling_design_20k/"
    "random_full_dataset_seed42/seed_42"
)
DEFAULT_OUT_ROOT = Path(
    "analysis/cohort_validation/29_seed42_20k_rng_degree_dimension_calibration/"
    "random_full_dataset_seed42/seed_42/degree_structure"
)
ALGORITHM_VERSION = 1


def load_calibration_module():
    path = PROJECT_ROOT / "scripts/validation/evaluate_seed42_rng_degree_dimension_calibration.py"
    spec = importlib.util.spec_from_file_location("seed42_rng_degree_calibration", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CALIBRATION = load_calibration_module()


def file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def edge_arrays_from_adjacency(adjacency) -> tuple[np.ndarray, np.ndarray]:
    source, target, _ = CALIBRATION.upper_edges(adjacency)
    return source.astype(np.int64, copy=False), target.astype(np.int64, copy=False)


def neighbors_from_edges(
    n_nodes: int, source: np.ndarray, target: np.ndarray
) -> list[set[int]]:
    neighbors = [set() for _ in range(n_nodes)]
    for left, right in zip(source.tolist(), target.tolist()):
        neighbors[left].add(right)
        neighbors[right].add(left)
    return neighbors


def induced_edge_count(
    active: np.ndarray, source: np.ndarray, target: np.ndarray
) -> int:
    return int(np.count_nonzero(active[source] & active[target]))


def graph_stage_row(
    stage: str,
    active: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    n_nodes = int(active.sum())
    n_edges = induced_edge_count(active, source, target)
    degrees = np.bincount(
        np.concatenate([source[active[source] & active[target]], target[active[source] & active[target]]]),
        minlength=len(active),
    )[active]
    return {
        "stage": stage,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density_m_over_n": float(n_edges / n_nodes) if n_nodes else np.nan,
        "mean_degree_2m_over_n": float(2 * n_edges / n_nodes) if n_nodes else np.nan,
        "median_degree": float(np.median(degrees)) if n_nodes else np.nan,
        "min_degree": int(degrees.min()) if n_nodes else 0,
        "max_degree": int(degrees.max()) if n_nodes else 0,
    }


def one_round_leaf_removal(
    n_nodes: int, source: np.ndarray, target: np.ndarray
) -> np.ndarray:
    degree = CALIBRATION.degree_vector(n_nodes, source, target)
    return degree != 1


def k_core_mask(neighbors: list[set[int]], minimum_degree: int = 2) -> np.ndarray:
    n_nodes = len(neighbors)
    active = np.ones(n_nodes, dtype=bool)
    degree = np.array([len(items) for items in neighbors], dtype=np.int64)
    queue = deque(np.flatnonzero(degree < minimum_degree).tolist())
    while queue:
        node = queue.popleft()
        if not active[node] or degree[node] >= minimum_degree:
            continue
        active[node] = False
        for neighbor in neighbors[node]:
            if active[neighbor]:
                degree[neighbor] -= 1
                if degree[neighbor] < minimum_degree:
                    queue.append(neighbor)
    return active


def charikar_peeling(
    neighbors: list[set[int]], source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, pd.DataFrame]:
    """Deterministic minimum-degree peeling with smallest node ID as tie-breaker."""
    n_nodes = len(neighbors)
    active = np.ones(n_nodes, dtype=bool)
    degree = np.array([len(items) for items in neighbors], dtype=np.int64)
    heap = [(int(degree[node]), node) for node in range(n_nodes)]
    heapq.heapify(heap)
    n_active = n_nodes
    n_edges = int(len(source))
    best_density = n_edges / n_active
    best_mask = active.copy()
    rows: list[dict[str, Any]] = []

    for step in range(n_nodes):
        while heap:
            candidate_degree, node = heapq.heappop(heap)
            if active[node] and candidate_degree == degree[node]:
                break
        else:
            raise RuntimeError("degree heap emptied before all vertices were peeled")

        rows.append(
            {
                "removal_step": step,
                "n_nodes_before_removal": n_active,
                "n_edges_before_removal": n_edges,
                "density_m_over_n": n_edges / n_active,
                "mean_degree_2m_over_n": 2 * n_edges / n_active,
                "minimum_degree_removed": int(candidate_degree),
                "removed_unique_coordinate_id": int(node),
            }
        )
        active[node] = False
        n_active -= 1
        n_edges -= int(candidate_degree)
        for neighbor in neighbors[node]:
            if active[neighbor]:
                degree[neighbor] -= 1
                heapq.heappush(heap, (int(degree[neighbor]), neighbor))

        if n_active:
            density = n_edges / n_active
            if density > best_density:
                best_density = density
                best_mask = active.copy()

    return best_mask, pd.DataFrame(rows)


def goldberg_exact_densest_subgraph(
    n_nodes: int,
    source: np.ndarray,
    target: np.ndarray,
    tolerance: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve unweighted densest subgraph with Goldberg's parametric min-cut."""
    n_edges = int(len(source))
    if n_nodes == 0 or n_edges == 0:
        return np.zeros(n_nodes, dtype=bool), {
            "iterations": 0,
            "density_m_over_n": 0.0,
            "mean_degree_2m_over_n": 0.0,
        }
    degree = CALIBRATION.degree_vector(n_nodes, source, target)
    low = 0.0
    high = float(degree.max() / 2.0)
    if tolerance is None:
        tolerance = 1.0 / (n_nodes * max(1, n_nodes - 1))
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n_nodes + 2))
    source_node = n_nodes
    sink_node = n_nodes + 1
    graph.add_edges_from(
        (source_node, node, {"capacity": float(n_edges)}) for node in range(n_nodes)
    )
    for left, right in zip(source.tolist(), target.tolist()):
        graph.add_edge(left, right, capacity=1.0)
        graph.add_edge(right, left, capacity=1.0)

    best_nodes: set[int] = set(range(n_nodes))
    iterations = 0
    baseline_cut = float(n_edges * n_nodes)
    while high - low > tolerance:
        guess = (low + high) / 2.0
        for node in range(n_nodes):
            graph.add_edge(
                node,
                sink_node,
                capacity=float(n_edges + 2.0 * guess - degree[node]),
            )
        cut_value, partition = nx.minimum_cut(
            graph, source_node, sink_node, capacity="capacity", flow_func=nx.algorithms.flow.preflow_push
        )
        candidate = set(partition[0]) - {source_node}
        if candidate and cut_value < baseline_cut - 1e-7:
            low = guess
            best_nodes = candidate
        else:
            high = guess
        iterations += 1

    active = np.zeros(n_nodes, dtype=bool)
    active[list(best_nodes)] = True
    exact_edges = induced_edge_count(active, source, target)
    density = exact_edges / int(active.sum())
    return active, {
        "iterations": iterations,
        "search_lower_bound": low,
        "search_upper_bound": high,
        "tolerance": tolerance,
        "density_m_over_n": density,
        "mean_degree_2m_over_n": 2.0 * density,
    }


def duplicate_audit(
    degree: np.ndarray, multiplicity: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows = []
    for label, keep in [
        ("degree_1", degree == 1),
        ("degree_2", degree == 2),
        ("degree_ge_2", degree >= 2),
        ("all_nodes", np.ones(len(degree), dtype=bool)),
    ]:
        values = multiplicity[keep]
        rows.append(
            {
                "node_group": label,
                "n_unique_coordinate_nodes": int(len(values)),
                "n_record_rows": int(values.sum()),
                "mean_coordinate_multiplicity": float(values.mean()),
                "median_coordinate_multiplicity": float(np.median(values)),
                "mean_additional_twins": float((values - 1).mean()),
                "n_nodes_with_duplicate_records": int(np.count_nonzero(values > 1)),
                "fraction_nodes_with_duplicate_records": float(np.mean(values > 1)),
                "max_coordinate_multiplicity": int(values.max()),
            }
        )

    by_degree_rows = []
    for value in np.unique(degree):
        values = multiplicity[degree == value]
        by_degree_rows.append(
            {
                "degree": int(value),
                "n_unique_coordinate_nodes": int(len(values)),
                "n_record_rows": int(values.sum()),
                "mean_coordinate_multiplicity": float(values.mean()),
                "median_coordinate_multiplicity": float(np.median(values)),
                "fraction_nodes_with_duplicate_records": float(np.mean(values > 1)),
                "max_coordinate_multiplicity": int(values.max()),
            }
        )

    leaf = multiplicity[degree == 1]
    nonleaf = multiplicity[degree >= 2]
    contingency = np.array(
        [
            [np.count_nonzero(leaf > 1), np.count_nonzero(leaf == 1)],
            [np.count_nonzero(nonleaf > 1), np.count_nonzero(nonleaf == 1)],
        ]
    )
    odds_ratio, fisher_p = fisher_exact(contingency, alternative="less")
    mann = mannwhitneyu(leaf, nonleaf, alternative="less", method="asymptotic")
    tests = {
        "hypothesis": "degree-1 unique-coordinate nodes have lower coordinate multiplicity than degree>=2 nodes",
        "fisher_alternative": "odds of any duplicate record are lower for degree-1 nodes",
        "fisher_odds_ratio": float(odds_ratio),
        "fisher_p_value": float(fisher_p),
        "mann_whitney_alternative": "coordinate multiplicity is lower for degree-1 nodes",
        "mann_whitney_u": float(mann.statistic),
        "mann_whitney_p_value": float(mann.pvalue),
    }
    return pd.DataFrame(rows), pd.DataFrame(by_degree_rows), tests


def upper_tail_audit(
    degree: np.ndarray,
    multiplicity: np.ndarray,
    inverse: np.ndarray,
    metadata: pd.DataFrame,
    neighbors: list[set[int]],
    one_round: np.ndarray,
    core: np.ndarray,
    n_nodes: int = 4,
) -> pd.DataFrame:
    selected = np.argsort(-degree, kind="stable")[:n_nodes]
    rows: list[dict[str, Any]] = []
    dates = pd.to_datetime(metadata["collection_date"], errors="coerce")
    for coordinate_id in selected:
        record_rows = np.flatnonzero(inverse == coordinate_id)
        group = metadata.iloc[record_rows]
        neighbor_ids = neighbors[int(coordinate_id)]
        leaf_neighbors = sum(degree[item] == 1 for item in neighbor_ids)
        retained_one_round_neighbors = sum(one_round[item] for item in neighbor_ids)
        retained_core_neighbors = sum(core[item] for item in neighbor_ids)
        lineages = sorted(set(group["lineage"].dropna().astype(str)))
        cohorts = sorted(set(group["cohort_name"].dropna().astype(str)))
        representative = group.sort_values("accession", kind="stable").iloc[0]
        group_dates = dates.iloc[record_rows].dropna()
        rows.append(
            {
                "unique_coordinate_id": int(coordinate_id),
                "original_degree": int(degree[coordinate_id]),
                "coordinate_multiplicity": int(multiplicity[coordinate_id]),
                "additional_twins": int(multiplicity[coordinate_id] - 1),
                "initial_degree_1_neighbors": int(leaf_neighbors),
                "degree_after_one_round_leaf_removal": int(retained_one_round_neighbors),
                "present_in_recursive_2_core": bool(core[coordinate_id]),
                "degree_within_recursive_2_core": int(retained_core_neighbors),
                "representative_accession": str(representative["accession"]),
                "lineages": "|".join(lineages),
                "cohort_names": "|".join(cohorts),
                "earliest_collection_date": group_dates.min().date().isoformat() if len(group_dates) else "",
                "latest_collection_date": group_dates.max().date().isoformat() if len(group_dates) else "",
            }
        )
    return pd.DataFrame(rows)


def plot_density_trace(trace: pd.DataFrame, stages: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    axis.plot(
        trace["n_nodes_before_removal"],
        trace["mean_degree_2m_over_n"],
        color="#1D4ED8",
        linewidth=1.4,
    )
    markers = stages[stages["stage"].isin(["original", "recursive_2_core", "charikar_best", "goldberg_exact"])]
    colors = {
        "original": "#111827",
        "recursive_2_core": "#059669",
        "charikar_best": "#DC2626",
        "goldberg_exact": "#7C3AED",
    }
    labels = {
        "original": "original",
        "recursive_2_core": "recursive 2-core",
        "charikar_best": "Charikar best",
        "goldberg_exact": "Goldberg exact",
    }
    offsets = {
        "original": (6, 8),
        "recursive_2_core": (6, 10),
        "charikar_best": (6, -18),
        "goldberg_exact": (6, 8),
    }
    for row in markers.itertuples(index=False):
        axis.scatter(row.n_nodes, row.mean_degree_2m_over_n, s=38, color=colors[row.stage], zorder=3)
        axis.annotate(
            labels[row.stage],
            (row.n_nodes, row.mean_degree_2m_over_n),
            xytext=offsets[row.stage],
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Vertices retained")
    axis.set_ylabel("Mean induced degree (2m/n)")
    axis.set_title("Minimum-degree peeling of the biological unique-coordinate RNG")
    axis.grid(alpha=0.22)
    axis.invert_xaxis()
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"charikar_density_peeling.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    args.out_root.mkdir(parents=True, exist_ok=True)
    embedding_file = CALIBRATION.embedding_path(args.panel_root)
    adjacency_file = CALIBRATION.original_rng_path(args.panel_root)
    metadata_file = args.panel_root / "inputs/pool_n20000/metadata.csv"

    coordinates = np.load(embedding_file, mmap_mode="r")
    adjacency = CALIBRATION.validate_adjacency(adjacency_file, int(coordinates.shape[0]))
    unique_coordinates, inverse, multiplicity = np.unique(
        np.asarray(coordinates), axis=0, return_inverse=True, return_counts=True
    )
    source, target, weight, collapse_qc = CALIBRATION.collapse_graph_to_unique_coordinates(
        adjacency, inverse, len(unique_coordinates)
    )
    del unique_coordinates
    n_nodes = int(len(multiplicity))
    neighbors = neighbors_from_edges(n_nodes, source, target)
    degree = CALIBRATION.degree_vector(n_nodes, source, target)
    if degree.min() < 1:
        raise ValueError("the collapsed biological RNG unexpectedly contains isolated vertices")

    original = np.ones(n_nodes, dtype=bool)
    one_round = one_round_leaf_removal(n_nodes, source, target)
    core = k_core_mask(neighbors, minimum_degree=2)
    charikar_best, trace = charikar_peeling(neighbors, source, target)
    stage_rows = [
        graph_stage_row("original", original, source, target),
        graph_stage_row("one_round_degree_1_removed", one_round, source, target),
        graph_stage_row("recursive_2_core", core, source, target),
        graph_stage_row("charikar_best", charikar_best, source, target),
    ]

    exact_qc: dict[str, Any] | None = None
    if args.exact_flow:
        exact, exact_qc = goldberg_exact_densest_subgraph(n_nodes, source, target)
        stage_rows.append(graph_stage_row("goldberg_exact", exact, source, target))
    stages = pd.DataFrame(stage_rows)

    metadata = pd.read_csv(metadata_file, low_memory=False)
    if len(metadata) != len(inverse):
        raise ValueError("metadata row count does not match embedding row count")
    if not np.array_equal(metadata["node_id"].to_numpy(dtype=np.int64), np.arange(len(metadata))):
        raise ValueError("metadata node_id is not aligned with embedding rows")

    duplicate_summary, duplicate_by_degree, duplicate_tests = duplicate_audit(degree, multiplicity)
    upper_tail = upper_tail_audit(
        degree,
        multiplicity,
        inverse,
        metadata,
        neighbors,
        one_round,
        core,
        n_nodes=args.upper_tail_nodes,
    )

    stages.to_csv(args.out_root / "graph_stage_summary.csv", index=False)
    trace.to_csv(args.out_root / "charikar_density_trace.csv", index=False)
    duplicate_summary.to_csv(args.out_root / "duplicate_multiplicity_group_summary.csv", index=False)
    duplicate_by_degree.to_csv(args.out_root / "duplicate_multiplicity_by_degree.csv", index=False)
    upper_tail.to_csv(args.out_root / "upper_tail_vertex_audit.csv", index=False)
    plot_density_trace(trace, stages, args.out_root / "figures")

    manifest = {
        "algorithm_version": ALGORITHM_VERSION,
        "completed_at_unix": time.time(),
        "graph_representation": "biological_unique_collapsed_original",
        "sequence_content_written": False,
        "coordinate_content_written": False,
        "one_round_rule": "remove vertices whose degree is exactly 1 in the original graph, once",
        "recursive_core_rule": "repeatedly remove vertices with current degree below 2",
        "charikar_rule": "remove one current minimum-degree vertex at a time; ties use smallest unique-coordinate ID; retain stage maximizing m/n",
        "density_definition": "m/n; reported mean induced degree is 2m/n",
        "exact_flow_enabled": bool(args.exact_flow),
        "exact_flow": exact_qc,
        "duplicate_definition": "coordinate multiplicity minus one among the 20,000 ESM-2 embedding rows",
        "duplicate_tests": duplicate_tests,
        "collapse_qc": collapse_qc,
        "inputs": {
            "embeddings": file_signature(embedding_file),
            "original_rng": file_signature(adjacency_file),
            "metadata": file_signature(metadata_file),
        },
        "outputs": sorted(str(path) for path in args.out_root.rglob("*") if path.is_file()),
    }
    write_json(args.out_root / "degree_structure_manifest.json", manifest)
    print(stages.to_string(index=False))
    print()
    print(duplicate_summary.to_string(index=False))
    print()
    print(json.dumps(duplicate_tests, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", type=Path, default=DEFAULT_PANEL_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--upper-tail-nodes", type=int, default=4)
    parser.add_argument("--exact-flow", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
