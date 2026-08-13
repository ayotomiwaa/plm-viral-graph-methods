#!/usr/bin/env python3
"""Directional filtering and intrinsic distances for the seed-42 20k ESM-2 graphs.

The workflow is deliberately resumable:

1. validate-inputs
2. filter-graphs
3. prepare-distances
4. summarize

For each node, nonzero embedding displacement vectors to its graph neighbors are
compared by absolute cosine similarity.  The lowest ``ceil(candidate_fraction*k)``
neighbors form a fixed candidate list.  Candidates are tested sequentially;
after a provisional removal all remaining directional scores are recomputed.
The local sequence stops at its first failure to improve the mean score by at
least ``delta``.  Zero-displacement edges and nodes with at most two usable
directions are left untouched.

An undirected edge becomes a global deletion candidate only when both endpoints
reject it.  Candidates are processed in a deterministic strongest-evidence-first
order.  A reverse-delete/union-find pass gives the exact result of sequentially
deleting a candidate only when doing so does not increase the graph's component
count.  Original edge weights are retained on every surviving edge.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from numba import njit
from scipy.sparse import csr_matrix, load_npz, save_npz
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.stats import spearmanr


DEFAULT_PANEL_ROOT = Path(
    "analysis/cohort_validation/07_sampling_design_20k/random_full_dataset_seed42/seed_42"
)
DEFAULT_OUT_ROOT = Path(
    "analysis/cohort_validation/24_seed42_20k_directional_intrinsic_distances/"
    "random_full_dataset_seed42/seed_42"
)


@dataclass(frozen=True)
class GraphSpec:
    key: str
    source_dirname: str


GRAPH_SPECS = {
    "knn5": GraphSpec("knn5", "embedding_knn_k05"),
    "knn50": GraphSpec("knn50", "embedding_knn_k50"),
    "rng": GraphSpec("rng", "embedding_rng_exact"),
}


@dataclass(frozen=True)
class Paths:
    panel_root: Path
    out_root: Path
    candidate_fraction: float
    delta: float

    @property
    def embeddings(self) -> Path:
        return self.panel_root / "embeddings/esm2_650M/pool_n20000/embeddings.npy"

    @property
    def graph_root(self) -> Path:
        return self.panel_root / "graphs/esm2_650M/cityblock/pool_n20000"

    @property
    def parameter_tag(self) -> str:
        return f"candidate_{float_tag(self.candidate_fraction)}_delta_{float_tag(self.delta)}"

    def source_graph(self, key: str) -> Path:
        return self.graph_root / GRAPH_SPECS[key].source_dirname

    def refined_graph(self, key: str) -> Path:
        return self.out_root / "refined_graphs" / self.parameter_tag / key

    def work_dir(self, key: str) -> Path:
        return self.out_root / "work" / self.parameter_tag / key

    def distance_dir(self, variant: str) -> Path:
        return self.out_root / "distance_matrices" / self.parameter_tag / variant


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def float_tag(value: float) -> str:
    return f"{value:.8g}".replace("-", "m").replace(".", "p")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def stable_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_csv_choices(value: str, allowed: Iterable[str]) -> list[str]:
    allowed_set = set(allowed)
    if value.strip().lower() == "all":
        return list(allowed)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(selected).difference(allowed_set))
    if unknown:
        raise ValueError(f"Unknown selections {unknown}; allowed={sorted(allowed_set)}")
    if not selected:
        raise ValueError("At least one selection is required")
    return selected


def structural_adjacency(adj: csr_matrix) -> csr_matrix:
    structure = adj.copy()
    structure.data = np.ones(structure.data.size, dtype=np.int8)
    return structure


def graph_qc(graph_dir: Path, expected_n: int | None = None) -> dict[str, Any]:
    adj_path = graph_dir / "adj.npz"
    nodes_path = graph_dir / "nodes.csv"
    if not adj_path.exists() or not nodes_path.exists():
        raise FileNotFoundError(f"Expected adj.npz and nodes.csv in {graph_dir}")
    adj = load_npz(adj_path).tocsr()
    adj.sort_indices()
    if adj.shape[0] != adj.shape[1]:
        raise ValueError(f"{adj_path}: adjacency is not square: {adj.shape}")
    if expected_n is not None and adj.shape != (expected_n, expected_n):
        raise ValueError(f"{adj_path}: expected {(expected_n, expected_n)}, observed {adj.shape}")
    if not np.isfinite(adj.data).all() or np.any(adj.data < 0):
        raise ValueError(f"{adj_path}: weights must be finite and nonnegative")
    structural_diff = structural_adjacency(adj) - structural_adjacency(adj.T.tocsr())
    if structural_diff.nnz:
        raise ValueError(f"{adj_path}: adjacency structure is asymmetric")
    weight_diff = adj - adj.T.tocsr()
    max_weight_asymmetry = float(np.max(np.abs(weight_diff.data))) if weight_diff.nnz else 0.0
    if max_weight_asymmetry > 1e-6:
        raise ValueError(f"{adj_path}: adjacency weights are asymmetric by {max_weight_asymmetry}")
    n_components, labels = connected_components(adj, directed=False, return_labels=True)
    sizes = np.bincount(labels, minlength=n_components)
    degree = np.diff(adj.indptr)
    return {
        "adj_path": str(adj_path),
        "n_nodes": int(adj.shape[0]),
        "n_edges": int(adj.nnz // 2),
        "n_components": int(n_components),
        "component_sizes": sorted((int(value) for value in sizes), reverse=True),
        "giant_component_size": int(sizes.max()) if sizes.size else 0,
        "mean_degree": float(degree.mean()) if degree.size else 0.0,
        "median_degree": float(np.median(degree)) if degree.size else 0.0,
        "max_degree": int(degree.max()) if degree.size else 0,
        "stored_zero_weight_entries": int(np.count_nonzero(adj.data == 0)),
        "max_weight_asymmetry": max_weight_asymmetry,
    }


def load_nodes(graph_dir: Path, expected_n: int) -> pd.DataFrame:
    nodes = pd.read_csv(graph_dir / "nodes.csv")
    required = {"node_id", "accession", "embedding_row"}
    missing = sorted(required.difference(nodes.columns))
    if missing:
        raise ValueError(f"{graph_dir / 'nodes.csv'} missing columns {missing}")
    nodes = nodes.sort_values("node_id").reset_index(drop=True)
    node_ids = nodes["node_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(node_ids, np.arange(expected_n, dtype=np.int64)):
        raise ValueError(f"{graph_dir / 'nodes.csv'}: node_id is not row-aligned 0..n-1")
    embedding_rows = nodes["embedding_row"].to_numpy(dtype=np.int64)
    if np.unique(embedding_rows).size != expected_n:
        raise ValueError(f"{graph_dir / 'nodes.csv'}: embedding_row is not unique")
    nodes["accession"] = nodes["accession"].astype(str).str.strip()
    if nodes["accession"].duplicated().any():
        raise ValueError(f"{graph_dir / 'nodes.csv'}: accession is not unique")
    return nodes


def input_contract(paths: Paths, graph_keys: list[str]) -> dict[str, Any]:
    if not 0 < paths.candidate_fraction <= 1:
        raise ValueError("candidate fraction must be in (0, 1]")
    if paths.delta < 0:
        raise ValueError("delta must be nonnegative")
    embeddings = np.load(paths.embeddings, mmap_mode="r")
    if embeddings.ndim != 2 or embeddings.shape[1] != 1280:
        raise ValueError(f"Expected n x 1280 embeddings, observed {embeddings.shape}")
    if embeddings.dtype != np.float32:
        raise ValueError(f"Expected float32 embeddings, observed {embeddings.dtype}")
    n_nodes = int(embeddings.shape[0])
    graph_reports: dict[str, Any] = {}
    canonical_accessions: pd.Series | None = None
    canonical_embedding_rows: np.ndarray | None = None
    for key in graph_keys:
        graph_dir = paths.source_graph(key)
        report = graph_qc(graph_dir, expected_n=n_nodes)
        nodes = load_nodes(graph_dir, expected_n=n_nodes)
        embedding_rows = nodes["embedding_row"].to_numpy(dtype=np.int64)
        if embedding_rows.min() < 0 or embedding_rows.max() >= n_nodes:
            raise ValueError(f"{graph_dir}: embedding_row falls outside embeddings.npy")
        if canonical_accessions is None:
            canonical_accessions = nodes["accession"]
            canonical_embedding_rows = embedding_rows
        else:
            if not canonical_accessions.equals(nodes["accession"]):
                raise ValueError(f"{key}: accession ordering differs from the first graph")
            if not np.array_equal(canonical_embedding_rows, embedding_rows):
                raise ValueError(f"{key}: embedding_row ordering differs from the first graph")
        report["nodes_path"] = str(graph_dir / "nodes.csv")
        graph_reports[key] = report
    return {
        "panel_root": str(paths.panel_root),
        "embeddings": file_signature(paths.embeddings),
        "embedding_shape": [int(x) for x in embeddings.shape],
        "embedding_dtype": str(embeddings.dtype),
        "candidate_fraction": float(paths.candidate_fraction),
        "delta": float(paths.delta),
        "low_degree_policy": "retain all edges when usable nonzero-direction degree <= 2",
        "zero_displacement_policy": "retain edge and exclude it from directional scoring",
        "endpoint_rule": "AND: both endpoints must provisionally reject",
        "connectivity_rule": "final component count must equal original component count",
        "graphs": graph_reports,
    }


def command_validate_inputs(args: argparse.Namespace) -> None:
    graph_keys = parse_csv_choices(args.graphs, GRAPH_SPECS)
    paths = Paths(args.panel_root, args.out_root, args.candidate_fraction, args.delta)
    report = input_contract(paths, graph_keys)
    report["validated_at_unix"] = time.time()
    out = paths.out_root / "input_validation.json"
    write_json(out, report)
    log(f"Validated {report['embedding_shape'][0]:,} x {report['embedding_shape'][1]:,} embeddings")
    for key in graph_keys:
        graph = report["graphs"][key]
        log(
            f"{key}: edges={graph['n_edges']:,}, components={graph['n_components']}, "
            f"zero-weight undirected edges={graph['stored_zero_weight_entries'] // 2:,}"
        )
    log(f"Wrote {out}")


def directional_decisions_for_node(
    embeddings: np.ndarray,
    embedding_rows: np.ndarray,
    neighbors: np.ndarray,
    node_id: int,
    candidate_fraction: float,
    delta: float,
    norm_epsilon: float,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Return a node summary and sequentially evaluated local candidates."""
    neighbors = np.asarray(neighbors, dtype=np.int64)
    center = np.asarray(embeddings[int(embedding_rows[node_id])], dtype=np.float32)
    neighbor_vectors = np.asarray(embeddings[embedding_rows[neighbors]], dtype=np.float32) - center
    norms = np.linalg.norm(neighbor_vectors, axis=1)
    usable_mask = norms > norm_epsilon
    usable_neighbors = neighbors[usable_mask]
    usable_vectors = neighbor_vectors[usable_mask]
    usable_norms = norms[usable_mask]
    directional_degree = int(usable_neighbors.size)
    summary = {
        "node_id": int(node_id),
        "graph_degree": int(neighbors.size),
        "directional_degree": directional_degree,
        "zero_direction_neighbors": int(neighbors.size - directional_degree),
        "candidate_pool_size": 0,
        "candidates_evaluated": 0,
        "local_rejections": 0,
    }
    if directional_degree <= 2:
        return summary, []

    unit = usable_vectors / usable_norms[:, None]
    similarities = np.abs(unit @ unit.T).astype(np.float64, copy=False)
    np.fill_diagonal(similarities, 0.0)
    active = np.ones(directional_degree, dtype=bool)
    row_sums = similarities.sum(axis=1)
    initial_scores = row_sums / (directional_degree - 1)
    order = np.lexsort((usable_neighbors, initial_scores))
    candidate_count = int(math.ceil(candidate_fraction * directional_degree))
    candidate_count = min(candidate_count, directional_degree - 2)
    candidate_indices = order[:candidate_count]
    summary["candidate_pool_size"] = candidate_count
    evaluations: list[dict[str, Any]] = []

    for one_based_rank, candidate_index in enumerate(candidate_indices, start=1):
        if not active[candidate_index]:
            raise RuntimeError("candidate list unexpectedly contains an inactive direction")
        n_active = int(active.sum())
        current_scores = row_sums[active] / (n_active - 1)
        mean_before = float(current_scores.mean())
        f_before = float(row_sums[candidate_index] / (n_active - 1))
        remaining = active.copy()
        remaining[candidate_index] = False
        remaining_indices = np.flatnonzero(remaining)
        updated_sums = row_sums[remaining_indices] - similarities[remaining_indices, candidate_index]
        new_n = n_active - 1
        mean_after = float(np.mean(updated_sums / (new_n - 1)))
        if mean_before <= norm_epsilon:
            improvement = float("nan")
            accepted = False
        else:
            improvement = (mean_after - mean_before) / mean_before
            accepted = bool(improvement + 1e-12 >= delta)
        evaluations.append(
            {
                "node_id": int(node_id),
                "neighbor_id": int(usable_neighbors[candidate_index]),
                "candidate_rank": int(one_based_rank),
                "candidate_pool_size": int(candidate_count),
                "directional_degree": int(directional_degree),
                "rank_fraction": float(one_based_rank / directional_degree),
                "f_before": f_before,
                "mean_before": mean_before,
                "mean_after": mean_after,
                "relative_improvement": float(improvement),
                "accepted": accepted,
            }
        )
        if not accepted:
            break
        row_sums[remaining_indices] = updated_sums
        active[candidate_index] = False

    summary["candidates_evaluated"] = len(evaluations)
    summary["local_rejections"] = sum(int(item["accepted"]) for item in evaluations)
    return summary, evaluations


SUMMARY_FIELDS = [
    "node_id",
    "graph_degree",
    "directional_degree",
    "zero_direction_neighbors",
    "candidate_pool_size",
    "candidates_evaluated",
    "local_rejections",
]
EVALUATION_FIELDS = [
    "node_id",
    "neighbor_id",
    "candidate_rank",
    "candidate_pool_size",
    "directional_degree",
    "rank_fraction",
    "f_before",
    "mean_before",
    "mean_after",
    "relative_improvement",
    "accepted",
]


def save_chunk(
    path: Path,
    fingerprint: str,
    summaries: list[dict[str, int]],
    evaluations: list[dict[str, Any]],
) -> None:
    payload: dict[str, np.ndarray] = {"fingerprint": np.asarray(fingerprint)}
    for field in SUMMARY_FIELDS:
        payload[f"summary_{field}"] = np.asarray([row[field] for row in summaries], dtype=np.int64)
    integer_evaluation_fields = {
        "node_id",
        "neighbor_id",
        "candidate_rank",
        "candidate_pool_size",
        "directional_degree",
    }
    for field in EVALUATION_FIELDS:
        values = [row[field] for row in evaluations]
        if field in integer_evaluation_fields:
            dtype = np.int64
        elif field == "accepted":
            dtype = bool
        else:
            dtype = np.float64
        payload[f"evaluation_{field}"] = np.asarray(values, dtype=dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def load_chunks(chunk_paths: list[Path], fingerprint: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_parts: dict[str, list[np.ndarray]] = {field: [] for field in SUMMARY_FIELDS}
    evaluation_parts: dict[str, list[np.ndarray]] = {field: [] for field in EVALUATION_FIELDS}
    for path in chunk_paths:
        with np.load(path, allow_pickle=False) as chunk:
            if str(chunk["fingerprint"].item()) != fingerprint:
                raise ValueError(f"{path}: fingerprint differs from this run configuration")
            for field in SUMMARY_FIELDS:
                summary_parts[field].append(chunk[f"summary_{field}"])
            for field in EVALUATION_FIELDS:
                evaluation_parts[field].append(chunk[f"evaluation_{field}"])
    summaries = pd.DataFrame(
        {field: np.concatenate(parts) if parts else np.array([]) for field, parts in summary_parts.items()}
    )
    evaluations = pd.DataFrame(
        {field: np.concatenate(parts) if parts else np.array([]) for field, parts in evaluation_parts.items()}
    )
    return summaries, evaluations


def mutual_rejection_queue(evaluations: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "source",
        "target",
        "source_f_before",
        "target_f_before",
        "source_rank_fraction",
        "target_rank_fraction",
        "source_improvement",
        "target_improvement",
        "max_f_before",
        "max_rank_fraction",
    ]
    if evaluations.empty:
        return pd.DataFrame(columns=columns)
    accepted = evaluations[evaluations["accepted"].astype(bool)].copy()
    if accepted.empty:
        return pd.DataFrame(columns=columns)
    accepted["source"] = np.minimum(accepted["node_id"], accepted["neighbor_id"]).astype(np.int64)
    accepted["target"] = np.maximum(accepted["node_id"], accepted["neighbor_id"]).astype(np.int64)
    source_side = accepted[accepted["node_id"] == accepted["source"]][
        ["source", "target", "f_before", "rank_fraction", "relative_improvement"]
    ].rename(
        columns={
            "f_before": "source_f_before",
            "rank_fraction": "source_rank_fraction",
            "relative_improvement": "source_improvement",
        }
    )
    target_side = accepted[accepted["node_id"] == accepted["target"]][
        ["source", "target", "f_before", "rank_fraction", "relative_improvement"]
    ].rename(
        columns={
            "f_before": "target_f_before",
            "rank_fraction": "target_rank_fraction",
            "relative_improvement": "target_improvement",
        }
    )
    mutual = source_side.merge(target_side, on=["source", "target"], how="inner", validate="one_to_one")
    mutual["max_f_before"] = mutual[["source_f_before", "target_f_before"]].max(axis=1)
    mutual["max_rank_fraction"] = mutual[["source_rank_fraction", "target_rank_fraction"]].max(axis=1)
    return mutual.sort_values(
        ["max_f_before", "max_rank_fraction", "source", "target"], kind="stable"
    ).reset_index(drop=True)


@njit(cache=False)
def connectivity_safe_delete_mask(
    n_nodes: int,
    sources: np.ndarray,
    targets: np.ndarray,
    queue_edge_indices: np.ndarray,
) -> np.ndarray:
    """Exact reverse-delete result for the supplied forward deletion queue."""
    n_edges = sources.size
    is_candidate = np.zeros(n_edges, dtype=np.uint8)
    for edge_index in queue_edge_indices:
        is_candidate[edge_index] = 1
    parent = np.arange(n_nodes, dtype=np.int64)
    rank = np.zeros(n_nodes, dtype=np.int8)

    def find(value: int) -> int:
        root = value
        while parent[root] != root:
            root = parent[root]
        while parent[value] != value:
            next_value = parent[value]
            parent[value] = root
            value = next_value
        return root

    def union(left: int, right: int) -> bool:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return False
        if rank[root_left] < rank[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        if rank[root_left] == rank[root_right]:
            rank[root_left] += 1
        return True

    for edge_index in range(n_edges):
        if is_candidate[edge_index] == 0:
            union(int(sources[edge_index]), int(targets[edge_index]))
    delete_mask = np.zeros(n_edges, dtype=np.bool_)
    for position in range(queue_edge_indices.size - 1, -1, -1):
        edge_index = int(queue_edge_indices[position])
        if union(int(sources[edge_index]), int(targets[edge_index])):
            delete_mask[edge_index] = False
        else:
            delete_mask[edge_index] = True
    return delete_mask


def upper_edges(adj: csr_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coo = adj.tocoo(copy=False)
    keep = coo.row < coo.col
    sources = coo.row[keep].astype(np.int64, copy=False)
    targets = coo.col[keep].astype(np.int64, copy=False)
    weights = coo.data[keep].astype(np.float32, copy=False)
    order = np.lexsort((targets, sources))
    return sources[order], targets[order], weights[order]


def edges_to_csr(
    n_nodes: int, sources: np.ndarray, targets: np.ndarray, weights: np.ndarray
) -> csr_matrix:
    rows = np.concatenate([sources, targets]).astype(np.int64, copy=False)
    cols = np.concatenate([targets, sources]).astype(np.int64, copy=False)
    data = np.concatenate([weights, weights]).astype(np.float32, copy=False)
    adjacency = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float32)
    adjacency.sort_indices()
    return adjacency


def write_dataframe_gzip(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)
    temporary.replace(path)


def filter_config(paths: Paths, key: str, norm_epsilon: float) -> dict[str, Any]:
    graph_dir = paths.source_graph(key)
    return {
        "algorithm_version": 1,
        "graph_key": key,
        "candidate_fraction": float(paths.candidate_fraction),
        "candidate_rounding": "ceil",
        "delta": float(paths.delta),
        "stop_at_first_failure": True,
        "recompute_scores_after_accepted_local_removal": True,
        "minimum_directional_degree": 3,
        "norm_epsilon": float(norm_epsilon),
        "zero_displacement_policy": "retain_and_exclude_from_directional_scoring",
        "endpoint_rule": "both_endpoints_AND",
        "global_queue_order": ["max_f_before_ascending", "max_rank_fraction_ascending", "source", "target"],
        "connectivity_rule": "reverse_delete_preserving_original_component_count",
        "embeddings": file_signature(paths.embeddings),
        "adjacency": file_signature(graph_dir / "adj.npz"),
        "nodes": file_signature(graph_dir / "nodes.csv"),
    }


def filter_one_graph(args: argparse.Namespace, paths: Paths, key: str) -> None:
    source_dir = paths.source_graph(key)
    output_dir = paths.refined_graph(key)
    work_dir = paths.work_dir(key)
    config = filter_config(paths, key, args.norm_epsilon)
    fingerprint = stable_fingerprint(config)
    completed_stats = output_dir / "stats.json"
    if completed_stats.exists():
        prior = json.loads(completed_stats.read_text(encoding="utf-8"))
        if prior.get("workflow_fingerprint") != fingerprint:
            raise ValueError(f"{output_dir}: completed output has a different workflow fingerprint")
        log(f"{key}: refined graph already complete; skipping")
        return

    embeddings = np.load(paths.embeddings, mmap_mode="r")
    n_nodes = int(embeddings.shape[0])
    nodes = load_nodes(source_dir, expected_n=n_nodes)
    embedding_rows = nodes["embedding_row"].to_numpy(dtype=np.int64)
    adj = load_npz(source_dir / "adj.npz").tocsr()
    adj.sort_indices()
    chunk_dir = work_dir / "local_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    write_json(work_dir / "filter_config.json", {**config, "workflow_fingerprint": fingerprint})

    expected_chunks: list[Path] = []
    for start in range(0, n_nodes, args.node_chunk_size):
        end = min(start + args.node_chunk_size, n_nodes)
        chunk_path = chunk_dir / f"nodes_{start:06d}_{end - 1:06d}.npz"
        expected_chunks.append(chunk_path)
        if chunk_path.exists():
            with np.load(chunk_path, allow_pickle=False) as chunk:
                if str(chunk["fingerprint"].item()) != fingerprint:
                    raise ValueError(f"{chunk_path}: stale chunk fingerprint")
            continue
        summaries: list[dict[str, int]] = []
        evaluations: list[dict[str, Any]] = []
        for node_id in range(start, end):
            neighbors = adj.indices[adj.indptr[node_id] : adj.indptr[node_id + 1]]
            summary, node_evaluations = directional_decisions_for_node(
                embeddings=embeddings,
                embedding_rows=embedding_rows,
                neighbors=neighbors,
                node_id=node_id,
                candidate_fraction=paths.candidate_fraction,
                delta=paths.delta,
                norm_epsilon=args.norm_epsilon,
            )
            summaries.append(summary)
            evaluations.extend(node_evaluations)
        save_chunk(chunk_path, fingerprint, summaries, evaluations)
        log(f"{key}: directional decisions checkpointed for nodes {start:,}-{end - 1:,}")

    summaries, evaluations = load_chunks(expected_chunks, fingerprint)
    if summaries.shape[0] != n_nodes:
        raise ValueError(f"{key}: expected {n_nodes:,} node summaries, observed {summaries.shape[0]:,}")
    write_dataframe_gzip(work_dir / "node_directional_summary.csv.gz", summaries)
    write_dataframe_gzip(work_dir / "local_candidate_evaluations.csv.gz", evaluations)
    mutual = mutual_rejection_queue(evaluations)

    sources, targets, weights = upper_edges(adj)
    edge_keys = sources * np.int64(n_nodes) + targets
    mutual_keys = (
        mutual["source"].to_numpy(dtype=np.int64) * np.int64(n_nodes)
        + mutual["target"].to_numpy(dtype=np.int64)
    )
    queue_edge_indices = np.searchsorted(edge_keys, mutual_keys).astype(np.int64)
    if mutual_keys.size:
        if np.any(queue_edge_indices >= edge_keys.size) or not np.array_equal(
            edge_keys[queue_edge_indices], mutual_keys
        ):
            raise ValueError(f"{key}: a mutually rejected edge is absent from the source graph")
    delete_mask = connectivity_safe_delete_mask(n_nodes, sources, targets, queue_edge_indices)
    queue_deleted = delete_mask[queue_edge_indices] if queue_edge_indices.size else np.array([], dtype=bool)
    mutual = mutual.copy()
    mutual["queue_position"] = np.arange(mutual.shape[0], dtype=np.int64)
    mutual["global_decision"] = np.where(queue_deleted, "removed", "retained_for_connectivity")
    write_dataframe_gzip(work_dir / "mutual_rejection_queue.csv.gz", mutual)

    keep_mask = ~delete_mask
    refined_sources = sources[keep_mask]
    refined_targets = targets[keep_mask]
    refined_weights = weights[keep_mask]
    refined_adj = edges_to_csr(n_nodes, refined_sources, refined_targets, refined_weights)
    before_components, before_labels = connected_components(adj, directed=False, return_labels=True)
    after_components, after_labels = connected_components(refined_adj, directed=False, return_labels=True)
    if int(before_components) != int(after_components):
        raise RuntimeError(
            f"{key}: connectivity safeguard failed: {before_components} -> {after_components} components"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_adj = output_dir / f".adj.tmp.{os.getpid()}.npz"
    save_npz(temporary_adj, refined_adj)
    temporary_adj.replace(output_dir / "adj.npz")
    edge_frame = pd.DataFrame(
        {
            "source": refined_sources,
            "target": refined_targets,
            "weight": refined_weights,
            "status": "kept",
        }
    )
    write_dataframe_gzip(output_dir / "edges.csv.gz", edge_frame)
    degree = np.diff(refined_adj.indptr)
    nodes_out = nodes.copy()
    nodes_out["component_id"] = after_labels
    nodes_out["degree"] = degree
    write_dataframe_gzip(output_dir / "nodes.csv.gz", nodes_out)

    before_sizes = np.bincount(before_labels, minlength=before_components)
    after_sizes = np.bincount(after_labels, minlength=after_components)
    stats = {
        **config,
        "workflow_fingerprint": fingerprint,
        "source_graph_dir": str(source_dir),
        "output_graph_dir": str(output_dir),
        "n_nodes": n_nodes,
        "n_edges_before": int(sources.size),
        "n_edges_after": int(refined_sources.size),
        "n_edges_removed": int(delete_mask.sum()),
        "n_components_before": int(before_components),
        "n_components_after": int(after_components),
        "component_sizes_before": sorted((int(x) for x in before_sizes), reverse=True),
        "component_sizes_after": sorted((int(x) for x in after_sizes), reverse=True),
        "local_candidate_evaluations": int(evaluations.shape[0]),
        "local_endpoint_rejections": int(evaluations["accepted"].sum()) if not evaluations.empty else 0,
        "mutually_rejected_edges": int(mutual.shape[0]),
        "mutually_rejected_edges_removed": int(queue_deleted.sum()),
        "mutually_rejected_edges_retained_for_connectivity": int((~queue_deleted).sum()),
        "nodes_with_directional_degree_le_2": int((summaries["directional_degree"] <= 2).sum()),
        "zero_direction_neighbor_incidents": int(summaries["zero_direction_neighbors"].sum()),
        "zero_weight_edges_before": int(np.count_nonzero(weights == 0)),
        "zero_weight_edges_after": int(np.count_nonzero(refined_weights == 0)),
        "mean_degree_after": float(degree.mean()),
        "median_degree_after": float(np.median(degree)),
        "max_degree_after": int(degree.max()),
        "completed_at_unix": time.time(),
    }
    write_json(completed_stats, stats)
    log(
        f"{key}: removed {stats['n_edges_removed']:,}/{stats['n_edges_before']:,} edges; "
        f"components preserved at {before_components}"
    )


def command_filter_graphs(args: argparse.Namespace) -> None:
    graph_keys = parse_csv_choices(args.graphs, GRAPH_SPECS)
    paths = Paths(args.panel_root, args.out_root, args.candidate_fraction, args.delta)
    input_contract(paths, graph_keys)
    for key in graph_keys:
        filter_one_graph(args, paths, key)


def distance_paths(paths: Paths, key: str, variant: str) -> tuple[Path, Path]:
    distance_dir = paths.distance_dir(variant)
    matrix = distance_dir / f"{key}_weighted_shortest_path_float32.npy"
    checkpoint = distance_dir / f"{key}_weighted_shortest_path_checkpoint.json"
    return matrix, checkpoint


def prepare_one_distance_matrix(
    args: argparse.Namespace,
    paths: Paths,
    key: str,
    variant: str,
) -> None:
    graph_dir = paths.source_graph(key) if variant == "baseline" else paths.refined_graph(key)
    adj_path = graph_dir / "adj.npz"
    if not adj_path.exists():
        raise FileNotFoundError(f"Missing {variant} graph for {key}: {adj_path}")
    adj = load_npz(adj_path).tocsr()
    adj.sort_indices()
    n_nodes = int(adj.shape[0])
    n_components, labels = connected_components(adj, directed=False, return_labels=True)
    graph_signature = file_signature(adj_path)
    matrix_path, checkpoint_path = distance_paths(paths, key, variant)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_existed = matrix_path.exists()
    if matrix_existed:
        rows = np.load(matrix_path, mmap_mode="r+")
        if rows.shape != (n_nodes, n_nodes) or rows.dtype != np.float32:
            raise ValueError(f"{matrix_path}: incompatible matrix; use a new output root")
    else:
        rows = np.lib.format.open_memmap(
            matrix_path, mode="w+", dtype=np.float32, shape=(n_nodes, n_nodes)
        )
    completed_rows = 0
    if checkpoint_path.exists() and matrix_existed:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("graph_signature") != graph_signature:
            raise ValueError(f"{checkpoint_path}: graph changed; use a new output root")
        completed_rows = int(checkpoint.get("completed_rows", 0))
    elif checkpoint_path.exists():
        raise ValueError(f"{checkpoint_path} exists but its matrix is absent")
    log(
        f"{variant}/{key}: {completed_rows:,}/{n_nodes:,} shortest-path rows already complete; "
        f"components={n_components}"
    )
    for start in range(completed_rows, n_nodes, args.batch_size):
        end = min(start + args.batch_size, n_nodes)
        source_ids = np.arange(start, end, dtype=np.int64)
        distances = dijkstra(adj, directed=False, indices=source_ids, unweighted=False)
        distances = np.asarray(distances, dtype=np.float64)
        if distances.ndim == 1:
            distances = distances.reshape(1, -1)
        expected_finite = labels[None, :] == labels[source_ids, None]
        observed_finite = np.isfinite(distances)
        if not np.array_equal(expected_finite, observed_finite):
            bad = int(np.count_nonzero(expected_finite != observed_finite))
            raise ValueError(f"{variant}/{key}: {bad:,} distances disagree with component membership")
        if np.any(distances[observed_finite] < 0):
            raise ValueError(f"{variant}/{key}: Dijkstra produced negative distances")
        distances[np.arange(source_ids.size), source_ids] = 0.0
        rows[start:end, :] = distances.astype(np.float32)
        rows.flush()
        write_json(
            checkpoint_path,
            {
                "graph_key": key,
                "variant": variant,
                "graph_dir": str(graph_dir),
                "graph_signature": graph_signature,
                "matrix_path": str(matrix_path),
                "shape": [n_nodes, n_nodes],
                "dtype": "float32",
                "n_components": int(n_components),
                "distance_definition": "weighted shortest path using retained original cityblock edge weights",
                "completed_rows": int(end),
                "complete": bool(end == n_nodes),
                "updated_at_unix": time.time(),
            },
        )
        log(f"{variant}/{key}: completed shortest-path rows {start:,}-{end - 1:,}")


def command_prepare_distances(args: argparse.Namespace) -> None:
    graph_keys = parse_csv_choices(args.graphs, GRAPH_SPECS)
    variants = parse_csv_choices(args.variants, ["baseline", "refined"])
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    paths = Paths(args.panel_root, args.out_root, args.candidate_fraction, args.delta)
    for variant in variants:
        for key in graph_keys:
            prepare_one_distance_matrix(args, paths, key, variant)


def command_summarize(args: argparse.Namespace) -> None:
    graph_keys = parse_csv_choices(args.graphs, GRAPH_SPECS)
    paths = Paths(args.panel_root, args.out_root, args.candidate_fraction, args.delta)
    graph_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.sample_seed)
    for key in graph_keys:
        source_qc = graph_qc(paths.source_graph(key))
        refined_stats_path = paths.refined_graph(key) / "stats.json"
        if not refined_stats_path.exists():
            raise FileNotFoundError(refined_stats_path)
        refined = json.loads(refined_stats_path.read_text(encoding="utf-8"))
        graph_rows.append(
            {
                "graph": key,
                "n_nodes": source_qc["n_nodes"],
                "edges_before": source_qc["n_edges"],
                "edges_after": refined["n_edges_after"],
                "edges_removed": refined["n_edges_removed"],
                "edge_removal_fraction": refined["n_edges_removed"] / source_qc["n_edges"],
                "components_before": source_qc["n_components"],
                "components_after": refined["n_components_after"],
                "local_endpoint_rejections": refined["local_endpoint_rejections"],
                "mutually_rejected_edges": refined["mutually_rejected_edges"],
                "mutual_edges_retained_for_connectivity": refined[
                    "mutually_rejected_edges_retained_for_connectivity"
                ],
            }
        )
        baseline_path, baseline_checkpoint = distance_paths(paths, key, "baseline")
        refined_path, refined_checkpoint = distance_paths(paths, key, "refined")
        if not all(path.exists() for path in [baseline_path, baseline_checkpoint, refined_path, refined_checkpoint]):
            continue
        baseline_status = json.loads(baseline_checkpoint.read_text(encoding="utf-8"))
        refined_status = json.loads(refined_checkpoint.read_text(encoding="utf-8"))
        if not baseline_status.get("complete") or not refined_status.get("complete"):
            continue
        baseline = np.load(baseline_path, mmap_mode="r")
        filtered = np.load(refined_path, mmap_mode="r")
        n_nodes = baseline.shape[0]
        pair_i = rng.integers(0, n_nodes, size=args.sample_pairs, dtype=np.int64)
        pair_j = rng.integers(0, n_nodes, size=args.sample_pairs, dtype=np.int64)
        unequal = pair_i != pair_j
        pair_i, pair_j = pair_i[unequal], pair_j[unequal]
        before = np.asarray(baseline[pair_i, pair_j], dtype=np.float64)
        after = np.asarray(filtered[pair_i, pair_j], dtype=np.float64)
        finite = np.isfinite(before) & np.isfinite(after)
        before_finite = before[finite]
        after_finite = after[finite]
        if before_finite.size == 0:
            raise ValueError(f"{key}: sampled no finite baseline/refined pairs")
        tolerance = np.maximum(1e-5, np.abs(before_finite) * 1e-6)
        monotonic_violations = int(np.count_nonzero(after_finite + tolerance < before_finite))
        if monotonic_violations:
            raise ValueError(f"{key}: {monotonic_violations} sampled refined distances decreased")
        changed = after_finite > before_finite + tolerance
        positive = before_finite > 0
        ratios = after_finite[positive] / before_finite[positive]
        comparison_rows.append(
            {
                "graph": key,
                "sampled_pairs": int(pair_i.size),
                "finite_in_both": int(finite.sum()),
                "finite_fraction": float(finite.mean()),
                "changed_distance_fraction": float(changed.mean()),
                "mean_additive_increase": float(np.mean(after_finite - before_finite)),
                "median_additive_increase": float(np.median(after_finite - before_finite)),
                "mean_ratio_positive_baseline": float(np.mean(ratios)) if ratios.size else np.nan,
                "median_ratio_positive_baseline": float(np.median(ratios)) if ratios.size else np.nan,
                "spearman_baseline_vs_refined": float(spearmanr(before_finite, after_finite).statistic),
                "sampled_monotonicity_violations": monotonic_violations,
            }
        )
    summary_dir = paths.out_root / "summaries" / paths.parameter_tag
    summary_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(graph_rows).to_csv(summary_dir / "graph_filtering_summary.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(
        summary_dir / "baseline_vs_refined_shortest_path_sample_summary.csv", index=False
    )
    write_json(
        summary_dir / "summary_manifest.json",
        {
            "candidate_fraction": float(paths.candidate_fraction),
            "delta": float(paths.delta),
            "sample_pairs_requested": int(args.sample_pairs),
            "sample_seed": int(args.sample_seed),
            "graphs": graph_keys,
            "distance_comparisons_available": [row["graph"] for row in comparison_rows],
            "completed_at_unix": time.time(),
        },
    )
    log(f"Wrote summaries under {summary_dir}")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--panel-root", type=Path, default=DEFAULT_PANEL_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--graphs", default="all", help="all or comma-separated: knn5,knn50,rng")
    parser.add_argument("--candidate-fraction", type=float, default=0.10)
    parser.add_argument("--delta", type=float, default=0.01)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-inputs", help="Validate embeddings, nodes, and source graphs")
    add_common_arguments(validate)
    validate.set_defaults(func=command_validate_inputs)

    filtering = subparsers.add_parser("filter-graphs", help="Build connectivity-preserving refined graphs")
    add_common_arguments(filtering)
    filtering.add_argument("--node-chunk-size", type=int, default=64)
    filtering.add_argument("--norm-epsilon", type=float, default=1e-12)
    filtering.set_defaults(func=command_filter_graphs)

    distances = subparsers.add_parser("prepare-distances", help="Build resumable all-pairs shortest-path matrices")
    add_common_arguments(distances)
    distances.add_argument("--variants", default="baseline,refined", help="baseline, refined, or both")
    distances.add_argument("--batch-size", type=int, default=16)
    distances.set_defaults(func=command_prepare_distances)

    summarize = subparsers.add_parser("summarize", help="Summarize graph filtering and completed matrices")
    add_common_arguments(summarize)
    summarize.add_argument("--sample-pairs", type=int, default=200_000)
    summarize.add_argument("--sample-seed", type=int, default=42)
    summarize.set_defaults(func=command_summarize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
