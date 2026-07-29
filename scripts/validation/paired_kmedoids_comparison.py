#!/usr/bin/env python3
"""Paired k-medoids comparison for the seed-42 20k cohort-validation panel.

This script is intentionally written as a resumable command-line workflow:

1. validate-inputs
2. prepare-design
3. prepare-rng-distances
4. run
5. summarize

The same repeat-level initial medoids are reused across all metrics. Raw Hamming
uses a profile/frequency medoid update over the cluster members. Raw embedding
uses an exact dense-matrix medoid update over the cluster members. RNG metrics
use weighted graph shortest-path distances. By default their medoid search space
is all nodes; an explicitly smaller candidate pool is an approximate smoke or
sensitivity mode and is labeled as such in every result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import load_npz
from scipy.sparse.csgraph import connected_components, dijkstra
from sklearn.metrics import adjusted_rand_score


DEFAULT_PANEL_ROOT = Path(
    "analysis/cohort_validation/07_sampling_design_20k/random_full_dataset_seed42/seed_42"
)
DEFAULT_OUT_ROOT = Path(
    "analysis/cohort_validation/16_seed42_20k_kmedoids/random_full_dataset_seed42/seed_42"
)

METRIC_DISPLAY = {
    "hamming": "raw Hamming dense matrix, profile medoid update",
    "embedding": "raw ESM-2 cityblock dense matrix, exact medoid update",
    "hamming_rng": "Hamming RNG weighted shortest-path k-medoids",
    "embedding_rng": "ESM-2 cityblock RNG weighted shortest-path k-medoids",
}


@dataclass(frozen=True)
class Paths:
    panel_root: Path
    out_root: Path

    @property
    def metadata_csv(self) -> Path:
        return self.panel_root / "inputs/pool_n20000/metadata.csv"

    @property
    def aligned_fasta(self) -> Path:
        return self.panel_root / "inputs/pool_n20000/spike_sequences_aligned_mafft.fasta"

    @property
    def hamming_nodes(self) -> Path:
        return self.panel_root / "graphs/hamming/pool_n20000/hamming_rng_exact/nodes.csv"

    @property
    def embedding_nodes(self) -> Path:
        return self.panel_root / "graphs/esm2_650M/cityblock/pool_n20000/embedding_rng_exact/nodes.csv"

    @property
    def hamming_matrix(self) -> Path:
        return (
            self.panel_root
            / "graphs/hamming/pool_n20000/distance_matrices/"
            / "hamming_count-gap-state_all_states_uint16.npy"
        )

    @property
    def embedding_matrix(self) -> Path:
        return (
            self.panel_root
            / "graphs/esm2_650M/cityblock/pool_n20000/distance_matrices/"
            / "embedding_cityblock_float32.npy"
        )

    @property
    def hamming_rng_graph(self) -> Path:
        return self.panel_root / "graphs/hamming/pool_n20000/hamming_rng_exact"

    @property
    def embedding_rng_graph(self) -> Path:
        return self.panel_root / "graphs/esm2_650M/cityblock/pool_n20000/embedding_rng_exact"

    @property
    def design_dir(self) -> Path:
        return self.out_root / "design"

    @property
    def rng_rows_dir(self) -> Path:
        return self.out_root / "distance_rows"

    @property
    def runs_dir(self) -> Path:
        return self.out_root / "runs"

    @property
    def initial_medoids_csv(self) -> Path:
        return self.design_dir / "initial_medoids.csv"

    @property
    def candidate_pool_csv(self) -> Path:
        return self.design_dir / "candidate_pool.csv"

    @property
    def design_manifest_json(self) -> Path:
        return self.design_dir / "manifest.json"


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ordered_int_fingerprint(values: np.ndarray) -> str:
    """Return a stable fingerprint for an ordered integer design vector."""
    arr = np.asarray(values, dtype="<i8")
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def candidate_fingerprint(candidates: pd.DataFrame) -> str:
    candidate_rows = candidates["candidate_row"].to_numpy(dtype=np.int64)
    expected_rows = np.arange(candidates.shape[0], dtype=np.int64)
    if not np.array_equal(candidate_rows, expected_rows):
        raise ValueError("candidate_row must be contiguous and row-aligned from 0..n_candidates-1")
    node_ids = candidates["node_id"].to_numpy(dtype=np.int64)
    if np.unique(node_ids).size != node_ids.size:
        raise ValueError("candidate pool contains duplicate node_id values")
    return ordered_int_fingerprint(node_ids)


def graph_signature(graph_dir: Path) -> dict[str, Any]:
    adj_path = (graph_dir / "adj.npz").resolve()
    stat = adj_path.stat()
    return {
        "adj_path": str(adj_path),
        "adj_size_bytes": int(stat.st_size),
        "adj_mtime_ns": int(stat.st_mtime_ns),
    }


def metric_paths(paths: Paths) -> dict[str, dict[str, Path]]:
    return {
        "hamming": {"matrix": paths.hamming_matrix, "nodes": paths.hamming_nodes},
        "embedding": {"matrix": paths.embedding_matrix, "nodes": paths.embedding_nodes},
        "hamming_rng": {"graph": paths.hamming_rng_graph, "nodes": paths.hamming_nodes},
        "embedding_rng": {"graph": paths.embedding_rng_graph, "nodes": paths.embedding_nodes},
    }


def read_nodes(nodes_path: Path, label_col: str) -> pd.DataFrame:
    nodes = pd.read_csv(nodes_path)
    required = {"node_id", "accession", label_col}
    missing = sorted(required.difference(nodes.columns))
    if missing:
        raise ValueError(f"{nodes_path} missing required columns: {missing}")
    nodes = nodes.sort_values("node_id").reset_index(drop=True)
    node_ids = nodes["node_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(node_ids, np.arange(len(nodes), dtype=np.int64)):
        raise ValueError(f"{nodes_path}: node_id must be contiguous and row-aligned from 0..n-1")
    nodes["accession"] = nodes["accession"].astype(str).str.strip()
    nodes[label_col] = nodes[label_col].astype(str).str.strip()
    if nodes["accession"].duplicated().any():
        raise ValueError(f"{nodes_path}: accession values must be unique")
    if nodes["accession"].isin(["", "nan", "None"]).any():
        raise ValueError(f"{nodes_path}: accession contains missing/blank values")
    if nodes[label_col].isin(["", "nan", "None"]).any():
        raise ValueError(f"{nodes_path}: {label_col} contains missing/blank values")
    return nodes


def load_canonical_nodes(paths: Paths, label_col: str) -> pd.DataFrame:
    h_nodes = read_nodes(paths.hamming_nodes, label_col)
    e_nodes = read_nodes(paths.embedding_nodes, label_col)
    if h_nodes.shape[0] != e_nodes.shape[0]:
        raise ValueError(f"node table row count mismatch: hamming={h_nodes.shape[0]}, embedding={e_nodes.shape[0]}")
    if not h_nodes["accession"].equals(e_nodes["accession"]):
        mismatch = np.flatnonzero(h_nodes["accession"].to_numpy() != e_nodes["accession"].to_numpy())
        raise ValueError(f"hamming/embedding node accession order mismatch; first mismatch rows={mismatch[:5].tolist()}")
    if not h_nodes[label_col].equals(e_nodes[label_col]):
        mismatch = np.flatnonzero(h_nodes[label_col].to_numpy() != e_nodes[label_col].to_numpy())
        raise ValueError(f"hamming/embedding {label_col} order mismatch; first mismatch rows={mismatch[:5].tolist()}")
    return h_nodes


def validate_metadata_alignment(metadata_path: Path, nodes: pd.DataFrame, label_col: str) -> dict[str, Any]:
    metadata = pd.read_csv(metadata_path, usecols=["accession", label_col])
    metadata["accession"] = metadata["accession"].astype(str).str.strip()
    metadata[label_col] = metadata[label_col].astype(str).str.strip()
    if metadata.shape[0] != nodes.shape[0]:
        raise ValueError(f"metadata/node row count mismatch: {metadata.shape[0]} vs {nodes.shape[0]}")
    if metadata["accession"].duplicated().any():
        raise ValueError(f"{metadata_path}: accession values must be unique")
    if not metadata["accession"].equals(nodes["accession"]):
        raise ValueError("metadata and graph nodes have different accession order")
    if not metadata[label_col].equals(nodes[label_col]):
        raise ValueError(f"metadata and graph nodes have different {label_col} order")
    return {
        "path": str(metadata_path),
        "n_rows": int(metadata.shape[0]),
        "accession_order_matches_nodes": True,
        f"{label_col}_order_matches_nodes": True,
    }


def validate_square_matrix(path: Path, expected_n: int) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.load(path, mmap_mode="r")
    if arr.shape != (expected_n, expected_n):
        raise ValueError(f"{path}: expected {(expected_n, expected_n)}, observed {arr.shape}")
    diagonal = np.asarray(arr[np.arange(expected_n), np.arange(expected_n)])
    rng = np.random.default_rng(20260728)
    sample_i = rng.integers(0, expected_n, size=min(20_000, expected_n * 2))
    sample_j = rng.integers(0, expected_n, size=sample_i.size)
    sample_j[sample_j == sample_i] = (sample_j[sample_j == sample_i] + 1) % expected_n
    forward = np.asarray(arr[sample_i, sample_j], dtype=np.float64)
    reverse = np.asarray(arr[sample_j, sample_i], dtype=np.float64)
    if not np.isfinite(forward).all():
        raise ValueError(f"{path}: sampled off-diagonal distances contain non-finite values")
    if (forward < 0).any():
        raise ValueError(f"{path}: sampled off-diagonal distances contain negative values")
    if not np.array_equal(forward, reverse):
        max_diff = float(np.max(np.abs(forward - reverse)))
        raise ValueError(f"{path}: sampled distances are asymmetric; max difference={max_diff}")
    finite_diagonal = diagonal[np.isfinite(diagonal)]
    return {
        "path": str(path),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "diagonal_finite_count": int(finite_diagonal.size),
        "diagonal_nonfinite_count": int(diagonal.size - finite_diagonal.size),
        "diagonal_finite_min": float(finite_diagonal.min()) if finite_diagonal.size else None,
        "diagonal_finite_max": float(finite_diagonal.max()) if finite_diagonal.size else None,
        "sampled_off_diagonal_pairs": int(forward.size),
        "sampled_off_diagonal_finite_nonnegative_symmetric": True,
    }


def validate_graph(graph_dir: Path, expected_n: int) -> dict[str, Any]:
    nodes_path = graph_dir / "nodes.csv"
    adj_path = graph_dir / "adj.npz"
    stats_path = graph_dir / "stats.json"
    if not nodes_path.exists():
        raise FileNotFoundError(nodes_path)
    if not adj_path.exists():
        raise FileNotFoundError(adj_path)
    adj = load_npz(adj_path).tocsr()
    if adj.shape != (expected_n, expected_n):
        raise ValueError(f"{adj_path}: expected {(expected_n, expected_n)}, observed {adj.shape}")
    if not np.isfinite(adj.data).all():
        raise ValueError(f"{adj_path}: adjacency contains non-finite edge weights")
    if (adj.data < 0).any():
        raise ValueError(f"{adj_path}: adjacency contains negative edge weights")
    adj.sort_indices()
    transpose = adj.transpose().tocsr()
    transpose.sort_indices()
    same_structure = np.array_equal(adj.indptr, transpose.indptr) and np.array_equal(
        adj.indices, transpose.indices
    )
    same_weights = same_structure and np.allclose(adj.data, transpose.data, rtol=1e-7, atol=1e-8)
    if not same_weights:
        raise ValueError(f"{adj_path}: adjacency is not symmetric")
    observed_components, component_labels = connected_components(adj, directed=False, return_labels=True)
    component_sizes = np.bincount(component_labels, minlength=observed_components)
    stats = read_json(stats_path) if stats_path.exists() else {}
    if observed_components != 1:
        raise ValueError(f"{adj_path}: expected a connected RNG graph, observed {observed_components} components")
    return {
        "graph_dir": str(graph_dir),
        "adj_shape": list(adj.shape),
        "adj_nnz": int(adj.nnz),
        "explicit_zero_weight_entries": int((adj.data == 0).sum()),
        "edge_weights_finite_nonnegative": True,
        "adjacency_symmetric": True,
        "observed_n_components": int(observed_components),
        "observed_giant_component_size": int(component_sizes.max()),
        "stats_n_components": stats.get("n_components"),
        "stats_giant_component_size": stats.get("giant_component_size"),
    }


def label_codes(labels: pd.Series) -> tuple[np.ndarray, list[str]]:
    categories = sorted(labels.astype(str).str.strip().unique().tolist())
    mapping = {label: idx for idx, label in enumerate(categories)}
    codes = labels.astype(str).str.strip().map(mapping).to_numpy(dtype=np.int64)
    return codes, categories


def command_validate_inputs(args: argparse.Namespace) -> None:
    paths = Paths(args.panel_root, args.out_root)
    nodes = load_canonical_nodes(paths, args.label_col)
    codes, categories = label_codes(nodes[args.label_col])
    if len(categories) != args.expected_k:
        raise ValueError(f"expected {args.expected_k} {args.label_col} values, observed {len(categories)}: {categories}")
    if not paths.metadata_csv.exists():
        raise FileNotFoundError(paths.metadata_csv)
    if not paths.aligned_fasta.exists():
        raise FileNotFoundError(paths.aligned_fasta)
    report = {
        "panel_root": str(paths.panel_root),
        "out_root": str(paths.out_root),
        "n_nodes": int(nodes.shape[0]),
        "label_col": args.label_col,
        "labels": {label: int((codes == idx).sum()) for idx, label in enumerate(categories)},
        "metadata": validate_metadata_alignment(paths.metadata_csv, nodes, args.label_col),
        "matrices": {
            "hamming": validate_square_matrix(paths.hamming_matrix, nodes.shape[0]),
            "embedding": validate_square_matrix(paths.embedding_matrix, nodes.shape[0]),
        },
        "graphs": {
            "hamming_rng": validate_graph(paths.hamming_rng_graph, nodes.shape[0]),
            "embedding_rng": validate_graph(paths.embedding_rng_graph, nodes.shape[0]),
        },
    }
    ensure_parent(paths.out_root / "input_validation.json")
    write_json(paths.out_root / "input_validation.json", report)
    log(f"wrote {paths.out_root / 'input_validation.json'}")
    print(json.dumps(report, indent=2, sort_keys=True))


def command_prepare_design(args: argparse.Namespace) -> None:
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    paths = Paths(args.panel_root, args.out_root)
    nodes = load_canonical_nodes(paths, args.label_col)
    codes, categories = label_codes(nodes[args.label_col])
    if len(categories) != args.expected_k:
        raise ValueError(f"expected {args.expected_k} labels, observed {len(categories)}: {categories}")

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    initial_node_ids: list[int] = []
    for repeat_id in range(args.repeats):
        for cluster_slot, label in enumerate(categories):
            members = np.flatnonzero(codes == cluster_slot)
            node_id = int(rng.choice(members))
            initial_node_ids.append(node_id)
            rows.append(
                {
                    "repeat_id": repeat_id,
                    "cluster_slot": cluster_slot,
                    "label_col": args.label_col,
                    "label_value": label,
                    "node_id": node_id,
                    "accession": nodes.loc[node_id, "accession"],
                }
            )
    initial = pd.DataFrame(rows)

    unique_initial = np.array(sorted(set(initial_node_ids)), dtype=np.int64)
    candidate_pool_size = nodes.shape[0] if args.candidate_pool_size == 0 else args.candidate_pool_size
    if candidate_pool_size < 1:
        raise ValueError("--candidate-pool-size must be 0 (all nodes) or a positive integer")
    if candidate_pool_size < unique_initial.size:
        raise ValueError(
            f"--candidate-pool-size {candidate_pool_size} is smaller than "
            f"the {unique_initial.size} unique initial medoids"
        )
    all_node_ids = np.arange(nodes.shape[0], dtype=np.int64)
    remaining = np.setdiff1d(all_node_ids, unique_initial, assume_unique=False)
    fill_n = candidate_pool_size - unique_initial.size
    if fill_n > remaining.size:
        raise ValueError(f"candidate pool size {candidate_pool_size} exceeds n={nodes.shape[0]}")
    fill = rng.choice(remaining, size=fill_n, replace=False) if fill_n else np.array([], dtype=np.int64)
    candidate_ids = np.array(sorted(np.concatenate([unique_initial, fill])), dtype=np.int64)
    candidate_pool = pd.DataFrame(
        {
            "candidate_row": np.arange(candidate_ids.size, dtype=np.int64),
            "node_id": candidate_ids,
            "accession": nodes.loc[candidate_ids, "accession"].to_numpy(),
            args.label_col: nodes.loc[candidate_ids, args.label_col].to_numpy(),
            "is_initial_medoid": np.isin(candidate_ids, unique_initial),
        }
    )

    paths.design_dir.mkdir(parents=True, exist_ok=True)
    initial.to_csv(paths.initial_medoids_csv, index=False)
    candidate_pool.to_csv(paths.candidate_pool_csv, index=False)
    candidate_hash = candidate_fingerprint(candidate_pool)
    initial_hash = ordered_int_fingerprint(initial["node_id"].to_numpy(dtype=np.int64))
    rng_medoid_scope = "exact_all_nodes" if candidate_ids.size == nodes.shape[0] else "approximate_candidate_pool"
    manifest = {
        "panel_root": str(paths.panel_root),
        "out_root": str(paths.out_root),
        "label_col": args.label_col,
        "labels": categories,
        "n_nodes": int(nodes.shape[0]),
        "repeats": int(args.repeats),
        "seed": int(args.seed),
        "expected_k": int(args.expected_k),
        "candidate_pool_size": int(candidate_ids.size),
        "candidate_pool_fingerprint": candidate_hash,
        "initial_medoids_fingerprint": initial_hash,
        "rng_medoid_scope": rng_medoid_scope,
        "unique_initial_medoids": int(unique_initial.size),
        "note": "Initial medoids are one random accession per label per repeat and are shared across all metrics.",
    }
    write_json(paths.design_manifest_json, manifest)
    log(f"wrote {paths.initial_medoids_csv}")
    log(f"wrote {paths.candidate_pool_csv}")
    log(f"wrote {paths.design_manifest_json}")


def load_design(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not paths.initial_medoids_csv.exists():
        raise FileNotFoundError(f"missing design file: {paths.initial_medoids_csv}")
    if not paths.candidate_pool_csv.exists():
        raise FileNotFoundError(f"missing candidate pool: {paths.candidate_pool_csv}")
    if not paths.design_manifest_json.exists():
        raise FileNotFoundError(f"missing design manifest: {paths.design_manifest_json}")
    initial = pd.read_csv(paths.initial_medoids_csv)
    candidates = pd.read_csv(paths.candidate_pool_csv)
    manifest = read_json(paths.design_manifest_json)
    required_manifest = {
        "n_nodes",
        "repeats",
        "expected_k",
        "candidate_pool_size",
        "candidate_pool_fingerprint",
        "initial_medoids_fingerprint",
        "rng_medoid_scope",
    }
    missing_manifest = sorted(required_manifest.difference(manifest))
    if missing_manifest:
        raise ValueError(f"design manifest is missing fields: {missing_manifest}; rerun prepare-design")
    if int(manifest["candidate_pool_size"]) != candidates.shape[0]:
        raise ValueError("candidate pool row count does not match design manifest")
    candidate_node_ids = candidates["node_id"].to_numpy(dtype=np.int64)
    n_nodes = int(manifest["n_nodes"])
    if ((candidate_node_ids < 0) | (candidate_node_ids >= n_nodes)).any():
        raise ValueError("candidate pool contains node IDs outside the panel")
    observed_scope = "exact_all_nodes" if candidates.shape[0] == n_nodes else "approximate_candidate_pool"
    if manifest["rng_medoid_scope"] != observed_scope:
        raise ValueError("RNG medoid scope does not match candidate pool size")
    observed_candidate_hash = candidate_fingerprint(candidates)
    if manifest.get("candidate_pool_fingerprint") != observed_candidate_hash:
        raise ValueError(
            "candidate pool does not match design manifest; rerun prepare-design and regenerate RNG caches/results"
        )
    observed_initial_hash = ordered_int_fingerprint(initial["node_id"].to_numpy(dtype=np.int64))
    if manifest.get("initial_medoids_fingerprint") != observed_initial_hash:
        raise ValueError(
            "initial medoids do not match design manifest; rerun prepare-design and regenerate results"
        )
    return initial, candidates, manifest


def validate_design_against_nodes(
    initial: pd.DataFrame,
    candidates: pd.DataFrame,
    manifest: dict[str, Any],
    nodes: pd.DataFrame,
) -> None:
    label_col = str(manifest["label_col"])
    categories = list(manifest["labels"])
    k = int(manifest["expected_k"])
    repeats = int(manifest["repeats"])
    required_initial = {"repeat_id", "cluster_slot", "label_value", "node_id", "accession"}
    missing = sorted(required_initial.difference(initial.columns))
    if missing:
        raise ValueError(f"initial medoid design is missing fields: {missing}")
    if nodes.shape[0] != int(manifest["n_nodes"]):
        raise ValueError("node count differs from design manifest")
    if initial.shape[0] != repeats * k:
        raise ValueError(f"initial medoid design has {initial.shape[0]} rows; expected {repeats * k}")
    expected_pairs = pd.MultiIndex.from_product(
        [range(repeats), range(k)], names=["repeat_id", "cluster_slot"]
    )
    observed_pairs = pd.MultiIndex.from_frame(initial[["repeat_id", "cluster_slot"]].astype(int))
    if not observed_pairs.is_unique or set(observed_pairs) != set(expected_pairs):
        raise ValueError("initial medoid design must contain each repeat/cluster slot exactly once")
    node_ids = initial["node_id"].to_numpy(dtype=np.int64)
    if ((node_ids < 0) | (node_ids >= nodes.shape[0])).any():
        raise ValueError("initial medoid design contains node IDs outside the panel")
    for row in initial.itertuples(index=False):
        slot = int(row.cluster_slot)
        node_id = int(row.node_id)
        expected_label = categories[slot]
        if str(row.label_value) != expected_label:
            raise ValueError(f"repeat {row.repeat_id}, slot {slot}: design label does not match manifest")
        if str(nodes.loc[node_id, label_col]) != expected_label:
            raise ValueError(f"repeat {row.repeat_id}, slot {slot}: initial node is not from {expected_label}")
        if str(nodes.loc[node_id, "accession"]) != str(row.accession):
            raise ValueError(f"repeat {row.repeat_id}, slot {slot}: accession does not match node ID")
    candidate_ids = set(candidates["node_id"].astype(int))
    if not set(node_ids.tolist()).issubset(candidate_ids):
        raise ValueError("candidate pool does not contain every initial medoid")


def rng_rows_path(paths: Paths, metric: str) -> Path:
    return paths.rng_rows_dir / f"{metric}_candidate_to_all_float32.npy"


def rng_checkpoint_path(paths: Paths, metric: str) -> Path:
    return paths.rng_rows_dir / f"{metric}_candidate_to_all_checkpoint.json"


def validate_complete_rng_cache(
    paths: Paths,
    metric: str,
    candidates: pd.DataFrame,
    n_nodes: int,
    graph_dir: Path,
) -> np.ndarray:
    rows_path = rng_rows_path(paths, metric)
    checkpoint_path = rng_checkpoint_path(paths, metric)
    if not rows_path.exists():
        raise FileNotFoundError(f"missing RNG distance cache: {rows_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing RNG distance checkpoint: {checkpoint_path}")
    rows = np.load(rows_path, mmap_mode="r")
    expected_shape = (candidates.shape[0], n_nodes)
    if rows.shape != expected_shape:
        raise ValueError(f"{rows_path}: shape {rows.shape} does not match {expected_shape}")
    checkpoint = read_json(checkpoint_path)
    if checkpoint.get("candidate_pool_fingerprint") != candidate_fingerprint(candidates):
        raise ValueError(f"{metric}: cached RNG rows do not match the current candidate pool")
    if checkpoint.get("graph_signature") != graph_signature(graph_dir):
        raise ValueError(f"{metric}: cached RNG rows do not match the current graph")
    completed = set(int(x) for x in checkpoint.get("completed_candidate_rows", []))
    expected_completed = set(range(candidates.shape[0]))
    if completed != expected_completed:
        raise ValueError(
            f"{metric}: RNG cache is incomplete ({len(completed):,}/{candidates.shape[0]:,} rows); "
            "finish prepare-rng-distances"
        )
    return rows


def command_prepare_rng_distances(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    paths = Paths(args.panel_root, args.out_root)
    _, candidates, manifest = load_design(paths)
    n_nodes = int(manifest["n_nodes"])
    candidate_hash = candidate_fingerprint(candidates)
    metric_to_graph = {
        "hamming_rng": paths.hamming_rng_graph,
        "embedding_rng": paths.embedding_rng_graph,
    }
    paths.rng_rows_dir.mkdir(parents=True, exist_ok=True)

    for metric in args.metrics:
        if metric not in metric_to_graph:
            raise ValueError(f"{metric} is not an RNG metric")
        graph_dir = metric_to_graph[metric]
        expected_graph_signature = graph_signature(graph_dir)
        adj = load_npz(graph_dir / "adj.npz").tocsr()
        if adj.shape != (n_nodes, n_nodes):
            raise ValueError(f"{graph_dir / 'adj.npz'} shape mismatch: {adj.shape} vs {(n_nodes, n_nodes)}")
        out_path = rng_rows_path(paths, metric)
        checkpoint = rng_checkpoint_path(paths, metric)
        if out_path.exists() and args.force:
            out_path.unlink()
        if checkpoint.exists() and args.force:
            checkpoint.unlink()

        rows_existed = out_path.exists()
        if rows_existed:
            rows = np.load(out_path, mmap_mode="r+")
            if rows.shape != (candidates.shape[0], n_nodes):
                raise ValueError(f"{out_path}: shape {rows.shape} does not match {(candidates.shape[0], n_nodes)}")
        else:
            rows = np.lib.format.open_memmap(
                out_path,
                mode="w+",
                dtype=np.float32,
                shape=(candidates.shape[0], n_nodes),
            )

        completed: set[int] = set()
        if checkpoint.exists() and rows_existed:
            checkpoint_payload = read_json(checkpoint)
            if checkpoint_payload.get("candidate_pool_fingerprint") != candidate_hash:
                raise ValueError(
                    f"{metric}: RNG cache candidate pool differs from the design; rerun with --force"
                )
            if checkpoint_payload.get("graph_signature") != expected_graph_signature:
                raise ValueError(f"{metric}: RNG graph changed since caching; rerun with --force")
            if int(checkpoint_payload.get("n_candidate_rows", -1)) != candidates.shape[0]:
                raise ValueError(f"{metric}: RNG cache candidate-row count differs; rerun with --force")
            if int(checkpoint_payload.get("n_nodes", -1)) != n_nodes:
                raise ValueError(f"{metric}: RNG cache node count differs; rerun with --force")
            completed = set(int(x) for x in checkpoint_payload.get("completed_candidate_rows", []))
        elif checkpoint.exists() and not rows_existed:
            log(f"{metric}: ignoring stale checkpoint because {out_path} did not exist")
        candidate_rows = candidates["candidate_row"].to_numpy(dtype=np.int64)
        node_ids = candidates["node_id"].to_numpy(dtype=np.int64)
        pending = [int(row) for row in candidate_rows if int(row) not in completed]
        log(f"{metric}: {len(completed):,} completed, {len(pending):,} pending candidate rows")

        for start in range(0, len(pending), args.batch_size):
            batch_rows = np.array(pending[start : start + args.batch_size], dtype=np.int64)
            sources = node_ids[batch_rows]
            dist = dijkstra(adj, directed=False, indices=sources, unweighted=False)
            dist = np.asarray(dist, dtype=np.float32)
            if dist.ndim == 1:
                dist = dist.reshape(1, -1)
            dist[np.arange(batch_rows.size), sources] = 0.0
            if not np.isfinite(dist).all():
                bad = int((~np.isfinite(dist)).sum())
                raise ValueError(f"{metric}: Dijkstra produced {bad:,} non-finite distances")
            rows[batch_rows, :] = dist
            rows.flush()
            completed.update(int(x) for x in batch_rows.tolist())
            write_json(
                checkpoint,
                {
                    "metric": metric,
                    "graph_dir": str(graph_dir),
                    "rows_path": str(out_path),
                    "candidate_pool_fingerprint": candidate_hash,
                    "graph_signature": expected_graph_signature,
                    "completed_candidate_rows": sorted(completed),
                    "n_completed": len(completed),
                    "n_candidate_rows": int(candidates.shape[0]),
                    "n_nodes": n_nodes,
                    "updated_at_unix": time.time(),
                },
            )
            log(f"{metric}: completed {len(completed):,}/{candidates.shape[0]:,} candidate rows")


def parse_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, list[str]] = {}
    current: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            current = line[1:].split()[0].strip()
            if current in seqs:
                raise ValueError(f"duplicate FASTA header accession: {current}")
            seqs[current] = []
        else:
            if current is None:
                raise ValueError(f"{path}: sequence line before first header")
            seqs[current].append(line.upper())
    return {acc: "".join(parts) for acc, parts in seqs.items()}


def load_encoded_alignment(paths: Paths, nodes: pd.DataFrame) -> np.ndarray:
    fasta = parse_fasta(paths.aligned_fasta)
    accessions = nodes["accession"].tolist()
    missing = [acc for acc in accessions if acc not in fasta]
    if missing:
        raise ValueError(f"{paths.aligned_fasta}: missing {len(missing):,} node accessions, examples={missing[:5]}")
    lengths = {len(fasta[acc]) for acc in accessions}
    if len(lengths) != 1:
        raise ValueError(f"aligned FASTA has multiple sequence lengths for node accessions: {sorted(lengths)[:5]}")
    n = len(accessions)
    l = lengths.pop()
    encoded = np.empty((n, l), dtype=np.uint8)
    for idx, acc in enumerate(accessions):
        encoded[idx, :] = np.frombuffer(fasta[acc].encode("ascii"), dtype=np.uint8)
    return encoded


def sanitize_medoid_distances(dense_matrix: np.ndarray, medoids: np.ndarray) -> np.ndarray:
    d = np.asarray(dense_matrix[medoids, :], dtype=np.float64)
    for slot, medoid in enumerate(medoids):
        d[slot, medoid] = 0.0
    if not np.isfinite(d).all():
        bad = ~np.isfinite(d)
        bad_rows, bad_cols = np.where(bad)
        offdiag_bad = [(int(medoids[r]), int(c)) for r, c in zip(bad_rows[:10], bad_cols[:10]) if int(medoids[r]) != int(c)]
        if offdiag_bad:
            raise ValueError(f"non-finite off-diagonal dense distances, examples={offdiag_bad[:5]}")
        d[bad] = 0.0
    return d


def assign_dense(dense_matrix: np.ndarray, medoids: np.ndarray) -> tuple[np.ndarray, float]:
    if np.unique(medoids).size != medoids.size:
        raise ValueError("medoid node IDs must be unique")
    d = sanitize_medoid_distances(dense_matrix, medoids)
    labels = np.argmin(d, axis=0).astype(np.int64)
    # A medoid owns its observation when distances tie (for example, identical
    # aligned sequences). This preserves k non-empty clusters without changing
    # the objective because every overridden assignment is tied at distance 0.
    labels[medoids] = np.arange(medoids.size, dtype=np.int64)
    objective = float(d[labels, np.arange(d.shape[1])].sum())
    return labels, objective


def profile_hamming_medoid(encoded: np.ndarray, members: np.ndarray, block_size: int) -> int:
    if members.size == 1:
        return int(members[0])
    cluster = encoded[members, :]
    n_cluster, seq_len = cluster.shape
    counts = np.zeros((seq_len, 256), dtype=np.int32)
    for pos in range(seq_len):
        counts[pos, :] = np.bincount(cluster[:, pos], minlength=256)
    pos_ix = np.arange(seq_len)
    best_node = int(members[0])
    best_score = math.inf
    for start in range(0, members.size, block_size):
        block_members = members[start : start + block_size]
        chars = encoded[block_members, :]
        matches = counts[pos_ix[None, :], chars].sum(axis=1)
        scores = (n_cluster * seq_len - matches).astype(np.float64)
        local_min = float(scores.min())
        if local_min < best_score:
            local_candidates = block_members[np.flatnonzero(scores == local_min)]
            best_node = int(local_candidates.min())
            best_score = local_min
        elif local_min == best_score:
            local_candidates = block_members[np.flatnonzero(scores == local_min)]
            best_node = min(best_node, int(local_candidates.min()))
    return best_node


def update_hamming_profile(encoded: np.ndarray, labels: np.ndarray, k: int, old_medoids: np.ndarray, block_size: int) -> np.ndarray:
    new_medoids = old_medoids.copy()
    for cluster_id in range(k):
        members = np.flatnonzero(labels == cluster_id).astype(np.int64)
        if not members.size:
            raise AssertionError(f"Hamming cluster {cluster_id} is empty after medoid-owned tie handling")
        new_medoids[cluster_id] = profile_hamming_medoid(encoded, members, block_size)
    return new_medoids


def dense_cluster_medoid(
    dense_matrix: np.ndarray,
    members: np.ndarray,
    candidate_block_size: int,
    member_block_size: int,
) -> int:
    if members.size == 1:
        return int(members[0])
    best_node = int(members[0])
    best_score = math.inf
    member_set = set(int(x) for x in members.tolist())
    for c_start in range(0, members.size, candidate_block_size):
        candidates = members[c_start : c_start + candidate_block_size]
        scores = np.zeros(candidates.size, dtype=np.float64)
        for m_start in range(0, members.size, member_block_size):
            member_block = members[m_start : m_start + member_block_size]
            block = np.asarray(dense_matrix[np.ix_(candidates, member_block)], dtype=np.float64)
            if not np.isfinite(block).all():
                for i, cand in enumerate(candidates):
                    hits = np.flatnonzero(member_block == cand)
                    if hits.size:
                        block[i, hits] = 0.0
                if not np.isfinite(block).all():
                    raise ValueError("dense medoid update found non-finite off-diagonal distances")
            scores += block.sum(axis=1)
        local_min = float(scores.min())
        if local_min < best_score:
            local_candidates = candidates[np.flatnonzero(scores == local_min)]
            best_node = int(local_candidates.min())
            best_score = local_min
        elif local_min == best_score:
            local_candidates = candidates[np.flatnonzero(scores == local_min)]
            best_node = min(best_node, int(local_candidates.min()))
    if best_node not in member_set:
        raise AssertionError("selected medoid is not a cluster member")
    return best_node


def update_dense_exact(
    dense_matrix: np.ndarray,
    labels: np.ndarray,
    k: int,
    old_medoids: np.ndarray,
    candidate_block_size: int,
    member_block_size: int,
) -> np.ndarray:
    new_medoids = old_medoids.copy()
    for cluster_id in range(k):
        members = np.flatnonzero(labels == cluster_id).astype(np.int64)
        if not members.size:
            raise AssertionError(f"dense cluster {cluster_id} is empty after medoid-owned tie handling")
        new_medoids[cluster_id] = dense_cluster_medoid(
            dense_matrix,
            members,
            candidate_block_size=candidate_block_size,
            member_block_size=member_block_size,
        )
    return new_medoids


def assign_candidate_rows(rows: np.ndarray, medoid_pool_rows: np.ndarray, medoid_node_ids: np.ndarray) -> tuple[np.ndarray, float]:
    if np.unique(medoid_node_ids).size != medoid_node_ids.size:
        raise ValueError("medoid node IDs must be unique")
    d = np.asarray(rows[medoid_pool_rows, :], dtype=np.float64)
    for slot, medoid in enumerate(medoid_node_ids):
        d[slot, medoid] = 0.0
    if not np.isfinite(d).all():
        raise ValueError("candidate-row assignment found non-finite RNG distances")
    labels = np.argmin(d, axis=0).astype(np.int64)
    labels[medoid_node_ids] = np.arange(medoid_node_ids.size, dtype=np.int64)
    objective = float(d[labels, np.arange(d.shape[1])].sum())
    return labels, objective


def update_candidate_rows(
    rows: np.ndarray,
    labels: np.ndarray,
    k: int,
    old_medoids: np.ndarray,
    candidate_node_ids: np.ndarray,
    node_to_candidate_row: dict[int, int],
    member_block_size: int,
) -> np.ndarray:
    new_medoids = old_medoids.copy()
    for cluster_id in range(k):
        members = np.flatnonzero(labels == cluster_id).astype(np.int64)
        if not members.size:
            raise AssertionError(f"RNG cluster {cluster_id} is empty after medoid-owned tie handling")
        eligible_node_ids = candidate_node_ids[labels[candidate_node_ids] == cluster_id]
        if not eligible_node_ids.size:
            raise AssertionError(f"RNG cluster {cluster_id} has no eligible medoid candidate")
        eligible_rows = np.array([node_to_candidate_row[int(x)] for x in eligible_node_ids], dtype=np.int64)
        scores = np.zeros(eligible_rows.size, dtype=np.float64)
        for m_start in range(0, members.size, member_block_size):
            member_block = members[m_start : m_start + member_block_size]
            block = np.asarray(rows[np.ix_(eligible_rows, member_block)], dtype=np.float64)
            if not np.isfinite(block).all():
                raise ValueError("candidate-row medoid update found non-finite RNG distances")
            scores += block.sum(axis=1)
        best_score = scores.min()
        best_nodes = eligible_node_ids[np.flatnonzero(scores == best_score)]
        new_medoids[cluster_id] = int(best_nodes.min())
    return new_medoids


def evaluate_labels(pred: np.ndarray, truth: np.ndarray, categories: list[str], k: int) -> dict[str, Any]:
    contingency = np.zeros((k, len(categories)), dtype=np.int64)
    for cluster_id in range(k):
        for label_id in range(len(categories)):
            contingency[cluster_id, label_id] = int(((pred == cluster_id) & (truth == label_id)).sum())
    row_ind, col_ind = linear_sum_assignment(-contingency)
    matched = int(contingency[row_ind, col_ind].sum())
    mapping = {int(row): categories[int(col)] for row, col in zip(row_ind, col_ind)}
    return {
        "ari": float(adjusted_rand_score(truth, pred)),
        "n_mislabeled": int(pred.size - matched),
        "error_rate": float((pred.size - matched) / pred.size),
        "hungarian_match_count": matched,
        "cluster_to_label": mapping,
        "contingency": contingency.tolist(),
    }


def run_one_metric(
    metric: str,
    initial_medoids: np.ndarray,
    truth: np.ndarray,
    categories: list[str],
    k: int,
    paths: Paths,
    nodes: pd.DataFrame,
    args: argparse.Namespace,
    encoded_alignment: np.ndarray | None,
    candidate_pool: pd.DataFrame,
) -> dict[str, Any]:
    medoids = initial_medoids.astype(np.int64).copy()
    labels = np.zeros(nodes.shape[0], dtype=np.int64)
    objective = math.nan
    converged = False
    notes: list[str] = []
    objective_history: list[float] = []

    if metric == "hamming":
        dense = np.load(paths.hamming_matrix, mmap_mode="r")
        if encoded_alignment is None:
            raise ValueError("encoded alignment is required for hamming profile updates")
        medoid_space = "all cluster members; update uses position-level residue frequency profile"
        for iteration in range(args.max_iter):
            labels, objective = assign_dense(dense, medoids)
            objective_history.append(float(objective))
            new_medoids = update_hamming_profile(
                encoded_alignment,
                labels,
                k,
                medoids,
                block_size=args.profile_block_size,
            )
            if np.array_equal(new_medoids, medoids):
                converged = True
                break
            medoids = new_medoids
        labels, objective = assign_dense(dense, medoids)
    elif metric == "embedding":
        dense = np.load(paths.embedding_matrix, mmap_mode="r")
        medoid_space = "all cluster members; update uses dense precomputed ESM-2 cityblock distances"
        notes.append("embedding matrix diagonal +inf values are sanitized in memory when encountered")
        for iteration in range(args.max_iter):
            labels, objective = assign_dense(dense, medoids)
            objective_history.append(float(objective))
            new_medoids = update_dense_exact(
                dense,
                labels,
                k,
                medoids,
                candidate_block_size=args.dense_candidate_block_size,
                member_block_size=args.member_block_size,
            )
            if np.array_equal(new_medoids, medoids):
                converged = True
                break
            medoids = new_medoids
        labels, objective = assign_dense(dense, medoids)
    elif metric in {"hamming_rng", "embedding_rng"}:
        graph_dir = paths.hamming_rng_graph if metric == "hamming_rng" else paths.embedding_rng_graph
        rows = validate_complete_rng_cache(
            paths,
            metric,
            candidate_pool,
            nodes.shape[0],
            graph_dir,
        )
        candidate_node_ids = candidate_pool["node_id"].to_numpy(dtype=np.int64)
        node_to_candidate_row = {
            int(node_id): int(row)
            for row, node_id in zip(
                candidate_pool["candidate_row"].to_numpy(dtype=np.int64),
                candidate_node_ids,
            )
        }
        missing_initial = [int(x) for x in medoids.tolist() if int(x) not in node_to_candidate_row]
        if missing_initial:
            raise ValueError(f"{metric}: initial medoids missing from candidate pool: {missing_initial[:5]}")
        exact_all_nodes = candidate_pool.shape[0] == nodes.shape[0]
        if exact_all_nodes:
            medoid_space = "all cluster members; exact weighted RNG shortest-path k-medoids update"
        else:
            medoid_space = "shared candidate pool only; approximate weighted RNG shortest-path k-medoids update"
            notes.append("RNG updates are candidate-pool approximations, not exact all-node graph k-medoids")
        for iteration in range(args.max_iter):
            medoid_pool_rows = np.array([node_to_candidate_row[int(x)] for x in medoids], dtype=np.int64)
            labels, objective = assign_candidate_rows(rows, medoid_pool_rows, medoids)
            objective_history.append(float(objective))
            new_medoids = update_candidate_rows(
                rows,
                labels,
                k,
                medoids,
                candidate_node_ids=candidate_node_ids,
                node_to_candidate_row=node_to_candidate_row,
                member_block_size=args.member_block_size,
            )
            if np.array_equal(new_medoids, medoids):
                converged = True
                break
            medoids = new_medoids
        medoid_pool_rows = np.array([node_to_candidate_row[int(x)] for x in medoids], dtype=np.int64)
        labels, objective = assign_candidate_rows(rows, medoid_pool_rows, medoids)
    else:
        raise ValueError(f"unknown metric: {metric}")

    objective_history.append(float(objective))
    for previous, current in zip(objective_history, objective_history[1:]):
        tolerance = max(1e-7, 1e-10 * abs(previous))
        if current > previous + tolerance:
            raise AssertionError(
                f"{metric}: k-medoids objective increased from {previous} to {current}"
            )
    eval_report = evaluate_labels(labels, truth, categories, k)
    return {
        "metric": metric,
        "metric_description": METRIC_DISPLAY[metric],
        "initial_medoids_node_id": initial_medoids.astype(int).tolist(),
        "initial_medoids_accession": nodes.loc[initial_medoids, "accession"].tolist(),
        "final_medoids_node_id": medoids.astype(int).tolist(),
        "final_medoids_accession": nodes.loc[medoids, "accession"].tolist(),
        "iterations": int(iteration + 1),
        "max_iter": int(args.max_iter),
        "converged": bool(converged),
        "objective": float(objective),
        "objective_history": objective_history,
        "medoid_search_space": medoid_space,
        "notes": notes,
        **eval_report,
    }


def repeat_result_path(paths: Paths, metric: str, repeat_id: int) -> Path:
    return paths.runs_dir / metric / f"repeat_{repeat_id:03d}.json"


def command_run(args: argparse.Namespace) -> None:
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if args.max_iter < 1:
        raise ValueError("--max-iter must be positive")
    paths = Paths(args.panel_root, args.out_root)
    initial, candidate_pool, manifest = load_design(paths)
    label_col = str(manifest["label_col"])
    nodes = load_canonical_nodes(paths, label_col)
    validate_design_against_nodes(initial, candidate_pool, manifest, nodes)
    truth, categories = label_codes(nodes[label_col])
    k = len(categories)
    if k != int(manifest["expected_k"]):
        raise ValueError(f"manifest expected_k={manifest['expected_k']} but observed {k}")
    if args.repeats > int(manifest["repeats"]):
        raise ValueError(
            f"requested {args.repeats} repeats but design contains only {manifest['repeats']}"
        )

    encoded_alignment = None
    if "hamming" in args.metrics:
        log("loading aligned FASTA for profile-based Hamming medoids")
        encoded_alignment = load_encoded_alignment(paths, nodes)
        log(f"encoded alignment shape={encoded_alignment.shape}")

    for repeat_id in range(args.repeats):
        repeat_rows = initial[initial["repeat_id"] == repeat_id].sort_values("cluster_slot")
        if repeat_rows.shape[0] != k:
            raise ValueError(f"repeat {repeat_id}: expected {k} initial medoids, observed {repeat_rows.shape[0]}")
        initial_medoids = repeat_rows["node_id"].to_numpy(dtype=np.int64)
        for metric in args.metrics:
            out_path = repeat_result_path(paths, metric, repeat_id)
            if out_path.exists() and args.resume:
                existing = read_json(out_path)
                expected_initial = initial_medoids.astype(int).tolist()
                if existing.get("metric") != metric or existing.get("initial_medoids_node_id") != expected_initial:
                    raise ValueError(f"{out_path}: existing result does not match the current paired design")
                if int(existing.get("max_iter", -1)) != args.max_iter:
                    raise ValueError(
                        f"{out_path}: existing max_iter={existing.get('max_iter')} differs from requested "
                        f"{args.max_iter}; remove/move the result or use the original setting"
                    )
                if existing.get("candidate_pool_fingerprint") != manifest["candidate_pool_fingerprint"]:
                    raise ValueError(f"{out_path}: existing result uses a different candidate pool")
                log(f"skip existing {out_path}")
                continue
            log(f"running metric={metric} repeat={repeat_id}")
            result = run_one_metric(
                metric=metric,
                initial_medoids=initial_medoids,
                truth=truth,
                categories=categories,
                k=k,
                paths=paths,
                nodes=nodes,
                args=args,
                encoded_alignment=encoded_alignment,
                candidate_pool=candidate_pool,
            )
            result.update(
                {
                    "repeat_id": int(repeat_id),
                    "label_col": label_col,
                    "labels": categories,
                    "n_nodes": int(nodes.shape[0]),
                    "design_manifest": str(paths.design_manifest_json),
                    "candidate_pool_size": int(candidate_pool.shape[0]),
                    "candidate_pool_fingerprint": manifest["candidate_pool_fingerprint"],
                    "rng_medoid_scope": manifest["rng_medoid_scope"],
                    "created_at_unix": time.time(),
                }
            )
            write_json(out_path, result)

def collect_run_rows(paths: Paths) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for metric_dir in sorted(paths.runs_dir.glob("*")):
        if not metric_dir.is_dir():
            continue
        for run_json in sorted(metric_dir.glob("repeat_*.json")):
            payload = read_json(run_json)
            records.append(
                {
                    "metric": payload["metric"],
                    "repeat_id": int(payload["repeat_id"]),
                    "ari": float(payload["ari"]),
                    "n_mislabeled": int(payload["n_mislabeled"]),
                    "error_rate": float(payload["error_rate"]),
                    "objective": float(payload["objective"]),
                    "iterations": int(payload["iterations"]),
                    "converged": bool(payload["converged"]),
                    "candidate_pool_size": int(payload.get("candidate_pool_size", 0)),
                    "candidate_pool_fingerprint": payload.get("candidate_pool_fingerprint", ""),
                    "rng_medoid_scope": payload.get("rng_medoid_scope", ""),
                    "medoid_search_space": payload.get("medoid_search_space", ""),
                    "result_path": str(run_json),
                }
            )
    if not records:
        raise FileNotFoundError(f"no repeat JSON files found under {paths.runs_dir}")
    return pd.DataFrame(records)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    rows = [[str(value) for value in row] for row in df.itertuples(index=False, name=None)]
    widths = [
        max(len(str(col)), *(len(row[idx]) for row in rows)) if rows else len(str(col))
        for idx, col in enumerate(columns)
    ]
    header = "| " + " | ".join(str(col).ljust(widths[idx]) for idx, col in enumerate(columns)) + " |"
    sep = "| " + " | ".join("-" * widths[idx] for idx in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(columns))) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def command_summarize(args: argparse.Namespace) -> None:
    paths = Paths(args.panel_root, args.out_root)
    _, _, manifest = load_design(paths)
    runs = collect_run_rows(paths)
    duplicates = runs.duplicated(["metric", "repeat_id"], keep=False)
    if duplicates.any():
        raise ValueError("duplicate metric/repeat results found")
    expected_repeats = set(range(int(manifest["repeats"])))
    expected_metrics = set(METRIC_DISPLAY)
    issues: list[str] = []
    if set(runs["metric"]) != expected_metrics:
        issues.append(
            f"metrics present={sorted(set(runs['metric']))}, expected={sorted(expected_metrics)}"
        )
    for metric in sorted(set(runs["metric"])):
        observed_repeats = set(runs.loc[runs["metric"] == metric, "repeat_id"].astype(int))
        if observed_repeats != expected_repeats:
            issues.append(
                f"{metric}: {len(observed_repeats)}/{len(expected_repeats)} expected repeat IDs present"
            )
    expected_candidate_hash = manifest["candidate_pool_fingerprint"]
    if set(runs["candidate_pool_fingerprint"]) != {expected_candidate_hash}:
        issues.append("one or more results use a different candidate pool than the design manifest")
    if issues and not args.allow_incomplete:
        raise ValueError("incomplete or mixed result set; " + "; ".join(issues))
    ensure_parent(paths.out_root / "kmedoids_runs.csv")
    runs.to_csv(paths.out_root / "kmedoids_runs.csv", index=False)
    summary = (
        runs.groupby("metric", as_index=False)
        .agg(
            n_repeats=("repeat_id", "count"),
            ari_mean=("ari", "mean"),
            ari_sd=("ari", "std"),
            ari_median=("ari", "median"),
            ari_min=("ari", "min"),
            ari_max=("ari", "max"),
            mislabeled_mean=("n_mislabeled", "mean"),
            mislabeled_median=("n_mislabeled", "median"),
            mislabeled_min=("n_mislabeled", "min"),
            mislabeled_max=("n_mislabeled", "max"),
            converged_rate=("converged", "mean"),
            iterations_mean=("iterations", "mean"),
        )
        .sort_values(["ari_mean", "mislabeled_mean"], ascending=[False, True])
    )
    summary.to_csv(paths.out_root / "summary_by_metric.csv", index=False)

    present_metrics = sorted(runs["metric"].unique().tolist())
    paired_rows: list[dict[str, Any]] = []
    for left in present_metrics:
        for right in present_metrics:
            if left >= right:
                continue
            merged = runs[runs["metric"] == left].merge(
                runs[runs["metric"] == right],
                on="repeat_id",
                suffixes=(f"_{left}", f"_{right}"),
            )
            if merged.empty:
                continue
            paired_rows.append(
                {
                    "metric_left": left,
                    "metric_right": right,
                    "n_paired_repeats": int(merged.shape[0]),
                    "ari_mean_delta_left_minus_right": float(
                        (merged[f"ari_{left}"] - merged[f"ari_{right}"]).mean()
                    ),
                    "mislabeled_mean_delta_left_minus_right": float(
                        (merged[f"n_mislabeled_{left}"] - merged[f"n_mislabeled_{right}"]).mean()
                    ),
                }
            )
    if paired_rows:
        pd.DataFrame(paired_rows).to_csv(paths.out_root / "paired_metric_deltas.csv", index=False)

    md_lines = [
        "# Seed-42 20k paired k-medoids summary",
        "",
        f"Output root: `{paths.out_root}`",
        "",
        (
            "RNG metrics use weighted shortest-path distances and exact all-node medoid updates."
            if manifest["rng_medoid_scope"] == "exact_all_nodes"
            else "RNG metrics use weighted shortest-path distances and approximate candidate-pool medoid updates."
        ),
        "",
        *( ["Validation warnings: " + "; ".join(issues), ""] if issues else [] ),
        dataframe_to_markdown(summary),
        "",
    ]
    (paths.out_root / "summary_by_metric.md").write_text("\n".join(md_lines), encoding="utf-8")
    log(f"wrote {paths.out_root / 'kmedoids_runs.csv'}")
    log(f"wrote {paths.out_root / 'summary_by_metric.csv'}")
    log(f"wrote {paths.out_root / 'summary_by_metric.md'}")


def parse_metrics(values: list[str], allowed: set[str]) -> list[str]:
    metrics: list[str] = []
    for value in values:
        if value == "all":
            metrics.extend(["hamming", "embedding", "hamming_rng", "embedding_rng"])
        else:
            metrics.append(value)
    bad = sorted(set(metrics).difference(allowed))
    if bad:
        raise ValueError(f"unknown metrics {bad}; allowed={sorted(allowed)}")
    deduped: list[str] = []
    for metric in metrics:
        if metric not in deduped:
            deduped.append(metric)
    return deduped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--panel-root", type=Path, default=DEFAULT_PANEL_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--label-col", default="cohort_id")
    parser.add_argument("--expected-k", type=int, default=9)

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("validate-inputs", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p_design = sub.add_parser("prepare-design", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p_design.add_argument("--repeats", type=int, default=200)
    p_design.add_argument("--seed", type=int, default=42)
    p_design.add_argument(
        "--candidate-pool-size",
        type=int,
        default=0,
        help="RNG medoid candidates; 0 uses all nodes (exact), smaller positive values are approximate",
    )

    p_rng = sub.add_parser("prepare-rng-distances", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p_rng.add_argument("--metrics", nargs="+", default=["hamming_rng", "embedding_rng"])
    p_rng.add_argument("--batch-size", type=int, default=16)
    p_rng.add_argument("--force", action="store_true")

    p_run = sub.add_parser("run", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p_run.add_argument("--metrics", nargs="+", default=["all"])
    p_run.add_argument("--repeats", type=int, default=200)
    p_run.add_argument("--max-iter", type=int, default=100)
    p_run.add_argument("--resume", action="store_true")
    p_run.add_argument("--profile-block-size", type=int, default=512)
    p_run.add_argument("--dense-candidate-block-size", type=int, default=256)
    p_run.add_argument("--member-block-size", type=int, default=4096)

    p_summary = sub.add_parser("summarize", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p_summary.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="summarize partial or mixed metric/repeat sets while recording a warning",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.panel_root = args.panel_root.resolve()
    args.out_root = args.out_root.resolve()

    if hasattr(args, "metrics"):
        allowed = {"hamming", "embedding", "hamming_rng", "embedding_rng"}
        if args.command == "prepare-rng-distances":
            allowed = {"hamming_rng", "embedding_rng"}
        args.metrics = parse_metrics(args.metrics, allowed)

    try:
        if args.command == "validate-inputs":
            command_validate_inputs(args)
        elif args.command == "prepare-design":
            command_prepare_design(args)
        elif args.command == "prepare-rng-distances":
            command_prepare_rng_distances(args)
        elif args.command == "run":
            command_run(args)
        elif args.command == "summarize":
            command_summarize(args)
        else:
            parser.error(f"unknown command {args.command}")
    except Exception as exc:
        log(f"ERROR: {exc}")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
