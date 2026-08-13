#!/usr/bin/env python3
"""Compute the diametrical-pair/farthest-first constructive ZST lower bound."""

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse import load_npz  # noqa: E402
from scipy.sparse.csgraph import dijkstra  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    family: str
    construction: str
    source_kind: str
    relative_path: str


METRIC_SPECS = (
    MetricSpec(
        "raw_hamming",
        "Raw Hamming",
        "Hamming",
        "Raw",
        "raw_global",
        "graphs/hamming/{sample}/distance_matrices/hamming_count-gap-state_all_states_uint16.npy",
    ),
    MetricSpec(
        "hamming_knn_k05",
        "Hamming k-NN (k=5)",
        "Hamming",
        "k-NN (k=5)",
        "knn_graph",
        "graphs/hamming/{sample}/hamming_knn_k05/adj.npz",
    ),
    MetricSpec(
        "hamming_knn_k50",
        "Hamming k-NN (k=50)",
        "Hamming",
        "k-NN (k=50)",
        "knn_graph",
        "graphs/hamming/{sample}/hamming_knn_k50/adj.npz",
    ),
    MetricSpec(
        "hamming_rng",
        "Hamming RNG",
        "Hamming",
        "RNG",
        "rng_global_cache",
        "graphs/hamming/{sample}/hamming_rng_exact/adj.npz",
    ),
    MetricSpec(
        "raw_embedding_cityblock",
        "Raw embedding cityblock",
        "Embedding cityblock",
        "Raw",
        "raw_global",
        "graphs/esm2_650M/cityblock/{sample}/distance_matrices/embedding_cityblock_float32.npy",
    ),
    MetricSpec(
        "embedding_knn_k05",
        "Embedding k-NN (k=5)",
        "Embedding cityblock",
        "k-NN (k=5)",
        "knn_graph",
        "graphs/esm2_650M/cityblock/{sample}/embedding_knn_k05/adj.npz",
    ),
    MetricSpec(
        "embedding_knn_k50",
        "Embedding k-NN (k=50)",
        "Embedding cityblock",
        "k-NN (k=50)",
        "knn_graph",
        "graphs/esm2_650M/cityblock/{sample}/embedding_knn_k50/adj.npz",
    ),
    MetricSpec(
        "embedding_rng",
        "Embedding RNG",
        "Embedding cityblock",
        "RNG",
        "rng_global_cache",
        "graphs/esm2_650M/cityblock/{sample}/embedding_rng_exact/adj.npz",
    ),
)

SPEC_BY_KEY = {spec.key: spec for spec in METRIC_SPECS}
KNN_KEYS = tuple(spec.key for spec in METRIC_SPECS if spec.source_kind == "knn_graph")
ALL_KEYS = tuple(spec.key for spec in METRIC_SPECS)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def parse_metric_keys(value: str, allowed: tuple[str, ...]) -> list[str]:
    if value.strip().lower() == "all":
        return list(allowed)
    keys = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(keys).difference(allowed))
    if unknown:
        raise ValueError(f"Unknown metrics: {unknown}; allowed values are {list(allowed)}")
    if not keys:
        raise ValueError("At least one metric is required")
    return keys


def ordered_int_fingerprint(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes(order="C")).hexdigest()


def file_signature(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def graph_signature(adj_path: Path) -> dict[str, Any]:
    signature = file_signature(adj_path)
    return {
        "adj_path": signature["path"],
        "adj_size_bytes": signature["size_bytes"],
        "adj_mtime_ns": signature["mtime_ns"],
    }


def panel_root(args: argparse.Namespace) -> Path:
    return Path(args.source_root) / args.panel / f"seed_{args.seed}"


def spec_path(spec: MetricSpec, args: argparse.Namespace) -> Path:
    return panel_root(args) / spec.relative_path.format(sample=args.sample_label)


def load_common_node_ids(path: Path, max_sinks: int | None = None) -> np.ndarray:
    table = pd.read_csv(path)
    if "node_id" not in table.columns:
        raise ValueError(f"{path} must contain a node_id column")
    node_ids = pd.to_numeric(table["node_id"], errors="raise").to_numpy(dtype=np.int64)
    if node_ids.size < 2 or np.unique(node_ids).size != node_ids.size:
        raise ValueError("Common node IDs must contain at least two unique values")
    if np.any(node_ids < 0):
        raise ValueError("Common node IDs cannot be negative")
    if max_sinks is not None:
        if max_sinks < 2:
            raise ValueError("--max-sinks must be at least 2")
        node_ids = node_ids[:max_sinks]
    return node_ids


def knn_cache_path(workspace: Path, metric_key: str) -> Path:
    return workspace / "distance_caches" / f"{metric_key}_weighted_shortest_path_common_float32.npy"


def knn_checkpoint_path(workspace: Path, metric_key: str) -> Path:
    return workspace / "distance_caches" / f"{metric_key}_weighted_shortest_path_common_checkpoint.json"


def validate_graph_weights(adj_path: Path, n_total_nodes: int):
    graph = load_npz(adj_path).tocsr()
    if graph.shape != (n_total_nodes, n_total_nodes):
        raise ValueError(f"{adj_path} has shape {graph.shape}, expected {(n_total_nodes, n_total_nodes)}")
    if not np.isfinite(graph.data).all() or np.any(graph.data < 0):
        raise ValueError(f"{adj_path} contains invalid edge weights")
    return graph


def prepare_one_knn_cache(
    spec: MetricSpec,
    args: argparse.Namespace,
    common_ids: np.ndarray,
) -> None:
    workspace = Path(args.workspace)
    cache_path = knn_cache_path(workspace, spec.key)
    checkpoint_path = knn_checkpoint_path(workspace, spec.key)
    adj_path = spec_path(spec, args)
    if not adj_path.exists():
        raise FileNotFoundError(adj_path)
    if args.force:
        cache_path.unlink(missing_ok=True)
        checkpoint_path.unlink(missing_ok=True)

    common_hash = ordered_int_fingerprint(common_ids)
    expected_graph_signature = graph_signature(adj_path)
    n_sinks = int(common_ids.size)
    n_total_nodes = int(max(common_ids.max() + 1, args.expected_total_nodes))
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache_existed = cache_path.exists()
    if cache_existed:
        rows = np.load(cache_path, mmap_mode="r+")
        if rows.shape != (n_sinks, n_sinks) or rows.dtype != np.float32:
            raise ValueError(f"{cache_path} has incompatible shape or dtype; use --force")
    else:
        rows = np.lib.format.open_memmap(
            cache_path,
            mode="w+",
            dtype=np.float32,
            shape=(n_sinks, n_sinks),
        )

    completed: set[int] = set()
    if checkpoint_path.exists() and cache_existed:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("common_node_fingerprint") != common_hash:
            raise ValueError(f"{spec.key}: common sink set changed; use --force")
        if checkpoint.get("graph_signature") != expected_graph_signature:
            raise ValueError(f"{spec.key}: graph changed; use --force")
        completed = set(int(value) for value in checkpoint.get("completed_local_rows", []))
    elif checkpoint_path.exists():
        log(f"{spec.label}: ignoring a stale checkpoint because its matrix is absent")

    pending = [row for row in range(n_sinks) if row not in completed]
    log(f"{spec.label}: {len(completed):,} cached rows, {len(pending):,} pending rows")
    if not pending:
        return

    graph = validate_graph_weights(adj_path, n_total_nodes=n_total_nodes)
    for start in range(0, len(pending), args.batch_size):
        local_rows = np.asarray(pending[start : start + args.batch_size], dtype=np.int64)
        source_ids = common_ids[local_rows]
        distances = dijkstra(graph, directed=False, indices=source_ids, unweighted=False)
        distances = np.asarray(distances)
        if distances.ndim == 1:
            distances = distances.reshape(1, -1)
        sink_distances = distances[:, common_ids]
        sink_distances[np.arange(local_rows.size), local_rows] = 0.0
        if not np.isfinite(sink_distances).all() or np.any(sink_distances < 0):
            n_bad = int(np.count_nonzero(~np.isfinite(sink_distances) | (sink_distances < 0)))
            raise ValueError(f"{spec.key}: Dijkstra produced {n_bad:,} invalid common-sink distances")
        rows[local_rows, :] = sink_distances.astype(np.float32)
        rows.flush()
        completed.update(int(value) for value in local_rows)
        write_json(
            checkpoint_path,
            {
                "metric_key": spec.key,
                "metric_label": spec.label,
                "distance_definition": "weighted shortest path on the original graph; sinks restricted to common_node_ids",
                "graph_signature": expected_graph_signature,
                "common_node_fingerprint": common_hash,
                "n_sinks": n_sinks,
                "n_total_graph_nodes": int(graph.shape[0]),
                "dtype": "float32",
                "completed_local_rows": sorted(completed),
                "n_completed": len(completed),
                "updated_at_unix": time.time(),
            },
        )
        log(f"{spec.label}: cached {len(completed):,}/{n_sinks:,} rows")


class DistanceProvider:
    def __init__(
        self,
        matrix_path: Path,
        common_ids: np.ndarray,
        row_space: str,
        conservative_float32: bool,
    ) -> None:
        self.matrix_path = matrix_path
        self.common_ids = common_ids
        self.row_space = row_space
        self.matrix = np.load(matrix_path, mmap_mode="r")
        self.n_sinks = int(common_ids.size)
        self.conservative_float32 = conservative_float32 and self.matrix.dtype == np.float32
        if row_space == "global":
            required = int(common_ids.max()) + 1
            if self.matrix.ndim != 2 or self.matrix.shape[0] < required or self.matrix.shape[1] < required:
                raise ValueError(f"{matrix_path} cannot index all requested global node IDs")
        elif row_space == "local":
            if self.matrix.shape != (self.n_sinks, self.n_sinks):
                raise ValueError(f"{matrix_path} has shape {self.matrix.shape}, expected {(self.n_sinks, self.n_sinks)}")
        else:
            raise ValueError(f"Unknown row space: {row_space}")

    def row(self, local_index: int) -> np.ndarray:
        if self.row_space == "global":
            source_id = int(self.common_ids[local_index])
            values = np.asarray(self.matrix[source_id, self.common_ids], dtype=np.float64)
        else:
            values = np.asarray(self.matrix[local_index, :], dtype=np.float64)
        values[local_index] = 0.0
        if self.conservative_float32:
            positive = values > 0
            values[positive] = np.nextafter(values[positive].astype(np.float32), -np.inf).astype(np.float64)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"{self.matrix_path}: row {local_index} contains invalid distances")
        return values


def validate_existing_rng_cache(spec: MetricSpec, args: argparse.Namespace) -> Path:
    cache_root = Path(args.rng_cache_root)
    cache_path = cache_root / f"{spec.key}_candidate_to_all_float32.npy"
    checkpoint_path = cache_root / f"{spec.key}_candidate_to_all_checkpoint.json"
    if not cache_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing complete RNG cache/checkpoint for {spec.key} under {cache_root}")
    matrix = np.load(cache_path, mmap_mode="r")
    if matrix.shape != (args.expected_total_nodes, args.expected_total_nodes) or matrix.dtype != np.float32:
        raise ValueError(f"{cache_path} has unexpected shape or dtype")
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if int(checkpoint.get("n_completed", -1)) != args.expected_total_nodes:
        raise ValueError(f"{checkpoint_path} is incomplete")
    expected_signature = graph_signature(spec_path(spec, args))
    if checkpoint.get("graph_signature") != expected_signature:
        raise ValueError(f"{spec.key}: RNG cache graph signature does not match the current graph")
    candidate_csv = Path(args.rng_candidate_csv)
    candidates = pd.read_csv(candidate_csv, usecols=["candidate_row", "node_id"])
    expected = np.arange(args.expected_total_nodes, dtype=np.int64)
    if not np.array_equal(candidates["candidate_row"].to_numpy(dtype=np.int64), expected):
        raise ValueError("RNG candidate rows are not contiguous")
    if not np.array_equal(candidates["node_id"].to_numpy(dtype=np.int64), expected):
        raise ValueError("RNG cache row order is not identical to global node_id order")
    return cache_path


def provider_for_metric(spec: MetricSpec, args: argparse.Namespace, common_ids: np.ndarray) -> DistanceProvider:
    if spec.source_kind == "raw_global":
        matrix_path = spec_path(spec, args)
        row_space = "global"
    elif spec.source_kind == "rng_global_cache":
        matrix_path = validate_existing_rng_cache(spec, args)
        row_space = "global"
    elif spec.source_kind == "knn_graph":
        matrix_path = knn_cache_path(Path(args.workspace), spec.key)
        checkpoint_path = knn_checkpoint_path(Path(args.workspace), spec.key)
        if not matrix_path.exists() or not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing k-NN cache for {spec.key}; run prepare-graph-distances first")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if int(checkpoint.get("n_completed", -1)) != common_ids.size:
            raise ValueError(f"The k-NN cache for {spec.key} is incomplete")
        if checkpoint.get("common_node_fingerprint") != ordered_int_fingerprint(common_ids):
            raise ValueError(f"The k-NN cache for {spec.key} uses a different sink set")
        matrix_path = matrix_path
        row_space = "local"
    else:
        raise ValueError(spec.source_kind)
    return DistanceProvider(
        matrix_path=matrix_path,
        common_ids=common_ids,
        row_space=row_space,
        conservative_float32=True,
    )


def exact_diametrical_pair(provider: DistanceProvider, progress_every: int = 500) -> tuple[int, int, float]:
    best_distance = -math.inf
    best_i = -1
    best_j = -1
    for i in range(provider.n_sinks - 1):
        row = provider.row(i)
        row[: i + 1] = -math.inf
        j = int(np.argmax(row))
        distance = float(row[j])
        if distance > best_distance:
            best_distance = distance
            best_i = i
            best_j = j
        if progress_every and (i + 1) % progress_every == 0:
            log(f"  diameter scan {i + 1:,}/{provider.n_sinks - 1:,}; current maximum={best_distance:.8g}")
    if best_i < 0 or not np.isfinite(best_distance):
        raise ValueError("Could not find a finite diametrical pair")
    return best_i, best_j, best_distance


def farthest_first_lower_bound(
    provider: DistanceProvider,
    first: int,
    second: int,
    diameter: float,
    progress_every: int = 500,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    n = provider.n_sinks
    selected = np.zeros(n, dtype=bool)
    selected[first] = True
    selected[second] = True
    nearest = np.minimum(provider.row(first), provider.row(second))
    nearest[selected] = -math.inf
    prefix_min = float(diameter)
    cumulative = float(diameter)
    rows: list[dict[str, Any]] = [
        {
            "rank": 1,
            "local_index": first,
            "node_id": int(provider.common_ids[first]),
            "insertion_distance": math.nan,
            "prefix_min_distance": math.nan,
            "bound_contribution": 0.0,
            "cumulative_lower_bound": 0.0,
        },
        {
            "rank": 2,
            "local_index": second,
            "node_id": int(provider.common_ids[second]),
            "insertion_distance": diameter,
            "prefix_min_distance": diameter,
            "bound_contribution": diameter,
            "cumulative_lower_bound": cumulative,
        },
    ]
    first_zero_rank: int | None = None

    for rank in range(3, n + 1):
        chosen = int(np.argmax(nearest))
        insertion_distance = float(nearest[chosen])
        if not np.isfinite(insertion_distance) or insertion_distance < 0:
            raise ValueError(f"Invalid farthest-first insertion distance at rank {rank}: {insertion_distance}")
        prefix_min = min(prefix_min, insertion_distance)
        contribution = 0.5 * prefix_min
        cumulative += contribution
        if prefix_min == 0.0 and first_zero_rank is None:
            first_zero_rank = rank
        rows.append(
            {
                "rank": rank,
                "local_index": chosen,
                "node_id": int(provider.common_ids[chosen]),
                "insertion_distance": insertion_distance,
                "prefix_min_distance": prefix_min,
                "bound_contribution": contribution,
                "cumulative_lower_bound": cumulative,
            }
        )
        selected[chosen] = True
        distances = provider.row(chosen)
        np.minimum(nearest, distances, out=nearest)
        nearest[selected] = -math.inf
        if progress_every and rank % progress_every == 0:
            log(f"  farthest-first {rank:,}/{n:,}; cumulative lower bound={cumulative:.8g}")

    ordering = pd.DataFrame(rows)
    result = {
        "diameter": diameter,
        "diameter_first_local_index": first,
        "diameter_second_local_index": second,
        "diameter_first_node_id": int(provider.common_ids[first]),
        "diameter_second_node_id": int(provider.common_ids[second]),
        "constructive_lower_bound": cumulative,
        "lower_bound_over_diameter": cumulative / diameter if diameter > 0 else math.nan,
        "n_positive_prefix_min_ranks": int(np.count_nonzero(ordering["prefix_min_distance"].fillna(0) > 0)),
        "first_zero_prefix_min_rank": first_zero_rank,
    }
    return ordering, result


def plot_lower_bound_bars(summary: pd.DataFrame, out_path: Path) -> None:
    family_colors = {"Hamming": "#2166AC", "Embedding cityblock": "#D6604D"}
    construction_alpha = {"Raw": 1.0, "k-NN (k=5)": 0.42, "k-NN (k=50)": 0.66, "RNG": 0.82}
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    for ax, family in zip(axes, ["Hamming", "Embedding cityblock"], strict=True):
        group = summary[summary["metric_family"] == family]
        labels = group["construction"].tolist()
        values = group["constructive_lower_bound"].to_numpy(dtype=float)
        bars = ax.bar(
            np.arange(len(group)),
            values,
            color=[matplotlib.colors.to_rgba(family_colors[family], construction_alpha[label]) for label in labels],
            edgecolor=family_colors[family],
            linewidth=1.2,
        )
        ax.bar_label(bars, labels=[f"{value:,.4g}" for value in values], padding=3, fontsize=9)
        ax.set_xticks(np.arange(len(group)), labels)
        ax.set_title(family)
        ax.set_ylabel("Constructive lower bound")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#E1E1E1", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("Zero-skew tree constructive lower bound by distance metric", fontsize=15)
    fig.text(
        0.5,
        0.01,
        f"Same {int(summary['n_sinks'].iloc[0]):,} sink-node policy; "
        "weighted graph shortest paths may traverse other panel nodes",
        ha="center",
        fontsize=9.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def downsample_ordering(ordering: pd.DataFrame, max_points: int = 1200) -> pd.DataFrame:
    if ordering.shape[0] <= max_points:
        return ordering
    indices = np.unique(np.linspace(0, ordering.shape[0] - 1, max_points).round().astype(int))
    return ordering.iloc[indices]


def plot_normalized_cumulative(orderings: dict[str, pd.DataFrame], summary: pd.DataFrame, out_path: Path) -> None:
    colors = {"Hamming": "#2166AC", "Embedding cityblock": "#D6604D"}
    styles = {"Raw": "-", "k-NN (k=5)": ":", "k-NN (k=50)": "--", "RNG": "-."}
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), sharex=True, sharey=True)
    for ax, family in zip(axes, ["Hamming", "Embedding cityblock"], strict=True):
        family_summary = summary[summary["metric_family"] == family]
        for row in family_summary.itertuples(index=False):
            ordering = downsample_ordering(orderings[row.metric_key])
            ax.plot(
                ordering["rank"] / row.n_sinks,
                ordering["cumulative_lower_bound"] / row.constructive_lower_bound,
                color=colors[family],
                linestyle=styles[row.construction],
                linewidth=2,
                label=row.construction,
            )
        ax.set_title(family)
        ax.set_xlabel("Fraction of sinks ordered")
        ax.set_ylabel("Fraction of final constructive lower bound")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(color="#E1E1E1", linewidth=0.8)
        ax.legend(frameon=False)
    fig.suptitle("Accumulation of the zero-skew constructive lower bound", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def command_prepare(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    common_ids = load_common_node_ids(Path(args.common_node_ids), args.max_sinks)
    keys = parse_metric_keys(args.metrics, KNN_KEYS)
    if common_ids.max() >= args.expected_total_nodes:
        raise ValueError("Common node IDs exceed --expected-total-nodes")
    for key in keys:
        prepare_one_knn_cache(SPEC_BY_KEY[key], args, common_ids)


def command_compute(args: argparse.Namespace) -> None:
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    common_ids = load_common_node_ids(Path(args.common_node_ids), args.max_sinks)
    common_hash = ordered_int_fingerprint(common_ids)
    keys = parse_metric_keys(args.metrics, ALL_KEYS)
    summary_rows: list[dict[str, Any]] = []
    orderings: dict[str, pd.DataFrame] = {}

    for key in keys:
        spec = SPEC_BY_KEY[key]
        log(f"Computing constructive lower bound for {spec.label}")
        start = time.perf_counter()
        provider = provider_for_metric(spec, args, common_ids)
        first, second, diameter = exact_diametrical_pair(provider, progress_every=args.progress_every)
        log(
            f"  diametrical pair node IDs {common_ids[first]} and {common_ids[second]}; "
            f"distance={diameter:.8g}"
        )
        ordering, result = farthest_first_lower_bound(
            provider,
            first=first,
            second=second,
            diameter=diameter,
            progress_every=args.progress_every,
        )
        ordering.insert(0, "metric_key", spec.key)
        ordering.insert(1, "metric_label", spec.label)
        ordering_path = workspace / "orderings" / f"{spec.key}_diametrical_farthest_first_ordering.csv"
        ordering_path.parent.mkdir(parents=True, exist_ok=True)
        ordering.to_csv(ordering_path, index=False)
        orderings[spec.key] = ordering
        elapsed = time.perf_counter() - start
        row = {
            "metric_key": spec.key,
            "metric_label": spec.label,
            "metric_family": spec.family,
            "construction": spec.construction,
            "distance_definition": "raw pairwise" if spec.source_kind == "raw_global" else "weighted graph shortest path",
            "n_sinks": int(common_ids.size),
            **result,
            "elapsed_seconds": elapsed,
            "distance_source": str(provider.matrix_path),
            "distance_source_signature": json.dumps(file_signature(provider.matrix_path), sort_keys=True),
            "positive_float32_adjustment": "one ULP toward -infinity" if provider.conservative_float32 else "none",
            "ordering_path": str(ordering_path),
        }
        summary_rows.append(row)
        write_json(
            workspace / "metric_results" / f"{spec.key}_constructive_lower_bound.json",
            {
                **row,
                "common_node_fingerprint": common_hash,
                "ordering_rule": (
                    "exact diametrical pair in the conservative sink-distance matrix, then maximize "
                    "distance to the selected set; ties use lowest local index"
                ),
                "bound_formula": "d(s1,s2) + 0.5 * sum_{k=3}^n MinDist({s1,...,sk})",
            },
        )
        log(f"  {spec.label}: lower bound={result['constructive_lower_bound']:.8g} in {elapsed:.1f}s")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(workspace / "zero_skew_constructive_lower_bound_summary.csv", index=False)
    combined = pd.concat(orderings.values(), ignore_index=True)
    combined.to_csv(workspace / "zero_skew_constructive_lower_bound_orderings.csv", index=False)
    if set(keys) == set(ALL_KEYS):
        ordered_summary = summary.set_index("metric_key").loc[list(ALL_KEYS)].reset_index()
        plot_lower_bound_bars(ordered_summary, workspace / "zero_skew_constructive_lower_bound_by_metric.png")
        plot_normalized_cumulative(
            orderings,
            ordered_summary,
            workspace / "zero_skew_constructive_lower_bound_cumulative_normalized.png",
        )
    write_json(
        workspace / "zero_skew_constructive_lower_bound_manifest.json",
        {
            "paper": {
                "title": "Practical Approximation Algorithms for Zero- and Bounded-Skew Trees",
                "authors": "Alexander Z. Zelikovsky and Ion I. Mandoiu",
                "doi": "10.1137/S0895480100378367",
                "lemma": "Lemma 2.1",
            },
            "metric_keys": keys,
            "n_sinks": int(common_ids.size),
            "common_node_ids_path": str(args.common_node_ids),
            "common_node_fingerprint": common_hash,
            "shortest_path_policy": "weighted shortest paths on each original graph; paths may traverse non-sink panel nodes",
            "ordering_rule": (
                "exact diametrical pair in the conservative sink-distance matrix, "
                "followed by deterministic farthest-first ordering"
            ),
            "bound_formula": "d(s1,s2) + 0.5 * sum_{k=3}^n MinDist({s1,...,sk})",
            "float32_policy": "positive float32 distances are shifted down one ULP before use for a conservative bound",
            "max_sinks": args.max_sinks,
        },
    )
    log(f"Wrote constructive lower-bound outputs to {workspace}")


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workspace",
        default="analysis/cohort_validation/15_seed42_20k/zero_skew_constructive_lower_bound/hamming_embedding_knn05_knn50_rng",
    )
    parser.add_argument("--source-root", default="analysis/cohort_validation/07_sampling_design_20k")
    parser.add_argument("--panel", default="random_full_dataset_seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-label", default="pool_n20000")
    parser.add_argument(
        "--common-node-ids",
        default=(
            "analysis/cohort_validation/15_seed42_20k/graph_box_counting/"
            "hamming_embedding_knn05_knn50_rng/common_node_ids.csv"
        ),
    )
    parser.add_argument("--expected-total-nodes", type=int, default=20000)
    parser.add_argument("--max-sinks", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-graph-distances", help="Create resumable k-NN shortest-path caches")
    add_shared_arguments(prepare)
    prepare.add_argument("--metrics", default=",".join(KNN_KEYS))
    prepare.add_argument("--batch-size", type=int, default=32)
    prepare.add_argument("--force", action="store_true")
    prepare.set_defaults(func=command_prepare)

    compute = subparsers.add_parser("compute", help="Compute the constructive lower bound from distance rows")
    add_shared_arguments(compute)
    compute.add_argument("--metrics", default="all")
    compute.add_argument(
        "--rng-cache-root",
        default=(
            "analysis/cohort_validation/16_seed42_20k_kmedoids/random_full_dataset_seed42/"
            "seed_42/distance_rows"
        ),
    )
    compute.add_argument(
        "--rng-candidate-csv",
        default=(
            "analysis/cohort_validation/16_seed42_20k_kmedoids/random_full_dataset_seed42/"
            "seed_42/design/candidate_pool.csv"
        ),
    )
    compute.add_argument("--progress-every", type=int, default=500)
    compute.set_defaults(func=command_compute)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
