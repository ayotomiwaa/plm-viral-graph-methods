#!/usr/bin/env python3
"""Local PCA dimension validation for seed-42 RNG f_j refinement.

For each selected node, this script compares local displacement spectra from:

1. original ESM-2 cityblock RNG neighbors;
2. f_j-refined ESM-2 cityblock RNG neighbors;
3. degree-matched random pruning of the original ESM-2 RNG neighborhood;
4. Hamming RNG neighbors represented as aligned-sequence one-hot differences.

The random-pruned condition controls for the fact that PCA rank is constrained by
neighborhood size.  Sequence strings are read locally only; outputs contain
aggregate dimensions, degrees, and eigenvalue fractions, not sequence content.
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
from scipy import stats
from scipy.sparse import load_npz

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_PANEL_ROOT = Path(
    "analysis/cohort_validation/07_sampling_design_20k/random_full_dataset_seed42/seed_42"
)
DEFAULT_DIRECTIONAL_ROOT = Path(
    "analysis/cohort_validation/24_seed42_20k_directional_intrinsic_distances/"
    "random_full_dataset_seed42/seed_42"
)
DEFAULT_OUT_ROOT = Path(
    "analysis/cohort_validation/28_seed42_20k_local_pca_dimension/"
    "random_full_dataset_seed42/seed_42"
)
DEFAULT_CANDIDATE_LABEL = "candidate_0p1_delta_0p01"


@dataclass(frozen=True)
class Spectrum:
    eigenvalues: np.ndarray
    degree_observed: int
    degree_used: int
    subsampled: bool


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**32)


def canonical_nodes_path(args: argparse.Namespace) -> Path:
    return args.panel_root / "graphs/esm2_650M/cityblock/pool_n20000/canonical_nodes.csv"


def embedding_path(args: argparse.Namespace) -> Path:
    return args.panel_root / "embeddings/esm2_650M/pool_n20000/embeddings.npy"


def alignment_path(args: argparse.Namespace) -> Path:
    return args.panel_root / "inputs/pool_n20000/spike_sequences_aligned_mafft.fasta"


def original_embedding_rng_adj(args: argparse.Namespace) -> Path:
    return (
        args.panel_root
        / "graphs/esm2_650M/cityblock/pool_n20000/embedding_rng_exact/adj.npz"
    )


def refined_embedding_rng_adj(args: argparse.Namespace) -> Path:
    return (
        args.directional_root
        / "refined_graphs"
        / args.candidate_label
        / "rng/adj.npz"
    )


def hamming_rng_adj(args: argparse.Namespace) -> Path:
    return args.panel_root / "graphs/hamming/pool_n20000/hamming_rng_exact/adj.npz"


def load_canonical_nodes(args: argparse.Namespace) -> pd.DataFrame:
    nodes = pd.read_csv(canonical_nodes_path(args), low_memory=False)
    nodes = nodes.sort_values("node_id").reset_index(drop=True)
    expected = np.arange(len(nodes), dtype=np.int64)
    if not np.array_equal(nodes["node_id"].to_numpy(dtype=np.int64), expected):
        raise ValueError(f"{canonical_nodes_path(args)}: node_id is not row-aligned")
    nodes["accession"] = nodes["accession"].astype(str).str.strip()
    if nodes["accession"].duplicated().any():
        raise ValueError("canonical accessions are not unique")
    return nodes


def read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    header: str | None = None
    chunks: list[str] = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n\r")
            if line.startswith(">"):
                if header is not None:
                    records[header] = "".join(chunks).upper()
                header = line[1:].strip().split()[0]
                chunks = []
            elif line:
                chunks.append(line.strip())
    if header is not None:
        records[header] = "".join(chunks).upper()
    if not records:
        raise ValueError(f"{path}: no FASTA records")
    return records


def encode_alignment(accessions: Iterable[str], records: dict[str, str]) -> tuple[np.ndarray, dict[str, Any]]:
    accessions = list(accessions)
    missing = [accession for accession in accessions if accession not in records]
    if missing:
        raise ValueError(f"{len(missing):,} accessions missing from alignment; examples={missing[:5]}")
    widths = {len(records[accession]) for accession in accessions}
    if len(widths) != 1:
        raise ValueError(f"aligned sequences are ragged; widths={sorted(widths)}")
    width = widths.pop()
    alphabet = sorted({char for accession in accessions for char in records[accession]})
    code_of = {char: index for index, char in enumerate(alphabet)}
    encoded = np.empty((len(accessions), width), dtype=np.uint8)
    for row, accession in enumerate(accessions):
        encoded[row] = np.fromiter((code_of[char] for char in records[accession]), dtype=np.uint8, count=width)
    qc = {
        "alignment_width": int(width),
        "alphabet_size": int(len(alphabet)),
        "alphabet_symbols": "".join(alphabet),
    }
    return encoded, qc


def adjacency_neighbors(adj_path: Path) -> tuple[list[np.ndarray], dict[str, Any]]:
    adj = load_npz(adj_path).tocsr()
    adj.sort_indices()
    if adj.shape[0] != adj.shape[1]:
        raise ValueError(f"{adj_path}: adjacency is not square")
    if np.any(adj.diagonal() != 0):
        raise ValueError(f"{adj_path}: adjacency contains self-loops")
    asymmetry = (adj != adj.T).nnz
    if asymmetry:
        raise ValueError(f"{adj_path}: adjacency is not symmetric ({asymmetry:,} unequal entries)")
    neighbors = [
        adj.indices[adj.indptr[node] : adj.indptr[node + 1]].astype(np.int64, copy=True)
        for node in range(adj.shape[0])
    ]
    degree = np.array([len(item) for item in neighbors], dtype=np.int64)
    qc = {
        "adjacency": file_signature(adj_path),
        "n_nodes": int(adj.shape[0]),
        "n_edges": int(adj.nnz // 2),
        "mean_degree": float(degree.mean()) if degree.size else 0.0,
        "median_degree": float(np.median(degree)) if degree.size else 0.0,
        "max_degree": int(degree.max()) if degree.size else 0,
    }
    return neighbors, qc


def validate_refined_subset(
    original_neighbors: list[np.ndarray],
    refined_neighbors: list[np.ndarray],
) -> dict[str, Any]:
    if len(original_neighbors) != len(refined_neighbors):
        raise ValueError("original and refined RNG have different node counts")
    extra_directed_edges = 0
    retained_directed_edges = 0
    for original, refined in zip(original_neighbors, refined_neighbors):
        original_set = set(np.asarray(original, dtype=np.int64).tolist())
        refined_set = set(np.asarray(refined, dtype=np.int64).tolist())
        extra_directed_edges += len(refined_set - original_set)
        retained_directed_edges += len(refined_set)
    if extra_directed_edges:
        raise ValueError(
            "refined RNG is not an edge-subset of the original RNG: "
            f"{extra_directed_edges:,} extra directed incidences"
        )
    return {
        "edge_subset_confirmed": True,
        "retained_undirected_edges": int(retained_directed_edges // 2),
        "removed_undirected_edges": int(
            (sum(len(item) for item in original_neighbors) - retained_directed_edges) // 2
        ),
    }


def choose_nodes(
    original_neighbors: list[np.ndarray],
    refined_neighbors: list[np.ndarray],
    min_degree: int,
    node_set: str,
    max_nodes: int | None,
    seed: int,
) -> np.ndarray:
    original_degree = np.array([len(item) for item in original_neighbors], dtype=np.int64)
    refined_degree = np.array([len(item) for item in refined_neighbors], dtype=np.int64)
    if node_set == "affected":
        mask = (refined_degree < original_degree) & (refined_degree >= min_degree)
    elif node_set == "all":
        mask = (original_degree >= min_degree) & (refined_degree >= min_degree)
    else:
        raise ValueError(f"Unknown node_set: {node_set}")
    nodes = np.flatnonzero(mask)
    if max_nodes is not None and max_nodes < len(nodes):
        rng = np.random.default_rng(seed)
        nodes = np.sort(rng.choice(nodes, size=max_nodes, replace=False))
    return nodes.astype(np.int64)


def deterministic_subset(values: np.ndarray, size: int, seed: int) -> tuple[np.ndarray, bool]:
    values = np.asarray(values, dtype=np.int64)
    if size >= len(values):
        return values, False
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(values, size=size, replace=False)), True


def eigenvalues_from_gram(gram: np.ndarray) -> np.ndarray:
    gram = 0.5 * (gram + gram.T)
    values = np.linalg.eigvalsh(gram).astype(np.float64, copy=False)
    values = values[::-1]
    tolerance = max(1e-10, float(np.max(values, initial=0.0)) * 1e-10)
    values[np.abs(values) <= tolerance] = 0.0
    if np.any(values < 0.0):
        values = np.maximum(values, 0.0)
    return values


def embedding_spectrum(
    embeddings: np.ndarray,
    node: int,
    neighbors: np.ndarray,
    max_degree: int,
    seed: int,
) -> Spectrum:
    used, subsampled = deterministic_subset(neighbors, min(len(neighbors), max_degree), seed)
    if len(used) == 0:
        return Spectrum(np.empty(0, dtype=np.float64), len(neighbors), 0, subsampled)
    V = np.asarray(embeddings[used], dtype=np.float64) - np.asarray(embeddings[node], dtype=np.float64)
    gram = V @ V.T
    return Spectrum(eigenvalues_from_gram(gram), len(neighbors), len(used), subsampled)


def onehot_spectrum(
    encoded: np.ndarray,
    node: int,
    neighbors: np.ndarray,
    max_degree: int,
    seed: int,
    chunk_size: int,
) -> Spectrum:
    used, subsampled = deterministic_subset(neighbors, min(len(neighbors), max_degree), seed)
    k = len(used)
    if k == 0:
        return Spectrum(np.empty(0, dtype=np.float64), len(neighbors), 0, subsampled)
    center = encoded[node]
    neighbor_codes = encoded[used]
    width = int(encoded.shape[1])
    center_matches = np.count_nonzero(neighbor_codes == center[None, :], axis=1).astype(np.float64)
    gram = np.empty((k, k), dtype=np.float64)
    for start in range(0, k, chunk_size):
        stop = min(start + chunk_size, k)
        matches = np.count_nonzero(
            neighbor_codes[start:stop, None, :] == neighbor_codes[None, :, :],
            axis=2,
        ).astype(np.float64)
        gram[start:stop] = matches - center_matches[start:stop, None] - center_matches[None, :] + width
    return Spectrum(eigenvalues_from_gram(gram), len(neighbors), k, subsampled)


def dimension_metrics(eigenvalues: np.ndarray, thresholds: tuple[float, ...] = (0.90, 0.95)) -> dict[str, Any]:
    values = np.asarray(eigenvalues, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0.0)]
    total = float(values.sum())
    out: dict[str, Any] = {
        "rank_positive": int(values.size),
        "variance_total": total,
    }
    if values.size == 0 or total <= 0.0:
        for threshold in thresholds:
            out[f"d{int(round(threshold * 100))}"] = 0
        out["d_pr"] = 0.0
        out["lambda1_fraction"] = math.nan
        out["lambda2_fraction"] = math.nan
        out["lambda3_fraction"] = math.nan
        return out
    cumulative = np.cumsum(values) / total
    for threshold in thresholds:
        out[f"d{int(round(threshold * 100))}"] = int(np.searchsorted(cumulative, threshold, side="left") + 1)
    denom = float(np.sum(values * values))
    out["d_pr"] = float(total * total / denom) if denom > 0.0 else 0.0
    fractions = values / total
    for index in range(3):
        out[f"lambda{index + 1}_fraction"] = float(fractions[index]) if index < fractions.size else 0.0
    return out


def metric_row(
    node: int,
    condition: str,
    spectrum: Spectrum,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "node_id": int(node),
        "condition": condition,
        "degree_observed": int(spectrum.degree_observed),
        "degree_used": int(spectrum.degree_used),
        "subsampled_for_svd": bool(spectrum.subsampled),
    }
    row.update(dimension_metrics(spectrum.eigenvalues))
    if spectrum.degree_used > 0:
        for key in ["d90", "d95", "d_pr"]:
            row[f"{key}_fraction_of_degree_used"] = float(row[key] / spectrum.degree_used)
    else:
        for key in ["d90", "d95", "d_pr"]:
            row[f"{key}_fraction_of_degree_used"] = math.nan
    if extra:
        row.update(extra)
    return row


def spectrum_rows(
    node: int,
    condition: str,
    spectrum: Spectrum,
    top_components: int,
) -> list[dict[str, Any]]:
    values = np.asarray(spectrum.eigenvalues, dtype=np.float64)
    positive = values[values > 0.0]
    total = float(positive.sum())
    if total <= 0.0:
        return []
    fractions = positive / total
    cumulative = np.cumsum(fractions)
    rows = []
    for index in range(min(top_components, fractions.size)):
        rows.append(
            {
                "node_id": int(node),
                "condition": condition,
                "pc_index": int(index + 1),
                "variance_fraction": float(fractions[index]),
                "cumulative_variance_fraction": float(cumulative[index]),
                "degree_used": int(spectrum.degree_used),
            }
        )
    return rows


def random_pruned_summary(
    embeddings: np.ndarray,
    node: int,
    original_neighbors: np.ndarray,
    target_degree: int,
    max_degree: int,
    replicates: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    if target_degree <= 0 or len(original_neighbors) < target_degree:
        return {}, rows
    for replicate in range(replicates):
        spectrum = embedding_spectrum(
            embeddings,
            node=node,
            neighbors=original_neighbors,
            max_degree=min(target_degree, max_degree),
            seed=stable_seed(seed, "random_pruned", node, replicate),
        )
        row = metric_row(
            node,
            "embedding_rng_random_pruned",
            spectrum,
            {"random_prune_replicate": int(replicate)},
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    summary: dict[str, Any] = {
        "node_id": int(node),
        "condition": "embedding_rng_random_pruned_mean",
        "degree_observed": int(len(original_neighbors)),
        "degree_used": int(target_degree),
        "subsampled_for_svd": bool(target_degree > max_degree),
        "random_prune_replicates": int(replicates),
    }
    for column in [
        "d90",
        "d95",
        "d_pr",
        "d90_fraction_of_degree_used",
        "d95_fraction_of_degree_used",
        "d_pr_fraction_of_degree_used",
        "lambda1_fraction",
        "lambda2_fraction",
        "lambda3_fraction",
    ]:
        values = pd.to_numeric(frame[column], errors="coerce")
        summary[column] = float(values.mean()) if values.notna().any() else math.nan
        summary[f"{column}_q10"] = float(values.quantile(0.10)) if values.notna().any() else math.nan
        summary[f"{column}_q90"] = float(values.quantile(0.90)) if values.notna().any() else math.nan
    return summary, rows


def summarize_conditions(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition, group in frame.groupby("condition", sort=False):
        row: dict[str, Any] = {
            "condition": condition,
            "nodes_analyzed": int(group["node_id"].nunique()),
            "median_degree_used": float(pd.to_numeric(group["degree_used"], errors="coerce").median()),
            "median_degree_observed": float(pd.to_numeric(group["degree_observed"], errors="coerce").median()),
            "fraction_subsampled_for_svd": float(group["subsampled_for_svd"].astype(bool).mean()),
        }
        for metric in ["d_pr", "d90", "d95", "d_pr_fraction_of_degree_used", "lambda1_fraction"]:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"median_{metric}"] = float(values.median()) if len(values) else math.nan
            row[f"q25_{metric}"] = float(values.quantile(0.25)) if len(values) else math.nan
            row[f"q75_{metric}"] = float(values.quantile(0.75)) if len(values) else math.nan
            row[f"iqr_{metric}"] = row[f"q75_{metric}"] - row[f"q25_{metric}"] if len(values) else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def paired_contrast(
    frame: pd.DataFrame,
    left: str,
    right: str,
    metrics: list[str],
) -> pd.DataFrame:
    left_frame = frame[frame["condition"] == left].set_index("node_id")
    right_frame = frame[frame["condition"] == right].set_index("node_id")
    common = left_frame.index.intersection(right_frame.index)
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        diff = (
            pd.to_numeric(left_frame.loc[common, metric], errors="coerce")
            - pd.to_numeric(right_frame.loc[common, metric], errors="coerce")
        ).dropna()
        values = diff.to_numpy(dtype=np.float64)
        row: dict[str, Any] = {
            "contrast": f"{left}_minus_{right}",
            "left_condition": left,
            "right_condition": right,
            "metric": metric,
            "n_nodes": int(values.size),
            "median_difference": float(np.median(values)) if values.size else math.nan,
            "mean_difference": float(values.mean()) if values.size else math.nan,
            "n_negative": int(np.count_nonzero(values < 0.0)),
            "n_positive": int(np.count_nonzero(values > 0.0)),
            "n_zero": int(np.count_nonzero(values == 0.0)),
        }
        nonzero = values[values != 0.0]
        if nonzero.size:
            row["sign_test_p_left_less"] = float(
                stats.binomtest(int(np.count_nonzero(nonzero < 0.0)), nonzero.size, 0.5, alternative="greater").pvalue
            )
        else:
            row["sign_test_p_left_less"] = math.nan
        if values.size >= 2 and np.any(values != 0.0):
            try:
                test = stats.wilcoxon(values, zero_method="wilcox", alternative="less")
                row["wilcoxon_p_left_less"] = float(test.pvalue)
            except ValueError:
                row["wilcoxon_p_left_less"] = math.nan
        else:
            row["wilcoxon_p_left_less"] = math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    if args.min_degree < 1:
        raise ValueError("--min-degree must be at least 1")
    if args.max_degree_for_svd < args.min_degree:
        raise ValueError("--max-degree-for-svd must be at least --min-degree")
    if args.random_prune_replicates < 1:
        raise ValueError("--random-prune-replicates must be at least 1")
    if args.onehot_chunk_size < 1:
        raise ValueError("--onehot-chunk-size must be at least 1")
    args.out_root.mkdir(parents=True, exist_ok=True)
    nodes = load_canonical_nodes(args)
    embeddings = np.load(embedding_path(args), mmap_mode="r")
    if embeddings.shape[0] != len(nodes):
        raise ValueError(f"embedding row count {embeddings.shape[0]} != node count {len(nodes)}")

    original_neighbors, original_qc = adjacency_neighbors(original_embedding_rng_adj(args))
    refined_neighbors, refined_qc = adjacency_neighbors(refined_embedding_rng_adj(args))
    hamming_neighbors, hamming_qc = adjacency_neighbors(hamming_rng_adj(args))
    if len(original_neighbors) != len(nodes) or len(refined_neighbors) != len(nodes) or len(hamming_neighbors) != len(nodes):
        raise ValueError("one or more graph adjacencies do not match canonical node count")
    refinement_qc = validate_refined_subset(original_neighbors, refined_neighbors)

    selected_nodes = choose_nodes(
        original_neighbors,
        refined_neighbors,
        min_degree=args.min_degree,
        node_set=args.node_set,
        max_nodes=args.max_nodes,
        seed=args.node_seed,
    )
    log(f"Selected {len(selected_nodes):,} nodes for local PCA (node_set={args.node_set})")
    if len(selected_nodes) == 0:
        raise ValueError("no nodes satisfy the requested node-set and degree criteria")

    original_degree = np.array([len(item) for item in original_neighbors], dtype=np.int64)
    refined_degree = np.array([len(item) for item in refined_neighbors], dtype=np.int64)
    hamming_degree = np.array([len(item) for item in hamming_neighbors], dtype=np.int64)
    node_set_frame = pd.DataFrame(
        {
            "node_id": selected_nodes,
            "embedding_rng_original_degree": original_degree[selected_nodes],
            "embedding_rng_refined_degree": refined_degree[selected_nodes],
            "embedding_rng_edges_removed": (
                original_degree[selected_nodes] - refined_degree[selected_nodes]
            ),
            "affected_by_refinement": refined_degree[selected_nodes] < original_degree[selected_nodes],
            "hamming_rng_degree": hamming_degree[selected_nodes],
        }
    )
    node_set_path = args.out_root / "local_pca_node_set.csv.gz"
    node_set_frame.to_csv(node_set_path, index=False)

    encoded: np.ndarray | None = None
    alignment_qc: dict[str, Any] | None = None
    if args.include_hamming:
        records = read_fasta(alignment_path(args))
        encoded, alignment_qc = encode_alignment(nodes["accession"].astype(str).tolist(), records)
        log(
            "Encoded alignment for Hamming one-hot PCA: "
            f"{encoded.shape[0]:,} sequences x {encoded.shape[1]:,} sites"
        )

    metric_rows: list[dict[str, Any]] = []
    spectrum_detail_rows: list[dict[str, Any]] = []
    random_rows: list[dict[str, Any]] = []

    for index, node in enumerate(selected_nodes, start=1):
        if args.progress_every and (index == 1 or index % args.progress_every == 0 or index == len(selected_nodes)):
            log(f"  local PCA node {index:,}/{len(selected_nodes):,}: node_id={int(node)}")
        node = int(node)
        original = embedding_spectrum(
            embeddings,
            node,
            original_neighbors[node],
            max_degree=args.max_degree_for_svd,
            seed=stable_seed(args.svd_seed, "embedding_original", node),
        )
        refined = embedding_spectrum(
            embeddings,
            node,
            refined_neighbors[node],
            max_degree=args.max_degree_for_svd,
            seed=stable_seed(args.svd_seed, "embedding_refined", node),
        )
        metric_rows.append(metric_row(node, "embedding_rng_original", original))
        metric_rows.append(metric_row(node, "embedding_rng_refined", refined))
        spectrum_detail_rows.extend(spectrum_rows(node, "embedding_rng_original", original, args.top_components))
        spectrum_detail_rows.extend(spectrum_rows(node, "embedding_rng_refined", refined, args.top_components))

        random_summary, replicate_rows = random_pruned_summary(
            embeddings,
            node,
            original_neighbors[node],
            target_degree=min(len(refined_neighbors[node]), args.max_degree_for_svd),
            max_degree=args.max_degree_for_svd,
            replicates=args.random_prune_replicates,
            seed=args.random_prune_seed,
        )
        if random_summary:
            metric_rows.append(random_summary)
            if args.write_random_replicates:
                random_rows.extend(replicate_rows)

        if args.include_hamming:
            if encoded is None:
                raise RuntimeError("encoded alignment was not loaded")
            hamming = onehot_spectrum(
                encoded,
                node,
                hamming_neighbors[node],
                max_degree=args.max_degree_for_svd,
                seed=stable_seed(args.svd_seed, "hamming", node),
                chunk_size=args.onehot_chunk_size,
            )
            metric_rows.append(metric_row(node, "hamming_rng_onehot", hamming))
            spectrum_detail_rows.extend(spectrum_rows(node, "hamming_rng_onehot", hamming, args.top_components))

    node_metrics = pd.DataFrame(metric_rows)
    spectrum_detail = pd.DataFrame(spectrum_detail_rows)
    out = args.out_root
    node_metrics_path = out / "local_pca_node_metrics.csv.gz"
    spectrum_detail_path = out / "local_pca_top_spectrum.csv.gz"
    node_metrics.to_csv(node_metrics_path, index=False)
    spectrum_detail.to_csv(spectrum_detail_path, index=False)
    if random_rows:
        pd.DataFrame(random_rows).to_csv(out / "local_pca_random_pruned_replicates.csv.gz", index=False)

    summary = summarize_conditions(node_metrics)
    summary.to_csv(out / "local_pca_condition_summary.csv", index=False)
    contrast_frames = []
    for left, right in [
        ("embedding_rng_refined", "embedding_rng_original"),
        ("embedding_rng_refined", "embedding_rng_random_pruned_mean"),
        ("embedding_rng_random_pruned_mean", "embedding_rng_original"),
    ]:
        contrast_frames.append(
            paired_contrast(node_metrics, left, right, ["d_pr", "d90", "d95", "d_pr_fraction_of_degree_used", "lambda1_fraction"])
        )
    contrasts = pd.concat(contrast_frames, ignore_index=True)
    contrasts.to_csv(out / "local_pca_paired_contrasts.csv", index=False)

    if not spectrum_detail.empty:
        spectrum_summary = (
            spectrum_detail.groupby(["condition", "pc_index"], as_index=False)
            .agg(
                median_variance_fraction=("variance_fraction", "median"),
                q25_variance_fraction=("variance_fraction", lambda x: float(pd.Series(x).quantile(0.25))),
                q75_variance_fraction=("variance_fraction", lambda x: float(pd.Series(x).quantile(0.75))),
                median_cumulative_variance_fraction=("cumulative_variance_fraction", "median"),
                n_nodes=("node_id", "nunique"),
            )
            .sort_values(["condition", "pc_index"])
        )
        spectrum_summary.to_csv(out / "local_pca_spectrum_summary.csv", index=False)

    write_json(
        out / "local_pca_manifest.json",
        {
            "completed_at_unix": time.time(),
            "sequence_content_written": False,
            "panel_root": args.panel_root,
            "directional_root": args.directional_root,
            "candidate_label": args.candidate_label,
            "node_set": args.node_set,
            "n_selected_nodes": int(len(selected_nodes)),
            "min_degree": int(args.min_degree),
            "max_degree_for_svd": int(args.max_degree_for_svd),
            "random_prune_replicates": int(args.random_prune_replicates),
            "local_pca_definition": "uncentered SVD/eigendecomposition of center-to-neighbor displacement rows",
            "degree_control": "random-pruned original RNG samples the same degree used by refined RNG after the SVD cap",
            "outputs": {
                "node_set": node_set_path,
                "node_metrics": node_metrics_path,
                "top_spectrum": spectrum_detail_path,
                "condition_summary": out / "local_pca_condition_summary.csv",
                "paired_contrasts": out / "local_pca_paired_contrasts.csv",
                "spectrum_summary": out / "local_pca_spectrum_summary.csv",
            },
            "inputs": {
                "canonical_nodes": file_signature(canonical_nodes_path(args)),
                "embeddings": file_signature(embedding_path(args)),
                "original_embedding_rng": original_qc,
                "refined_embedding_rng": refined_qc,
                "refinement_qc": refinement_qc,
                "hamming_rng": hamming_qc,
                "alignment": file_signature(alignment_path(args)) if args.include_hamming else None,
                "alignment_qc": alignment_qc,
            },
        },
    )
    log(f"Wrote local PCA node metrics: {node_metrics_path}")
    log(f"Wrote local PCA condition summary: {out / 'local_pca_condition_summary.csv'}")
    log(f"Wrote local PCA paired contrasts: {out / 'local_pca_paired_contrasts.csv'}")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel-root", type=Path, default=DEFAULT_PANEL_ROOT)
    ap.add_argument("--directional-root", type=Path, default=DEFAULT_DIRECTIONAL_ROOT)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--candidate-label", default=DEFAULT_CANDIDATE_LABEL)
    ap.add_argument("--node-set", choices=["affected", "all"], default="affected")
    ap.add_argument("--max-nodes", type=int, default=None)
    ap.add_argument("--node-seed", type=int, default=42)
    ap.add_argument("--min-degree", type=int, default=3)
    ap.add_argument("--max-degree-for-svd", type=int, default=256)
    ap.add_argument("--svd-seed", type=int, default=42)
    ap.add_argument("--random-prune-replicates", type=int, default=50)
    ap.add_argument("--random-prune-seed", type=int, default=42)
    ap.add_argument("--include-hamming", action="store_true", default=True)
    ap.add_argument("--no-include-hamming", dest="include_hamming", action="store_false")
    ap.add_argument("--onehot-chunk-size", type=int, default=64)
    ap.add_argument("--top-components", type=int, default=20)
    ap.add_argument("--write-random-replicates", action="store_true")
    ap.add_argument("--progress-every", type=int, default=250)
    return ap


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
