#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy.sparse import csr_matrix, save_npz
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial.distance import cdist

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


COHORT_DIRS = [
    "A_early_US_ancestral_B1x",
    "B_alpha_clean",
    "C_delta_dominant",
    "D_early_omicron_BA1_BA2_BA2121",
    "E1_BA4_BA5_era",
    "E2_BQ_XBB_transition",
    "E3_XBB_JN1_era",
    "F_beta_focused",
    "G_gamma_focused",
]

ACCESSION_RE = re.compile(r"EPI_ISL_\d+")
MISSING_LABELS = {"", "nan", "none", "null", "nat"}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_accession(value: str) -> str:
    parts = str(value).strip().split("|")
    for part in parts:
        part = part.strip()
        if part.startswith("EPI_ISL_"):
            return part
    match = ACCESSION_RE.search(str(value))
    return match.group(0) if match else str(value).strip()


def parse_virus_name_key(value: object) -> str:
    """Normalize FASTA and GISAID virus names for accession fallback joins."""
    text = str(value).strip()
    parts = [part.strip() for part in text.split("|")]
    if parts and parts[0] == "Spike" and len(parts) > 1:
        text = parts[1]
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"/\d{4}$", "", text)
    return text


def fill_accessions_from_original_metadata(ids: pd.DataFrame, input_dir: Path, meta_accessions: set[str]) -> pd.DataFrame:
    missing_mask = ~ids["accession"].isin(meta_accessions)
    if not missing_mask.any():
        return ids

    original_metadata_path = input_dir / "metadata_original.csv"
    if not original_metadata_path.exists():
        return ids

    original = pd.read_csv(
        original_metadata_path,
        usecols=["Virus name", "Accession ID"],
        low_memory=False,
    )
    original = original.dropna(subset=["Virus name", "Accession ID"]).copy()
    original["virus_name_key"] = original["Virus name"].map(parse_virus_name_key)
    original["accession_fallback"] = original["Accession ID"].astype(str).str.strip()

    counts = original["virus_name_key"].value_counts()
    unique_original = original[original["virus_name_key"].map(counts) == 1]
    virus_to_accession = dict(zip(unique_original["virus_name_key"], unique_original["accession_fallback"]))

    ids = ids.copy()
    ids.loc[missing_mask, "virus_name_key"] = ids.loc[missing_mask, "id_raw"].map(parse_virus_name_key)
    fallback = ids.loc[missing_mask, "virus_name_key"].map(virus_to_accession)
    usable = fallback.isin(meta_accessions)
    ids.loc[fallback[usable].index, "accession"] = fallback[usable]
    resolved = int(usable.sum())
    if resolved:
        log(f"Resolved {resolved:,} FASTA ids via metadata_original virus-name fallback")
    return ids


def fill_accessions_from_fasta_headers(ids: pd.DataFrame, input_dir: Path, meta_accessions: set[str]) -> pd.DataFrame:
    missing_mask = ~ids["accession"].isin(meta_accessions)
    if not missing_mask.any():
        return ids

    fasta_path = input_dir / "spike_sequences.fasta"
    if not fasta_path.exists():
        return ids

    headers = [
        line[1:].strip()
        for line in fasta_path.open("r", encoding="utf-8", errors="replace")
        if line.startswith(">")
    ]
    if len(headers) != len(ids):
        log(f"Skipping FASTA row-order accession fallback: {len(headers):,} FASTA headers != {len(ids):,} embedding ids")
        return ids

    fasta_accessions = pd.Series(headers, index=ids.index).map(parse_accession)
    usable = missing_mask & fasta_accessions.isin(meta_accessions)
    ids = ids.copy()
    ids.loc[usable, "accession"] = fasta_accessions.loc[usable]
    resolved = int(usable.sum())
    if resolved:
        log(f"Resolved {resolved:,} FASTA ids via row-aligned spike_sequences.fasta headers")
    return ids


def fix_partial_date(value: object) -> str:
    s = str(value)
    if s.endswith("-00-00"):
        return s[:4] + "-06-15"
    if s.endswith("-00"):
        return s[:7] + "-15"
    return s


def load_cohort_panel(cohort_root: Path, sample_label: str, embed_tag: str) -> tuple[np.ndarray, pd.DataFrame, dict[str, object]]:
    input_dir = cohort_root / "inputs" / sample_label
    embed_dir = cohort_root / "embeddings" / embed_tag / sample_label
    embeddings_path = embed_dir / "embeddings.npy"
    ids_path = embed_dir / "ids.txt"
    metadata_path = input_dir / "metadata.csv"
    manifest_path = embed_dir / "embed_manifest.json"

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Missing embeddings: {embeddings_path}")
    if not ids_path.exists():
        raise FileNotFoundError(f"Missing ids: {ids_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")

    X_all = np.load(embeddings_path, mmap_mode="r")
    ids = pd.DataFrame(
        {
            "embedding_row": np.arange(X_all.shape[0], dtype=int),
            "id_raw": [line.strip() for line in ids_path.open("r", encoding="utf-8", errors="replace") if line.strip()],
        }
    )
    if len(ids) != X_all.shape[0]:
        raise ValueError(f"ids rows {len(ids):,} != embedding rows {X_all.shape[0]:,} for {cohort_root}")
    ids["accession"] = ids["id_raw"].map(parse_accession)

    meta = pd.read_csv(metadata_path, low_memory=False)
    if "accession" not in meta.columns:
        raise ValueError(f"{metadata_path} must contain accession")
    meta = meta.copy()
    meta["accession"] = meta["accession"].astype(str).str.strip()
    ids = fill_accessions_from_original_metadata(ids, input_dir, set(meta["accession"]))
    ids = fill_accessions_from_fasta_headers(ids, input_dir, set(meta["accession"]))
    merged = ids.merge(meta, on="accession", how="inner", suffixes=("", "_meta"))
    if len(merged) != len(ids):
        missing = sorted(set(ids["accession"]) - set(merged["accession"]))
        raise ValueError(f"Only matched {len(merged):,}/{len(ids):,} ids to metadata in {cohort_root}; missing examples={missing[:5]}")

    row_idx = merged["embedding_row"].astype(int).to_numpy()
    X = np.asarray(X_all[row_idx], dtype=np.float32)
    dates = pd.to_datetime(merged["collection_date"].map(fix_partial_date), errors="coerce")
    nodes = pd.DataFrame(
        {
            "node_id": np.arange(len(merged), dtype=int),
            "accession": merged["accession"].astype(str).to_numpy(),
            "id_raw": merged["id_raw"].astype(str).to_numpy(),
            "collection_date": dates.dt.strftime("%Y-%m-%d").fillna("").to_numpy(),
            "collection_month": merged.get("collection_month", dates.dt.strftime("%Y-%m")).astype(str).to_numpy(),
            "lineage": merged.get("lineage", "").astype(str).to_numpy(),
            "within_lineage_label": merged.get("within_lineage_label", "").astype(str).to_numpy(),
            "region": merged.get("region", "").astype(str).to_numpy(),
            "location": merged.get("location", "").astype(str).to_numpy(),
            "cohort_id": merged.get("cohort_id", "").astype(str).to_numpy(),
            "cohort_name": merged.get("cohort_name", "").astype(str).to_numpy(),
            "embedding_row": row_idx,
        }
    )
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    log(f"Loaded {cohort_root.name}: n={X.shape[0]:,}, d={X.shape[1]:,}")
    return X, nodes, manifest


def compute_pairwise(X: np.ndarray, metric: str) -> np.ndarray:
    log(f"Computing pairwise {metric}: n={X.shape[0]:,}, d={X.shape[1]:,}")
    D = cdist(X, X, metric=metric).astype(np.float32, copy=False)
    np.fill_diagonal(D, np.inf)
    log(f"Distance matrix ready: shape={D.shape}, size={D.nbytes / 1e9:.2f} GB")
    return D


def load_or_compute_pairwise(
    X: np.ndarray,
    metric: str,
    cache_path: Path | None,
    refresh_cache: bool,
) -> np.ndarray:
    if cache_path is not None and cache_path.exists() and not refresh_cache:
        log(f"Loading cached pairwise {metric}: {cache_path}")
        D = np.load(cache_path)
        expected_shape = (X.shape[0], X.shape[0])
        if D.shape != expected_shape or D.dtype != np.float32:
            log(f"Ignoring invalid distance cache: shape={D.shape}, dtype={D.dtype}, expected={expected_shape}/float32")
        else:
            log(f"Distance matrix loaded: shape={D.shape}, size={D.nbytes / 1e9:.2f} GB")
            return D

    D = compute_pairwise(X, metric=metric)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(f"{cache_path.name}.tmp.{os.getpid()}")
        log(f"Saving distance matrix cache: {cache_path}")
        with tmp_path.open("wb") as handle:
            np.save(handle, D)
        tmp_path.replace(cache_path)
        with cache_path.with_suffix(cache_path.suffix + ".json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "metric": metric,
                    "dtype": str(D.dtype),
                    "shape": [int(D.shape[0]), int(D.shape[1])],
                    "size_gb": D.nbytes / 1e9,
                },
                handle,
                indent=2,
            )
    return D


def mst_edges(D: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log("Building MST")
    D_mst = np.asarray(D, dtype=np.float32).copy()
    np.fill_diagonal(D_mst, 0.0)
    mst = minimum_spanning_tree(D_mst).tocoo()
    sources = mst.row.astype(np.int32)
    targets = mst.col.astype(np.int32)
    weights = mst.data.astype(np.float32)
    mask = sources != targets
    return sources[mask], targets[mask], weights[mask]


def symmetric_knn_edges(D: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log(f"Building symmetric kNN k={k}")
    n = D.shape[0]
    idx = np.argpartition(D, kth=k - 1, axis=1)[:, :k]
    row_ids = np.arange(n)[:, None]
    dist = D[row_ids, idx]
    order = np.argsort(dist, axis=1, kind="stable")
    idx = np.take_along_axis(idx, order, axis=1)
    dist = np.take_along_axis(dist, order, axis=1)
    edges: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j, dij in zip(idx[i], dist[i]):
            j = int(j)
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            value = float(dij)
            previous = edges.get((a, b))
            if previous is None or value < previous:
                edges[(a, b)] = value
    items = sorted(edges.items())
    sources = np.array([i for (i, _), _ in items], dtype=np.int32)
    targets = np.array([j for (_, j), _ in items], dtype=np.int32)
    weights = np.array([w for _, w in items], dtype=np.float32)
    log(f"Symmetric kNN k={k} kept {len(sources):,} undirected edges")
    return sources, targets, weights


@njit(cache=True)
def exact_rng_rows_block(
    D: np.ndarray,
    order_block: np.ndarray,
    row_start: int,
    row_end: int,
    max_edges: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    sources = np.empty(max_edges, dtype=np.int32)
    targets = np.empty(max_edges, dtype=np.int32)
    weights = np.empty(max_edges, dtype=np.float32)
    kept = 0
    pruned = 0
    checked = 0
    for local_i in range(row_end - row_start):
        i = row_start + local_i
        oi = order_block[local_i]
        for j in range(i + 1, D.shape[0]):
            dij = D[i, j]
            witness = False
            for pos in range(D.shape[0]):
                k = oi[pos]
                if D[i, k] >= dij:
                    break
                if D[j, k] < dij:
                    witness = True
                    break
            checked += 1
            if witness:
                pruned += 1
            else:
                if kept >= max_edges:
                    return sources, targets, weights, kept, pruned, checked
                sources[kept] = i
                targets[kept] = j
                weights[kept] = float(dij)
                kept += 1
    return sources, targets, weights, kept, pruned, checked


def exact_rng_edges_blockwise_order(
    D: np.ndarray,
    row_block_size: int,
    max_block_edges: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    n = D.shape[0]
    candidate_edges = n * (n - 1) // 2
    sources_all = []
    targets_all = []
    weights_all = []
    checked_total = 0
    pruned_total = 0
    kept_total = 0
    t0 = time.time()
    for row_start in range(0, n - 1, row_block_size):
        row_end = min(row_start + row_block_size, n - 1)
        log(f"Sorting distance rows {row_start:,}-{row_end - 1:,} for exact RNG witness scan")
        order_block = np.argsort(D[row_start:row_end], axis=1).astype(np.int32, copy=False)
        sources, targets, weights, kept, pruned, checked = exact_rng_rows_block(
            D,
            order_block,
            row_start,
            row_end,
            max_block_edges,
        )
        if kept >= max_block_edges:
            raise RuntimeError(f"RNG block {row_start}:{row_end} reached max_block_edges={max_block_edges}")
        if kept:
            sources_all.append(sources[:kept].copy())
            targets_all.append(targets[:kept].copy())
            weights_all.append(weights[:kept].copy())
        checked_total += int(checked)
        pruned_total += int(pruned)
        kept_total += int(kept)
        elapsed = time.time() - t0
        rate = checked_total / elapsed if elapsed else 0.0
        eta = (candidate_edges - checked_total) / rate if rate else 0.0
        log(
            f"RNG rows {row_start:,}-{row_end - 1:,}: checked={checked_total:,}/{candidate_edges:,} "
            f"kept={kept_total:,} pruned={pruned_total:,} rate={rate:,.0f}/s eta={eta / 3600:.2f}h"
        )
    sources_out = np.concatenate(sources_all) if sources_all else np.array([], dtype=np.int32)
    targets_out = np.concatenate(targets_all) if targets_all else np.array([], dtype=np.int32)
    weights_out = np.concatenate(weights_all) if weights_all else np.array([], dtype=np.float32)
    return sources_out, targets_out, weights_out, checked_total, pruned_total


def edges_to_csr(n_nodes: int, sources: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> csr_matrix:
    rows = np.concatenate([sources, targets]).astype(np.int32, copy=False)
    cols = np.concatenate([targets, sources]).astype(np.int32, copy=False)
    data = np.concatenate([weights, weights]).astype(np.float32, copy=False)
    return csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float32)


def graph_stats(adj: csr_matrix) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    n_comp, labels = connected_components(adj, directed=False, return_labels=True)
    sizes = np.bincount(labels, minlength=n_comp) if len(labels) else np.array([], dtype=int)
    giant_label = int(sizes.argmax()) if len(sizes) else -1
    giant_size = int(sizes[giant_label]) if giant_label >= 0 else 0
    degree = np.asarray(adj.getnnz(axis=1)).ravel().astype(int)
    return labels, degree, {
        "n_nodes": int(adj.shape[0]),
        "n_edges": int(adj.nnz // 2),
        "n_components": int(n_comp),
        "giant_label": giant_label,
        "giant_component_size": giant_size,
        "giant_component_frac": float(giant_size / adj.shape[0]) if adj.shape[0] else 0.0,
        "mean_degree": float(degree.mean()) if len(degree) else 0.0,
        "median_degree": float(np.median(degree)) if len(degree) else 0.0,
        "max_degree": int(degree.max()) if len(degree) else 0,
    }


def label_metrics(nodes: pd.DataFrame, sources: np.ndarray, targets: np.ndarray, label_col: str) -> dict[str, object]:
    if label_col not in nodes.columns:
        return {}
    labels = nodes[label_col].astype(str).str.strip()
    valid = ~labels.str.casefold().isin(MISSING_LABELS)
    edge_mask = valid.to_numpy()[sources] & valid.to_numpy()[targets]
    src = sources[edge_mask]
    tgt = targets[edge_mask]
    m = int(len(src))
    prefix = f"{label_col}_"
    if m == 0:
        return {
            prefix + "n_edges_used": 0,
            prefix + "observed_same_fraction": np.nan,
            prefix + "nodepair_expected_same_fraction": np.nan,
            prefix + "nodepair_enrichment_ratio": np.nan,
            prefix + "n_labels": int(labels[valid].nunique()),
        }
    label_values = labels.to_numpy()
    same = np.fromiter((label_values[int(i)] == label_values[int(j)] for i, j in zip(src, tgt)), dtype=bool, count=m)
    observed = float(same.mean())
    counts = labels[valid].value_counts()
    n = int(counts.sum())
    total_pairs = n * (n - 1) / 2
    same_pairs = float(sum(count * (count - 1) / 2 for count in counts.astype(int)))
    expected = same_pairs / total_pairs if total_pairs else np.nan
    enrichment = observed / expected if expected and np.isfinite(expected) else np.nan
    return {
        prefix + "n_edges_used": m,
        prefix + "observed_same_fraction": observed,
        prefix + "nodepair_expected_same_fraction": expected,
        prefix + "nodepair_enrichment_ratio": enrichment,
        prefix + "n_labels": int(counts.size),
        prefix + "top_label": str(counts.index[0]) if len(counts) else "",
        prefix + "top_label_count": int(counts.iloc[0]) if len(counts) else 0,
    }


def write_graph(
    out_dir: Path,
    nodes: pd.DataFrame,
    sources: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    graph_type: str,
    metric: str,
    k: int | None,
    cohort_dir: str,
    sample_label: str,
    embed_tag: str,
    embedding_manifest: dict[str, object],
    label_cols: list[str],
    extra_stats: dict[str, object] | None = None,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    adj = edges_to_csr(len(nodes), sources, targets, weights)
    comp_labels, degree, stats = graph_stats(adj)
    save_npz(out_dir / "adj.npz", adj)

    nodes_out = nodes.copy()
    nodes_out["graph_type"] = graph_type
    nodes_out["metric_family"] = "embedding"
    nodes_out["metric"] = metric
    nodes_out["k"] = "" if k is None else int(k)
    nodes_out["component_id"] = comp_labels
    nodes_out["degree"] = degree
    nodes_out.to_csv(out_dir / "nodes.csv", index=False)

    with (out_dir / "edges.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "target", "weight", "status"])
        for i, j, w in zip(sources, targets, weights):
            writer.writerow([int(i), int(j), f"{float(w):.12g}", "kept"])

    meta = {
        "cohort_dir": cohort_dir,
        "sample_label": sample_label,
        "embed_tag": embed_tag,
        "embedding_model": embedding_manifest.get("model", ""),
        "graph_type": graph_type,
        "metric_family": "embedding",
        "metric": metric,
        "k": None if k is None else int(k),
        **stats,
    }
    if extra_stats:
        meta.update(extra_stats)
    for label_col in label_cols:
        meta.update(label_metrics(nodes, sources, targets, label_col))
    with (out_dir / "stats.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    log(f"Wrote {out_dir}: edges={stats['n_edges']:,}, components={stats['n_components']}, giant={stats['giant_component_frac']:.3f}")
    return meta


def graph_specs(k_values: list[int], include_rng: bool) -> list[tuple[str, str, int | None]]:
    specs = [("embedding_mst", "embedding_mst", None)]
    if include_rng:
        specs.append(("embedding_rng_exact", "embedding_rng_exact", None))
    for k in k_values:
        specs.append((f"embedding_knn_k{k:02d}", "embedding_knn", k))
    return specs


def build_one_cohort(
    cohort_root: Path,
    sample_label: str,
    embed_tag: str,
    metric: str,
    k_values: list[int],
    label_cols: list[str],
    include_rng: bool,
    rng_row_block_size: int,
    max_block_edges: int,
    cache_distance_matrix: bool,
    refresh_distance_cache: bool,
    overwrite: bool,
    graph_metric_subdir: str | None = None,
) -> list[dict[str, object]]:
    X, nodes, embed_manifest = load_cohort_panel(cohort_root, sample_label, embed_tag)
    input_dir = cohort_root / "inputs" / sample_label
    embed_dir = cohort_root / "embeddings" / embed_tag / sample_label
    dependency_paths = [
        input_dir / "metadata.csv",
        embed_dir / "embeddings.npy",
        embed_dir / "ids.txt",
    ]
    newest_dependency = max(path.stat().st_mtime for path in dependency_paths if path.exists())
    if graph_metric_subdir:
        graph_root = cohort_root / "graphs" / embed_tag / graph_metric_subdir / sample_label
    else:
        graph_root = cohort_root / "graphs" / embed_tag / sample_label
    graph_root.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(graph_root / "canonical_nodes.csv", index=False)

    specs = graph_specs(k_values, include_rng)
    if not overwrite:
        existing_rows = []
        for dirname, _graph_type, _k in specs:
            out_dir = graph_root / dirname
            stats_path = out_dir / "stats.json"
            if not stats_path.exists() or stats_path.stat().st_mtime < newest_dependency:
                existing_rows = []
                break
            stats = json.loads(stats_path.read_text())
            existing_rows.append({"cohort_dir": cohort_root.name, "graph_dir": str(out_dir), **stats})
        if existing_rows:
            log(f"Skipping {cohort_root.name}: all requested graphs already exist")
            return existing_rows

    cache_path = None
    if cache_distance_matrix:
        cache_path = graph_root / "distance_matrices" / f"embedding_{metric}_float32.npy"
        if cache_path.exists() and cache_path.stat().st_mtime < newest_dependency:
            log(f"Refreshing stale embedding distance cache: {cache_path}")
            refresh_distance_cache = True
    D = load_or_compute_pairwise(
        X,
        metric=metric,
        cache_path=cache_path,
        refresh_cache=refresh_distance_cache,
    )
    rows = []
    for dirname, graph_type, k in specs:
        out_dir = graph_root / dirname
        stats_path = out_dir / "stats.json"
        if (
            out_dir.exists()
            and stats_path.exists()
            and stats_path.stat().st_mtime >= newest_dependency
            and not overwrite
        ):
            log(f"Skipping existing graph: {out_dir}")
            stats = json.loads(stats_path.read_text())
        else:
            extra_stats = None
            if graph_type == "embedding_rng_exact":
                sources, targets, weights, checked, pruned = exact_rng_edges_blockwise_order(
                    D,
                    row_block_size=rng_row_block_size,
                    max_block_edges=max_block_edges,
                )
                extra_stats = {
                    "candidate_edges": int(checked),
                    "n_pruned_edges": int(pruned),
                    "rng_row_block_size": int(rng_row_block_size),
                    "max_block_edges": int(max_block_edges),
                }
            elif k is None:
                sources, targets, weights = mst_edges(D)
            else:
                sources, targets, weights = symmetric_knn_edges(D, k)
            stats = write_graph(
                out_dir=out_dir,
                nodes=nodes,
                sources=sources,
                targets=targets,
                weights=weights,
                graph_type=graph_type,
                metric=metric,
                k=k,
                cohort_dir=cohort_root.name,
                sample_label=sample_label,
                embed_tag=embed_tag,
                embedding_manifest=embed_manifest,
                label_cols=label_cols,
                extra_stats=extra_stats,
            )
        rows.append({"cohort_dir": cohort_root.name, "graph_dir": str(out_dir), **stats})
    return rows


def parse_cohorts(value: str) -> list[str]:
    if value.strip().lower() in {"all", ""}:
        return COHORT_DIRS
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build per-cohort ESM-2 embedding MST/RNG/kNN graph families.")
    ap.add_argument("--cohort-root", type=Path, default=Path("analysis/cohort_validation/04_cohort_runs"))
    ap.add_argument("--sample-label", default="main_n20000_seed42")
    ap.add_argument("--embed-tag", default="esm2_650M")
    ap.add_argument("--cohorts", default="all", help="Comma-separated cohort directory names, or all.")
    ap.add_argument("--metric", choices=["cityblock", "manhattan", "euclidean"], default="cityblock")
    ap.add_argument("--k-values", default="5,50")
    ap.add_argument("--label-cols", default="within_lineage_label,lineage,collection_month,region")
    ap.add_argument("--include-rng", action="store_true", help="Also build exact embedding RNG.")
    ap.add_argument("--rng-row-block-size", type=int, default=100)
    ap.add_argument("--max-block-edges", type=int, default=2_000_000)
    ap.add_argument("--cache-distance-matrix", action="store_true", help="Save/load the per-cohort pairwise embedding distance matrix.")
    ap.add_argument("--refresh-distance-cache", action="store_true", help="Recompute and replace the cached distance matrix.")
    ap.add_argument(
        "--graph-metric-subdir",
        default="",
        help="Optional graph subdirectory under graphs/<embed-tag>/ for keeping metrics such as cityblock and euclidean separate.",
    )
    ap.add_argument(
        "--summary",
        type=Path,
        default=Path("analysis/cohort_validation/05_cross_cohort_comparison/esm2_main_n20000_embedding_graph_summary.csv"),
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    metric = "cityblock" if args.metric == "manhattan" else args.metric
    k_values = [int(k.strip()) for k in args.k_values.split(",") if k.strip()]
    label_cols = [col.strip() for col in args.label_cols.split(",") if col.strip()]
    rows = []
    for cohort_dir in parse_cohorts(args.cohorts):
        cohort_root = args.cohort_root / cohort_dir
        if not cohort_root.exists():
            raise FileNotFoundError(f"Missing cohort directory: {cohort_root}")
        log(f"=== Building graphs for {cohort_dir} ===")
        rows.extend(
            build_one_cohort(
                cohort_root=cohort_root,
                sample_label=args.sample_label,
                embed_tag=args.embed_tag,
                metric=metric,
                k_values=k_values,
                label_cols=label_cols,
                include_rng=args.include_rng,
                rng_row_block_size=args.rng_row_block_size,
                max_block_edges=args.max_block_edges,
                cache_distance_matrix=args.cache_distance_matrix,
                refresh_distance_cache=args.refresh_distance_cache,
                overwrite=args.overwrite,
                graph_metric_subdir=args.graph_metric_subdir.strip() or None,
            )
        )

    summary = pd.DataFrame(rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False)
    log(f"Wrote summary: {args.summary}")


if __name__ == "__main__":
    main()
