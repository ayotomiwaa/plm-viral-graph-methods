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
from scipy.sparse import csr_matrix, save_npz
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.graph_construction.build_cohort_embedding_graphs import (
    COHORT_DIRS,
    exact_rng_edges_blockwise_order,
    fix_partial_date,
    graph_stats,
    label_metrics,
)


ACCESSION_RE = re.compile(r"EPI_ISL_\d+")


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_accession(value: object) -> str:
    text = str(value)
    match = ACCESSION_RE.search(text)
    return match.group(0) if match else text.split()[0].strip()


def read_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    cur_id: str | None = None
    cur: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    seqs[cur_id] = "".join(cur).upper()
                cur_id = parse_accession(line[1:])
                cur = []
            else:
                cur.append(line)
    if cur_id is not None:
        seqs[cur_id] = "".join(cur).upper()
    return seqs


def load_cohort_alignment(
    cohort_root: Path,
    sample_label: str,
    aligned_fasta_name: str,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, object]]:
    input_dir = cohort_root / "inputs" / sample_label
    aligned_fasta = input_dir / aligned_fasta_name
    metadata_path = input_dir / "metadata.csv"
    if not aligned_fasta.exists():
        raise FileNotFoundError(f"Missing aligned FASTA: {aligned_fasta}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")

    seqs_by_accession = read_fasta(aligned_fasta)
    meta = pd.read_csv(metadata_path, low_memory=False)
    if "accession" not in meta.columns:
        raise ValueError(f"{metadata_path} must contain accession")
    meta = meta.copy()
    meta["accession"] = meta["accession"].astype(str).str.strip()
    meta = meta[meta["accession"].isin(seqs_by_accession)].copy()
    if meta.empty:
        raise ValueError(f"No metadata accessions from {metadata_path} were present in {aligned_fasta}")
    if len(meta) != len(seqs_by_accession):
        missing_meta = sorted(set(seqs_by_accession) - set(meta["accession"]))
        missing_seq = sorted(set(pd.read_csv(metadata_path, usecols=["accession"])["accession"].astype(str)) - set(seqs_by_accession))
        raise ValueError(
            f"Alignment/metadata mismatch for {cohort_root}: aligned={len(seqs_by_accession):,}, "
            f"metadata_matched={len(meta):,}, missing_meta_examples={missing_meta[:5]}, missing_seq_examples={missing_seq[:5]}"
        )

    seqs = [seqs_by_accession[acc] for acc in meta["accession"].astype(str)]
    lengths = {len(seq) for seq in seqs}
    if len(lengths) != 1:
        counts = pd.Series([len(seq) for seq in seqs]).value_counts().head().to_dict()
        raise ValueError(f"Selected FASTA records are not aligned to one length. Top lengths: {counts}")

    dates = pd.to_datetime(meta["collection_date"].map(fix_partial_date), errors="coerce")
    nodes = pd.DataFrame(
        {
            "node_id": np.arange(len(meta), dtype=int),
            "accession": meta["accession"].astype(str).to_numpy(),
            "collection_date": dates.dt.strftime("%Y-%m-%d").fillna("").to_numpy(),
            "collection_month": meta.get("collection_month", dates.dt.strftime("%Y-%m")).astype(str).to_numpy(),
            "lineage": meta.get("lineage", "").astype(str).to_numpy(),
            "within_lineage_label": meta.get("within_lineage_label", "").astype(str).to_numpy(),
            "region": meta.get("region", "").astype(str).to_numpy(),
            "location": meta.get("location", "").astype(str).to_numpy(),
            "cohort_id": meta.get("cohort_id", "").astype(str).to_numpy(),
            "cohort_name": meta.get("cohort_name", "").astype(str).to_numpy(),
            "alignment_row": np.arange(len(meta), dtype=int),
        }
    )
    arr = np.frombuffer("".join(seqs).encode("ascii"), dtype=np.uint8).reshape(len(seqs), len(seqs[0]))
    manifest = {
        "aligned_fasta": str(aligned_fasta),
        "alignment_length": int(arr.shape[1]),
    }
    log(f"Loaded {cohort_root.name}: n={arr.shape[0]:,}, alignment_length={arr.shape[1]:,}")
    return arr, nodes, manifest


def compute_hamming_distance(seqs: np.ndarray, block_size: int, gap_policy: str, ignore_ambiguous: bool) -> np.ndarray:
    n = seqs.shape[0]
    D = np.zeros((n, n), dtype=np.uint16)
    gap = ord("-")
    ambiguous = np.frombuffer(b"XBZJUO?", dtype=np.uint8)
    valid = seqs != gap if gap_policy == "ignore-any-gap" else np.ones_like(seqs, dtype=bool)
    if ignore_ambiguous:
        valid &= ~np.isin(seqs, ambiguous)

    log(f"Computing Hamming distances: n={n:,}, block_size={block_size}, gap_policy={gap_policy}")
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        diff = seqs[start:stop, None, :] != seqs[None, :, :]
        if gap_policy == "ignore-any-gap" or ignore_ambiguous:
            diff &= valid[start:stop, None, :] & valid[None, :, :]
        D[start:stop, :] = diff.sum(axis=2, dtype=np.int32).astype(np.uint16)
        log(f"Hamming rows {start:,}-{stop - 1:,}/{n:,}")
    np.fill_diagonal(D, np.iinfo(np.uint16).max)
    log(f"Hamming distance matrix ready: shape={D.shape}, size={D.nbytes / 1e9:.2f} GB")
    return D


def load_or_compute_hamming_distance(
    seqs: np.ndarray,
    block_size: int,
    gap_policy: str,
    ignore_ambiguous: bool,
    cache_path: Path | None,
    refresh_cache: bool,
) -> np.ndarray:
    if cache_path is not None and cache_path.exists() and not refresh_cache:
        log(f"Loading cached Hamming distance matrix: {cache_path}")
        D = np.load(cache_path)
        expected_shape = (seqs.shape[0], seqs.shape[0])
        if D.shape != expected_shape or D.dtype != np.uint16:
            log(f"Ignoring invalid Hamming cache: shape={D.shape}, dtype={D.dtype}, expected={expected_shape}/uint16")
        else:
            log(f"Hamming distance matrix loaded: shape={D.shape}, size={D.nbytes / 1e9:.2f} GB")
            return D

    D = compute_hamming_distance(seqs, block_size=block_size, gap_policy=gap_policy, ignore_ambiguous=ignore_ambiguous)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(f"{cache_path.name}.tmp.{os.getpid()}")
        log(f"Saving Hamming distance matrix cache: {cache_path}")
        with tmp_path.open("wb") as handle:
            np.save(handle, D)
        tmp_path.replace(cache_path)
        with cache_path.with_suffix(cache_path.suffix + ".json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "metric": "hamming",
                    "gap_policy": gap_policy,
                    "ignore_ambiguous": bool(ignore_ambiguous),
                    "dtype": str(D.dtype),
                    "shape": [int(D.shape[0]), int(D.shape[1])],
                    "size_gb": D.nbytes / 1e9,
                },
                handle,
                indent=2,
            )
    return D


def mst_edges(D: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log("Building Hamming MST")
    D_mst = np.asarray(D, dtype=np.float32).copy()
    np.fill_diagonal(D_mst, 0.0)
    mst = minimum_spanning_tree(D_mst).tocoo()
    mask = mst.row != mst.col
    return mst.row[mask].astype(np.int32), mst.col[mask].astype(np.int32), mst.data[mask].astype(np.float32)


def symmetric_knn_edges(D: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log(f"Building Hamming symmetric kNN k={k}")
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
    log(f"Hamming symmetric kNN k={k} kept {len(sources):,} undirected edges")
    return sources, targets, weights


def edges_to_csr(n_nodes: int, sources: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> csr_matrix:
    rows = np.concatenate([sources, targets]).astype(np.int32, copy=False)
    cols = np.concatenate([targets, sources]).astype(np.int32, copy=False)
    data = np.concatenate([weights, weights]).astype(np.float32, copy=False)
    return csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float32)


def write_graph(
    out_dir: Path,
    nodes: pd.DataFrame,
    sources: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    graph_type: str,
    k: int | None,
    cohort_dir: str,
    sample_label: str,
    alignment_manifest: dict[str, object],
    label_cols: list[str],
    gap_policy: str,
    ignore_ambiguous: bool,
    extra_stats: dict[str, object] | None = None,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    adj = edges_to_csr(len(nodes), sources, targets, weights)
    comp_labels, degree, stats = graph_stats(adj)
    save_npz(out_dir / "adj.npz", adj)

    nodes_out = nodes.copy()
    nodes_out["graph_type"] = graph_type
    nodes_out["metric_family"] = "hamming"
    nodes_out["metric"] = "hamming"
    nodes_out["k"] = "" if k is None else int(k)
    nodes_out["hamming_gap_policy"] = gap_policy
    nodes_out["ignore_ambiguous"] = bool(ignore_ambiguous)
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
        "graph_type": graph_type,
        "metric_family": "hamming",
        "metric": "hamming",
        "k": None if k is None else int(k),
        "hamming_gap_policy": gap_policy,
        "ignore_ambiguous": bool(ignore_ambiguous),
        **alignment_manifest,
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
    specs = [("hamming_mst", "hamming_mst", None)]
    if include_rng:
        specs.append(("hamming_rng_exact", "hamming_rng_exact", None))
    for k in k_values:
        specs.append((f"hamming_knn_k{k:02d}", "hamming_knn", k))
    return specs


def build_one_cohort(
    cohort_root: Path,
    sample_label: str,
    aligned_fasta_name: str,
    k_values: list[int],
    label_cols: list[str],
    include_rng: bool,
    hamming_block_size: int,
    rng_row_block_size: int,
    max_block_edges: int,
    gap_policy: str,
    ignore_ambiguous: bool,
    cache_distance_matrix: bool,
    refresh_distance_cache: bool,
    overwrite: bool,
) -> list[dict[str, object]]:
    seqs, nodes, alignment_manifest = load_cohort_alignment(cohort_root, sample_label, aligned_fasta_name)
    input_dir = cohort_root / "inputs" / sample_label
    dependency_paths = [
        input_dir / "metadata.csv",
        input_dir / aligned_fasta_name,
    ]
    newest_dependency = max(path.stat().st_mtime for path in dependency_paths if path.exists())
    graph_root = cohort_root / "graphs" / "hamming" / sample_label
    graph_root.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(graph_root / "canonical_nodes.csv", index=False)

    specs = graph_specs(k_values, include_rng)
    if not overwrite:
        existing_rows = []
        for dirname, _graph_type, _k in specs:
            stats_path = graph_root / dirname / "stats.json"
            if not stats_path.exists() or stats_path.stat().st_mtime < newest_dependency:
                existing_rows = []
                break
            stats = json.loads(stats_path.read_text())
            existing_rows.append({"cohort_dir": cohort_root.name, "graph_dir": str(graph_root / dirname), **stats})
        if existing_rows:
            log(f"Skipping {cohort_root.name}: all requested Hamming graphs already exist")
            return existing_rows

    cache_path = None
    if cache_distance_matrix:
        suffix = "ignore_ambiguous" if ignore_ambiguous else "all_states"
        cache_path = graph_root / "distance_matrices" / f"hamming_{gap_policy}_{suffix}_uint16.npy"
        if cache_path.exists() and cache_path.stat().st_mtime < newest_dependency:
            log(f"Refreshing stale Hamming distance cache: {cache_path}")
            refresh_distance_cache = True
    D = load_or_compute_hamming_distance(
        seqs,
        block_size=hamming_block_size,
        gap_policy=gap_policy,
        ignore_ambiguous=ignore_ambiguous,
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
            if graph_type == "hamming_rng_exact":
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
            elif graph_type == "hamming_mst":
                sources, targets, weights = mst_edges(D)
            else:
                sources, targets, weights = symmetric_knn_edges(D, int(k))
            stats = write_graph(
                out_dir=out_dir,
                nodes=nodes,
                sources=sources,
                targets=targets,
                weights=weights,
                graph_type=graph_type,
                k=k,
                cohort_dir=cohort_root.name,
                sample_label=sample_label,
                alignment_manifest=alignment_manifest,
                label_cols=label_cols,
                gap_policy=gap_policy,
                ignore_ambiguous=ignore_ambiguous,
                extra_stats=extra_stats,
            )
        rows.append({"cohort_dir": cohort_root.name, "graph_dir": str(out_dir), **stats})
    return rows


def parse_cohorts(value: str) -> list[str]:
    if value.strip().lower() in {"all", ""}:
        return COHORT_DIRS
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build per-cohort aligned-sequence Hamming MST/RNG/kNN graph families.")
    ap.add_argument("--cohort-root", type=Path, default=Path("analysis/cohort_validation/04_cohort_runs"))
    ap.add_argument("--sample-label", default="main_n20000_seed42")
    ap.add_argument("--aligned-fasta-name", default="spike_sequences_aligned_mafft.fasta")
    ap.add_argument("--cohorts", default="all")
    ap.add_argument("--k-values", default="5,50")
    ap.add_argument("--label-cols", default="within_lineage_label,lineage,collection_month,region")
    ap.add_argument("--include-rng", action="store_true")
    ap.add_argument("--hamming-block-size", type=int, default=24)
    ap.add_argument("--rng-row-block-size", type=int, default=100)
    ap.add_argument("--max-block-edges", type=int, default=5_000_000)
    ap.add_argument("--gap-policy", choices=["count-gap-state", "ignore-any-gap"], default="count-gap-state")
    ap.add_argument("--ignore-ambiguous", action="store_true")
    ap.add_argument("--cache-distance-matrix", action="store_true")
    ap.add_argument("--refresh-distance-cache", action="store_true")
    ap.add_argument(
        "--summary",
        type=Path,
        default=Path("analysis/cohort_validation/05_cross_cohort_comparison/hamming_main_n20000_graph_summary.csv"),
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    k_values = [int(k.strip()) for k in args.k_values.split(",") if k.strip()]
    label_cols = [col.strip() for col in args.label_cols.split(",") if col.strip()]
    rows = []
    for cohort_dir in parse_cohorts(args.cohorts):
        cohort_root = args.cohort_root / cohort_dir
        if not cohort_root.exists():
            raise FileNotFoundError(f"Missing cohort directory: {cohort_root}")
        log(f"=== Building Hamming graphs for {cohort_dir} ===")
        rows.extend(
            build_one_cohort(
                cohort_root=cohort_root,
                sample_label=args.sample_label,
                aligned_fasta_name=args.aligned_fasta_name,
                k_values=k_values,
                label_cols=label_cols,
                include_rng=args.include_rng,
                hamming_block_size=args.hamming_block_size,
                rng_row_block_size=args.rng_row_block_size,
                max_block_edges=args.max_block_edges,
                gap_policy=args.gap_policy,
                ignore_ambiguous=args.ignore_ambiguous,
                cache_distance_matrix=args.cache_distance_matrix,
                refresh_distance_cache=args.refresh_distance_cache,
                overwrite=args.overwrite,
            )
        )

    summary = pd.DataFrame(rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False)
    log(f"Wrote summary: {args.summary}")


if __name__ == "__main__":
    main()
