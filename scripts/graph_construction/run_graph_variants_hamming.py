#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def fix_partial_date(x: object) -> str:
    s = str(x)
    if s.endswith("-00-00"):
        return s[:4] + "-06-15"
    if s.endswith("-00"):
        return s[:7] + "-15"
    return s


def canonical_fasta_id(header: str) -> str:
    token = header.split()[0]
    parts = [p.strip() for p in token.split("|") if p.strip()]
    for part in parts:
        if part.startswith(("NC_", "EPI_ISL_", "OP", "ON", "OM", "MZ", "MW", "MT")):
            return part
    return token


def read_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    cur_id: str | None = None
    cur: list[str] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    seqs[cur_id] = "".join(cur).upper()
                cur_id = canonical_fasta_id(line[1:])
                cur = []
            else:
                cur.append(line)
    if cur_id is not None:
        seqs[cur_id] = "".join(cur).upper()
    return seqs


def load_variant_alignment(
    aligned_fasta: Path,
    metadata_path: Path,
    variant: str,
    variant_col: str,
    include_accessions: set[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    meta = pd.read_csv(metadata_path, low_memory=False)
    if "accession" not in meta.columns:
        raise ValueError(f"{metadata_path} must contain an 'accession' column")
    if variant_col not in meta.columns:
        raise ValueError(f"{metadata_path} must contain variant column '{variant_col}'")
    if "collection_date" not in meta.columns:
        raise ValueError(f"{metadata_path} must contain a 'collection_date' column")

    meta = meta.copy()
    meta["accession"] = meta["accession"].astype(str).str.strip()
    meta_idx = meta.set_index("accession", drop=False)

    seq_by_id = read_fasta(aligned_fasta)
    if not seq_by_id:
        raise RuntimeError(f"No sequences found in {aligned_fasta}")

    missing_ref = sorted(a for a in include_accessions if a not in seq_by_id)
    if missing_ref:
        raise RuntimeError(
            "Missing required reference accession(s) in aligned FASTA: "
            + ", ".join(missing_ref)
            + ". Build an augmented aligned FASTA first; do not append an ungapped reference to an MSA."
        )

    variant_mask = meta[variant_col].astype(str).str.casefold() == variant.casefold()
    include_mask = meta["accession"].isin(include_accessions)
    selected = meta.loc[variant_mask | include_mask].copy()
    selected = selected[selected["accession"].isin(seq_by_id)]
    if selected.empty:
        available = sorted(meta[variant_col].dropna().astype(str).unique().tolist())
        raise RuntimeError(f"No rows matched variant={variant!r}; available variants include {available[:20]}")

    # Keep deterministic metadata order, with any included reference retained from metadata.
    ids = selected["accession"].astype(str).tolist()
    seqs = [seq_by_id[a] for a in ids]
    lengths = {len(s) for s in seqs}
    if len(lengths) != 1:
        by_len = pd.Series([len(s) for s in seqs]).value_counts().head().to_dict()
        raise RuntimeError(f"Selected sequences are not aligned to one length. Top lengths: {by_len}")

    selected["collection_date"] = pd.to_datetime(selected["collection_date"].map(fix_partial_date), errors="coerce")
    if "lineage" not in selected.columns:
        selected["lineage"] = ""

    nodes = pd.DataFrame({
        "node_id": np.arange(len(selected), dtype=int),
        "accession": selected["accession"].astype(str).to_numpy(),
        "collection_date": selected["collection_date"].to_numpy(),
        "lineage": selected["lineage"].astype(str).to_numpy(),
        "variant_bucket": selected[variant_col].astype(str).to_numpy(),
        "alignment_row": np.arange(len(selected), dtype=int),
    })
    if "location" in selected.columns:
        nodes["location"] = selected["location"].astype(str).to_numpy()

    arr = np.frombuffer("".join(seqs).encode("ascii"), dtype=np.uint8).reshape(len(seqs), len(seqs[0]))
    print(f"[LOAD hamming] {variant}: n={arr.shape[0]:,}, alignment_length={arr.shape[1]:,}")
    return arr, nodes


def hamming_distance_matrix(
    seqs: np.ndarray,
    block_size: int,
    gap_policy: str,
    ignore_ambiguous: bool,
) -> np.ndarray:
    n, _ = seqs.shape
    D = np.zeros((n, n), dtype=np.float32)
    gap = ord("-")
    ambiguous = np.frombuffer(b"XBZJUO?", dtype=np.uint8)
    valid = seqs != gap if gap_policy == "ignore-any-gap" else np.ones_like(seqs, dtype=bool)
    if ignore_ambiguous:
        valid &= ~np.isin(seqs, ambiguous)

    print(f"[DIST] computing exact aligned-sequence Hamming distances, block_size={block_size}...")
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        diff = seqs[start:stop, None, :] != seqs[None, :, :]
        if gap_policy == "ignore-any-gap" or ignore_ambiguous:
            pair_valid = valid[start:stop, None, :] & valid[None, :, :]
            diff &= pair_valid
        D[start:stop, :] = diff.sum(axis=2, dtype=np.int32).astype(np.float32)
        print(f"[DIST] rows {start:,}-{stop - 1:,} / {n:,}")
    print("[DIST] pairwise Hamming matrix ready")
    return D


def exact_knn_from_distance(D: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n = D.shape[0]
    kk = min(k + 1, n)
    idx = np.argpartition(D, kth=kk - 1, axis=1)[:, :kk]
    dist = np.take_along_axis(D, idx, axis=1)
    order = np.argsort(dist, axis=1, kind="stable")
    idx = np.take_along_axis(idx, order, axis=1)
    dist = np.take_along_axis(dist, order, axis=1)

    out_idx = np.empty((n, min(k, n - 1)), dtype=np.int32)
    out_dist = np.empty_like(out_idx, dtype=np.float32)
    for i in range(n):
        keep = idx[i] != i
        row_idx = idx[i, keep][: out_idx.shape[1]]
        row_dist = dist[i, keep][: out_idx.shape[1]]
        out_idx[i, : len(row_idx)] = row_idx
        out_dist[i, : len(row_dist)] = row_dist
    print(f"[KNN] built exact Hamming kNN with k={k}")
    return out_idx, out_dist


def build_symmetric_knn_edges(knn_idx: np.ndarray, knn_dist: np.ndarray) -> Dict[Tuple[int, int], float]:
    edges: Dict[Tuple[int, int], float] = {}
    n, _ = knn_idx.shape
    for i in range(n):
        for j, d in zip(knn_idx[i], knn_dist[i]):
            j = int(j)
            a, b = (i, j) if i < j else (j, i)
            d = float(d)
            prev = edges.get((a, b))
            if prev is None or d < prev:
                edges[(a, b)] = d
    print(f"[GRAPH] hamming_knn: edges={len(edges):,}")
    return edges


def build_hamming_rng_edges(D: np.ndarray) -> Dict[Tuple[int, int], float]:
    n = D.shape[0]
    edges: Dict[Tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            dij = D[i, j]
            lune = np.maximum(D[i, :], D[j, :]) < dij
            lune[i] = False
            lune[j] = False
            if not np.any(lune):
                edges[(i, j)] = float(dij)
    print(f"[GRAPH] hamming_rng_exact: edges={len(edges):,}")
    return edges


def rng_prune_edges(
    candidate_edges: Dict[Tuple[int, int], float],
    D: np.ndarray,
) -> tuple[Dict[Tuple[int, int], float], List[Tuple[int, int, float]]]:
    kept: Dict[Tuple[int, int], float] = {}
    pruned: List[Tuple[int, int, float]] = []
    for (i, j), dij in candidate_edges.items():
        lune = np.maximum(D[i, :], D[j, :]) < dij
        lune[i] = False
        lune[j] = False
        if np.any(lune):
            pruned.append((i, j, float(dij)))
        else:
            kept[(i, j)] = float(dij)
    print(f"[PRUNE] hamming_knn_rng_exact_witness: kept={len(kept):,}, pruned={len(pruned):,}")
    return kept, pruned


def edges_to_csr(n_nodes: int, edges: Dict[Tuple[int, int], float]) -> csr_matrix:
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    for (i, j), w in edges.items():
        rows.extend([i, j])
        cols.extend([j, i])
        vals.extend([w, w])
    return csr_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float32)


def graph_stats(adj: csr_matrix) -> dict:
    n_comp, labels = connected_components(adj, directed=False, return_labels=True)
    deg = np.asarray(adj.getnnz(axis=1)).ravel().astype(int)
    if len(labels):
        vc = pd.Series(labels).value_counts().sort_values(ascending=False)
        giant_label = int(vc.index[0])
        giant_size = int(vc.iloc[0])
    else:
        giant_label = 0
        giant_size = 0
    return {
        "n_nodes": int(adj.shape[0]),
        "n_edges": int(adj.nnz // 2),
        "n_components": int(n_comp),
        "giant_label": giant_label,
        "giant_component_size": giant_size,
        "giant_component_frac": float(giant_size / adj.shape[0] if adj.shape[0] else 0.0),
        "component_labels": labels,
        "degree_counts": deg,
    }


def spectral_layout_giant(adj: csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    n = adj.shape[0]
    coords = np.full((n, 2), np.nan, dtype=np.float32)
    if adj.nnz == 0:
        return coords, np.array([], dtype=np.float32)
    _, labels = connected_components(adj, directed=False, return_labels=True)
    vc = pd.Series(labels).value_counts().sort_values(ascending=False)
    if len(vc) == 0:
        return coords, np.array([], dtype=np.float32)
    giant_nodes = np.where(labels == int(vc.index[0]))[0]
    if len(giant_nodes) == 1:
        coords[giant_nodes[0], :] = 0.0
        return coords, np.array([], dtype=np.float32)
    sub = adj[giant_nodes][:, giant_nodes].astype(np.float64)
    deg = np.asarray(sub.sum(axis=1)).ravel()
    L = csr_matrix(np.diag(deg)) - sub
    k_eigs = min(3, sub.shape[0] - 1)
    if k_eigs < 1:
        coords[giant_nodes, :] = 0.0
        return coords, np.array([], dtype=np.float32)
    vals, vecs = eigsh(L, k=k_eigs, which="SM")
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]
    if vecs.shape[1] >= 3:
        xy = vecs[:, 1:3]
    elif vecs.shape[1] == 2:
        xy = np.column_stack([vecs[:, 1], np.zeros(sub.shape[0])])
    else:
        xy = np.zeros((sub.shape[0], 2), dtype=np.float64)
    coords[giant_nodes, :] = xy.astype(np.float32)
    return coords, vals.astype(np.float32)


def write_run(
    out_dir: Path,
    nodes: pd.DataFrame,
    edges: Dict[Tuple[int, int], float],
    graph_type: str,
    k: int | None = None,
    pruned_edges: List[Tuple[int, int, float]] | None = None,
    gap_policy: str = "count-gap-state",
) -> None:
    out_dir = ensure_dir(out_dir)
    adj = edges_to_csr(len(nodes), edges)
    stats = graph_stats(adj)
    coords, eigvals = spectral_layout_giant(adj)

    nodes_out = nodes.copy()
    nodes_out["graph_type"] = graph_type
    nodes_out["metric"] = "hamming"
    nodes_out["hamming_gap_policy"] = gap_policy
    nodes_out["k"] = np.nan if k is None else int(k)
    nodes_out["component_id"] = stats["component_labels"]
    nodes_out["degree"] = stats["degree_counts"]
    nodes_out["x_spec"] = coords[:, 0]
    nodes_out["y_spec"] = coords[:, 1]
    valid = np.isfinite(nodes_out["x_spec"].to_numpy())
    x_rank = np.full(len(nodes_out), np.nan, dtype=np.float32)
    valid_idx = np.where(valid)[0]
    if len(valid_idx):
        order = np.argsort(nodes_out.loc[valid_idx, "x_spec"].to_numpy())
        x_rank[valid_idx[order]] = np.arange(len(valid_idx), dtype=np.float32)
    nodes_out["x_rank"] = x_rank

    nodes_out.to_csv(out_dir / "nodes.csv", index=False)
    nodes_out[["node_id", "x_spec", "y_spec"]].to_csv(out_dir / "layout_spectral.csv", index=False)
    save_npz(out_dir / "adj.npz", adj)
    np.save(out_dir / "spectral_eigenvalues.npy", eigvals)

    edge_rows = [{"source": i, "target": j, "weight": w, "status": "kept"} for (i, j), w in edges.items()]
    if pruned_edges:
        edge_rows.extend({"source": i, "target": j, "weight": w, "status": "pruned"} for (i, j, w) in pruned_edges)
    pd.DataFrame(edge_rows).to_csv(out_dir / "edges.csv", index=False)

    meta = {
        "graph_type": graph_type,
        "witness_type": "full_dataset" if graph_type == "hamming_knn_rng_exact_witness" else None,
        "metric": "hamming",
        "hamming_gap_policy": gap_policy,
        "k": None if k is None else int(k),
        "n_nodes": int(stats["n_nodes"]),
        "n_edges": int(stats["n_edges"]),
        "n_components": int(stats["n_components"]),
        "giant_label": int(stats["giant_label"]),
        "giant_component_size": int(stats["giant_component_size"]),
        "giant_component_frac": float(stats["giant_component_frac"]),
        "n_pruned_edges": int(len(pruned_edges) if pruned_edges else 0),
    }
    with open(out_dir / "stats.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(
        f"[DONE] {out_dir.name}: nodes={meta['n_nodes']:,} edges={meta['n_edges']:,} "
        f"giant={meta['giant_component_size']:,} ({meta['giant_component_frac']:.1%})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build aligned-sequence Hamming graph variants")
    ap.add_argument("--aligned-fasta", required=True, help="Aligned FASTA containing variant sequences and reference")
    ap.add_argument("--metadata", required=True, help="metadata.csv with accession, collection_date, lineage, variant column")
    ap.add_argument("--variant", required=True, help="Variant label, e.g. Alpha")
    ap.add_argument("--variant-col", default="variant_bucket")
    ap.add_argument("--include-accession", action="append", default=["NC_045512.2"])
    ap.add_argument("--no-default-reference", action="store_true")
    ap.add_argument("--dataset-label", default="Hamming_20k_with_wuhan_ref")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--ks", nargs="+", type=int, default=[5, 10, 15, 25, 50])
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--gap-policy", choices=["count-gap-state", "ignore-any-gap"], default="count-gap-state")
    ap.add_argument("--ignore-ambiguous", action="store_true")
    ap.add_argument("--max-hamming-rng-n", type=int, default=5000)
    ap.add_argument("--skip-hamming-rng", action="store_true")
    args = ap.parse_args()

    include_accessions = set([] if args.no_default_reference else args.include_accession)
    seqs, nodes = load_variant_alignment(
        aligned_fasta=Path(args.aligned_fasta),
        metadata_path=Path(args.metadata),
        variant=args.variant,
        variant_col=args.variant_col,
        include_accessions=include_accessions,
    )

    root = ensure_dir(Path(args.out_root) / args.dataset_label / args.variant)
    nodes.to_csv(root / "canonical_nodes.csv", index=False)
    with open(root / "input_manifest.json", "w") as f:
        json.dump({
            "dataset_label": args.dataset_label,
            "variant": args.variant,
            "metric": "hamming",
            "hamming_gap_policy": args.gap_policy,
            "ignore_ambiguous": bool(args.ignore_ambiguous),
            "ks": args.ks,
            "aligned_fasta": args.aligned_fasta,
            "metadata": args.metadata,
            "variant_col": args.variant_col,
            "include_accession": [] if args.no_default_reference else args.include_accession,
            "n_nodes": int(seqs.shape[0]),
            "alignment_length": int(seqs.shape[1]),
        }, f, indent=2)

    D = hamming_distance_matrix(
        seqs=seqs,
        block_size=args.block_size,
        gap_policy=args.gap_policy,
        ignore_ambiguous=args.ignore_ambiguous,
    )

    if args.skip_hamming_rng:
        print("[SKIP] hamming_rng_exact skipped by --skip-hamming-rng")
    elif D.shape[0] <= args.max_hamming_rng_n:
        rng_edges = build_hamming_rng_edges(D)
        write_run(
            out_dir=root / "hamming_rng_exact",
            nodes=nodes,
            edges=rng_edges,
            graph_type="hamming_rng_exact",
            k=None,
            pruned_edges=None,
            gap_policy=args.gap_policy,
        )
    else:
        print(f"[SKIP] hamming_rng_exact skipped because n={D.shape[0]:,} > max_hamming_rng_n={args.max_hamming_rng_n:,}")

    for k in args.ks:
        knn_idx, knn_dist = exact_knn_from_distance(D, k=k)
        sym_edges = build_symmetric_knn_edges(knn_idx, knn_dist)
        write_run(
            out_dir=root / f"hamming_knn_k{k:02d}",
            nodes=nodes,
            edges=sym_edges,
            graph_type="hamming_knn",
            k=k,
            pruned_edges=None,
            gap_policy=args.gap_policy,
        )
        rng_edges, pruned_edges = rng_prune_edges(sym_edges, D)
        write_run(
            out_dir=root / f"hamming_knn_rng_exact_witness_k{k:02d}",
            nodes=nodes,
            edges=rng_edges,
            graph_type="hamming_knn_rng_exact_witness",
            k=k,
            pruned_edges=pruned_edges,
            gap_policy=args.gap_policy,
        )


if __name__ == "__main__":
    main()
