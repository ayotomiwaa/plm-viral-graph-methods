#!/usr/bin/env python3
"""Reproduce the Lamb et al. UK RNG from their shipped embeddings and verify equivalence.

Inputs (from ``Lamb_et_al/Lamb2026_Fig6-7_Reproduction_ESM2_SARS-CoV-2_Spike/fig7``):

* ``06_embeddings.npy``        (11272 x 2560 float32)  -- ESM-2 3B, 11,272 haplotypes
* ``09_wuhan_embeddings.npy``  (1 x 2560 float32)      -- Wuhan-Hu-1
* ``06_ids.csv``               (accession, collection_date, pango_lineage, variant, cluster_size, row_index)
* ``rng/14_rng_l1_adj.npz``    (11273 x 11273, their RNG, Wuhan at node 11272)

Outputs a self-contained UK panel under ``--out-root``:

    embeddings/embeddings.npy                (11273 x 2560, haplotypes + Wuhan last)
    nodes/nodes.csv                          canonical nodes
    graphs/cityblock/embedding_rng_exact/    faithful exact-witness RNG (our build)
    graphs/cityblock/embedding_knn_k{5,10,50,100}/  symmetric kNN baselines
    comparison/rng_vs_lamb_l1.{json,md}      edge-overlap report vs their adj

The RNG construction mirrors ``spectral_backbone_revised/build_all_variant_knn_rng.py``:
symmetric kNN candidates -> exact full-dataset lune witness test
``witness = any(D[i] < dij & D[j] < dij)``.  Candidate k is ``--k-rng`` (default 7000, which
covers their recorded ``max_surviving_rank`` of 6106, so the result is exact).

Run with the ``hmm`` env (numpy/scipy/pandas only; no numba needed).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def symmetric_knn_edges(D: np.ndarray, k: int) -> dict[tuple[int, int], float]:
    """Union of both endpoints' k nearest neighbors. Mirrors build_all_variant_knn_rng.py."""
    n = D.shape[0]
    k = max(1, min(k, n - 1))
    idx = np.argpartition(D, kth=k - 1, axis=1)[:, :k]
    row_ids = np.arange(n)[:, None]
    dist = D[row_ids, idx]
    order = np.argsort(dist, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    dist = np.take_along_axis(dist, order, axis=1)
    edges: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j, dij in zip(idx[i], dist[i]):
            j = int(j)
            if j == i:
                continue
            a, b = (i, j) if i < j else (j, i)
            previous = edges.get((a, b))
            if previous is None or float(dij) < previous:
                edges[(a, b)] = float(dij)
    return edges


def exact_witness_rng_prune(
    edges: dict[tuple[int, int], float], D: np.ndarray, progress_every: int
) -> tuple[dict[tuple[int, int], float], list[tuple[int, int, float]]]:
    """Drop candidate edge (i,j) iff some w has D[i,w] < dij and D[j,w] < dij (strict)."""
    kept: dict[tuple[int, int], float] = {}
    pruned: list[tuple[int, int, float]] = []
    items = list(edges.items())
    t0 = time.time()
    for pos, ((i, j), dij) in enumerate(items, start=1):
        if np.any((D[i] < dij) & (D[j] < dij)):
            pruned.append((i, j, float(dij)))
        else:
            kept[(i, j)] = float(dij)
        if progress_every > 0 and pos % progress_every == 0:
            el = time.time() - t0
            log(f"  prune {pos:,}/{len(items):,} kept={len(kept):,} pruned={len(pruned):,} {el:.0f}s")
    return kept, pruned


def edges_to_csr(n_nodes: int, edges: dict[tuple[int, int], float]) -> csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for (i, j), w in edges.items():
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([w, w])
    return csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float32)


def write_graph(out_dir: Path, nodes: pd.DataFrame, edges: dict, kind: str, k: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    adj = edges_to_csr(len(nodes), edges)
    n_comp, labels = connected_components(adj, directed=False, return_labels=True)
    deg = np.asarray(adj.getnnz(axis=1)).ravel().astype(int)
    nodes_out = nodes.copy()
    nodes_out["component_id"] = labels
    nodes_out["degree"] = deg
    nodes_out.to_csv(out_dir / "nodes.csv", index=False)
    save_npz(out_dir / "adj.npz", adj)
    pd.DataFrame(
        [{"source": i, "target": j, "weight": w} for (i, j), w in edges.items()]
    ).to_csv(out_dir / "edges.csv.gz", index=False, compression="gzip")
    stats = {
        "kind": kind,
        "metric": "cityblock",
        "k": int(k),
        "n_nodes": int(len(nodes)),
        "n_edges": int(adj.nnz // 2),
        "n_components": int(n_comp),
        "mean_degree": float(deg.mean()),
    }
    with (out_dir / "stats.json").open("w") as fh:
        json.dump(stats, fh, indent=2)


def load_reference_edges(adj_path: Path) -> set[tuple[int, int]]:
    adj = load_npz(adj_path).tocsr()
    coo = adj.tocoo(copy=False)
    mask = coo.row < coo.col
    return {(int(i), int(j)) for i, j in zip(coo.row[mask], coo.col[mask])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lamb-fig7", type=Path, required=True)
    ap.add_argument("--out-root", type=Path,
                    default=Path("analysis/cohort_validation/39_lamb_uk_rng"))
    ap.add_argument("--k-rng", type=int, default=7000,
                    help="candidate k for the faithful RNG (>= their max_surviving_rank 6106)")
    ap.add_argument("--knn", type=int, nargs="+", default=[5, 10, 50, 100])
    ap.add_argument("--progress-every", type=int, default=25000)
    args = ap.parse_args()

    fig7 = args.lamb_fig7
    emb = np.load(fig7 / "06_embeddings.npy").astype(np.float32)
    wuhan = np.load(fig7 / "09_wuhan_embeddings.npy").astype(np.float32)
    ids = pd.read_csv(fig7 / "06_ids.csv")
    log(f"Loaded embeddings {emb.shape} + wuhan {wuhan.shape}; ids {ids.shape}")

    assert emb.shape[0] == len(ids) == 11272
    assert wuhan.shape == (1, emb.shape[1])
    assert emb.shape[1] == 2560, "expected ESM-2 3B 2560-dim embeddings"

    # Stack haplotypes (0..11271) + Wuhan (11272).
    X = np.vstack([emb, wuhan]).astype(np.float32)
    n = X.shape[0]

    dates = pd.to_datetime(ids["collection_date"], errors="coerce")
    nodes = pd.DataFrame({
        "node_id": np.arange(n, dtype=int),
        "embedding_row": np.arange(n, dtype=int),
        "accession": list(ids["accession"].astype(str)) + ["Wuhan-Hu-1"],
        "collection_date": list(dates.dt.strftime("%Y-%m-%d").fillna("")) + ["2019-12-30"],
        "lineage": list(ids["pango_lineage"].astype(str)) + ["Wuhan"],
        "variant": list(ids["variant"].astype(str)) + ["Reference"],
        "cluster_size": list(ids["cluster_size"].astype(int)) + [1],
    })

    out = args.out_root
    (out / "embeddings").mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings" / "embeddings.npy", X)
    (out / "nodes").mkdir(parents=True, exist_ok=True)
    nodes.to_csv(out / "nodes" / "nodes.csv", index=False)
    log(f"Wrote embeddings {X.shape} and canonical nodes to {out}")

    D = cdist(X, X, metric="cityblock").astype(np.float32)
    np.fill_diagonal(D, np.inf)
    log(f"Pairwise cityblock D ready: {D.shape}, {D.nbytes/1e9:.2f} GB")

    graph_root = out / "graphs" / "cityblock"

    # --- kNN baselines ---
    for k in args.knn:
        log(f"Building symmetric kNN k={k}")
        edges = symmetric_knn_edges(D, k)
        write_graph(graph_root / f"embedding_knn_k{k:02d}", nodes, edges, "symmetric_knn", k)

    # --- faithful exact-witness RNG ---
    log(f"Building faithful RNG with candidate k={args.k_rng}")
    candidates = symmetric_knn_edges(D, args.k_rng)
    kept, pruned = exact_witness_rng_prune(candidates, D, args.progress_every)
    write_graph(graph_root / "embedding_rng_exact", nodes, kept, "exact_witness_rng", args.k_rng)
    log(f"RNG kept {len(kept):,} / candidates {len(candidates):,} (pruned {len(pruned):,})")

    # --- overlap vs their L1 adj ---
    ref_path = fig7 / "rng" / "14_rng_l1_adj.npz"
    if ref_path.exists():
        ref = load_reference_edges(ref_path)
        ours = set(kept.keys())
        inter = ref & ours
        only_ref = ref - ours
        only_ours = ours - ref
        jac = len(inter) / len(ref | ours) if (ref | ours) else 0.0
        precision = len(inter) / len(ours) if ours else 0.0
        recall = len(inter) / len(ref) if ref else 0.0
        report = {
            "reference_path": str(ref_path),
            "reference_n_edges": len(ref),
            "ours_n_edges": len(ours),
            "intersection": len(inter),
            "only_in_reference": len(only_ref),
            "only_in_ours": len(only_ours),
            "jaccard": jac,
            "precision_ours": precision,
            "recall_vs_reference": recall,
            "sample_differing_edges": sorted(list(only_ref | only_ours))[:50],
        }
        (out / "comparison").mkdir(parents=True, exist_ok=True)
        with (out / "comparison" / "rng_vs_lamb_l1.json").open("w") as fh:
            json.dump(report, fh, indent=2)
        with (out / "comparison" / "rng_vs_lamb_l1.md").open("w") as fh:
            fh.write(f"# RNG reproduction vs Lamb L1 adjacency\n\n"
                     f"- reference edges: {len(ref):,}\n"
                     f"- ours edges: {len(ours):,}\n"
                     f"- intersection: {len(inter):,}\n"
                     f"- only reference: {len(only_ref):,}\n"
                     f"- only ours: {len(only_ours):,}\n"
                     f"- Jaccard: {jac:.6f}\n"
                     f"- precision (ours): {precision:.6f}\n"
                     f"- recall (vs reference): {recall:.6f}\n")
        log(f"Overlap vs reference: Jaccard={jac:.6f} precision={precision:.6f} recall={recall:.6f}")
    else:
        log(f"Reference adj not found at {ref_path}; skipping overlap")

    log("Done.")


if __name__ == "__main__":
    main()
