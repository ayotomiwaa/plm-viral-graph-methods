#!/usr/bin/env python3
"""Build symmetric kNN graphs at arbitrary k on the Wuhan-ref embedding bank.

The mixed 20k + Wuhan panel (`graph_runs_coleman_20k_with_wuhan_root_full_rng`) already
has `knn_k05` and `knn_k50`; this fills in `knn_k10`/`knn_k15`/`knn_k25` so the
distance-vs-time correlation can be swept over k. Matches the existing panel exactly:
cityblock metric, union-symmetrized kNN, and the `embedding/rng_exact/nodes.csv` node
ordering (node `embedding_row` indexes into the Wuhan-ref bank).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_env, "1")

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def symmetric_knn_edges(D: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            prev = edges.get((a, b))
            if prev is None or value < prev:
                edges[(a, b)] = value
    items = sorted(edges.items())
    sources = np.array([i for (i, _), _ in items], dtype=np.int32)
    targets = np.array([j for (_, j), _ in items], dtype=np.int32)
    weights = np.array([w for _, w in items], dtype=np.float32)
    return sources, targets, weights


def graph_stats(adj: csr_matrix) -> dict:
    n_comp, labels = connected_components(adj, directed=False, return_labels=True)
    sizes = np.bincount(labels, minlength=n_comp) if len(labels) else np.array([], dtype=int)
    degree = np.asarray(adj.getnnz(axis=1)).ravel().astype(int)
    giant = int(sizes.max()) if len(sizes) else 0
    return {
        "n_nodes": int(adj.shape[0]),
        "n_edges": int(adj.nnz // 2),
        "n_components": int(n_comp),
        "giant_component_size": giant,
        "giant_component_frac": float(giant / adj.shape[0]) if adj.shape[0] else 0.0,
        "mean_degree": float(degree.mean()) if len(degree) else 0.0,
        "max_degree": int(degree.max()) if len(degree) else 0,
        "component_labels": labels,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embeddings", type=Path, default=Path("ref_embeddins/embeddings_esm2_650M_with_wuhan_ref/embeddings.npy"))
    ap.add_argument("--nodes", type=Path, default=Path("graph_runs_coleman_20k_with_wuhan_root_full_rng/embedding/rng_exact/nodes.csv"))
    ap.add_argument("--out-root", type=Path, default=Path("graph_runs_coleman_20k_with_wuhan_root_full_rng/embedding"))
    ap.add_argument("--k", type=int, nargs="+", default=[10, 15, 25], help="k values to build (space-separated)")
    ap.add_argument("--metric", choices=["cityblock", "euclidean"], default="cityblock")
    args = ap.parse_args()

    ks = args.k if args.k else [10, 15, 25]
    nodes = pd.read_csv(args.nodes, low_memory=False)
    X = np.load(args.embeddings, mmap_mode="r")
    erow = nodes["embedding_row"].astype(int).to_numpy()
    Xp = np.asarray(X[erow], dtype=np.float32)
    n, d = Xp.shape
    log(f"Loaded embeddings n={n} d={d} metric={args.metric}")

    log(f"Computing pairwise {args.metric} distance matrix ({n}x{n})...")
    t0 = time.time()
    D = cdist(Xp, Xp, metric=args.metric).astype(np.float32, copy=False)
    np.fill_diagonal(D, np.inf)
    log(f"Distance matrix ready in {time.time()-t0:.1f}s ({D.nbytes/1e9:.2f} GB)")

    for k in ks:
        log(f"Building symmetric kNN k={k}")
        src, tgt, wgt = symmetric_knn_edges(D, k)
        rows = np.concatenate([src, tgt]).astype(np.int32)
        cols = np.concatenate([tgt, src]).astype(np.int32)
        data = np.concatenate([wgt, wgt]).astype(np.float32)
        adj = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
        stats = graph_stats(adj)
        labels = stats.pop("component_labels")

        out_dir = args.out_root / f"knn_k{k:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_npz(out_dir / "adj.npz", adj)
        nodes_out = nodes.copy()
        nodes_out["node_id"] = np.arange(n, dtype=int)
        nodes_out["degree"] = np.asarray(adj.getnnz(axis=1)).ravel().astype(int)
        nodes_out["component_id"] = labels
        nodes_out["graph_type"] = "embedding_knn"
        nodes_out["metric_family"] = "embedding"
        nodes_out["metric"] = args.metric
        nodes_out["k"] = k
        nodes_out.to_csv(out_dir / "nodes.csv", index=False)
        import csv

        with (out_dir / "edges.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["source", "target", "weight", "status"])
            for i, j, wt in zip(src, tgt, wgt):
                w.writerow([int(i), int(j), f"{float(wt):.12g}", "kept"])
        (out_dir / "stats.json").write_text(
            json.dumps(
                {
                    "graph_type": "embedding_knn",
                    "metric_family": "embedding",
                    "metric": args.metric,
                    "k": k,
                    **stats,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        log(f"Wrote {out_dir}: edges={stats['n_edges']:,} components={stats['n_components']} giant={stats['giant_component_frac']:.4f}")


if __name__ == "__main__":
    main()
