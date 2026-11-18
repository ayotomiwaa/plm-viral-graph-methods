#!/usr/bin/env python3
"""RNG-out-of-kNN: restrict the RNG candidate set to symmetric kNN and score.

Per the meeting (transcript "Epsilon MST and RNG Graph Analysis.txt"): instead of
building the RNG over the *complete* graph, build it over the kNN candidate edges
(symmetric kNN, exact full-dataset lune witness), for k in {5, 10, 50, 100}, and
compute Coleman assortativity + correlation-with-time (clock).

Because the exact-witness RNG test ``no w with d(u,w)<dij and d(v,w)<dij`` is
independent of the candidate set,  RNG-out-of-kNN_k == fullRNG ∩ kNN_k.  Both
adjacencies are already on disk, so this is computed by intersection (no distance
matrix, no re-pruning).

Scores every graph on the two project axes:
    * assortativity : Coleman homophily h by WHO variant (+ observed same-variant)
    * clock         : Spearman/Pearson of (geodesic from Wuhan) vs collection date

Run with the ``hmm`` env.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.stats import pearsonr, spearmanr


def coleman_h(labels: np.ndarray, sources: np.ndarray, targets: np.ndarray) -> dict:
    src = labels[sources]
    tgt = labels[targets]
    valid = (src != "") & (src != "nan") & (tgt != "") & (tgt != "nan")
    s, t = src[valid], tgt[valid]
    m = int(len(s))
    if m == 0:
        return {"coleman_h": np.nan, "observed": np.nan, "expected": np.nan, "n_edges_used": 0}
    observed = float((s == t).mean())
    from collections import Counter
    stubs: Counter = Counter()
    for a, b in zip(s, t):
        stubs[a] += 1
        stubs[b] += 1
    frac = {k: v / (2 * m) for k, v in stubs.items()}
    expected = float(sum(v * v for v in frac.values()))
    h = (observed - expected) / (1.0 - expected) if expected != 1.0 else np.nan
    return {"coleman_h": float(h), "observed": observed, "expected": expected, "n_edges_used": m}


def edge_arrays(adj) -> tuple[np.ndarray, np.ndarray]:
    coo = adj.tocoo(copy=False)
    m = coo.row < coo.col
    return coo.row[m].astype(np.int64), coo.col[m].astype(np.int64)


def score(adj, w: int, days: np.ndarray, labels: np.ndarray) -> dict:
    n = adj.shape[0]
    ncomp, lab = connected_components(adj, directed=False, return_labels=True)
    deg = np.diff(adj.indptr)
    src, tgt = edge_arrays(adj)
    ch = coleman_h(labels, src, tgt)
    reachable = lab[:w] == lab[w]
    dist = dijkstra(adj, directed=False, indices=[w]).ravel()[:w]
    mask = np.isfinite(dist) & reachable & np.isfinite(days)
    p = pearsonr(dist[mask], days[mask]).statistic
    sp = spearmanr(dist[mask], days[mask]).statistic
    return {
        "n_edges": int(adj.nnz // 2),
        "mean_degree": float(deg.mean()),
        "n_components": int(ncomp),
        "n_unreachable": int((~reachable).sum()),
        "coleman_h": ch["coleman_h"],
        "observed_same_variant": ch["observed"],
        "pearson": float(p),
        "spearman": float(sp),
    }


def intersect(rng, knn):
    """Elementwise intersection keeping the RNG weights."""
    inter = rng.multiply((knn != 0).astype(np.float32))
    inter.eliminate_zeros()
    return inter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-root", type=Path,
                    default=Path("analysis/cohort_validation/39_lamb_uk_rng"))
    ap.add_argument("--out-root", type=Path,
                    default=Path("analysis/cohort_validation/43_lamb_uk_knn_rng"))
    ap.add_argument("--k-list", type=int, nargs="+", default=[5, 10, 50, 100])
    ap.add_argument("--wuhan-index", type=int, default=11272)
    ap.add_argument("--variant-col", type=str, default="variant")
    args = ap.parse_args()

    out = args.out_root
    out.mkdir(parents=True, exist_ok=True)
    w = args.wuhan_index
    graph_root = args.panel_root / "graphs" / "cityblock"

    nodes = pd.read_csv(args.panel_root / "nodes" / "nodes.csv")
    labels = nodes[args.variant_col].astype(str).str.strip().to_numpy()
    dates = pd.to_datetime(nodes["collection_date"], errors="coerce")
    days = (dates - dates.min()).dt.days.to_numpy(dtype=float)[:w]

    rng = load_npz(graph_root / "embedding_rng_exact" / "adj.npz").tocsr()
    rows = []

    # --- plain kNN baselines ---
    for k in args.k_list:
        knn = load_npz(graph_root / f"embedding_knn_k{k:02d}" / "adj.npz").tocsr()
        sc = score(knn, w, days, labels)
        sc["graph"] = f"knn{k}"
        sc["candidate_k"] = k
        rows.append(sc)
        print(f"knn{k}:   edges={sc['n_edges']:>7,} mean_deg={sc['mean_degree']:5.2f} "
              f"comp={sc['n_components']:>3} unreach={sc['n_unreachable']:>4} "
              f"h={sc['coleman_h']:.4f} pearson={sc['pearson']:.4f} spearman={sc['spearman']:.4f}")

    # --- full RNG baseline ---
    sc = score(rng, w, days, labels)
    sc["graph"] = "rng_full"
    sc["candidate_k"] = -1
    rows.append(sc)
    print(f"rng_full: edges={sc['n_edges']:>7,} mean_deg={sc['mean_degree']:5.2f} "
          f"comp={sc['n_components']:>3} unreach={sc['n_unreachable']:>4} "
          f"h={sc['coleman_h']:.4f} pearson={sc['pearson']:.4f} spearman={sc['spearman']:.4f}")

    # --- RNG-out-of-kNN_k = fullRNG ∩ kNN_k ---
    for k in args.k_list:
        knn = load_npz(graph_root / f"embedding_knn_k{k:02d}" / "adj.npz").tocsr()
        inter = intersect(rng, knn)
        sc = score(inter, w, days, labels)
        sc["graph"] = f"rng_of_knn{k}"
        sc["candidate_k"] = k
        rows.append(sc)
        print(f"rng_of_knn{k}: edges={sc['n_edges']:>7,} mean_deg={sc['mean_degree']:5.2f} "
              f"comp={sc['n_components']:>3} unreach={sc['n_unreachable']:>4} "
              f"h={sc['coleman_h']:.4f} pearson={sc['pearson']:.4f} spearman={sc['spearman']:.4f}")

    df = pd.DataFrame(rows)
    col_order = ["graph", "candidate_k", "n_edges", "mean_degree", "n_components",
                 "n_unreachable", "coleman_h", "observed_same_variant", "pearson", "spearman"]
    df = df[col_order]
    df.to_csv(out / "knn_rng_comparison.csv", index=False)
    with (out / "knn_rng_comparison.json").open("w") as fh:
        json.dump({"rows": rows}, fh, indent=2)
    print("\n" + df.to_string(index=False))
    print(f"\nwrote {out/'knn_rng_comparison.csv'} and .json")


if __name__ == "__main__":
    main()
