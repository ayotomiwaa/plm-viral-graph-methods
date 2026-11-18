#!/usr/bin/env python3
"""Extended RNG-out-of-kNN sweep + degree-vs-multiplicity correlation (UK 11,272).

Two meeting follow-ups:

1. Sweep candidate k = 125, 150, 175, 200, 250, 500 (and 5,10,50,100) for the
   RNG-out-of-kNN graph, to find where correlation-with-time peaks.
   RNG-out-of-kNN_k == fullRNG ∩ kNN_k == {RNG edges with min_rank < k}, where
   min_rank(e) = min(#nodes strictly closer to u than v, #nodes strictly closer
   to v than u).  So no kNN graphs need rebuilding: we compute min_rank once for
   every RNG edge and threshold.

2. For every graph, the Spearman/Pearson correlation between node degree and node
   multiplicity (= cluster_size, the number of GISAID records collapsed into each
   unique haplotype node).

Scores each graph on:
    * clock        : Spearman/Pearson of (geodesic from Wuhan) vs collection date
    * assortativity: Coleman homophily h by WHO variant
    * degree x multiplicity correlation

Run with the ``hmm`` env.  min_rank needs the cached cityblock D (memmap, read-only).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.stats import pearsonr, spearmanr


def log(msg: str) -> None:
    print(msg, flush=True)


def coleman_h(labels: np.ndarray, sources: np.ndarray, targets: np.ndarray) -> dict:
    src = labels[sources]
    tgt = labels[targets]
    valid = (src != "") & (src != "nan") & (tgt != "") & (tgt != "nan")
    s, t = src[valid], tgt[valid]
    m = int(len(s))
    if m == 0:
        return {"coleman_h": np.nan, "observed": np.nan}
    observed = float((s == t).mean())
    stubs: Counter = Counter()
    for a, b in zip(s, t):
        stubs[a] += 1
        stubs[b] += 1
    frac = {k: v / (2 * m) for k, v in stubs.items()}
    expected = float(sum(v * v for v in frac.values()))
    h = (observed - expected) / (1.0 - expected) if expected != 1.0 else np.nan
    return {"coleman_h": float(h), "observed": observed}


def score(adj, w: int, days: np.ndarray, labels: np.ndarray, mult: np.ndarray) -> dict:
    n = adj.shape[0]
    ncomp, lab = connected_components(adj, directed=False, return_labels=True)
    deg = np.diff(adj.indptr).astype(float)
    coo = adj.tocoo(copy=False)
    m = coo.row < coo.col
    src, tgt = coo.row[m].astype(np.int64), coo.col[m].astype(np.int64)
    ch = coleman_h(labels, src, tgt)
    reachable = lab[:w] == lab[w]
    dist = dijkstra(adj, directed=False, indices=[w]).ravel()[:w]
    mask = np.isfinite(dist) & reachable & np.isfinite(days)
    p = pearsonr(dist[mask], days[mask]).statistic
    sp = spearmanr(dist[mask], days[mask]).statistic
    # degree x multiplicity over the 11,272 haplotypes (exclude Wuhan node w)
    keep = np.arange(n) != w
    d = deg[keep]
    mm = mult[keep]
    spearman_dm = spearmanr(d, mm).statistic
    pearson_dm = pearsonr(d, mm).statistic
    pearson_d_logm = pearsonr(d, np.log10(np.maximum(mm, 1.0))).statistic
    return {
        "n_edges": int(adj.nnz // 2),
        "mean_degree": float(deg.mean()),
        "degree_std": float(deg.std()),
        "degree_min": int(deg.min()),
        "degree_max": int(deg.max()),
        "n_components": int(ncomp),
        "n_unreachable": int((~reachable).sum()),
        "coleman_h": ch["coleman_h"],
        "observed_same_variant": ch["observed"],
        "pearson_clock": float(p),
        "spearman_clock": float(sp),
        "spearman_deg_mult": float(spearman_dm),
        "pearson_deg_mult": float(pearson_dm),
        "pearson_deg_logmult": float(pearson_d_logm),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-root", type=Path,
                    default=Path("analysis/cohort_validation/39_lamb_uk_rng"))
    ap.add_argument("--out-root", type=Path,
                    default=Path("analysis/cohort_validation/43_lamb_uk_knn_rng"))
    ap.add_argument("--distance-matrix", type=Path,
                    default=Path("analysis/cohort_validation/40_lamb_uk_relaxed_rng/D_cityblock.npy"))
    ap.add_argument("--k-list", type=int, nargs="+",
                    default=[5, 10, 50, 100, 125, 150, 175, 200, 250, 500])
    ap.add_argument("--wuhan-index", type=int, default=11272)
    ap.add_argument("--variant-col", type=str, default="variant")
    args = ap.parse_args()

    out = args.out_root
    out.mkdir(parents=True, exist_ok=True)
    w = args.wuhan_index
    graph_root = args.panel_root / "graphs" / "cityblock"

    nodes = pd.read_csv(args.panel_root / "nodes" / "nodes.csv")
    n = len(nodes)
    labels = nodes[args.variant_col].astype(str).str.strip().to_numpy()
    dates = pd.to_datetime(nodes["collection_date"], errors="coerce")
    days = (dates - dates.min()).dt.days.to_numpy(dtype=float)[:w]
    mult = nodes["cluster_size"].astype(float).to_numpy()

    rng = load_npz(graph_root / "embedding_rng_exact" / "adj.npz").tocsr()
    rows = []

    # --- min_rank for every RNG edge (cached) ---
    mrank_path = out / "rng_edge_min_rank.npy"
    edge_path = out / "rng_edges.npz"
    if mrank_path.exists() and edge_path.exists():
        coo = rng.tocoo(copy=False)
        mm = coo.row < coo.col
        src, tgt = coo.row[mm].astype(np.int64), coo.col[mm].astype(np.int64)
        min_rank = np.load(mrank_path)
        log(f"Reusing cached min_rank for {len(src):,} RNG edges")
    else:
        coo = rng.tocoo(copy=False)
        mm = coo.row < coo.col
        src = coo.row[mm].astype(np.int64)
        tgt = coo.col[mm].astype(np.int64)
        D = np.load(args.distance_matrix, mmap_mode="r")
        min_rank = np.empty(len(src), dtype=np.int32)
        for e in range(len(src)):
            i, j = int(src[e]), int(tgt[e])
            dij = D[i, j]
            ri = int(np.count_nonzero(D[i] < dij))
            rj = int(np.count_nonzero(D[j] < dij))
            min_rank[e] = ri if ri < rj else rj
            if (e + 1) % 5000 == 0:
                log(f"  min_rank {e+1:,}/{len(src):,}")
        np.save(mrank_path, min_rank)
        np.savez(edge_path, src=src, tgt=tgt)
        log(f"Computed min_rank for {len(src):,} RNG edges")

    # --- plain kNN baselines (degree x multiplicity + clock) ---
    for k in [5, 10, 50, 100]:
        knn = load_npz(graph_root / f"embedding_knn_k{k:02d}" / "adj.npz").tocsr()
        sc = score(knn, w, days, labels, mult)
        sc["graph"] = f"knn{k}"
        sc["candidate_k"] = k
        rows.append(sc)
        log(f"knn{k}: clock P={sc['pearson_clock']:.4f} S={sc['spearman_clock']:.4f} "
            f"degxmult Spearman={sc['spearman_deg_mult']:+.4f}")

    # --- full RNG baseline ---
    sc = score(rng, w, days, labels, mult)
    sc["graph"] = "rng_full"
    sc["candidate_k"] = -1
    rows.append(sc)
    log(f"rng_full: clock P={sc['pearson_clock']:.4f} S={sc['spearman_clock']:.4f} "
        f"degxmult Spearman={sc['spearman_deg_mult']:+.4f}")

    # --- RNG-out-of-kNN_k for every k ---
    for k in args.k_list:
        keep = min_rank < k
        s = src[keep]
        t = tgt[keep]
        wt = rng[s, t]  # cityblock weights (float32)
        wt = np.asarray(wt).ravel().astype(np.float32)
        adj = csr_matrix((np.r_[wt, wt], (np.r_[s, t], np.r_[t, s])), shape=(n, n))
        sc = score(adj, w, days, labels, mult)
        sc["graph"] = f"rng_of_knn{k}"
        sc["candidate_k"] = k
        rows.append(sc)
        log(f"rng_of_knn{k}: edges={sc['n_edges']:,} clock P={sc['pearson_clock']:.4f} "
            f"S={sc['spearman_clock']:.4f} degxmult Spearman={sc['spearman_deg_mult']:+.4f}")

    df = pd.DataFrame(rows)
    col_order = ["graph", "candidate_k", "n_edges", "mean_degree", "degree_std",
                 "degree_min", "degree_max", "n_components", "n_unreachable",
                 "coleman_h", "observed_same_variant", "pearson_clock", "spearman_clock",
                 "spearman_deg_mult", "pearson_deg_mult", "pearson_deg_logmult"]
    df = df[col_order]
    df.to_csv(out / "knn_rng_extended_sweep.csv", index=False)
    with (out / "knn_rng_extended_sweep.json").open("w") as fh:
        json.dump({"rows": rows}, fh, indent=2)
    print("\n" + df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nwrote {out/'knn_rng_extended_sweep.csv'} and .json")


if __name__ == "__main__":
    main()
