#!/usr/bin/env python3
"""Tau (distance-threshold / epsilon-ball) graph: keep edge (A,B) iff d(A,B) <= Tau.

This is the "Tau" construction from the meeting (short edges carry the clock, long
edges kill it) and from Knyazev's eMST.  Unlike kNN (fixed *count* per node) or the
bottleneck eMST, Tau uses a fixed *distance* threshold, so it adapts to local density.

Sweeps Tau and scores every graph on the two axes that matter:
    * assortativity  : Coleman homophily h by WHO variant
    * time clock     : Spearman/Pearson of (geodesic from Wuhan) vs collection date

alongside the RNG / kNN5 / kNN50 baselines already on disk.

Run with the ``hmm`` env.  Cheap: it just thresholds D_cityblock.npy.
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
from scipy.spatial.distance import cdist
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
    stubs: Counter = Counter()
    for a, b in zip(s, t):
        stubs[a] += 1
        stubs[b] += 1
    frac = {k: v / (2 * m) for k, v in stubs.items()}
    expected = float(sum(v * v for v in frac.values()))
    h = (observed - expected) / (1.0 - expected) if expected != 1.0 else np.nan
    return {"coleman_h": float(h), "observed": observed, "expected": expected, "n_edges_used": m}


def score(adj, w, days, labels) -> dict:
    n = adj.shape[0]
    ncomp, lab = connected_components(adj, directed=False, return_labels=True)
    deg = np.diff(adj.indptr)
    coo = adj.tocoo(copy=False)
    m = coo.row < coo.col
    src, tgt = coo.row[m].astype(np.int64), coo.col[m].astype(np.int64)
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
        "pearson": float(p),
        "spearman": float(sp),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-root", type=Path,
                    default=Path("analysis/cohort_validation/39_lamb_uk_rng"))
    ap.add_argument("--out-root", type=Path,
                    default=Path("analysis/cohort_validation/42_lamb_uk_tau"))
    ap.add_argument("--distance-matrix", type=Path,
                    default=Path("analysis/cohort_validation/40_lamb_uk_relaxed_rng/D_cityblock.npy"))
    ap.add_argument("--tau", type=str,
                    default="0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.1,1.3,1.6,2.0")
    ap.add_argument("--wuhan-index", type=int, default=11272)
    ap.add_argument("--variant-col", type=str, default="variant")
    args = ap.parse_args()

    tau_list = [float(x) for x in args.tau.split(",") if x.strip()]
    out = args.out_root
    out.mkdir(parents=True, exist_ok=True)
    w = args.wuhan_index

    nodes = pd.read_csv(args.panel_root / "nodes" / "nodes.csv")
    n = len(nodes)
    labels = nodes[args.variant_col].astype(str).str.strip().to_numpy()
    dates = pd.to_datetime(nodes["collection_date"], errors="coerce")
    days = (dates - dates.min()).dt.days.to_numpy(dtype=float)[:w]

    if args.distance_matrix.exists():
        D = np.load(args.distance_matrix, mmap_mode="r")
    else:
        emb = np.load(args.panel_root / "embeddings" / "embeddings.npy").astype(np.float32)
        D = cdist(emb, emb, metric="cityblock").astype(np.float32)
        np.fill_diagonal(D, np.inf)

    iu, ju = np.triu_indices(n, k=1)
    D_up = np.asarray(D[iu, ju], dtype=np.float32)
    finite = np.isfinite(D_up)
    print("distance distribution (off-diagonal): "
          f"min={D_up[finite].min():.4f} p50={np.percentile(D_up[finite],50):.4f} "
          f"p90={np.percentile(D_up[finite],90):.4f} p99={np.percentile(D_up[finite],99):.4f} "
          f"max={D_up[finite].max():.4f}")

    rows = []
    for tau in tau_list:
        keep = (D_up <= tau) & finite
        s = iu[keep]
        t = ju[keep]
        wt = D_up[keep].astype(np.float32)
        adj = csr_matrix((np.r_[wt, wt], (np.r_[s, t], np.r_[t, s])), shape=(n, n))
        sc = score(adj, w, days, labels)
        sc["tau"] = tau
        rows.append(sc)
        print(f"tau={tau:.3g}: edges={sc['n_edges']:>9,} mean_deg={sc['mean_degree']:7.2f} "
              f"coleman_h={sc['coleman_h']:.4f} pearson={sc['pearson']:.4f} spearman={sc['spearman']:.4f}")

    for name, gdir in [
        ("rng", args.panel_root / "graphs/cityblock/embedding_rng_exact"),
        ("knn5", args.panel_root / "graphs/cityblock/embedding_knn_k05"),
        ("knn50", args.panel_root / "graphs/cityblock/embedding_knn_k50"),
    ]:
        adj = load_npz(gdir / "adj.npz").tocsr()
        sc = score(adj, w, days, labels)
        sc["graph"] = name
        sc["tau"] = np.nan
        rows.append(sc)
        print(f"{name}: edges={sc['n_edges']:>9,} mean_deg={sc['mean_degree']:7.2f} "
              f"coleman_h={sc['coleman_h']:.4f} pearson={sc['pearson']:.4f} spearman={sc['spearman']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "tau_comparison.csv", index=False)
    with (out / "tau_comparison.json").open("w") as fh:
        json.dump({"tau": tau_list, "rows": rows}, fh, indent=2)
    print("Comparison table:\n" + df.to_string(index=False))
    print(f"wrote {out / 'tau_comparison.csv'}")


if __name__ == "__main__":
    main()
