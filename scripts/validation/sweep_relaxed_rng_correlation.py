#!/usr/bin/env python3
"""Sweep the RNG lune-aggressiveness parameter epsilon and score correlation-with-time.

Relaxed RNG(eps): keep edge (A,B) iff there is NO witness X with BOTH
    d(A,X) < eps * d(A,B)  AND  d(B,X) < eps * d(A,B).

    eps = 1.0  -> standard RNG (most aggressive)
    eps -> 0    -> complete graph (keep everything)

Implemented exactly over ALL pairs via a single precomputed quantity
    s_ratio(A,B) = min_X max(d(A,X), d(B,X)) / d(A,B)  in [0,1]
    keep (A,B) iff s_ratio(A,B) >= eps.

Computing s_ratio is O(n^3) and is the expensive step (~1-1.5 h at n=11,273);
it is saved to a memmap so the eps sweep itself is instant and resumable.

Self-check: at eps=1.0 the graph must reproduce the exact RNG (17,331 edges,
Spearman ~0.780, Pearson ~0.639), because keep iff s_ratio>=1.0 == standard lune.

Run with the ``hmm`` env. Give this to tmux -- it is long-running.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def corr(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return {"pearson": float(pearsonr(x, y).statistic),
            "spearman": float(spearmanr(x, y).statistic)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-root", type=Path,
                    default=Path("analysis/cohort_validation/39_lamb_uk_rng"))
    ap.add_argument("--out-root", type=Path,
                    default=Path("analysis/cohort_validation/40_lamb_uk_relaxed_rng"))
    ap.add_argument("--eps", type=str,
                    default="1.0,0.5,0.25,0.1,0.05,0.02,0.01,0.0")
    ap.add_argument("--wuhan-index", type=int, default=11272)
    ap.add_argument("--save-graphs", action="store_true",
                    help="also write each eps adjacency as adj.npz")
    args = ap.parse_args()

    eps_list = [float(x) for x in args.eps.split(",") if x.strip()]
    out = args.out_root
    out.mkdir(parents=True, exist_ok=True)

    emb = np.load(args.panel_root / "embeddings" / "embeddings.npy").astype(np.float32)
    nodes = pd.read_csv(args.panel_root / "nodes" / "nodes.csv")
    n = len(nodes)
    w = args.wuhan_index
    assert emb.shape == (n, 2560), f"expected (n,2560), got {emb.shape}"
    assert w == n - 1

    # ---- distance matrix (resume if already written) ----
    d_path = out / "D_cityblock.npy"
    if d_path.exists():
        D = np.load(d_path, mmap_mode="r+")
        log(f"Reusing distance matrix {D.shape} from {d_path}")
    else:
        D = cdist(emb, emb, metric="cityblock").astype(np.float32)
        np.fill_diagonal(D, np.inf)
        dmm = np.lib.format.open_memmap(d_path, mode="w+", dtype=np.float32, shape=D.shape)
        dmm[:] = D
        dmm.flush()
        log(f"Wrote distance matrix {D.shape} ({D.nbytes/1e9:.2f} GB)")

    # ---- s_ratio(A,B) = min_X max(d(A,X), d(B,X)) / d(A,B), upper triangle ----
    s_path = out / "s_ratio.npy"
    if s_path.exists():
        s = np.load(s_path, mmap_mode="r")
        log(f"Reusing s_ratio matrix {s.shape} from {s_path}")
    else:
        s = np.lib.format.open_memmap(s_path, mode="w+", dtype=np.float32, shape=(n, n))
        t0 = time.time()
        for a in range(n):
            # for all b >= a: min over X of max(D[a,X], D[b,X])
            M = np.maximum(D[a][None, :], D[a:, :]).min(axis=1)  # (n-a,)
            vals = (M / D[a, a:]).astype(np.float32)
            s[a, a:] = vals
            s[a:, a] = vals  # symmetric for convenience
            if a % 500 == 0 or a == n - 1:
                s.flush()
                el = time.time() - t0
                log(f"s_ratio row {a:,}/{n:,} ({100*a/n:.1f}%) elapsed {el/60:.1f} min")
        s.flush()
        log(f"s_ratio computed in {(time.time()-t0)/60:.1f} min")

    # ---- time axis (same convention as the reference) ----
    dates = pd.to_datetime(nodes["collection_date"], errors="coerce")
    days = (dates - dates.min()).dt.days.to_numpy(dtype=np.float64)[:w]
    valid = np.isfinite(days)

    iu, ju = np.triu_indices(n, k=1)
    D_upper = np.asarray(D[iu, ju], dtype=np.float32)
    svals_all = np.asarray(s[iu, ju], dtype=np.float32)  # read once, reuse for every eps

    rows = []
    for eps in eps_list:
        keep = svals_all >= eps
        sources = iu[keep]
        targets = ju[keep]
        weights = D_upper[keep]
        n_edges = int(keep.sum())

        # symmetric CSR (weights = cityblock edge length)
        rows_idx = np.concatenate([sources, targets])
        cols_idx = np.concatenate([targets, sources])
        data = np.concatenate([weights, weights])
        adj = csr_matrix((data, (rows_idx, cols_idx)), shape=(n, n), dtype=np.float32)

        ncomp, labels = connected_components(adj, directed=False, return_labels=True)
        deg = np.diff(adj.indptr)
        reachable = labels[:w] == labels[w]
        n_unreachable = int((~reachable).sum())

        dist = dijkstra(adj, directed=False, indices=[w]).ravel()[:w]
        mask = np.isfinite(dist) & reachable & valid
        c = corr(dist[mask], days[mask])

        rows.append({
            "eps": eps,
            "n_edges": n_edges,
            "mean_degree": float(deg.mean()),
            "n_components": int(ncomp),
            "n_unreachable": n_unreachable,
            "pearson": c["pearson"],
            "spearman": c["spearman"],
        })
        log(f"eps={eps:.3g}: edges={n_edges:,} mean_deg={deg.mean():.2f} "
            f"components={ncomp} unreachable={n_unreachable} "
            f"pearson={c['pearson']:.4f} spearman={c['spearman']:.4f}")

        if args.save_graphs:
            gdir = out / "graphs" / f"eps_{str(eps).replace('.', 'p')}"
            gdir.mkdir(parents=True, exist_ok=True)
            save_npz(gdir / "adj.npz", adj)

    df = pd.DataFrame(rows)
    df.to_csv(out / "eps_sweep_summary.csv", index=False)
    with (out / "eps_sweep_summary.json").open("w") as fh:
        json.dump({"eps": eps_list, "rows": rows, "wuhan_index": w, "n_nodes": n},
                  fh, indent=2)
    log("Summary:\n" + df.to_string(index=False))

    # self-check vs standard RNG
    r = df[df["eps"] == 1.0]
    if not r.empty:
        r = r.iloc[0]
        ok = (r["n_edges"] == 17331 and abs(r["spearman"] - 0.7800) < 0.005
              and abs(r["pearson"] - 0.6391) < 0.005)
        log(f"self-check eps=1.0 vs standard RNG (17331 edges, 0.780/0.639): "
            f"edges={r['n_edges']} spearman={r['spearman']:.4f} pearson={r['pearson']:.4f} "
            f"-> {'OK' if ok else 'MISMATCH'}")

    log("Done.")


if __name__ == "__main__":
    main()
