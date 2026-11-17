#!/usr/bin/env python3
"""Build the eMST (extended minimum spanning tree) and compare it to RNG / kNN5.

Construction (per the meeting):
    1. Build the MST over the cityblock embedding distances.
    2. For every pair (A,B), let L(A,B) = the LONGEST edge on the MST path A->B
       (the "bottleneck" / minimax distance).
    3. Keep edge (A,B) iff  d(A,B) <= (1 + eps) * L(A,B).

    ratio(A,B) = d(A,B) / L(A,B)  >= 1  always (MST cycle property):
        = 1  on MST edges (always kept)
        > 1  long redundant edges (kept only if ratio <= 1+eps; dropped otherwise)

Scores every graph on the two axes the project cares about:
    * assortativity  : Coleman homophily h by WHO variant
    * time clock     : Spearman/Pearson of (geodesic from Wuhan) vs collection date

compared against the RNG and kNN5 already on disk.

Run with the ``hmm`` env.  The bottleneck matrix is ~n^2 (a ~500 MB memmap) and is
resumable; the rest is seconds.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def prim_mst(D: np.ndarray) -> list[tuple[int, int, float]]:
    """Prim's MST on a dense distance matrix (diagonal = inf)."""
    n = D.shape[0]
    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True
    min_dist = D[0].astype(np.float64).copy()
    min_dist[0] = np.inf
    parent = np.zeros(n, dtype=np.int64)
    edges: list[tuple[int, int, float]] = []
    for _ in range(n - 1):
        u = int(np.argmin(min_dist))
        edges.append((int(parent[u]), u, float(min_dist[u])))
        in_tree[u] = True
        min_dist[u] = np.inf
        du = D[u].astype(np.float64)
        upd = (~in_tree) & (du < min_dist)
        min_dist[upd] = du[upd]
        parent[upd] = u
    return edges


def coleman_h(labels: np.ndarray, sources: np.ndarray, targets: np.ndarray) -> dict:
    """Coleman homophily index (degree-weighted expected). labels: str array."""
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


def edge_arrays(adj) -> tuple[np.ndarray, np.ndarray]:
    coo = adj.tocoo(copy=False)
    m = coo.row < coo.col
    return coo.row[m].astype(np.int64), coo.col[m].astype(np.int64)


def score(adj, w, days, labels) -> dict:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-root", type=Path,
                    default=Path("analysis/cohort_validation/39_lamb_uk_rng"))
    ap.add_argument("--out-root", type=Path,
                    default=Path("analysis/cohort_validation/41_lamb_uk_emst"))
    ap.add_argument("--distance-matrix", type=Path,
                    default=Path("analysis/cohort_validation/40_lamb_uk_relaxed_rng/D_cityblock.npy"))
    ap.add_argument("--eps", type=str, default="0.0,0.1,0.25,0.5,1.0")
    ap.add_argument("--wuhan-index", type=int, default=11272)
    ap.add_argument("--variant-col", type=str, default="variant")
    args = ap.parse_args()

    eps_list = [float(x) for x in args.eps.split(",") if x.strip()]
    out = args.out_root
    out.mkdir(parents=True, exist_ok=True)
    w = args.wuhan_index

    nodes = pd.read_csv(args.panel_root / "nodes" / "nodes.csv")
    n = len(nodes)
    labels = nodes[args.variant_col].astype(str).str.strip().to_numpy()
    dates = pd.to_datetime(nodes["collection_date"], errors="coerce")
    days = (dates - dates.min()).dt.days.to_numpy(dtype=float)[:w]

    # --- distance matrix ---
    if args.distance_matrix.exists():
        D = np.load(args.distance_matrix, mmap_mode="r")
        log(f"Reusing D {D.shape}")
    else:
        emb = np.load(args.panel_root / "embeddings" / "embeddings.npy").astype(np.float32)
        D = cdist(emb, emb, metric="cityblock").astype(np.float32)
        np.fill_diagonal(D, np.inf)

    # --- MST ---
    t0 = time.time()
    mst_edges = prim_mst(D)
    log(f"MST built: {len(mst_edges):,} edges in {time.time()-t0:.1f}s")

    # sanity: the MST must be a single connected spanning tree
    if len(mst_edges) != n - 1:
        raise RuntimeError(f"MST has {len(mst_edges):,} edges, expected {n-1:,}")
    _rows = [e[0] for e in mst_edges] + [e[1] for e in mst_edges]
    _cols = [e[1] for e in mst_edges] + [e[0] for e in mst_edges]
    _tmp = csr_matrix((np.ones(len(_rows)), (_rows, _cols)), shape=(n, n))
    _ncomp = connected_components(_tmp, directed=False, return_labels=False)
    if _ncomp != 1:
        raise RuntimeError(f"MST is not connected: {_ncomp} components")
    log("MST connectivity check: 1 component, OK")

    # --- bottleneck matrix L(A,B) = max edge on MST path ---
    L_path = out / "bottleneck.npy"
    if L_path.exists():
        L = np.load(L_path, mmap_mode="r")
        log(f"Reusing bottleneck matrix {L.shape}")
    else:
        adj_list = [[] for _ in range(n)]
        for u, v, wt in mst_edges:
            adj_list[u].append((v, wt))
            adj_list[v].append((u, wt))
        L = np.lib.format.open_memmap(L_path, mode="w+", dtype=np.float32, shape=(n, n))
        t0 = time.time()
        for src in range(n):
            # iterative DFS propagating the running max edge weight; write the whole
            # row in one go (single memmap write, far faster than per-element).
            row = np.zeros(n, dtype=np.float32)
            stack = [(src, 0.0, -1)]
            while stack:
                node, maxw, parent = stack.pop()
                for nbr, wt in adj_list[node]:
                    if nbr == parent:
                        continue
                    nmax = max(maxw, wt)
                    row[nbr] = nmax
                    stack.append((nbr, nmax, node))
            L[src] = row
            if src % 1000 == 0 or src == n - 1:
                L.flush()
                log(f"bottleneck row {src:,}/{n:,} ({100*src/n:.1f}%) {time.time()-t0:.0f}s")
        log(f"bottleneck matrix done in {(time.time()-t0)/60:.1f} min")

    # --- ratio = d / L, threshold per eps ---
    iu, ju = np.triu_indices(n, k=1)
    D_up = np.asarray(D[iu, ju], dtype=np.float32)
    L_up = np.asarray(L[iu, ju], dtype=np.float32)
    ratio = D_up / L_up
    log(f"ratio distribution: min={ratio.min():.4f} median={np.median(ratio):.4f} "
        f"max={ratio.max():.4f}  p90={np.percentile(ratio,90):.4f} p99={np.percentile(ratio,99):.4f}")

    rows = []
    for eps in eps_list:
        keep = ratio <= (1.0 + eps)
        s = iu[keep]
        t = ju[keep]
        wt = D_up[keep].astype(np.float32)
        adj = csr_matrix((np.r_[wt, wt], (np.r_[s, t], np.r_[t, s])), shape=(n, n))
        sc = score(adj, w, days, labels)
        sc["eps"] = eps
        rows.append(sc)
        log(f"eMST eps={eps:.3g}: edges={sc['n_edges']:,} mean_deg={sc['mean_degree']:.2f} "
            f"coleman_h={sc['coleman_h']:.4f} pearson={sc['pearson']:.4f} spearman={sc['spearman']:.4f}")

    # --- baselines: RNG and kNN5 ---
    for name, gdir in [
        ("rng", args.panel_root / "graphs/cityblock/embedding_rng_exact"),
        ("knn5", args.panel_root / "graphs/cityblock/embedding_knn_k05"),
    ]:
        adj = load_npz(gdir / "adj.npz").tocsr()
        sc = score(adj, w, days, labels)
        sc["graph"] = name
        sc["eps"] = np.nan
        rows.append(sc)
        log(f"{name}: edges={sc['n_edges']:,} mean_deg={sc['mean_degree']:.2f} "
            f"coleman_h={sc['coleman_h']:.4f} pearson={sc['pearson']:.4f} spearman={sc['spearman']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "emst_comparison.csv", index=False)
    with (out / "emst_comparison.json").open("w") as fh:
        json.dump({"eps": eps_list, "rows": rows}, fh, indent=2)
    log("Comparison table:\n" + df.to_string(index=False))
    log("Done.")


if __name__ == "__main__":
    main()
