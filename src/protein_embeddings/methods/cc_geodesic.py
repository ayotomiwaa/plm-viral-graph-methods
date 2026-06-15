#!/usr/bin/env python3
import argparse
import os
import sys
from io import StringIO

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.sparse.csgraph import dijkstra, connected_components


# I/O helpers
def read_ids(path):
    with open(path) as f:
        return [x.strip() for x in f if x.strip()]


# Tree builders
def build_nj_tree(D, labels, prefer="auto"):
    """
    Returns (tree_builder_name, tree_object)
    tree_builder_name in {"skbio","biopython"}.
    """
    if prefer in ("auto", "skbio"):
        try:
            from skbio import DistanceMatrix
            from skbio.tree import nj

            dm = DistanceMatrix(D, labels)
            tree = nj(dm)  # TreeNode
            return "skbio", tree
        except Exception:
            if prefer == "skbio":
                raise

    # Biopython fallback
    from Bio.Phylo.TreeConstruction import DistanceMatrix as BioDM
    from Bio.Phylo.TreeConstruction import DistanceTreeConstructor

    n = len(labels)
    lower = [[float(D[i, j]) for j in range(i + 1)] for i in range(n)]
    dm = BioDM(names=list(labels), matrix=lower)
    tree = DistanceTreeConstructor().nj(dm)
    return "biopython", tree


def clip_negative_branch_lengths(tree_kind, tree):
    if tree_kind == "skbio":
        for n in tree.postorder():
            if n.length is not None and n.length < 0:
                n.length = 0.0
    else:
        for clade in tree.find_clades():
            if clade.branch_length is not None and clade.branch_length < 0:
                clade.branch_length = 0.0


def save_newick(tree_kind, tree, outpath):
    # scikit-bio: TreeNode.write(...) (more stable than to_newick across versions)
    if tree_kind == "skbio":
        buf = StringIO()
        try:
            tree.write(buf, format="newick")
        except TypeError:
            tree.write(buf)  # older signatures
        nwk = buf.getvalue().strip()
        if not nwk.endswith(";"):
            nwk += ";"
        with open(outpath, "w") as f:
            f.write(nwk + "\n")
        return

    # biopython recursive writer: bump recursion limit
    try:
        n_tips = tree.count_terminals()
    except Exception:
        n_tips = 5000
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10_000 + 5 * n_tips))

    from Bio import Phylo
    Phylo.write(tree, outpath, "newick")


def patristic_distance_matrix(tree_kind, tree, labels):
    n = len(labels)
    T = np.zeros((n, n), dtype=np.float64)

    if tree_kind == "skbio":
        tips = {tip.name: tip for tip in tree.tips()}
        for i in range(n):
            a = tips[labels[i]]
            for j in range(i + 1, n):
                b = tips[labels[j]]
                d = a.distance(b)
                T[i, j] = T[j, i] = float(d)
    else:
        for i in range(n):
            for j in range(i + 1, n):
                d = tree.distance(labels[i], labels[j])
                T[i, j] = T[j, i] = float(d)

    return T


def rsd(D, T):
    iu = np.triu_indices_from(D, k=1)
    num = np.sum((D[iu] - T[iu]) ** 2)
    den = np.sum((D[iu]) ** 2)
    return float(np.sqrt(num / den)) if den > 0 else float("nan")


# Core pipeline
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Folder with selected_ids.txt and D_manhattan.npy")
    ap.add_argument("--graph-npz", required=True, help="Pruned CSR graph (save_npz output) built on ~20k")
    ap.add_argument("--graph-ids", required=True, help="IDs in the exact node order used to build the graph")
    ap.add_argument("--selected", default="selected_ids.txt")
    ap.add_argument("--manhattan", default="D_manhattan.npy")

    ap.add_argument("--out-tag", default="cc", help="suffix: selected_ids_<tag>.txt, D_*_<tag>.npy")
    ap.add_argument("--prefer-tree-builder", default="auto", choices=["auto", "skbio", "biopython"])
    ap.add_argument("--clip-negative-branches", action="store_true")
    args = ap.parse_args()

    run = args.run_dir.rstrip("/")

    # Load graph + component labels (global)
    G = load_npz(args.graph_npz).tocsr()
    n_comp, comp = connected_components(G, directed=False, return_labels=True)
    print(f"[INFO] graph: N={G.shape[0]} nnz={G.nnz} components={n_comp}")

    graph_ids = read_ids(args.graph_ids)
    id2i = {s: i for i, s in enumerate(graph_ids)}

    # Load selected IDs from run folder, map to graph indices
    sel = read_ids(os.path.join(run, args.selected))
    idx = []
    missing = []
    for s in sel:
        if s in id2i:
            idx.append(id2i[s])
        else:
            missing.append(s)
    if missing:
        raise SystemExit(f"[ERROR] {len(missing)} selected IDs not found in graph-ids. Example: {missing[:5]}")
    idx = np.array(idx, dtype=int)

    # Dominant component among selected
    comp_sel = comp[idx]
    major = np.bincount(comp_sel).argmax()
    keep = (comp_sel == major)

    kept = int(keep.sum())
    dropped = int((~keep).sum())
    print(f"[INFO] selected={len(sel)} kept_in_major_component={kept} dropped={dropped} (component={major})")
    if kept < 3:
        raise SystemExit("[ERROR] Major component too small; cannot build NJ tree.")

    idx_cc = idx[keep]
    sel_cc = [s for s, k in zip(sel, keep) if k]

    tag = args.out_tag
    selected_cc_path = os.path.join(run, f"selected_ids_{tag}.txt")
    with open(selected_cc_path, "w") as f:
        f.write("\n".join(sel_cc) + "\n")

    np.save(os.path.join(run, f"keep_mask_{tag}.npy"), keep)

    # (2) rerun Dijkstra only from CC nodes, but on full graph (paths can use any nodes)
    print(f"[INFO] dijkstra: sources m={len(idx_cc)} over full graph N={G.shape[0]} ...")
    D_rows = dijkstra(G, directed=False, indices=idx_cc)  # (m, N)
    D_geo = D_rows[:, idx_cc]                             # (m, m)

    # sanitize
    D_geo = D_geo.astype(np.float64, copy=False)
    D_geo = 0.5 * (D_geo + D_geo.T)
    np.fill_diagonal(D_geo, 0.0)

    if not np.isfinite(D_geo).all():
        n_inf = np.sum(~np.isfinite(D_geo))
        raise SystemExit(f"[ERROR] D_geodesic_{tag}.npy still non-finite ({n_inf} entries).")

    geo_path = os.path.join(run, f"D_geodesic_{tag}.npy")
    np.save(geo_path, D_geo)

    # (3) slice Manhattan to same CC node set
    D_man = np.load(os.path.join(run, args.manhattan))
    D_man_cc = D_man[np.ix_(keep, keep)].astype(np.float64, copy=False)
    D_man_cc = 0.5 * (D_man_cc + D_man_cc.T)
    np.fill_diagonal(D_man_cc, 0.0)

    man_cc_path = os.path.join(run, f"D_manhattan_{tag}.npy")
    np.save(man_cc_path, D_man_cc)

    # (4) build NJ + compute RSD for both CC matrices
    rows = []
    for dist_name, Dmat in [(f"geodesic_{tag}", D_geo), (f"manhattan_{tag}", D_man_cc)]:
        tree_kind, tree = build_nj_tree(Dmat, sel_cc, prefer=args.prefer_tree_builder)
        if args.clip_negative_branches:
            clip_negative_branch_lengths(tree_kind, tree)

        T = patristic_distance_matrix(tree_kind, tree, sel_cc)
        r = rsd(Dmat, T)

        np.save(os.path.join(run, f"T_{dist_name}.npy"), T)
        save_newick(tree_kind, tree, os.path.join(run, f"tree_{dist_name}.nwk"))

        rows.append({"distance": dist_name, "tree_builder": tree_kind, "n": len(sel_cc), "rsd": r})

    # update results.csv (append/replace these two distance rows)
    results_path = os.path.join(run, "results.csv")
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        df = df[~df["distance"].astype(str).isin([r["distance"] for r in rows])]
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    else:
        df = pd.DataFrame(rows)

    df.to_csv(results_path, index=False)

    print("\n[OK] Wrote:")
    print(" ", selected_cc_path)
    print(" ", geo_path)
    print(" ", man_cc_path)
    print(" ", results_path)
    print("\n[SUMMARY] (sorted by rsd)")
    show = df.sort_values("rsd")[["distance", "tree_builder", "n", "rsd"]]
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()