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
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


# ============================================================
# Utilities
# ============================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def fix_partial_date(x: object) -> str:
    s = str(x)
    if s.endswith('-00-00'):
        return s[:4] + '-06-15'
    if s.endswith('-00'):
        return s[:7] + '-15'
    return s


def parse_embedding_ids(ids_path: Path) -> pd.DataFrame:
    rows = []
    with open(ids_path, 'r') as f:
        for row, line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            parts = s.split('|')
            if len(parts) >= 3:
                lineage, date, accession = parts[0].strip(), parts[1].strip(), parts[2].strip()
            else:
                lineage, date, accession = '', '', parts[-1].strip()
            rows.append({
                'embedding_row': row,
                'accession': accession,
                'collection_date_from_ids': date,
                'lineage_from_ids': lineage,
            })
    return pd.DataFrame(rows)


def load_npz(npz_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    data = np.load(npz_path, allow_pickle=True)

    X = np.asarray(data['embeddings_mean'], dtype=np.float32)
    accessions = np.asarray(data['accession_ids']).astype(str)
    dates = pd.to_datetime(pd.Series(data['dates']).map(fix_partial_date), errors='coerce')

    nodes = pd.DataFrame({
        'node_id': np.arange(X.shape[0], dtype=int),
        'accession': accessions,
        'collection_date': dates,
    })

    if 'repr_layer' in data:
        try:
            nodes['repr_layer'] = str(data['repr_layer'])
        except Exception:
            pass
    if 'model_ckpt' in data:
        try:
            nodes['model_ckpt'] = str(data['model_ckpt'])
        except Exception:
            pass

    print(f"[LOAD] {npz_path.name}: n={X.shape[0]:,}, dim={X.shape[1]}")
    return X, nodes


def load_benchmark_npy(
    embeddings_path: Path,
    ids_path: Path,
    metadata_path: Path,
    variant: str,
    variant_col: str = 'variant_bucket',
    include_accessions: set[str] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    X_all = np.load(embeddings_path, mmap_mode='r')
    id_df = parse_embedding_ids(ids_path)
    if len(id_df) != X_all.shape[0]:
        raise ValueError(f"ID count {len(id_df):,} != embedding rows {X_all.shape[0]:,}")

    meta = pd.read_csv(metadata_path, low_memory=False)
    if 'accession' not in meta.columns:
        raise ValueError(f"{metadata_path} must contain an 'accession' column")
    if variant_col not in meta.columns:
        raise ValueError(f"{metadata_path} must contain variant column '{variant_col}'")

    meta = meta.copy()
    meta['accession'] = meta['accession'].astype(str).str.strip()
    id_df['accession'] = id_df['accession'].astype(str).str.strip()
    merged = id_df.merge(meta, on='accession', how='left', suffixes=('', '_meta'))

    include_accessions = include_accessions or set()
    variant_mask = merged[variant_col].astype(str).str.casefold() == variant.casefold()
    include_mask = merged['accession'].astype(str).isin(include_accessions)
    mask = variant_mask | include_mask
    sub = merged.loc[mask].copy()
    if sub.empty:
        available = sorted(meta[variant_col].dropna().astype(str).unique().tolist())
        raise RuntimeError(f"No rows matched variant={variant!r}; available variants include {available[:20]}")

    sub['collection_date'] = pd.to_datetime(
        sub.get('collection_date', sub['collection_date_from_ids']).map(fix_partial_date),
        errors='coerce',
    )
    if 'lineage' not in sub.columns:
        sub['lineage'] = sub['lineage_from_ids']

    row_idx = sub['embedding_row'].astype(int).to_numpy()
    X = np.asarray(X_all[row_idx], dtype=np.float32)

    nodes = pd.DataFrame({
        'node_id': np.arange(len(sub), dtype=int),
        'accession': sub['accession'].astype(str).to_numpy(),
        'collection_date': sub['collection_date'].to_numpy(),
        'lineage': sub['lineage'].astype(str).to_numpy(),
        'variant_bucket': sub[variant_col].astype(str).to_numpy(),
        'embedding_row': row_idx,
    })
    if 'location' in sub.columns:
        nodes['location'] = sub['location'].astype(str).to_numpy()

    print(
        f"[LOAD benchmark] {embeddings_path.name}: total n={X_all.shape[0]:,}, dim={X_all.shape[1]}, "
        f"{variant} n={X.shape[0]:,}, included_reference={int(include_mask.loc[mask].sum())}"
    )
    return X, nodes


# ============================================================
# Distance / kNN
# ============================================================

def normalize_metric(metric: str) -> str:
    return 'cityblock' if metric == 'manhattan' else metric


def exact_knn(X: np.ndarray, k: int, metric: str) -> tuple[np.ndarray, np.ndarray]:
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm='auto')
    nn.fit(X)
    dist, idx = nn.kneighbors(X, return_distance=True)
    # drop self
    idx = idx[:, 1:]
    dist = dist[:, 1:]
    print(f"[KNN] built exact {metric} kNN with k={k}")
    return idx.astype(np.int32), dist.astype(np.float32)


def pairwise_matrix(X: np.ndarray, metric: str) -> np.ndarray:
    print(f"[DIST] computing exact pairwise {metric} distances...")
    D = cdist(X, X, metric=metric).astype(np.float32)
    print('[DIST] pairwise matrix ready')
    return D


# ============================================================
# Graph builders
# ============================================================

def build_symmetric_knn_edges(knn_idx: np.ndarray, knn_dist: np.ndarray) -> Dict[Tuple[int, int], float]:
    """
    Union-symmetrized kNN:
      keep undirected edge (i,j) if i->j OR j->i exists.
      edge weight = min observed directed distance.
    """
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
    print(f"[GRAPH] symmetric_knn: edges={len(edges):,}")
    return edges


def build_embedding_rng_edges(D: np.ndarray) -> Dict[Tuple[int, int], float]:
    """
    Exact RNG on the complete graph:
      keep edge (i,j) iff there is NO witness r with
      max(D[i,r], D[j,r]) < D[i,j]
    WARNING: O(n^3) in the straightforward form; intended only for modest N.
    """
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
    print(f"[GRAPH] embedding_rng: edges={len(edges):,}")
    return edges


def rng_prune_edges(
    candidate_edges: Dict[Tuple[int, int], float],
    D: np.ndarray,
) -> tuple[Dict[Tuple[int, int], float], List[Tuple[int, int, float]]]:
    """
    Exact RNG pruning on a candidate edge set using ALL points as witnesses.

    Prune edge (i,j) if there exists r != i,j such that:
      max(D[i,r], D[j,r]) < D[i,j]
    """
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

    print(f"[PRUNE] symmetric_knn_rng_exact_witness: kept={len(kept):,}, pruned={len(pruned):,}")
    return kept, pruned


# ============================================================
# Sparse graph helpers / stats / spectral layout
# ============================================================

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
        'n_nodes': int(adj.shape[0]),
        'n_edges': int(adj.nnz // 2),
        'n_components': int(n_comp),
        'giant_label': giant_label,
        'giant_component_size': giant_size,
        'giant_component_frac': float(giant_size / adj.shape[0] if adj.shape[0] else 0.0),
        'component_labels': labels,
        'degree_counts': deg,
    }


def spectral_layout_giant(adj: csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute spectral coordinates on the giant component only.
    Non-giant nodes receive NaN coordinates.
    """
    n = adj.shape[0]
    coords = np.full((n, 2), np.nan, dtype=np.float32)

    if adj.nnz == 0:
        return coords, np.array([], dtype=np.float32)

    n_comp, labels = connected_components(adj, directed=False, return_labels=True)
    vc = pd.Series(labels).value_counts().sort_values(ascending=False)
    if len(vc) == 0:
        return coords, np.array([], dtype=np.float32)

    giant_label = int(vc.index[0])
    giant_nodes = np.where(labels == giant_label)[0]

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

    vals, vecs = eigsh(L, k=k_eigs, which='SM')
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


# ============================================================
# Export
# ============================================================

def write_run(
    out_dir: Path,
    nodes: pd.DataFrame,
    edges: Dict[Tuple[int, int], float],
    graph_type: str,
    metric: str,
    k: int | None = None,
    pruned_edges: List[Tuple[int, int, float]] | None = None,
) -> None:
    out_dir = ensure_dir(out_dir)

    adj = edges_to_csr(len(nodes), edges)
    stats = graph_stats(adj)
    coords, eigvals = spectral_layout_giant(adj)

    nodes_out = nodes.copy()
    nodes_out['graph_type'] = graph_type
    nodes_out['metric'] = metric
    nodes_out['k'] = np.nan if k is None else int(k)
    nodes_out['component_id'] = stats['component_labels']
    nodes_out['degree'] = stats['degree_counts']
    nodes_out['x_spec'] = coords[:, 0]
    nodes_out['y_spec'] = coords[:, 1]

    valid = np.isfinite(nodes_out['x_spec'].to_numpy())
    x_rank = np.full(len(nodes_out), np.nan, dtype=np.float32)
    valid_idx = np.where(valid)[0]
    if len(valid_idx):
        order = np.argsort(nodes_out.loc[valid_idx, 'x_spec'].to_numpy())
        x_rank[valid_idx[order]] = np.arange(len(valid_idx), dtype=np.float32)
    nodes_out['x_rank'] = x_rank

    nodes_out.to_csv(out_dir / 'nodes.csv', index=False)
    nodes_out[['node_id', 'x_spec', 'y_spec']].to_csv(out_dir / 'layout_spectral.csv', index=False)
    save_npz(out_dir / 'adj.npz', adj)
    np.save(out_dir / 'spectral_eigenvalues.npy', eigvals)

    edge_rows = [
        {'source': i, 'target': j, 'weight': w, 'status': 'kept'}
        for (i, j), w in edges.items()
    ]
    if pruned_edges:
        edge_rows.extend(
            {'source': i, 'target': j, 'weight': w, 'status': 'pruned'}
            for (i, j, w) in pruned_edges
        )
    pd.DataFrame(edge_rows).to_csv(out_dir / 'edges.csv', index=False)

    meta = {
        'graph_type': graph_type,
        'witness_type': 'full_dataset' if graph_type == 'symmetric_knn_rng_exact_witness' else None,
        'metric': metric,
        'k': None if k is None else int(k),
        'n_nodes': int(stats['n_nodes']),
        'n_edges': int(stats['n_edges']),
        'n_components': int(stats['n_components']),
        'giant_label': int(stats['giant_label']),
        'giant_component_size': int(stats['giant_component_size']),
        'giant_component_frac': float(stats['giant_component_frac']),
        'n_pruned_edges': int(len(pruned_edges) if pruned_edges else 0),
    }
    with open(out_dir / 'stats.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(
        f"[DONE] {out_dir.name}: nodes={meta['n_nodes']:,} edges={meta['n_edges']:,} "
        f"giant={meta['giant_component_size']:,} ({meta['giant_component_frac']:.1%})"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(description='Build exact-witness RNG graph variants')
    input_group = ap.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--npz', help='Path to Maryam-style .npz file')
    input_group.add_argument('--embeddings-npy', help='Path to benchmark embeddings .npy')
    ap.add_argument('--ids', help='ids.txt for --embeddings-npy; format lineage|date|accession')
    ap.add_argument('--metadata', help='metadata.csv for --embeddings-npy')
    ap.add_argument('--variant-col', default='variant_bucket', help='Variant column in metadata.csv')
    ap.add_argument(
        '--include-accession',
        action='append',
        default=['NC_045512.2'],
        help='Always include this accession after variant filtering. Can be repeated. Default: NC_045512.2',
    )
    ap.add_argument(
        '--no-default-reference',
        action='store_true',
        help='Do not automatically include NC_045512.2 when filtering benchmark embeddings.',
    )
    ap.add_argument('--dataset-label', default=None, help='Output dataset label, e.g. ESM2_20k')
    ap.add_argument('--variant', required=True, help='Variant label, e.g. Alpha')
    ap.add_argument('--out-root', required=True, help='Root output directory')
    ap.add_argument('--metric', default='euclidean', choices=['euclidean', 'manhattan', 'cityblock', 'cosine'])
    ap.add_argument('--ks', nargs='+', type=int, default=[5, 10, 15, 25, 50], help='k values for symmetric kNN variants')
    ap.add_argument('--max-embedding-rng-n', type=int, default=5000, help='Skip exact embedding RNG if n exceeds this cutoff')
    ap.add_argument('--skip-embedding-rng', action='store_true', help='Skip complete-graph exact RNG reference')
    args = ap.parse_args()

    metric = normalize_metric(args.metric)
    if args.npz:
        X, nodes = load_npz(Path(args.npz))
        dataset_label = args.dataset_label or Path(args.npz).stem
    else:
        if not args.ids or not args.metadata:
            raise SystemExit('--embeddings-npy requires --ids and --metadata')
        X, nodes = load_benchmark_npy(
            embeddings_path=Path(args.embeddings_npy),
            ids_path=Path(args.ids),
            metadata_path=Path(args.metadata),
            variant=args.variant,
            variant_col=args.variant_col,
            include_accessions=set([] if args.no_default_reference else args.include_accession),
        )
        dataset_label = args.dataset_label or Path(args.embeddings_npy).parent.name

    root = ensure_dir(Path(args.out_root) / dataset_label / args.variant)
    nodes.to_csv(root / 'canonical_nodes.csv', index=False)
    with open(root / 'input_manifest.json', 'w') as f:
        json.dump({
            'dataset_label': dataset_label,
            'variant': args.variant,
            'metric': metric,
            'ks': args.ks,
            'npz': args.npz,
            'embeddings_npy': args.embeddings_npy,
            'ids': args.ids,
            'metadata': args.metadata,
            'variant_col': args.variant_col,
            'include_accession': [] if args.no_default_reference else args.include_accession,
            'n_nodes': int(X.shape[0]),
            'embedding_dim': int(X.shape[1]),
            'graph_naming': {
                'symmetric_knn': 'plain symmetric kNN baseline',
                'symmetric_knn_rng_exact_witness': 'symmetric kNN candidate edges pruned with all points as RNG witnesses',
                'embedding_rng_exact': 'complete-graph exact RNG reference',
            },
        }, f, indent=2)

    # full pairwise distance matrix once
    D = pairwise_matrix(X, metric=metric)

    # 1) embedding_rng on the complete graph (exact RNG)
    if args.skip_embedding_rng:
        print("[SKIP] embedding_rng_exact skipped by --skip-embedding-rng")
    elif X.shape[0] <= args.max_embedding_rng_n:
        emb_rng_edges = build_embedding_rng_edges(D)
        write_run(
            out_dir=root / 'embedding_rng_exact',
            nodes=nodes,
            edges=emb_rng_edges,
            graph_type='embedding_rng_exact',
            metric=metric,
            k=None,
            pruned_edges=None,
        )
    else:
        print(f"[SKIP] embedding_rng skipped because n={X.shape[0]:,} > max_embedding_rng_n={args.max_embedding_rng_n:,}")

    # 2) symmetric_knn and 3) symmetric_knn_rng
    for k in args.ks:
        knn_idx, knn_dist = exact_knn(X, k=k, metric=metric)
        sym_edges = build_symmetric_knn_edges(knn_idx, knn_dist)

        write_run(
            out_dir=root / f'symmetric_knn_k{k:02d}',
            nodes=nodes,
            edges=sym_edges,
            graph_type='symmetric_knn',
            metric=metric,
            k=k,
            pruned_edges=None,
        )

        rng_edges, pruned_edges = rng_prune_edges(sym_edges, D)
        write_run(
            out_dir=root / f'symmetric_knn_rng_exact_witness_k{k:02d}',
            nodes=nodes,
            edges=rng_edges,
            graph_type='symmetric_knn_rng_exact_witness',
            metric=metric,
            k=k,
            pruned_edges=pruned_edges,
        )

    print(f"\n[ALL DONE] results under {root}")


if __name__ == '__main__':
    main()
