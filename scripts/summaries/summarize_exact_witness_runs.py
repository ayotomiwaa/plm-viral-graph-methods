#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


K_RE = re.compile(r'_k(\d+)$')


def load_stats(path: Path) -> dict:
    with open(path, 'r') as f:
        stats = json.load(f)
    run_dir = path.parent.name
    m = K_RE.search(run_dir)
    if stats.get('k') is None and m:
        stats['k'] = int(m.group(1))
    stats['run_dir'] = run_dir
    stats['stats_path'] = str(path)
    return stats


def graph_sort_key(row: dict) -> tuple[int, str]:
    k = row.get('k')
    k_sort = int(k) if k is not None else -1
    return k_sort, str(row.get('graph_type', row.get('run_dir', '')))


def summarize(root: Path) -> list[dict]:
    rows = [load_stats(path) for path in sorted(root.glob('*/stats.json'))]
    rows.sort(key=graph_sort_key)

    knn_edges_by_k = {
        int(row['k']): int(row['n_edges'])
        for row in rows
        if row.get('graph_type') == 'symmetric_knn' and row.get('k') is not None
    }

    out = []
    for row in rows:
        k = row.get('k')
        n_nodes = int(row.get('n_nodes', 0))
        n_edges = int(row.get('n_edges', 0))
        n_pruned = int(row.get('n_pruned_edges', 0))
        mean_degree = (2.0 * n_edges / n_nodes) if n_nodes else 0.0
        graph_type = row.get('graph_type', row.get('run_dir', ''))

        baseline_edges = knn_edges_by_k.get(int(k)) if k is not None else None
        if graph_type == 'symmetric_knn_rng_exact_witness' and baseline_edges:
            pruned_frac = n_pruned / baseline_edges
            kept_frac = n_edges / baseline_edges
        else:
            pruned_frac = 0.0
            kept_frac = 1.0

        out.append({
            'run_dir': row.get('run_dir', ''),
            'graph_type': graph_type,
            'witness_type': row.get('witness_type') or '',
            'k': '' if k is None else int(k),
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'mean_degree': round(mean_degree, 6),
            'n_components': int(row.get('n_components', 0)),
            'giant_component_size': int(row.get('giant_component_size', 0)),
            'giant_component_frac': float(row.get('giant_component_frac', 0.0)),
            'n_pruned_edges': n_pruned,
            'pruned_frac_vs_knn': round(pruned_frac, 6),
            'kept_frac_vs_knn': round(kept_frac, 6),
        })
    return out


def write_csv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        raise RuntimeError('No rows to write')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict]) -> None:
    cols = [
        'graph_type',
        'k',
        'n_edges',
        'mean_degree',
        'n_components',
        'giant_component_frac',
        'n_pruned_edges',
        'pruned_frac_vs_knn',
    ]
    widths = {
        col: max(len(col), *(len(str(row[col])) for row in rows))
        for col in cols
    }
    print('  '.join(col.ljust(widths[col]) for col in cols))
    print('  '.join('-' * widths[col] for col in cols))
    for row in rows:
        print('  '.join(str(row[col]).ljust(widths[col]) for col in cols))


def main() -> None:
    ap = argparse.ArgumentParser(description='Summarize symmetric kNN vs exact-witness kNN-RNG stats.json files')
    ap.add_argument('--root', required=True, help='Run root containing */stats.json, e.g. .../ESM2_20k_benchmark/Alpha')
    ap.add_argument('--out-csv', default=None, help='Optional output CSV path. Defaults to <root>/graph_diagnostics_summary.csv')
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f'Root does not exist: {root}')

    rows = summarize(root)
    if not rows:
        raise SystemExit(f'No stats.json files found under: {root}')

    out_csv = Path(args.out_csv) if args.out_csv else root / 'graph_diagnostics_summary.csv'
    write_csv(rows, out_csv)
    print_table(rows)
    print(f'\nSaved: {out_csv}')


if __name__ == '__main__':
    main()

