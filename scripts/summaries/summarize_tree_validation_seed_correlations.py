#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


def summarize(
    frame: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    sort_cols: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = pd.to_numeric(group[value_col], errors="coerce").dropna()
        n = int(values.shape[0])
        mean = float(values.mean()) if n else math.nan
        sd = float(values.std(ddof=1)) if n > 1 else 0.0 if n == 1 else math.nan
        se = float(sd / math.sqrt(n)) if n > 0 else math.nan
        ci95 = float(1.96 * se) if n > 1 else 0.0 if n == 1 else math.nan
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update(
            {
                "n_seeds": n,
                f"{value_col}_mean": mean,
                f"{value_col}_sd": sd,
                f"{value_col}_se": se,
                f"{value_col}_ci95_halfwidth": ci95,
                f"{value_col}_min": float(values.min()) if n else math.nan,
                f"{value_col}_max": float(values.max()) if n else math.nan,
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Summarize tree-validation Spearman correlations across repeated seeds."
    )
    ap.add_argument(
        "--workspace",
        type=Path,
        default=Path("analysis/cohort_validation/12_repeated_2k_spike_tree_validation"),
    )
    args = ap.parse_args()

    workspace = args.workspace
    raw_path = workspace / "all_raw_distance_nextstrain_correlations.csv"
    graph_path = workspace / "all_graph_geodesic_nextstrain_correlations.csv"
    delta_path = workspace / "all_paired_delta_rho_summary.csv"

    if raw_path.exists():
        raw = pd.read_csv(raw_path)
        summarize(
            raw,
            ["panel", "baseline", "metric_family", "metric"],
            "spearman_rho",
            ["panel", "spearman_rho_mean"],
        ).sort_values(["panel", "spearman_rho_mean"], ascending=[True, False]).to_csv(
            workspace / "raw_distance_spearman_seed_summary.csv", index=False
        )

    if graph_path.exists():
        graph = pd.read_csv(graph_path)
        summarize(
            graph,
            ["panel", "graph_name", "metric_family", "embedding_metric", "graph_family"],
            "spearman_rho",
            ["panel", "spearman_rho_mean"],
        ).sort_values(["panel", "spearman_rho_mean"], ascending=[True, False]).to_csv(
            workspace / "graph_geodesic_spearman_seed_summary.csv", index=False
        )

    if delta_path.exists():
        delta = pd.read_csv(delta_path)
        summarize(
            delta,
            ["panel", "comparison_type", "embedding_metric", "graph_family", "embedding_graph", "hamming_graph"],
            "delta_rho_embedding_minus_hamming",
            ["panel", "comparison_type", "embedding_metric", "graph_family"],
        ).to_csv(workspace / "paired_delta_rho_seed_summary.csv", index=False)


if __name__ == "__main__":
    main()
