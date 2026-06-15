#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.graph_construction.build_panel_nj_distance_reference_trees import (  # noqa: E402
    parse_seed_list,
)
from scripts.graph_construction.build_panel_spike_reference_tree import (  # noqa: E402
    tree_arrays,
)
from scripts.validation.time_dated_tree_validation import decimal_year  # noqa: E402


DEFAULT_ROOT_STRATEGIES = ["newick_root", "oldest_tip", "best_tip"]
SUMMARY_VALUE_COLS = [
    "pearson_r",
    "pearson_r2",
    "spearman_rho",
    "slope_distance_per_year",
    "root_to_tip_distance_mean",
    "root_to_tip_distance_max",
]


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def load_dates(panel_root: Path, sample_label: str) -> dict[str, dict[str, Any]]:
    metadata_path = panel_root / "inputs" / sample_label / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    meta = pd.read_csv(metadata_path, usecols=["accession", "collection_date"], low_memory=False)
    out: dict[str, dict[str, Any]] = {}
    for _, row in meta.iterrows():
        accession = str(row["accession"]).strip()
        if not accession:
            continue
        dec = decimal_year(row["collection_date"])
        out[accession] = {
            "collection_date": str(row["collection_date"]),
            "decimal_year": float(dec) if dec is not None else math.nan,
        }
    return out


def regression_score(
    distances: np.ndarray,
    dates: np.ndarray,
    root_index: int | None,
    root_strategy: str,
    root_accession: str,
    root_collection_date: str,
    root_decimal_year: float,
) -> dict[str, Any]:
    distances = np.asarray(distances, dtype=np.float64)
    dates = np.asarray(dates, dtype=np.float64)
    mask = np.isfinite(distances) & np.isfinite(dates)
    if root_index is not None and 0 <= root_index < mask.size:
        mask[root_index] = False

    x = dates[mask]
    y = distances[mask]
    n = int(mask.sum())
    if n >= 3 and np.unique(x).size >= 2 and np.unique(y).size >= 2:
        lr = linregress(x, y)
        pr = pearsonr(x, y)
        sr = spearmanr(x, y)
        residuals = y - (float(lr.intercept) + float(lr.slope) * x)
        residual_sd = float(np.sqrt(np.sum(residuals**2) / max(1, n - 2)))
        pearson_r_value = float(pr.statistic)
        pearson_pvalue = float(pr.pvalue)
        spearman_rho = float(sr.statistic)
        spearman_pvalue = float(sr.pvalue)
        slope = float(lr.slope)
        intercept = float(lr.intercept)
    else:
        residual_sd = math.nan
        pearson_r_value = math.nan
        pearson_pvalue = math.nan
        spearman_rho = math.nan
        spearman_pvalue = math.nan
        slope = math.nan
        intercept = math.nan

    return {
        "root_strategy": root_strategy,
        "root_accession": root_accession,
        "root_collection_date": root_collection_date,
        "root_decimal_year": root_decimal_year,
        "n_dated_tips_used": n,
        "n_tips_with_missing_date_or_distance": int(mask.size - n - (1 if root_index is not None and 0 <= root_index < mask.size else 0)),
        "date_decimal_year_min": float(np.nanmin(x)) if n else math.nan,
        "date_decimal_year_max": float(np.nanmax(x)) if n else math.nan,
        "root_to_tip_distance_min": float(np.nanmin(y)) if n else math.nan,
        "root_to_tip_distance_mean": float(np.nanmean(y)) if n else math.nan,
        "root_to_tip_distance_max": float(np.nanmax(y)) if n else math.nan,
        "slope_distance_per_year": slope,
        "intercept": intercept,
        "pearson_r": pearson_r_value,
        "pearson_r2": float(pearson_r_value**2) if np.isfinite(pearson_r_value) else math.nan,
        "pearson_pvalue": pearson_pvalue,
        "spearman_rho": spearman_rho,
        "spearman_pvalue": spearman_pvalue,
        "residual_sd": residual_sd,
    }


def best_tip_root_index(D: np.ndarray, dates: np.ndarray) -> int | None:
    dates = np.asarray(dates, dtype=np.float64)
    dated_idx = np.flatnonzero(np.isfinite(dates))
    if dated_idx.size < 3:
        return None

    sub = np.asarray(D[np.ix_(dated_idx, dated_idx)], dtype=np.float64)
    if not np.isfinite(sub).all():
        best_idx: int | None = None
        best_r = -np.inf
        for idx in dated_idx:
            distances = np.asarray(D[idx, :], dtype=np.float64)
            mask = np.isfinite(dates) & np.isfinite(distances)
            mask[idx] = False
            if int(mask.sum()) < 3 or np.unique(distances[mask]).size < 2:
                continue
            r = float(pearsonr(dates[mask], distances[mask]).statistic)
            if np.isfinite(r) and r > best_r:
                best_r = r
                best_idx = int(idx)
        return best_idx

    y = dates[dated_idx]
    m = int(y.size)
    n_eff = m - 1
    diag = np.diag(sub)
    sum_y = float(np.sum(y))
    sum_y2 = float(np.sum(y * y))
    row_sum = np.sum(sub, axis=1) - diag
    row_sum2 = np.sum(sub * sub, axis=1) - diag * diag
    row_xy = sub @ y - diag * y
    y_sum_excluding_root = sum_y - y
    y2_sum_excluding_root = sum_y2 - y * y

    cov = row_xy - (row_sum * y_sum_excluding_root / n_eff)
    var_x = row_sum2 - (row_sum * row_sum / n_eff)
    var_y = y2_sum_excluding_root - (y_sum_excluding_root * y_sum_excluding_root / n_eff)
    denom = np.sqrt(var_x * var_y)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = cov / denom
    r[~np.isfinite(r)] = -np.inf
    if not np.isfinite(r).any():
        return None
    return int(dated_idx[int(np.argmax(r))])


def oldest_tip_root_index(dates: np.ndarray) -> int | None:
    finite = np.flatnonzero(np.isfinite(dates))
    if finite.size == 0:
        return None
    values = dates[finite]
    return int(finite[int(np.argmin(values))])


def tip_frame_from_nodes(nodes_path: Path, date_map: dict[str, dict[str, Any]]) -> pd.DataFrame:
    nodes = pd.read_csv(nodes_path, low_memory=False)
    if "accession" not in nodes.columns:
        raise ValueError(f"{nodes_path} must contain an accession column")
    accessions = nodes["accession"].astype(str).str.strip().tolist()
    rows = []
    for idx, accession in enumerate(accessions):
        date_info = date_map.get(accession, {"collection_date": "", "decimal_year": math.nan})
        rows.append(
            {
                "row_index": idx,
                "accession": accession,
                "collection_date": date_info["collection_date"],
                "decimal_year": date_info["decimal_year"],
            }
        )
    return pd.DataFrame(rows)


def newick_root_distances(newick_path: Path, accessions: list[str]) -> np.ndarray:
    arrays = tree_arrays(newick_path)
    tips = arrays["tips"].copy()
    tips = tips[tips["accession"] != ""].drop_duplicates("accession", keep="first")
    tip_map = dict(zip(tips["accession"], tips["tree_node_index"].astype(int)))
    root_dist = arrays["root_dist"]
    out = np.full(len(accessions), np.nan, dtype=np.float64)
    missing = []
    for idx, accession in enumerate(accessions):
        tree_idx = tip_map.get(accession)
        if tree_idx is None:
            missing.append(accession)
            continue
        out[idx] = float(root_dist[int(tree_idx)])
    if missing:
        raise ValueError(f"{newick_path}: {len(missing):,} accessions missing from tree tips; examples={missing[:5]}")
    return out


def root_metadata(tips: pd.DataFrame, root_index: int | None) -> tuple[str, str, float]:
    if root_index is None or root_index < 0 or root_index >= len(tips):
        return "", "", math.nan
    row = tips.iloc[int(root_index)]
    return str(row["accession"]), str(row["collection_date"]), float(row["decimal_year"])


def score_tree_row(
    row: pd.Series,
    date_map: dict[str, dict[str, Any]],
    root_strategies: set[str],
) -> list[dict[str, Any]]:
    newick_path = Path(str(row["newick_path"]))
    nodes_path = Path(str(row["patristic_nodes"]))
    matrix_path = Path(str(row["patristic_matrix"]))
    if not newick_path.exists():
        raise FileNotFoundError(f"Missing NJ Newick: {newick_path}")
    if not nodes_path.exists():
        raise FileNotFoundError(f"Missing NJ node table: {nodes_path}")

    tips = tip_frame_from_nodes(nodes_path, date_map)
    accessions = tips["accession"].astype(str).tolist()
    dates = tips["decimal_year"].astype(float).to_numpy()
    out: list[dict[str, Any]] = []

    if "newick_root" in root_strategies:
        distances = newick_root_distances(newick_path, accessions)
        score = regression_score(
            distances=distances,
            dates=dates,
            root_index=None,
            root_strategy="newick_root",
            root_accession="",
            root_collection_date="",
            root_decimal_year=math.nan,
        )
        out.append(score)

    matrix_needed = bool(root_strategies & {"oldest_tip", "best_tip"})
    D = None
    if matrix_needed:
        if not matrix_path.exists():
            raise FileNotFoundError(f"Missing NJ patristic matrix: {matrix_path}")
        D = np.load(matrix_path, mmap_mode="r")

    if D is not None and "oldest_tip" in root_strategies:
        root_idx = oldest_tip_root_index(dates)
        root_acc, root_date, root_dec = root_metadata(tips, root_idx)
        distances = np.asarray(D[int(root_idx), :], dtype=np.float64) if root_idx is not None else np.full(len(tips), np.nan)
        out.append(
            regression_score(
                distances=distances,
                dates=dates,
                root_index=root_idx,
                root_strategy="oldest_tip",
                root_accession=root_acc,
                root_collection_date=root_date,
                root_decimal_year=root_dec,
            )
        )

    if D is not None and "best_tip" in root_strategies:
        root_idx = best_tip_root_index(D, dates)
        root_acc, root_date, root_dec = root_metadata(tips, root_idx)
        distances = np.asarray(D[int(root_idx), :], dtype=np.float64) if root_idx is not None else np.full(len(tips), np.nan)
        out.append(
            regression_score(
                distances=distances,
                dates=dates,
                root_index=root_idx,
                root_strategy="best_tip",
                root_accession=root_acc,
                root_collection_date=root_date,
                root_decimal_year=root_dec,
            )
        )

    return out


def evaluate_raw(seed_out: Path, panel_root: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    raw_tree_path = seed_out / "nj_self_tree_likeness_correlations.csv"
    if not raw_tree_path.exists():
        raise FileNotFoundError(f"Missing raw NJ tree-likeness file: {raw_tree_path}")
    raw_rows = pd.read_csv(raw_tree_path)
    wanted = {item.strip() for item in args.raw_baselines.split(",") if item.strip()}
    date_map = load_dates(panel_root, args.sample_label)
    root_strategies = {item.strip() for item in args.root_strategies.split(",") if item.strip()}

    rows: list[dict[str, Any]] = []
    for _, row in raw_rows.iterrows():
        baseline = str(row["baseline"])
        if baseline not in wanted:
            continue
        log(f"Root-to-tip regression, raw NJ tree: {baseline}")
        scores = score_tree_row(row, date_map, root_strategies)
        for score in scores:
            rows.append(
                {
                    "panel": row["panel"],
                    "seed": int(row["seed"]),
                    "sample_label": row["sample_label"],
                    "comparison_type": "raw_nj_root_to_tip",
                    "baseline": baseline,
                    "metric_family": row["metric_family"],
                    "metric": row["metric"],
                    "n_accessions": int(row["n_accessions"]),
                    "newick_path": row["newick_path"],
                    "patristic_matrix": row["patristic_matrix"],
                    "patristic_nodes": row["patristic_nodes"],
                    **score,
                }
            )
    return rows


def evaluate_graphs(seed_out: Path, panel_root: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    graph_tree_path = seed_out / "graph_tree_distortion_metrics.csv"
    if not graph_tree_path.exists():
        raise FileNotFoundError(f"Missing graph NJ tree metric file: {graph_tree_path}")
    graph_rows = pd.read_csv(graph_tree_path)
    wanted = {item.strip() for item in args.graph_names.split(",") if item.strip()}
    if wanted:
        graph_rows = graph_rows[graph_rows["graph_name"].astype(str).isin(wanted)].copy()
    date_map = load_dates(panel_root, args.sample_label)
    root_strategies = {item.strip() for item in args.root_strategies.split(",") if item.strip()}

    rows: list[dict[str, Any]] = []
    for _, row in graph_rows.iterrows():
        graph_name = str(row["graph_name"])
        log(f"Root-to-tip regression, graph NJ tree: {graph_name}")
        scores = score_tree_row(row, date_map, root_strategies)
        for score in scores:
            rows.append(
                {
                    "panel": row["panel"],
                    "seed": int(row["seed"]),
                    "sample_label": row["sample_label"],
                    "comparison_type": "graph_geodesic_nj_root_to_tip",
                    "graph_name": graph_name,
                    "metric_family": row["metric_family"],
                    "embedding_metric": row["embedding_metric"],
                    "graph_family": row["graph_family"],
                    "n_accessions": int(row["n_accessions"]),
                    "graph_distance_n_components": row.get("graph_distance_n_components", ""),
                    "graph_distance_giant_component_size": row.get("graph_distance_giant_component_size", ""),
                    "graph_component_mode": row.get("graph_component_mode", ""),
                    "n_nodes_dropped_for_nj": row.get("n_nodes_dropped_for_nj", ""),
                    "newick_path": row["newick_path"],
                    "patristic_matrix": row["patristic_matrix"],
                    "patristic_nodes": row["patristic_nodes"],
                    **score,
                }
            )
    return rows


def summarize(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        out = {col: key for col, key in zip(group_cols, keys)}
        out["n_seeds"] = int(group["seed"].nunique()) if "seed" in group.columns else int(group.shape[0])
        for value_col in SUMMARY_VALUE_COLS:
            if value_col not in group.columns:
                continue
            values = pd.to_numeric(group[value_col], errors="coerce").dropna()
            n = int(values.shape[0])
            sd = float(values.std(ddof=1)) if n > 1 else 0.0 if n == 1 else math.nan
            se = float(sd / math.sqrt(n)) if n else math.nan
            out[f"{value_col}_mean"] = float(values.mean()) if n else math.nan
            out[f"{value_col}_sd"] = sd
            out[f"{value_col}_se"] = se
            out[f"{value_col}_ci95_halfwidth"] = float(1.96 * se) if n > 1 else 0.0 if n == 1 else math.nan
            out[f"{value_col}_min"] = float(values.min()) if n else math.nan
            out[f"{value_col}_max"] = float(values.max()) if n else math.nan
        rows.append(out)
    result = pd.DataFrame(rows)
    if not result.empty and "pearson_r2_mean" in result.columns:
        result = result.sort_values(["panel", "root_strategy", "pearson_r2_mean"], ascending=[True, True, False])
    return result


def aggregate_workspace(workspace: Path) -> None:
    raw_frames = [pd.read_csv(path) for path in workspace.glob("*/seed_*/raw_root_to_tip_regression_metrics.csv")]
    graph_frames = [pd.read_csv(path) for path in workspace.glob("*/seed_*/graph_root_to_tip_regression_metrics.csv")]

    if raw_frames:
        raw = pd.concat(raw_frames, ignore_index=True)
        raw.to_csv(workspace / "all_raw_root_to_tip_regression_metrics.csv", index=False)
        summarize(raw, ["panel", "baseline", "metric_family", "metric", "root_strategy"]).to_csv(
            workspace / "raw_root_to_tip_regression_seed_summary.csv", index=False
        )

    if graph_frames:
        graph = pd.concat(graph_frames, ignore_index=True)
        graph.to_csv(workspace / "all_graph_root_to_tip_regression_metrics.csv", index=False)
        summarize(graph, ["panel", "graph_name", "metric_family", "embedding_metric", "graph_family", "root_strategy"]).to_csv(
            workspace / "graph_root_to_tip_regression_seed_summary.csv", index=False
        )


def infer_workspace_and_panel(args: argparse.Namespace) -> tuple[Path, list[str]]:
    if args.panel_workspace:
        panel_path = args.panel_workspace
        return panel_path.parent, [panel_path.name]
    panels = [item.strip() for item in args.panels.split(",") if item.strip()]
    return args.workspace, panels


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate TempEst-style root-to-tip regression for existing raw and graph NJ trees."
    )
    ap.add_argument("--workspace", type=Path, default=Path("analysis/cohort_validation/13_random_full_dataset_2k_nj_tree_validation"))
    ap.add_argument("--panel-workspace", type=Path, default=None)
    ap.add_argument("--source-root", type=Path, default=None)
    ap.add_argument("--panels", default="random_full_dataset_2k")
    ap.add_argument("--seeds", default="0-199")
    ap.add_argument("--sample-label", default="pool_n2000")
    ap.add_argument("--raw-baselines", default="raw_hamming,raw_esm2_cityblock,raw_esm2_euclidean")
    ap.add_argument("--graph-names", default="", help="Comma-separated graph names; empty means all graph families with NJ trees.")
    ap.add_argument("--root-strategies", default=",".join(DEFAULT_ROOT_STRATEGIES))
    ap.add_argument("--skip-raw", action="store_true")
    ap.add_argument("--skip-graphs", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    valid_root_strategies = set(DEFAULT_ROOT_STRATEGIES)
    requested = {item.strip() for item in args.root_strategies.split(",") if item.strip()}
    unknown = requested - valid_root_strategies
    if unknown:
        raise ValueError(f"Unknown root strategies: {sorted(unknown)}; valid={sorted(valid_root_strategies)}")

    workspace, panels = infer_workspace_and_panel(args)
    source_root = args.source_root or (workspace / "source")
    seeds = parse_seed_list(args.seeds)

    if not args.aggregate_only:
        for panel in panels:
            for seed in seeds:
                panel_root = source_root / panel / f"seed_{seed}"
                seed_out = workspace / panel / f"seed_{seed}"
                if not panel_root.exists():
                    log(f"Skipping missing panel seed root: {panel_root}")
                    continue
                if not seed_out.exists():
                    log(f"Skipping missing NJ validation seed output: {seed_out}")
                    continue
                if not args.skip_raw:
                    raw_rows = evaluate_raw(seed_out, panel_root, args)
                    pd.DataFrame(raw_rows).to_csv(seed_out / "raw_root_to_tip_regression_metrics.csv", index=False)
                if not args.skip_graphs:
                    graph_rows = evaluate_graphs(seed_out, panel_root, args)
                    pd.DataFrame(graph_rows).to_csv(seed_out / "graph_root_to_tip_regression_metrics.csv", index=False)

    aggregate_workspace(workspace)
    log(f"Wrote root-to-tip regression outputs under {workspace}")


if __name__ == "__main__":
    main()
