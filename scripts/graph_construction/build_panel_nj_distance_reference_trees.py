#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from protein_embeddings.methods.cc_geodesic import (  # noqa: E402
    build_nj_tree,
    clip_negative_branch_lengths,
    save_newick,
)

from scripts.graph_construction.build_panel_spike_reference_tree import (  # noqa: E402
    lca_many,
    load_panel_accessions,
    tree_arrays,
)
from scripts.validation.nextstrain_spike_tree_validation import (  # noqa: E402
    raw_distance_paths,
    subset_dense_matrix,
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_seed_list(value: str) -> list[int]:
    seeds: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, stop = [int(x.strip()) for x in part.split("-", 1)]
            seeds.extend(range(start, stop + 1))
        else:
            seeds.append(int(part))
    return sorted(dict.fromkeys(seeds))


def read_node_accessions(path: Path) -> set[str]:
    if not path.exists():
        return set()
    nodes = pd.read_csv(path, usecols=["accession"], low_memory=False)
    return set(nodes["accession"].astype(str).str.strip())


def metric_specs(panel_root: Path, sample_label: str, baselines: set[str]) -> list[dict[str, Any]]:
    specs = []
    for spec in raw_distance_paths(panel_root, sample_label):
        if spec["baseline"] in baselines:
            specs.append(spec)
    missing = baselines - {str(spec["baseline"]) for spec in specs}
    if missing:
        raise ValueError(f"Unknown raw distance baseline(s): {sorted(missing)}")
    return specs


def shared_accessions(panel_root: Path, sample_label: str, specs: list[dict[str, Any]]) -> list[str]:
    selected = load_panel_accessions(panel_root, sample_label)
    node_sets = [read_node_accessions(spec["nodes"]) for spec in specs]
    for spec, node_set in zip(specs, node_sets):
        if not node_set:
            raise FileNotFoundError(f"Missing or empty raw node table for {spec['baseline']}: {spec['nodes']}")
    keep_sets = node_sets
    accessions = [acc for acc in selected if all(acc in node_set for node_set in keep_sets)]
    return accessions


def count_negative_branch_lengths(tree_kind: str, tree: Any) -> dict[str, Any]:
    lengths: list[float] = []
    if tree_kind == "skbio":
        for node in tree.postorder():
            if node.length is not None:
                lengths.append(float(node.length))
    else:
        for clade in tree.find_clades():
            if clade.branch_length is not None:
                lengths.append(float(clade.branch_length))
    negatives = [value for value in lengths if value < 0]
    return {
        "n_branches": int(len(lengths)),
        "n_negative_branches": int(len(negatives)),
        "negative_branch_length_sum": float(sum(negatives)) if negatives else 0.0,
        "negative_branch_length_min": float(min(negatives)) if negatives else 0.0,
    }


def compute_patristic_matrix(
    newick_path: Path,
    accessions: list[str],
    out_dir: Path,
    matrix_name: str,
    nodes_name: str,
    qc_name: str,
    block_size: int,
    overwrite: bool,
) -> tuple[Path, Path, dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = out_dir / matrix_name
    nodes_path = out_dir / nodes_name
    qc_path = out_dir / qc_name
    if matrix_path.exists() and nodes_path.exists() and qc_path.exists() and not overwrite:
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        log(f"Using existing NJ patristic matrix: {matrix_path}")
        return matrix_path, nodes_path, qc

    arrays = tree_arrays(newick_path)
    tips = arrays["tips"].copy()
    tips = tips[tips["accession"] != ""].drop_duplicates("accession", keep="first")
    tip_map = dict(zip(tips["accession"], tips["tree_node_index"].astype(int)))
    matched = [acc for acc in accessions if acc in tip_map]
    missing = [acc for acc in accessions if acc not in tip_map]
    if missing:
        raise ValueError(f"{newick_path}: {len(missing):,} NJ tips missing after Newick parse; examples={missing[:5]}")

    tip_indices = np.array([tip_map[acc] for acc in matched], dtype=np.int32)
    n = len(tip_indices)
    D = np.lib.format.open_memmap(matrix_path, mode="w+", dtype=np.float32, shape=(n, n))
    root_dist = arrays["root_dist"]
    up = arrays["up"]
    depth = arrays["depth"]
    all_tips = tip_indices
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        left = tip_indices[start:stop]
        a = np.repeat(left[:, None], n, axis=1).ravel()
        b = np.repeat(all_tips[None, :], stop - start, axis=0).ravel()
        lca = lca_many(a, b, up, depth)
        block = root_dist[a] + root_dist[b] - 2.0 * root_dist[lca]
        D[start:stop, :] = block.reshape(stop - start, n).astype(np.float32, copy=False)
        D.flush()
        log(f"NJ patristic rows {start:,}-{stop - 1:,}/{n:,}")
    del D

    tip_lookup = tips.set_index("accession")
    nodes = pd.DataFrame(
        {
            "node_id": np.arange(n, dtype=int),
            "accession": matched,
            "tree_node_index": tip_indices,
            "tree_tip_name": [tip_lookup.loc[acc, "tree_tip_name"] for acc in matched],
        }
    )
    nodes.to_csv(nodes_path, index=False)
    qc = {
        "newick_path": str(newick_path),
        "n_accessions": int(len(accessions)),
        "n_matched_tree_tips": int(n),
        "n_missing_tree_tips": int(len(missing)),
        "matrix_path": str(matrix_path),
        "nodes_path": str(nodes_path),
        "matrix_shape": [int(n), int(n)],
        "matrix_dtype": "float32",
        "matrix_size_gb": float((n * n * np.dtype(np.float32).itemsize) / 1e9),
    }
    qc_path.write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
    return matrix_path, nodes_path, qc


def upper_values(D_a: np.ndarray, D_b: np.ndarray, pair_mode: str, sample_size: int, seed: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    n = D_a.shape[0]
    if D_b.shape != (n, n):
        raise ValueError(f"Matrix shape mismatch: {D_a.shape} vs {D_b.shape}")
    if pair_mode == "all":
        i, j = np.triu_indices(n, k=1)
    else:
        total_pairs = n * (n - 1) // 2
        m = min(int(sample_size), total_pairs)
        rng = np.random.default_rng(seed)
        i = rng.integers(0, n, size=m, dtype=np.int64)
        j = rng.integers(0, n - 1, size=m, dtype=np.int64)
        j = j + (j >= i)
    x = np.asarray(D_a[i, j])
    y = np.asarray(D_b[i, j])
    raw_pairs = int(len(x))
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask], raw_pairs, int(mask.sum())


def score_raw_vs_nj(
    raw: np.ndarray,
    patristic_path: Path,
    pair_mode: str,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    patristic = np.load(patristic_path, mmap_mode="r")
    x, y, raw_pairs, finite_pairs = upper_values(raw, patristic, pair_mode, sample_size, seed)
    if finite_pairs < 3:
        rho = np.nan
        pvalue = np.nan
    else:
        rho, pvalue = spearmanr(x, y)
    if finite_pairs and np.sum(x * x) > 0:
        rsd = float(np.sqrt(np.sum((x - y) ** 2) / np.sum(x**2)))
    else:
        rsd = np.nan
    return {
        "pair_mode": pair_mode,
        "n_pairs_raw": raw_pairs,
        "n_pairs_used": finite_pairs,
        "finite_pair_fraction": float(finite_pairs / raw_pairs) if raw_pairs else np.nan,
        "spearman_raw_vs_own_nj_patristic": float(rho) if np.isfinite(rho) else np.nan,
        "spearman_pvalue": float(pvalue) if np.isfinite(pvalue) else np.nan,
        "rsd_raw_vs_own_nj_patristic": float(rsd) if np.isfinite(rsd) else np.nan,
    }


def output_paths(seed_out: Path, baseline: str, reference_baseline: str) -> dict[str, Path | str]:
    if baseline == reference_baseline:
        out_dir = seed_out / "reference_tree"
        return {
            "out_dir": out_dir,
            "newick": out_dir / f"spike_reference_nj_{baseline}.nwk",
            "matrix_name": "D_reference_spike_float32.npy",
            "nodes_name": "D_reference_spike_nodes.csv",
            "qc_name": "D_reference_spike_qc.json",
        }
    out_dir = seed_out / "nj_metric_trees" / baseline
    return {
        "out_dir": out_dir,
        "newick": out_dir / f"{baseline}_nj.nwk",
        "matrix_name": "D_nj_patristic_float32.npy",
        "nodes_name": "D_nj_patristic_nodes.csv",
        "qc_name": "D_nj_patristic_qc.json",
    }


def build_one_metric(
    panel: str,
    seed: int,
    panel_root: Path,
    seed_out: Path,
    sample_label: str,
    spec: dict[str, Any],
    accessions: list[str],
    reference_baseline: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    baseline = str(spec["baseline"])
    paths = output_paths(seed_out, baseline, reference_baseline)
    out_dir = Path(paths["out_dir"])
    newick_path = Path(paths["newick"])
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"{panel}/seed_{seed}: loading {baseline} matrix for n={len(accessions):,}")
    raw = subset_dense_matrix(Path(spec["matrix"]), Path(spec["nodes"]), accessions).astype(np.float64, copy=False)
    raw = 0.5 * (raw + raw.T)
    np.fill_diagonal(raw, 0.0)

    tree_qc_path = out_dir / "nj_tree_manifest.json"
    if newick_path.exists() and newick_path.stat().st_size > 0 and tree_qc_path.exists() and not args.overwrite_tree:
        tree_qc = json.loads(tree_qc_path.read_text(encoding="utf-8"))
        log(f"Using existing NJ Newick: {newick_path}")
    else:
        log(f"{panel}/seed_{seed}: building NJ tree for {baseline}")
        tree_kind, tree = build_nj_tree(raw, accessions, prefer=args.prefer_tree_builder)
        neg_before = count_negative_branch_lengths(tree_kind, tree)
        if args.clip_negative_branches:
            clip_negative_branch_lengths(tree_kind, tree)
        neg_after = count_negative_branch_lengths(tree_kind, tree)
        save_newick(tree_kind, tree, str(newick_path))
        tree_qc = {
            "panel": panel,
            "seed": int(seed),
            "sample_label": sample_label,
            "baseline": baseline,
            "metric_family": spec["metric_family"],
            "metric": spec["metric"],
            "tree_builder": "neighbor_joining",
            "tree_builder_backend": tree_kind,
            "raw_matrix": str(spec["matrix"]),
            "raw_nodes": str(spec["nodes"]),
            "n_accessions": int(len(accessions)),
            "newick_path": str(newick_path),
            "clip_negative_branches": bool(args.clip_negative_branches),
            "branch_lengths_before_clip": neg_before,
            "branch_lengths_after_clip": neg_after,
        }
        tree_qc_path.write_text(json.dumps(tree_qc, indent=2) + "\n", encoding="utf-8")

    matrix_path, nodes_path, pat_qc = compute_patristic_matrix(
        newick_path=newick_path,
        accessions=accessions,
        out_dir=out_dir,
        matrix_name=str(paths["matrix_name"]),
        nodes_name=str(paths["nodes_name"]),
        qc_name=str(paths["qc_name"]),
        block_size=args.patristic_block_size,
        overwrite=args.overwrite_patristic,
    )
    score = score_raw_vs_nj(
        raw=raw,
        patristic_path=matrix_path,
        pair_mode=args.pair_mode,
        sample_size=args.pair_sample_size,
        seed=args.pair_seed + seed,
    )
    del raw
    return {
        "panel": panel,
        "seed": int(seed),
        "sample_label": sample_label,
        "baseline": baseline,
        "metric_family": spec["metric_family"],
        "metric": spec["metric"],
        "n_accessions": int(len(accessions)),
        "newick_path": str(newick_path),
        "patristic_matrix": str(matrix_path),
        "patristic_nodes": str(nodes_path),
        "used_as_panel_tree_reference": bool(baseline == reference_baseline),
        "tree_builder_backend": tree_qc.get("tree_builder_backend", ""),
        "clip_negative_branches": tree_qc.get("clip_negative_branches", ""),
        "n_negative_branches_before_clip": tree_qc.get("branch_lengths_before_clip", {}).get("n_negative_branches", ""),
        "negative_branch_length_sum_before_clip": tree_qc.get("branch_lengths_before_clip", {}).get("negative_branch_length_sum", ""),
        "n_negative_branches_after_clip": tree_qc.get("branch_lengths_after_clip", {}).get("n_negative_branches", ""),
        "matrix_size_gb": pat_qc.get("matrix_size_gb", ""),
        **score,
    }


def run_panel_seed(panel: str, seed: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    panel_root = args.source_root / panel / f"seed_{seed}"
    if not panel_root.exists():
        log(f"Skipping missing panel seed root: {panel_root}")
        return []
    baselines = {item.strip() for item in args.baselines.split(",") if item.strip()}
    specs = metric_specs(panel_root, args.sample_label, baselines)
    accessions = shared_accessions(panel_root, args.sample_label, specs)
    if len(accessions) < 3:
        raise ValueError(f"{panel}/seed_{seed}: fewer than 3 shared accessions")
    seed_out = args.workspace / panel / f"seed_{seed}"
    seed_out.mkdir(parents=True, exist_ok=True)
    intersection_qc = {
        "panel": panel,
        "seed": int(seed),
        "sample_label": args.sample_label,
        "source_panel_root": str(panel_root),
        "baselines": sorted(baselines),
        "n_selected_accessions": int(len(load_panel_accessions(panel_root, args.sample_label))),
        "n_shared_raw_metric_accessions": int(len(accessions)),
    }
    (seed_out / "nj_raw_metric_intersection_qc.json").write_text(
        json.dumps(intersection_qc, indent=2) + "\n",
        encoding="utf-8",
    )
    rows = []
    for spec in specs:
        rows.append(
            build_one_metric(
                panel=panel,
                seed=seed,
                panel_root=panel_root,
                seed_out=seed_out,
                sample_label=args.sample_label,
                spec=spec,
                accessions=accessions,
                reference_baseline=args.reference_baseline,
                args=args,
            )
        )
    pd.DataFrame(rows).to_csv(seed_out / "nj_self_tree_likeness_correlations.csv", index=False)
    return rows


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    grouped = rows.groupby(["baseline", "metric_family", "metric"], dropna=False)
    summary = grouped.agg(
        n_seeds=("seed", "nunique"),
        mean_spearman_raw_vs_own_nj_patristic=("spearman_raw_vs_own_nj_patristic", "mean"),
        sd_spearman_raw_vs_own_nj_patristic=("spearman_raw_vs_own_nj_patristic", "std"),
        min_spearman_raw_vs_own_nj_patristic=("spearman_raw_vs_own_nj_patristic", "min"),
        max_spearman_raw_vs_own_nj_patristic=("spearman_raw_vs_own_nj_patristic", "max"),
        mean_rsd_raw_vs_own_nj_patristic=("rsd_raw_vs_own_nj_patristic", "mean"),
        sd_rsd_raw_vs_own_nj_patristic=("rsd_raw_vs_own_nj_patristic", "std"),
        mean_n_negative_branches_before_clip=("n_negative_branches_before_clip", "mean"),
    ).reset_index()
    summary["se_spearman_raw_vs_own_nj_patristic"] = (
        summary["sd_spearman_raw_vs_own_nj_patristic"] / np.sqrt(summary["n_seeds"].clip(lower=1))
    )
    summary["ci95_spearman_half_width"] = 1.96 * summary["se_spearman_raw_vs_own_nj_patristic"]
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Build per-panel Neighbor Joining trees from raw distance matrices.")
    ap.add_argument("--workspace", type=Path, default=Path("analysis/cohort_validation/13_random_full_dataset_2k_nj_tree_validation"))
    ap.add_argument("--source-root", type=Path, default=Path("analysis/cohort_validation/08_sampling_design_2k/random_full_dataset"))
    ap.add_argument("--panels", default="random_full_dataset_2k")
    ap.add_argument("--seeds", default="0-199")
    ap.add_argument("--sample-label", default="pool_n2000")
    ap.add_argument("--baselines", default="raw_hamming,raw_esm2_cityblock,raw_esm2_euclidean")
    ap.add_argument("--reference-baseline", default="raw_hamming")
    ap.add_argument("--prefer-tree-builder", default="auto", choices=["auto", "skbio", "biopython"])
    ap.add_argument("--clip-negative-branches", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--patristic-block-size", type=int, default=128)
    ap.add_argument("--pair-mode", default="all", choices=["all", "sample"])
    ap.add_argument("--pair-sample-size", type=int, default=5_000_000)
    ap.add_argument("--pair-seed", type=int, default=12345)
    ap.add_argument("--overwrite-tree", action="store_true")
    ap.add_argument("--overwrite-patristic", action="store_true")
    args = ap.parse_args()

    panels = [panel.strip() for panel in args.panels.split(",") if panel.strip()]
    seeds = parse_seed_list(args.seeds)
    all_rows: list[dict[str, Any]] = []
    for panel in panels:
        for seed in seeds:
            log(f"=== NJ {panel}/seed_{seed} ===")
            all_rows.extend(run_panel_seed(panel, seed, args))

    existing = sorted(args.workspace.glob("*/seed_*/nj_self_tree_likeness_correlations.csv"))
    if existing:
        frame = pd.concat([pd.read_csv(path) for path in existing], ignore_index=True)
    else:
        frame = pd.DataFrame(all_rows)
    args.workspace.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.workspace / "all_nj_self_tree_likeness_correlations.csv", index=False)
    summarize(frame).to_csv(args.workspace / "nj_self_tree_likeness_seed_summary.csv", index=False)
    log(f"Wrote NJ aggregate outputs under {args.workspace}")


if __name__ == "__main__":
    main()
