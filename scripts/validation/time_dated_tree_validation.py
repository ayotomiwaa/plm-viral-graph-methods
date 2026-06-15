#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from Bio import Phylo

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.graph_construction.build_panel_spike_reference_tree import (  # noqa: E402
    lca_many,
    load_panel_accessions,
    parse_accession,
    parse_seed_list,
)
from scripts.validation.nextstrain_spike_tree_validation import (  # noqa: E402
    aggregate_workspace,
    score_graph_distances,
    score_partial_correlations,
    score_raw_distances,
    write_delta_summary,
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def decimal_year(value: object) -> float | None:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(text, fmt).date()
            break
        except ValueError:
            dt = None
    if dt is None:
        return None
    start = date(dt.year, 1, 1)
    stop = date(dt.year + 1, 1, 1)
    return dt.year + ((dt - start).days / ((stop - start).days))


def detect_tree_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".nex", ".nexus"}:
        return "nexus"
    return "newick"


def tree_arrays(tree_path: Path, tree_format: str | None = None) -> dict[str, Any]:
    fmt = tree_format or detect_tree_format(tree_path)
    tree = Phylo.read(str(tree_path), fmt)
    clades = list(tree.find_clades(order="preorder"))
    index = {id(clade): i for i, clade in enumerate(clades)}
    n_nodes = len(clades)
    parent = np.full(n_nodes, -1, dtype=np.int32)
    depth = np.zeros(n_nodes, dtype=np.int32)
    root_dist = np.zeros(n_nodes, dtype=np.float64)
    tip_rows: list[dict[str, Any]] = []

    def visit(clade: Any, parent_idx: int) -> None:
        idx = index[id(clade)]
        parent[idx] = idx if parent_idx < 0 else parent_idx
        if parent_idx >= 0:
            depth[idx] = depth[parent_idx] + 1
            root_dist[idx] = root_dist[parent_idx] + float(clade.branch_length or 0.0)
        if clade.is_terminal():
            name = "" if clade.name is None else str(clade.name)
            tip_rows.append(
                {
                    "accession": parse_accession(name),
                    "tree_node_index": idx,
                    "tree_tip_name": name,
                }
            )
        for child in clade.clades:
            visit(child, idx)

    visit(tree.root, -1)
    max_log = max(1, math.ceil(math.log2(max(2, n_nodes))) + 1)
    up = np.empty((max_log, n_nodes), dtype=np.int32)
    up[0] = parent
    for level in range(1, max_log):
        up[level] = up[level - 1, up[level - 1]]

    return {
        "parent": parent,
        "depth": depth,
        "root_dist": root_dist,
        "up": up,
        "tips": pd.DataFrame(tip_rows),
        "n_nodes": n_nodes,
        "format": fmt,
    }


def write_dates(panel_root: Path, sample_label: str, out_dir: Path) -> pd.DataFrame:
    input_dir = panel_root / "inputs" / sample_label
    metadata_path = input_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    meta = pd.read_csv(metadata_path, low_memory=False)
    if "accession" not in meta.columns or "collection_date" not in meta.columns:
        raise ValueError(f"{metadata_path} must contain accession and collection_date columns")
    rows = []
    for _, row in meta[["accession", "collection_date"]].iterrows():
        accession = str(row["accession"]).strip()
        dec = decimal_year(row["collection_date"])
        if accession and dec is not None:
            rows.append(
                {
                    "name": accession,
                    "date": dec,
                    "collection_date": str(row["collection_date"]),
                }
            )
    dates = pd.DataFrame(rows).drop_duplicates("name", keep="first").sort_values("name")
    out_dir.mkdir(parents=True, exist_ok=True)
    dates[["name", "date"]].to_csv(out_dir / "dates.tsv", sep="\t", index=False)
    dates.to_csv(out_dir / "dates_with_collection_date.csv", index=False)
    qc = {
        "metadata_path": str(metadata_path),
        "n_metadata_rows": int(len(meta)),
        "n_dates_written": int(len(dates)),
        "n_missing_or_partial_dates": int(len(meta) - len(dates)),
        "date_min": float(dates["date"].min()) if len(dates) else None,
        "date_max": float(dates["date"].max()) if len(dates) else None,
    }
    (out_dir / "dates_qc.json").write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
    log(f"Wrote TreeTime dates table: {out_dir / 'dates.tsv'}")
    return dates


def run_treetime(
    input_tree: Path,
    alignment: Path,
    dates_tsv: Path,
    out_dir: Path,
    treetime_bin: str,
    extra_args: list[str],
    force: bool,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dated = find_dated_tree(out_dir)
    if dated is not None and not force:
        log(f"Using existing dated tree: {dated}")
        return dated
    binary = shutil.which(treetime_bin)
    if binary is None:
        raise FileNotFoundError(
            f"Could not find TreeTime binary '{treetime_bin}'. Install TreeTime in the run environment "
            "or set TREETIME_BIN to its full path."
        )
    cmd = [
        binary,
        "--tree",
        str(input_tree),
        "--aln",
        str(alignment),
        "--dates",
        str(dates_tsv),
        "--outdir",
        str(out_dir),
        "--reroot",
        "least-squares",
        "--clock-filter",
        "0",
        "--aa",
        *extra_args,
    ]
    log("Running TreeTime: " + " ".join(cmd))
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    (out_dir / "treetime.stdout.log").write_text(proc.stdout, encoding="utf-8")
    (out_dir / "treetime.stderr.log").write_text(proc.stderr, encoding="utf-8")
    (out_dir / "treetime_command.json").write_text(json.dumps({"command": cmd}, indent=2) + "\n", encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"TreeTime failed with exit code {proc.returncode}; see {out_dir / 'treetime.stderr.log'}")
    dated = find_dated_tree(out_dir)
    if dated is None:
        files = sorted(str(p.name) for p in out_dir.iterdir())
        raise FileNotFoundError(f"TreeTime finished but no dated tree file was recognized in {out_dir}: {files}")
    log(f"TreeTime dated tree: {dated}")
    return dated


def find_dated_tree(out_dir: Path) -> Path | None:
    candidates = [
        out_dir / "timetree.nexus",
        out_dir / "timetree.nwk",
        out_dir / "timetree.newick",
        out_dir / "annotated_tree.nexus",
        out_dir / "annotated_tree.nwk",
        out_dir / "annotated_tree.newick",
    ]
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def compute_time_patristic_matrix(
    tree_path: Path,
    panel_accessions: list[str],
    out_dir: Path,
    block_size: int,
    overwrite: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    matrix_path = out_dir / "D_time_dated_tree_float32.npy"
    nodes_path = out_dir / "D_time_dated_tree_nodes.csv"
    qc_path = out_dir / "D_time_dated_tree_qc.json"
    if matrix_path.exists() and nodes_path.exists() and not overwrite:
        log(f"Using existing time-dated patristic matrix: {matrix_path}")
        return pd.read_csv(nodes_path), np.load(matrix_path, mmap_mode="r")

    arrays = tree_arrays(tree_path)
    tips = arrays["tips"].copy()
    tips = tips[tips["accession"] != ""].drop_duplicates("accession", keep="first")
    tip_map = dict(zip(tips["accession"], tips["tree_node_index"].astype(int)))
    matched = [acc for acc in panel_accessions if acc in tip_map]
    missing = [acc for acc in panel_accessions if acc not in tip_map]
    if not matched:
        raise ValueError(f"No panel accessions were present in {tree_path}")

    tip_indices = np.array([tip_map[acc] for acc in matched], dtype=np.int32)
    n = len(tip_indices)
    log(f"Computing time-dated tree patristic matrix: matched={n:,}, missing={len(missing):,}")

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
        log(f"Time-tree patristic rows {start:,}-{stop - 1:,}/{n:,}")
    del D

    nodes = pd.DataFrame(
        {
            "node_id": np.arange(n, dtype=int),
            "accession": matched,
            "tree_node_index": tip_indices,
            "tree_tip_name": [tips.set_index("accession").loc[acc, "tree_tip_name"] for acc in matched],
        }
    )
    nodes.to_csv(nodes_path, index=False)
    qc = {
        "tree_path": str(tree_path),
        "tree_format": arrays["format"],
        "n_panel_accessions": int(len(panel_accessions)),
        "n_matched_tree_tips": int(n),
        "n_missing_tree_tips": int(len(missing)),
        "missing_tree_tip_examples": missing[:10],
        "n_tree_nodes": int(arrays["n_nodes"]),
        "matrix_path": str(matrix_path),
        "matrix_shape": [int(n), int(n)],
        "matrix_dtype": "float32",
        "matrix_size_gb": float((n * n * np.dtype(np.float32).itemsize) / 1e9),
    }
    qc_path.write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
    log(f"Wrote time-dated tree patristic matrix: {matrix_path}")
    return nodes, np.load(matrix_path, mmap_mode="r")


def load_time_reference(out_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    nodes_path = out_dir / "D_time_dated_tree_nodes.csv"
    matrix_path = out_dir / "D_time_dated_tree_float32.npy"
    if not nodes_path.exists() or not matrix_path.exists():
        raise FileNotFoundError(f"Missing time-dated tree reference outputs under {out_dir}")
    return pd.read_csv(nodes_path), np.load(matrix_path, mmap_mode="r")


def write_time_control(
    panel_root: Path,
    sample_label: str,
    matched: pd.DataFrame,
    D_ref: np.ndarray,
    out_dir: Path,
    pair_mode: str,
    pair_sample_size: int,
    pair_seed: int,
) -> None:
    meta = pd.read_csv(panel_root / "inputs" / sample_label / "metadata.csv", usecols=["accession", "collection_date"])
    date_map = {
        str(row["accession"]).strip(): decimal_year(row["collection_date"])
        for _, row in meta.iterrows()
    }
    dates = np.array([date_map.get(acc) if date_map.get(acc) is not None else np.nan for acc in matched["accession"]], dtype=float)
    if np.isnan(dates).any():
        log("Skipping time-control correlation because matched accessions include missing dates")
        return
    from scipy.stats import spearmanr
    from scripts.validation.nextstrain_spike_tree_validation import make_pair_indices

    pair_i, pair_j = make_pair_indices(len(dates), pair_mode, pair_sample_size, pair_seed)
    x = np.asarray(D_ref[pair_i, pair_j])
    y = np.abs(dates[pair_i] - dates[pair_j])
    raw_pairs = int(len(x))
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        rho = np.nan
        pvalue = np.nan
    else:
        rho, pvalue = spearmanr(x[mask], y[mask])
    score = {
        "pair_mode": pair_mode,
        "n_pairs_raw": raw_pairs,
        "n_pairs_used": int(mask.sum()),
        "finite_pair_fraction": float(mask.sum() / raw_pairs) if raw_pairs else np.nan,
        "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
        "spearman_pvalue": float(pvalue) if np.isfinite(pvalue) else np.nan,
    }
    pd.DataFrame(
        [
            {
                "control": "absolute_collection_date_difference_decimal_years",
                "n_matched": int(len(matched)),
                **score,
            }
        ]
    ).to_csv(out_dir / "time_tree_vs_collection_date_difference.csv", index=False)
    log(f"Wrote time control correlation: {out_dir / 'time_tree_vs_collection_date_difference.csv'}")


def run_panel_seed(
    panel: str,
    seed: int,
    source_root: Path,
    workspace: Path,
    reference_workspace: Path,
    sample_label: str,
    stages: set[str],
    treetime_bin: str,
    treetime_extra_args: list[str],
    input_tree: Path | None,
    input_alignment: Path | None,
    dated_tree: Path | None,
    block_size: int,
    pair_mode: str,
    pair_sample_size: int,
    pair_seed: int,
    force_treetime: bool,
    overwrite_patristic: bool,
) -> None:
    panel_root = source_root / panel / f"seed_{seed}"
    out_dir = workspace / panel / f"seed_{seed}"
    tree_dir = out_dir / "time_dated_tree"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not panel_root.exists():
        raise FileNotFoundError(f"Missing panel seed root: {panel_root}")

    reference_tree_dir = reference_workspace / panel / f"seed_{seed}" / "reference_tree"
    topology_tree = input_tree or reference_tree_dir / "spike_reference_fasttree.nwk"
    alignment = input_alignment or reference_tree_dir / "spike_sequences_aligned.fasta"
    if not topology_tree.exists():
        raise FileNotFoundError(f"Missing panel topology tree: {topology_tree}")
    if not alignment.exists():
        alignment = panel_root / "inputs" / sample_label / "spike_sequences_aligned_mafft.fasta"
    if not alignment.exists():
        raise FileNotFoundError(f"Missing aligned FASTA: {alignment}")

    if "prepare" in stages:
        write_dates(panel_root, sample_label, out_dir)
    dates_path = out_dir / "dates.tsv"
    if not dates_path.exists():
        raise FileNotFoundError(f"Missing dates table; run prepare stage first: {dates_path}")

    downstream_stages = {"timetree", "patristic", "time_control", "raw", "graphs", "partial", "summary"}
    if not (stages & downstream_stages):
        return

    if "timetree" in stages:
        active_dated_tree = run_treetime(
            input_tree=topology_tree,
            alignment=alignment,
            dates_tsv=dates_path,
            out_dir=tree_dir,
            treetime_bin=treetime_bin,
            extra_args=treetime_extra_args,
            force=force_treetime,
        )
    else:
        active_dated_tree = dated_tree or find_dated_tree(tree_dir)
        if active_dated_tree is None:
            raise FileNotFoundError(f"Missing dated tree; run timetree stage first or pass --dated-tree under {tree_dir}")

    if "patristic" in stages:
        accessions = load_panel_accessions(panel_root, sample_label)
        matched, D_ref = compute_time_patristic_matrix(active_dated_tree, accessions, out_dir, block_size, overwrite_patristic)
    else:
        matched, D_ref = load_time_reference(out_dir)

    score_stages = {"time_control", "raw", "graphs", "partial", "summary"}
    if stages & score_stages:
        if "time_control" in stages:
            write_time_control(panel_root, sample_label, matched, D_ref, out_dir, pair_mode, pair_sample_size, pair_seed)
        if "raw" in stages:
            score_raw_distances(panel, seed, panel_root, sample_label, matched, D_ref, out_dir, pair_mode, pair_sample_size, pair_seed)
        if "graphs" in stages:
            score_graph_distances(panel, seed, panel_root, sample_label, matched, D_ref, out_dir, pair_mode, pair_sample_size, pair_seed)
        if "partial" in stages:
            score_partial_correlations(panel, seed, panel_root, sample_label, matched, D_ref, out_dir, pair_mode, pair_sample_size, pair_seed)
        if "summary" in stages:
            write_delta_summary(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build and score time-dated panel spike trees against 20k graph distances.")
    ap.add_argument("--workspace", type=Path, default=Path("analysis/cohort_validation/11_time_dated_tree_validation"))
    ap.add_argument("--source-root", type=Path, default=Path("analysis/cohort_validation/07_sampling_design_20k"))
    ap.add_argument("--reference-workspace", type=Path, default=Path("analysis/cohort_validation/09_nextstrain_spike_tree_validation"))
    ap.add_argument("--panels", default="random_full_dataset_seed42")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--sample-label", default="pool_n20000")
    ap.add_argument("--stages", default="prepare,timetree,patristic,time_control,raw,graphs,partial,summary,aggregate")
    ap.add_argument("--treetime-bin", default="treetime")
    ap.add_argument("--treetime-extra-args", default="")
    ap.add_argument("--input-tree", type=Path, default=None)
    ap.add_argument("--input-alignment", type=Path, default=None)
    ap.add_argument("--dated-tree", type=Path, default=None)
    ap.add_argument("--patristic-block-size", type=int, default=128)
    ap.add_argument("--pair-mode", choices=["all", "sample"], default="sample")
    ap.add_argument("--pair-sample-size", type=int, default=5_000_000)
    ap.add_argument("--pair-seed", type=int, default=12345)
    ap.add_argument("--force-treetime", action="store_true")
    ap.add_argument("--overwrite-patristic", action="store_true")
    args = ap.parse_args()

    args.workspace.mkdir(parents=True, exist_ok=True)
    stages = {stage.strip() for stage in args.stages.split(",") if stage.strip()}
    extra_args = [arg for arg in args.treetime_extra_args.split(" ") if arg]

    for panel in [p.strip() for p in args.panels.split(",") if p.strip()]:
        for seed in parse_seed_list(args.seeds):
            run_panel_seed(
                panel=panel,
                seed=seed,
                source_root=args.source_root,
                workspace=args.workspace,
                reference_workspace=args.reference_workspace,
                sample_label=args.sample_label,
                stages=stages,
                treetime_bin=args.treetime_bin,
                treetime_extra_args=extra_args,
                input_tree=args.input_tree,
                input_alignment=args.input_alignment,
                dated_tree=args.dated_tree,
                block_size=args.patristic_block_size,
                pair_mode=args.pair_mode,
                pair_sample_size=args.pair_sample_size,
                pair_seed=args.pair_seed,
                force_treetime=args.force_treetime,
                overwrite_patristic=args.overwrite_patristic,
            )

    if "aggregate" in stages:
        aggregate_workspace(args.workspace)
        log(f"Wrote aggregate summaries under {args.workspace}")


if __name__ == "__main__":
    main()
