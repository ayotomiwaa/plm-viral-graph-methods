#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from Bio import Phylo


ACCESSION_RE = re.compile(r"EPI_ISL_\d+")


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


def parse_accession(value: object) -> str:
    text = str(value).strip()
    match = ACCESSION_RE.search(text)
    if match:
        return match.group(0)
    return text.split()[0].split("|")[0].strip()


def read_accessions(path: Path) -> list[str]:
    return [line.strip() for line in path.open("r", encoding="utf-8", errors="replace") if line.strip()]


def fasta_accessions(path: Path) -> list[str]:
    accessions: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(">"):
                accessions.append(parse_accession(line[1:]))
    return accessions


def ensure_alignment(
    input_dir: Path,
    out_dir: Path,
    aligned_name: str,
    unaligned_name: str,
    force_align: bool,
    mafft_bin: str,
    threads: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    aligned_in = input_dir / aligned_name
    unaligned_in = input_dir / unaligned_name
    aligned_out = out_dir / "spike_sequences_aligned.fasta"
    manifest_path = out_dir / "alignment_manifest.json"

    if aligned_out.exists() and not force_align:
        log(f"Using existing copied alignment: {aligned_out}")
        return aligned_out

    if aligned_in.exists() and not force_align:
        shutil.copy2(aligned_in, aligned_out)
        manifest = {
            "alignment_source": str(aligned_in),
            "alignment_output": str(aligned_out),
            "alignment_method": "existing_mafft_alignment_copied",
            "n_sequences": len(fasta_accessions(aligned_out)),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        log(f"Copied existing alignment: {aligned_in} -> {aligned_out}")
        return aligned_out

    if not unaligned_in.exists():
        raise FileNotFoundError(f"Missing aligned and unaligned FASTA inputs in {input_dir}")

    mafft = shutil.which(mafft_bin)
    if mafft is None:
        raise FileNotFoundError(f"Could not find MAFFT binary: {mafft_bin}")

    log(f"Running MAFFT alignment: {unaligned_in}")
    cmd = [mafft, "--thread", str(threads), "--auto", str(unaligned_in)]
    with aligned_out.open("w", encoding="utf-8") as stdout:
        proc = subprocess.run(cmd, stdout=stdout, stderr=subprocess.PIPE, text=True, check=False)
    (out_dir / "mafft.stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"MAFFT failed with exit code {proc.returncode}; see {out_dir / 'mafft.stderr.log'}")

    manifest = {
        "alignment_source": str(unaligned_in),
        "alignment_output": str(aligned_out),
        "alignment_method": "mafft_auto",
        "mafft_bin": mafft,
        "threads": int(threads),
        "n_sequences": len(fasta_accessions(aligned_out)),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log(f"Wrote MAFFT alignment: {aligned_out}")
    return aligned_out


def run_fasttree(alignment: Path, out_dir: Path, fasttree_bin: str, overwrite: bool) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    tree_path = out_dir / "spike_reference_fasttree.nwk"
    stderr_path = out_dir / "fasttree.stderr.log"
    manifest_path = out_dir / "tree_manifest.json"
    if tree_path.exists() and tree_path.stat().st_size > 0 and not overwrite:
        log(f"Using existing FastTree Newick: {tree_path}")
        return tree_path

    fasttree = shutil.which(fasttree_bin)
    if fasttree is None:
        raise FileNotFoundError(f"Could not find FastTree binary: {fasttree_bin}")

    log(f"Running FastTree protein tree for {alignment}")
    cmd = [fasttree, "-quiet", str(alignment)]
    with tree_path.open("w", encoding="utf-8") as stdout:
        proc = subprocess.run(cmd, stdout=stdout, stderr=subprocess.PIPE, text=True, check=False)
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"FastTree failed with exit code {proc.returncode}; see {stderr_path}")

    manifest = {
        "tree_builder": "FastTree",
        "fasttree_bin": fasttree,
        "alignment": str(alignment),
        "tree_path": str(tree_path),
        "command": cmd,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log(f"Wrote FastTree Newick: {tree_path}")
    return tree_path


def load_panel_accessions(panel_root: Path, sample_label: str) -> list[str]:
    input_dir = panel_root / "inputs" / sample_label
    accessions_path = input_dir / "accessions.txt"
    if accessions_path.exists():
        return read_accessions(accessions_path)
    metadata_path = input_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing panel accession inputs under {input_dir}")
    meta = pd.read_csv(metadata_path, usecols=["accession"], low_memory=False)
    return meta["accession"].astype(str).str.strip().tolist()


def tree_arrays(newick_path: Path) -> dict[str, Any]:
    tree = Phylo.read(str(newick_path), "newick")
    clades = list(tree.find_clades(order="preorder"))
    index = {id(clade): i for i, clade in enumerate(clades)}
    n_nodes = len(clades)
    parent = np.full(n_nodes, -1, dtype=np.int32)
    depth = np.zeros(n_nodes, dtype=np.int32)
    root_dist = np.zeros(n_nodes, dtype=np.float64)
    names = [""] * n_nodes
    tip_rows: list[dict[str, Any]] = []

    def visit(clade: Any, parent_idx: int) -> None:
        idx = index[id(clade)]
        parent[idx] = idx if parent_idx < 0 else parent_idx
        names[idx] = "" if clade.name is None else str(clade.name)
        if parent_idx >= 0:
            depth[idx] = depth[parent_idx] + 1
            root_dist[idx] = root_dist[parent_idx] + float(clade.branch_length or 0.0)
        if clade.is_terminal():
            accession = parse_accession(names[idx])
            tip_rows.append({"accession": accession, "tree_node_index": idx, "tree_tip_name": names[idx]})
        for child in clade.clades:
            visit(child, idx)

    visit(tree.root, -1)
    max_log = max(1, math.ceil(math.log2(max(2, n_nodes))) + 1)
    up = np.empty((max_log, n_nodes), dtype=np.int32)
    up[0] = parent
    for level in range(1, max_log):
        up[level] = up[level - 1, up[level - 1]]

    tips = pd.DataFrame(tip_rows)
    return {
        "parent": parent,
        "depth": depth,
        "root_dist": root_dist,
        "up": up,
        "tips": tips,
        "n_nodes": n_nodes,
    }


def lca_many(a: np.ndarray, b: np.ndarray, up: np.ndarray, depth: np.ndarray) -> np.ndarray:
    a = a.astype(np.int32, copy=True)
    b = b.astype(np.int32, copy=True)
    swap = depth[a] < depth[b]
    if swap.any():
        tmp = a[swap].copy()
        a[swap] = b[swap]
        b[swap] = tmp

    diff = depth[a] - depth[b]
    for level in range(up.shape[0]):
        mask = ((diff >> level) & 1).astype(bool)
        if mask.any():
            a[mask] = up[level, a[mask]]

    for level in range(up.shape[0] - 1, -1, -1):
        mask = (a != b) & (up[level, a] != up[level, b])
        if mask.any():
            a[mask] = up[level, a[mask]]
            b[mask] = up[level, b[mask]]

    out = a.copy()
    mask = a != b
    if mask.any():
        out[mask] = up[0, a[mask]]
    return out


def compute_patristic_matrix(
    newick_path: Path,
    panel_accessions: list[str],
    out_dir: Path,
    block_size: int,
    overwrite: bool,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = out_dir / "D_reference_spike_float32.npy"
    nodes_path = out_dir / "D_reference_spike_nodes.csv"
    qc_path = out_dir / "D_reference_spike_qc.json"
    if matrix_path.exists() and nodes_path.exists() and not overwrite:
        log(f"Using existing patristic matrix: {matrix_path}")
        return matrix_path

    arrays = tree_arrays(newick_path)
    tips = arrays["tips"].copy()
    tips = tips[tips["accession"] != ""].drop_duplicates("accession", keep="first")
    tip_map = dict(zip(tips["accession"], tips["tree_node_index"].astype(int)))
    matched = [acc for acc in panel_accessions if acc in tip_map]
    missing = [acc for acc in panel_accessions if acc not in tip_map]
    if not matched:
        raise ValueError(f"No panel accessions were present in {newick_path}")

    tip_indices = np.array([tip_map[acc] for acc in matched], dtype=np.int32)
    n = len(tip_indices)
    log(f"Computing reference spike patristic matrix: matched={n:,}, missing={len(missing):,}")

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
        log(f"Patristic rows {start:,}-{stop - 1:,}/{n:,}")
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
        "newick_path": str(newick_path),
        "n_panel_accessions": int(len(panel_accessions)),
        "n_matched_tree_tips": int(n),
        "n_missing_tree_tips": int(len(missing)),
        "missing_tree_tip_examples": missing[:10],
        "n_tree_nodes": int(arrays["n_nodes"]),
        "n_tree_tips_with_accessions": int(len(tips)),
        "matrix_path": str(matrix_path),
        "matrix_shape": [int(n), int(n)],
        "matrix_dtype": "float32",
        "matrix_size_gb": float((n * n * np.dtype(np.float32).itemsize) / 1e9),
    }
    qc_path.write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
    log(f"Wrote reference spike patristic matrix: {matrix_path}")
    return matrix_path


def run_panel_seed(
    panel: str,
    seed: int,
    source_root: Path,
    workspace: Path,
    sample_label: str,
    stages: set[str],
    args: argparse.Namespace,
) -> None:
    panel_root = source_root / panel / f"seed_{seed}"
    if not panel_root.exists():
        log(f"Skipping missing panel seed root: {panel_root}")
        return
    input_dir = panel_root / "inputs" / sample_label
    out_dir = workspace / panel / f"seed_{seed}" / "reference_tree"
    out_dir.mkdir(parents=True, exist_ok=True)

    alignment_path = out_dir / "spike_sequences_aligned.fasta"
    if "prepare" in stages:
        alignment_path = ensure_alignment(
            input_dir=input_dir,
            out_dir=out_dir,
            aligned_name=args.aligned_fasta_name,
            unaligned_name=args.unaligned_fasta_name,
            force_align=args.force_align,
            mafft_bin=args.mafft_bin,
            threads=args.threads,
        )
    elif not alignment_path.exists():
        src_alignment = input_dir / args.aligned_fasta_name
        if src_alignment.exists():
            alignment_path = src_alignment
        else:
            raise FileNotFoundError(f"Missing prepared alignment: {alignment_path}")

    if "patristic" in stages:
        tree_path = out_dir / "spike_reference_fasttree.nwk"
        if args.input_newick:
            tree_path = args.input_newick
        elif "tree" in stages:
            tree_path = run_fasttree(
                alignment=alignment_path,
                out_dir=out_dir,
                fasttree_bin=args.fasttree_bin,
                overwrite=args.overwrite_tree,
            )
        elif not tree_path.exists():
            raise FileNotFoundError(f"Missing tree; run tree stage or pass --input-newick: {tree_path}")
        panel_accessions = load_panel_accessions(panel_root, sample_label)
        compute_patristic_matrix(
            newick_path=tree_path,
            panel_accessions=panel_accessions,
            out_dir=out_dir,
            block_size=args.patristic_block_size,
            overwrite=args.overwrite_patristic,
        )
    elif "tree" in stages and not args.input_newick:
        run_fasttree(
            alignment=alignment_path,
            out_dir=out_dir,
            fasttree_bin=args.fasttree_bin,
            overwrite=args.overwrite_tree,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build per-panel spike reference trees and patristic matrices.")
    ap.add_argument("--workspace", type=Path, default=Path("analysis/cohort_validation/09_nextstrain_spike_tree_validation"))
    ap.add_argument("--source-root", type=Path, default=Path("analysis/cohort_validation/07_sampling_design_20k"))
    ap.add_argument("--panels", default="random_full_dataset_seed42,monthly_stratified_full_dataset_seed42")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--sample-label", default="pool_n20000")
    ap.add_argument("--stages", default="prepare,tree,patristic")
    ap.add_argument("--aligned-fasta-name", default="spike_sequences_aligned_mafft.fasta")
    ap.add_argument("--unaligned-fasta-name", default="spike_sequences.fasta")
    ap.add_argument("--force-align", action="store_true")
    ap.add_argument("--mafft-bin", default="mafft")
    ap.add_argument("--fasttree-bin", default="FastTree")
    ap.add_argument("--input-newick", type=Path, default=None)
    ap.add_argument("--threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    ap.add_argument("--patristic-block-size", type=int, default=128)
    ap.add_argument("--overwrite-tree", action="store_true")
    ap.add_argument("--overwrite-patristic", action="store_true")
    args = ap.parse_args()

    stages = {stage.strip() for stage in args.stages.split(",") if stage.strip()}
    panels = [panel.strip() for panel in args.panels.split(",") if panel.strip()]
    seeds = parse_seed_list(args.seeds)
    for panel in panels:
        for seed in seeds:
            log(f"=== {panel}/seed_{seed} ===")
            run_panel_seed(
                panel=panel,
                seed=seed,
                source_root=args.source_root,
                workspace=args.workspace,
                sample_label=args.sample_label,
                stages=stages,
                args=args,
            )


if __name__ == "__main__":
    main()
