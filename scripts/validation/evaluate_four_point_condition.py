#!/usr/bin/env python3
"""Four-point-condition tree-likeness for the frozen seed-42 2k distance matrices.

A distance matrix is an additive (exact) tree metric if and only if, for every
four points a, b, c, d, the two largest of

    S1 = d(a,b) + d(c,d)
    S2 = d(a,c) + d(b,d)
    S3 = d(a,d) + d(b,c)

are equal.  Sampling quartets measures that directly, without going through
D -> NJ -> D_tree, and complements the Gromov delta already reported by the
paired 2k tree-geometry workflow.

Reported per representation, over the same quartets for every representation:

- delta_q  = (S_max - S_mid) / (S_max - S_min), the Holland delta-plot statistic;
  0 means the quartet is exactly additive, 1 means maximally non-additive
- the raw four-point gap (S_max - S_mid) / 2 and its scale-normalised version
- the fraction of quartets that are exactly additive and the fraction resolved
- quartet topology agreement with the ML reference tree, restricted to quartets
  both sides resolve

The ML patristic matrix is included as a positive control: an additive tree
metric must score delta_q = 0.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validation.build_reference_ml_tree_panel import (  # noqa: E402
    DEFAULT_FROZEN_2K_ROOT,
    DEFAULT_OUT_ROOT,
    file_signature,
    log,
    read_json,
    stable_fingerprint,
    write_json,
)

STAGES = ["prepare-quartets", "evaluate", "summarize"]

FROZEN_2K_REPRESENTATIONS = [
    "raw_hamming",
    "hamming_knn5",
    "hamming_knn50",
    "hamming_rng",
    "raw_embedding_cityblock",
    "embedding_knn5",
    "embedding_knn50",
    "embedding_rng",
    "refined_embedding_knn5",
    "refined_embedding_knn50",
    "refined_embedding_rng",
]

ML_PATRISTIC_KEY = "ml_patristic_reference"


def sample_quartets(n_tips: int, n_samples: int, seed: int, chunk: int = 500_000) -> np.ndarray:
    """Uniform quartets of distinct tips, drawn once and shared by all matrices."""
    if n_tips < 4:
        return np.empty((0, 4), dtype=np.int32)
    rng = np.random.default_rng(seed)
    out = np.empty((n_samples, 4), dtype=np.int32)
    filled = 0
    while filled < n_samples:
        size = min(chunk, n_samples - filled)
        draw = rng.integers(0, n_tips, size=(int(size * 1.3) + 16, 4), dtype=np.int64)
        distinct = (
            (draw[:, 0] != draw[:, 1])
            & (draw[:, 0] != draw[:, 2])
            & (draw[:, 0] != draw[:, 3])
            & (draw[:, 1] != draw[:, 2])
            & (draw[:, 1] != draw[:, 3])
            & (draw[:, 2] != draw[:, 3])
        )
        usable = draw[distinct][:size]
        out[filled : filled + len(usable)] = usable.astype(np.int32)
        filled += len(usable)
    return out


def quartet_sums(D: np.ndarray, quartets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the three pairing sums and the six pairwise distances per quartet."""
    a, b, c, d = quartets[:, 0], quartets[:, 1], quartets[:, 2], quartets[:, 3]
    ab = D[a, b]
    cd = D[c, d]
    ac = D[a, c]
    bd = D[b, d]
    ad = D[a, d]
    bc = D[b, c]
    sums = np.column_stack([ab + cd, ac + bd, ad + bc]).astype(np.float64, copy=False)
    pairwise = np.column_stack([ab, cd, ac, bd, ad, bc]).astype(np.float64, copy=False)
    return sums, pairwise


def four_point_chunk(
    D: np.ndarray, quartets: np.ndarray, tolerance: float
) -> dict[str, np.ndarray]:
    sums, pairwise = quartet_sums(D, quartets)
    order = np.argsort(sums, axis=1)
    ordered = np.take_along_axis(sums, order, axis=1)
    s_min, s_mid, s_max = ordered[:, 0], ordered[:, 1], ordered[:, 2]
    spread = s_max - s_min
    gap = s_max - s_mid
    delta_q = np.where(spread > 0.0, gap / np.where(spread > 0.0, spread, 1.0), 0.0)
    scale = pairwise.max(axis=1)
    normalised_gap = np.where(scale > 0.0, 0.5 * gap / np.where(scale > 0.0, scale, 1.0), 0.0)
    # the minimum-sum pairing is the quartet topology; -1 when it is not unique
    topology = order[:, 0].astype(np.int8)
    unresolved = (s_mid - s_min) <= tolerance * np.maximum(scale, 1e-12)
    topology = np.where(unresolved, np.int8(-1), topology)
    return {
        "delta_q": delta_q,
        "gap": gap,
        "normalised_gap": normalised_gap,
        "scale": scale,
        "topology": topology,
        "degenerate": spread <= 0.0,
        "exactly_additive": gap <= tolerance * np.maximum(scale, 1e-12),
        "all_pairs_positive": (pairwise > 0.0).all(axis=1),
    }


def weighted_quantiles(
    counts: np.ndarray, bin_edges: np.ndarray, quantiles: list[float]
) -> list[float]:
    total = counts.sum()
    if total == 0:
        return [math.nan for _ in quantiles]
    cumulative = np.cumsum(counts) / total
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return [
        float(centers[min(int(np.searchsorted(cumulative, q)), centers.size - 1)])
        for q in quantiles
    ]


class Accumulator:
    """Streaming statistics for one representation and one quartet subset."""

    def __init__(self, n_bins: int = 200) -> None:
        self.n = 0
        self.sum_delta = 0.0
        self.sum_delta_sq = 0.0
        self.sum_normalised_gap = 0.0
        self.n_degenerate = 0
        self.n_exactly_additive = 0
        self.n_resolved = 0
        self.bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        self.hist = np.zeros(n_bins, dtype=np.int64)

    def update(self, chunk: dict[str, np.ndarray], mask: np.ndarray) -> None:
        if not mask.any():
            return
        delta = chunk["delta_q"][mask]
        self.n += int(delta.size)
        self.sum_delta += float(delta.sum())
        self.sum_delta_sq += float(np.square(delta).sum())
        self.sum_normalised_gap += float(chunk["normalised_gap"][mask].sum())
        self.n_degenerate += int(np.count_nonzero(chunk["degenerate"][mask]))
        self.n_exactly_additive += int(np.count_nonzero(chunk["exactly_additive"][mask]))
        self.n_resolved += int(np.count_nonzero(chunk["topology"][mask] >= 0))
        self.hist += np.histogram(delta, bins=self.bin_edges)[0]

    def summary(self) -> dict[str, Any]:
        if self.n == 0:
            return {"n_quartets": 0}
        mean = self.sum_delta / self.n
        variance = max(0.0, self.sum_delta_sq / self.n - mean**2)
        q25, q50, q75, q95 = weighted_quantiles(self.hist, self.bin_edges, [0.25, 0.5, 0.75, 0.95])
        return {
            "n_quartets": int(self.n),
            "mean_delta_q": float(mean),
            "sd_delta_q": float(math.sqrt(variance)),
            "median_delta_q": q50,
            "q25_delta_q": q25,
            "q75_delta_q": q75,
            "q95_delta_q": q95,
            "mean_normalised_four_point_gap": float(self.sum_normalised_gap / self.n),
            "fraction_degenerate": float(self.n_degenerate / self.n),
            "fraction_exactly_additive": float(self.n_exactly_additive / self.n),
            "fraction_resolved": float(self.n_resolved / self.n),
        }


def representation_matrices(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for key in FROZEN_2K_REPRESENTATIONS:
        path = args.frozen_2k_root / "matrices" / key / "D_input_float32.npy"
        specs[key] = {"key": key, "path": path, "family": "frozen_2k_representation"}
    specs[ML_PATRISTIC_KEY] = {
        "key": ML_PATRISTIC_KEY,
        "path": args.out_root / "tree_evaluation/D_ml_patristic_float32.npy",
        "family": "reference_tree_positive_control",
    }
    return specs


def load_frozen_2k_rows(args: argparse.Namespace) -> np.ndarray:
    """Rows of the panel patristic matrix that correspond to the frozen 2k tips."""
    panel = pd.read_csv(args.out_root / "design/panel_tips.csv")
    frozen = pd.read_csv(args.frozen_2k_root / "design/selected_tips.csv", usecols=["node_id"])
    panel_row_of_node = dict(
        zip(panel["node_id"].astype(int).tolist(), panel["panel_row"].astype(int).tolist())
    )
    missing = [node for node in frozen["node_id"].astype(int) if node not in panel_row_of_node]
    if missing:
        raise ValueError(
            f"{len(missing):,} frozen 2k tips are not in the ML panel; rebuild the panel with "
            f"--include-frozen-2k"
        )
    return np.array(
        [panel_row_of_node[int(node)] for node in frozen["node_id"].astype(int)], dtype=np.int64
    )


def stage_prepare_quartets(args: argparse.Namespace) -> np.ndarray:
    frozen = pd.read_csv(args.frozen_2k_root / "design/selected_tips.csv", usecols=["node_id"])
    n_tips = len(frozen)
    quartets = sample_quartets(n_tips, args.quartet_samples, args.quartet_seed)
    design_dir = args.out_root / "four_point"
    design_dir.mkdir(parents=True, exist_ok=True)
    path = design_dir / "quartets_int32.npy"
    np.save(path, quartets)
    write_json(
        design_dir / "quartet_design.json",
        {
            "prepared_at_unix": time.time(),
            "n_tips": int(n_tips),
            "n_quartets": int(len(quartets)),
            "quartet_seed": int(args.quartet_seed),
            "quartets_path": str(path),
            "sampling_policy": "uniform quartets of four distinct tips, rejection sampled",
            "shared_across_representations": True,
            "design_fingerprint": stable_fingerprint(
                {
                    "n_tips": int(n_tips),
                    "n_quartets": int(len(quartets)),
                    "quartet_seed": int(args.quartet_seed),
                }
            ),
        },
    )
    log(f"Wrote {len(quartets):,} quartets over {n_tips:,} tips: {path}")
    return quartets


def load_matrix(spec: dict[str, Any], args: argparse.Namespace) -> np.ndarray:
    path = spec["path"]
    if not path.exists():
        raise FileNotFoundError(f"Missing matrix for {spec['key']}: {path}")
    D = np.load(path, mmap_mode="r")
    if spec["key"] == ML_PATRISTIC_KEY:
        rows = load_frozen_2k_rows(args)
        D = np.asarray(D[np.ix_(rows, rows)], dtype=np.float64)
    else:
        D = np.asarray(D, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"{path}: expected a square matrix, observed {D.shape}")
    return D


def stage_evaluate(args: argparse.Namespace, keys: list[str]) -> None:
    quartet_path = args.out_root / "four_point/quartets_int32.npy"
    if not quartet_path.exists():
        raise FileNotFoundError("Missing quartets; run --stages prepare-quartets first")
    quartets = np.load(quartet_path)
    specs = representation_matrices(args)

    reference_topology: np.ndarray | None = None
    reference_path = args.out_root / "four_point" / ML_PATRISTIC_KEY / "topology_int8.npy"
    if reference_path.exists():
        candidate = np.load(reference_path, mmap_mode="r")
        # a stale reference from an earlier quartet design must not be reused
        reference_topology = candidate if len(candidate) == len(quartets) else None

    ordered_keys = [ML_PATRISTIC_KEY] + [key for key in keys if key != ML_PATRISTIC_KEY]
    for key in ordered_keys:
        spec = specs[key]
        if key == ML_PATRISTIC_KEY and not spec["path"].exists():
            if key in keys:
                raise FileNotFoundError(
                    f"Missing ML patristic matrix {spec['path']}; run the edge-validation "
                    f"--stages patristic first"
                )
            log("ML patristic matrix absent; skipping the reference-tree quartet agreement column")
            continue
        out_dir = args.out_root / "four_point" / key
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = out_dir / "metrics.json"
        topology_path = out_dir / "topology_int8.npy"
        if metrics_path.exists() and topology_path.exists() and not args.overwrite:
            log(f"Using existing four-point metrics: {metrics_path}")
            if key == ML_PATRISTIC_KEY:
                reference_topology = np.load(topology_path, mmap_mode="r")
            continue

        log(f"Four-point condition: {key}")
        D = load_matrix(spec, args)
        if quartets.size and int(quartets.max()) >= D.shape[0]:
            raise ValueError(
                f"{key}: matrix has {D.shape[0]} rows but the quartet design indexes tip "
                f"{int(quartets.max())}"
            )
        overall = Accumulator()
        distinct_only = Accumulator()
        scale_bins: dict[int, Accumulator] = {}
        topology = np.empty(len(quartets), dtype=np.int8)
        agreement_total = 0
        agreement_hits = 0
        scale_edges: np.ndarray | None = None

        for start in range(0, len(quartets), args.chunk_size):
            stop = min(start + args.chunk_size, len(quartets))
            block = quartets[start:stop]
            chunk = four_point_chunk(D, block, tolerance=args.tolerance)
            topology[start:stop] = chunk["topology"]
            everything = np.ones(stop - start, dtype=bool)
            overall.update(chunk, everything)
            distinct_only.update(chunk, chunk["all_pairs_positive"])
            if scale_edges is None:
                positive = chunk["scale"][chunk["scale"] > 0.0]
                scale_edges = (
                    np.quantile(positive, [0.0, 0.25, 0.5, 0.75, 1.0])
                    if positive.size
                    else np.array([0.0, 1.0])
                )
            bins = np.clip(
                np.searchsorted(scale_edges[1:-1], chunk["scale"], side="right"),
                0,
                max(0, len(scale_edges) - 2),
            )
            for bin_index in np.unique(bins):
                scale_bins.setdefault(int(bin_index), Accumulator()).update(
                    chunk, bins == bin_index
                )
            if reference_topology is not None and key != ML_PATRISTIC_KEY:
                reference_block = np.asarray(reference_topology[start:stop])
                comparable = (reference_block >= 0) & (chunk["topology"] >= 0)
                agreement_total += int(np.count_nonzero(comparable))
                agreement_hits += int(
                    np.count_nonzero(reference_block[comparable] == chunk["topology"][comparable])
                )

        np.save(topology_path, topology)
        if key == ML_PATRISTIC_KEY:
            reference_topology = topology

        metrics: dict[str, Any] = {
            "evaluated_at_unix": time.time(),
            "representation": key,
            "representation_family": spec["family"],
            "matrix_path": str(spec["path"]),
            "matrix_signature": file_signature(spec["path"]),
            "n_tips": int(D.shape[0]),
            "tolerance": float(args.tolerance),
            "quartets_path": str(quartet_path),
            "all_quartets": overall.summary(),
            "distinct_sequence_quartets": distinct_only.summary(),
            "scale_quartile_bins": {
                str(index): accumulator.summary() for index, accumulator in sorted(scale_bins.items())
            },
            "scale_bin_edges": [float(value) for value in (scale_edges if scale_edges is not None else [])],
        }
        if key != ML_PATRISTIC_KEY:
            metrics["reference_tree_quartet_agreement"] = {
                "n_comparable_quartets": int(agreement_total),
                "fraction_matching_ml_tree_topology": float(agreement_hits / agreement_total)
                if agreement_total
                else math.nan,
            }
        np.savez_compressed(
            out_dir / "delta_q_histogram.npz",
            bin_edges=overall.bin_edges,
            all_quartets=overall.hist,
            distinct_sequence_quartets=distinct_only.hist,
        )
        write_json(metrics_path, metrics)
        summary = metrics["all_quartets"]
        log(
            f"  {key}: mean delta_q={summary.get('mean_delta_q', float('nan')):.4f} "
            f"exactly additive={summary.get('fraction_exactly_additive', float('nan')):.4f} "
            f"resolved={summary.get('fraction_resolved', float('nan')):.4f}"
        )


def stage_summarize(args: argparse.Namespace, keys: list[str]) -> None:
    rows: list[dict[str, Any]] = []
    strata_rows: list[dict[str, Any]] = []
    for key in [ML_PATRISTIC_KEY] + [item for item in keys if item != ML_PATRISTIC_KEY]:
        metrics_path = args.out_root / "four_point" / key / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = read_json(metrics_path)
        row: dict[str, Any] = {
            "representation": key,
            "representation_family": metrics["representation_family"],
            "n_tips": metrics["n_tips"],
        }
        for prefix, block in [
            ("all", metrics["all_quartets"]),
            ("distinct", metrics["distinct_sequence_quartets"]),
        ]:
            for name, value in block.items():
                row[f"{prefix}_{name}"] = value
        agreement = metrics.get("reference_tree_quartet_agreement", {})
        row["ml_tree_quartet_agreement"] = agreement.get("fraction_matching_ml_tree_topology", math.nan)
        row["ml_tree_quartet_comparable"] = agreement.get("n_comparable_quartets", 0)
        rows.append(row)
        for bin_index, block in metrics["scale_quartile_bins"].items():
            strata_rows.append({"representation": key, "scale_quartile": int(bin_index), **block})
    if not rows:
        raise FileNotFoundError("No four-point metrics found; run --stages evaluate first")
    summary_dir = args.out_root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(summary_dir / "four_point_metrics.csv", index=False)
    pd.DataFrame(strata_rows).to_csv(summary_dir / "four_point_scale_strata.csv", index=False)
    write_json(
        summary_dir / "four_point_manifest.json",
        {
            "summarized_at_unix": time.time(),
            "representations": [row["representation"] for row in rows],
            "positive_control": ML_PATRISTIC_KEY,
            "lower_is_better_metrics": [
                "all_mean_delta_q",
                "all_mean_normalised_four_point_gap",
                "distinct_mean_delta_q",
            ],
            "higher_is_better_metrics": [
                "all_fraction_exactly_additive",
                "ml_tree_quartet_agreement",
            ],
            "interpretation": (
                "delta_q is 0 for an exact additive tree metric and 1 for a maximally "
                "non-additive quartet; the ML patristic row is the positive control"
            ),
        },
    )
    log(f"Wrote four-point metrics: {summary_dir / 'four_point_metrics.csv'}")
    columns = [
        "representation",
        "all_mean_delta_q",
        "all_fraction_exactly_additive",
        "distinct_mean_delta_q",
        "ml_tree_quartet_agreement",
    ]
    print(frame[[column for column in columns if column in frame.columns]].to_string(index=False))


def run_stages(args: argparse.Namespace) -> None:
    if args.representations.strip().lower() == "all":
        keys = list(FROZEN_2K_REPRESENTATIONS) + [ML_PATRISTIC_KEY]
    else:
        keys = [item.strip() for item in args.representations.split(",") if item.strip()]
        allowed = set(FROZEN_2K_REPRESENTATIONS) | {ML_PATRISTIC_KEY}
        unknown = sorted(set(keys).difference(allowed))
        if unknown:
            raise ValueError(f"Unknown representation(s) {unknown}; allowed={sorted(allowed)}")
    stages = {stage.strip() for stage in args.stages.split(",") if stage.strip()}
    if "all" in stages:
        stages = set(STAGES)
    unknown_stages = stages.difference(STAGES)
    if unknown_stages:
        raise ValueError(f"Unknown stage(s): {sorted(unknown_stages)}; allowed={STAGES}")
    args.out_root.mkdir(parents=True, exist_ok=True)
    if "prepare-quartets" in stages:
        stage_prepare_quartets(args)
    if "evaluate" in stages:
        stage_evaluate(args, keys)
    if "summarize" in stages:
        stage_summarize(args, keys)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frozen-2k-root", type=Path, default=DEFAULT_FROZEN_2K_ROOT)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--stages", default="all")
    ap.add_argument("--representations", default="all")
    ap.add_argument("--quartet-samples", type=int, default=2_000_000)
    ap.add_argument("--quartet-seed", type=int, default=42)
    ap.add_argument("--chunk-size", type=int, default=250_000)
    ap.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
        help="relative tolerance for calling a quartet exactly additive or unresolved",
    )
    ap.add_argument("--overwrite", action="store_true")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    run_stages(args)


if __name__ == "__main__":
    main()
