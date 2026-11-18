#!/usr/bin/env python3
"""Re-score purified RNG graphs on the paper's two headline stats.

For each (seed, metric) exact RNG graph, compute on BOTH the original and the
geodesic-convex-hull-purified subgraph:

* **Coleman-h assortativity** (cohort_id primary, cohort_name secondary) — via a
  verbatim copy of ``coleman_index`` from ``build_graph_family_coleman.py`` so the
  numbers are directly comparable to the stored per-seed summaries.
* **Tree geometry** — graph-geodesic distance matrix, own-NJ tree, then RSD and max
  tree distortion (all pairs), Gromov hyperbolicity delta (+ normalized delta), and
  Spearman rho between geodesic and patristic upper triangles.

Original-side tree geometry is read from the stored
``13_random_full_dataset_2k_nj_tree_validation/`` per-seed rows (built with the
skbio backend); purified-side is computed here with the pure-NumPy NJ fallback.
``--validate-tree-seeds`` recomputes original tree geometry with the numpy-NJ
backend on a few seeds to confirm the two backends agree within tolerance.

Self-contained on purpose: the paper's validation modules import Biopython/numba,
which are absent from the analysis env, so the metric definitions are inlined here
verbatim (see comments).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_env, "1")

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.stats import spearmanr

import sys
from pathlib import Path as _Path

_SRC_ROOT = _Path(__file__).resolve().parents[2] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from protein_embeddings.methods.cc_geodesic import build_nj_tree, patristic_distance_matrix

MISSING_LABELS = {"", "nan", "none", "null", "nat"}
TREE_ROOT = Path("analysis/cohort_validation/13_random_full_dataset_2k_nj_tree_validation")


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_seeds(value: str) -> list[int]:
    seeds: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


# ---------------------------------------------------------------- graph paths


def rng_graph_dir(panel_root: Path, seed: int, metric: str, sample_label: str) -> Path:
    if metric == "hamming":
        return panel_root / f"seed_{seed}" / "graphs" / "hamming" / sample_label / "hamming_rng_exact"
    return panel_root / f"seed_{seed}" / "graphs" / "esm2_650M" / metric / sample_label / "embedding_rng_exact"


def purified_graph_dir(out_root: Path, seed: int, metric: str) -> Path:
    return out_root / "purified_graphs" / metric / f"seed_{seed}" / "embedding_rng_exact_purified"


def stored_graph_name(metric: str) -> str:
    if metric == "hamming":
        return "hamming_rng_exact"
    return f"embedding_{metric}_rng_exact"


def load_graph_edges_nodes(graph_dir: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    nodes = pd.read_csv(graph_dir / "nodes.csv", low_memory=False)
    adj = load_npz(graph_dir / "adj.npz").tocsr()
    coo = adj.tocoo()
    src = np.asarray(coo.row, dtype=np.int64)
    tgt = np.asarray(coo.col, dtype=np.int64)
    # dedupe to one undirected copy (adj is symmetric)
    src, tgt = np.minimum(src, tgt), np.maximum(src, tgt)
    keep = src != tgt
    pairs = np.stack([src[keep], tgt[keep]], axis=1)
    pairs = np.unique(pairs, axis=0)
    return nodes, pairs[:, 0].astype(np.int32), pairs[:, 1].astype(np.int32)


# ---------------------------------------------------------------- coleman h
# Verbatim from scripts/graph_construction/build_graph_family_coleman.py::coleman_index.


def apply_label_group(value, groups):
    label = str(value).strip()
    for new_label, members in groups.items():
        if label in members:
            return new_label
    return label


def coleman_index(nodes, sources, targets, label_col, label_groups):
    labels = [apply_label_group(value, label_groups) for value in nodes[label_col].astype(str)]
    valid = np.array([label.casefold() not in MISSING_LABELS for label in labels], dtype=bool)
    edge_mask = valid[sources] & valid[targets]
    sources = sources[edge_mask]
    targets = targets[edge_mask]
    m = int(len(sources))
    if m == 0:
        return {"coleman_h": np.nan, "observed_same_fraction": np.nan, "expected_same_fraction": np.nan, "n_edges_used": 0}
    same = sum(labels[int(i)] == labels[int(j)] for i, j in zip(sources, targets))
    observed = same / m
    stubs: Counter[str] = Counter()
    for i, j in zip(sources, targets):
        stubs[labels[int(i)]] += 1
        stubs[labels[int(j)]] += 1
    stub_fractions = {label: count / (2 * m) for label, count in stubs.items()}
    expected = sum(v * v for v in stub_fractions.values())
    denom = 1.0 - expected
    h = (observed - expected) / denom if denom != 0 else np.nan
    return {"coleman_h": h, "observed_same_fraction": observed, "expected_same_fraction": expected, "n_edges_used": m}


# ---------------------------------------------------------------- tree geometry
# Verbatim metric definitions from the paper's validation modules (inlined because
# those modules import Biopython/numba, which are absent in this environment).


def robust_iqr(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    q25, q75 = np.percentile(finite, [25, 75])
    iqr = float(q75 - q25)
    if iqr == 0.0:
        sd = float(np.std(finite))
        return sd if sd > 0.0 else 1.0
    return iqr


def gromov_delta_hyperbolicity(D: np.ndarray, n_samples: int, seed: int, max_attempt_multiplier: int = 20) -> dict:
    n = int(D.shape[0])
    if n < 4:
        return {"gromov_delta": math.nan, "gromov_delta_norm_iqr": math.nan}
    iu = np.triu_indices(n, k=1)
    iqr = robust_iqr(np.asarray(D[iu], dtype=np.float64))
    rng = np.random.default_rng(seed)
    delta_max = 0.0
    used = 0
    attempts = 0
    max_attempts = max(int(n_samples) * max_attempt_multiplier, int(n_samples))
    while used < n_samples and attempts < max_attempts:
        attempts += 1
        a, b, c, d = rng.choice(n, size=4, replace=False)
        vals = np.array([D[a, b] + D[c, d], D[a, c] + D[b, d], D[a, d] + D[b, c]], dtype=np.float64)
        if not np.isfinite(vals).all():
            continue
        vals.sort()
        delta = float(0.5 * (vals[2] - vals[1]))
        if delta > delta_max:
            delta_max = delta
        used += 1
    return {
        "gromov_delta": float(delta_max) if used else math.nan,
        "gromov_delta_norm_iqr": float(delta_max / iqr) if used and np.isfinite(iqr) and iqr != 0.0 else math.nan,
    }


def distance_tree_metrics_all(D: np.ndarray, T: np.ndarray, eps: float = 1e-12) -> dict:
    """RSD + max tree distortion over all upper-triangle pairs (pair_mode='all')."""
    iu = np.triu_indices_from(D, k=1)
    d = np.asarray(D[iu], dtype=np.float64)
    t = np.asarray(T[iu], dtype=np.float64)
    finite = np.isfinite(d) & np.isfinite(t)
    d_f = d[finite]
    t_f = t[finite]
    denom = float(np.sum(d_f * d_f))
    rsd = float(np.sqrt(np.sum((d_f - t_f) ** 2) / denom)) if denom > eps else math.nan
    denom_s = float(np.dot(t_f, t_f))
    s_star = float(np.dot(d_f, t_f) / denom_s) if denom_s > eps else math.nan
    positive = finite & (d > eps) & (t > eps) & np.isfinite(s_star) & (s_star > eps)
    if positive.any():
        ratio = d[positive] / (s_star * t[positive] + eps)
        distortion = float(np.maximum(ratio, 1.0 / (ratio + eps)).max())
        log_distortion = float(np.abs(np.log(ratio + eps)).max())
    else:
        distortion = math.nan
        log_distortion = math.nan
    return {
        "relative_square_deviation": rsd,
        "tree_dist_scale_s_star": s_star,
        "max_tree_distortion": distortion,
        "tree_distortion_log_delta": log_distortion,
    }


def geodesic_matrix(adj) -> np.ndarray:
    return np.asarray(shortest_path(adj, directed=False, unweighted=False), dtype=np.float64)


def largest_component_subset(D: np.ndarray, accessions: list[str]) -> tuple[np.ndarray, list[str], int]:
    n = D.shape[0]
    finite = np.isfinite(D)
    np.fill_diagonal(finite, False)
    from scipy.sparse import csr_matrix

    n_comp, labels = connected_components(csr_matrix(finite), directed=False, return_labels=True)
    sizes = np.bincount(labels, minlength=n_comp) if n else np.array([], dtype=int)
    if n_comp <= 1:
        return D, accessions, 0
    keep_label = int(np.argmax(sizes))
    idx = np.flatnonzero(labels == keep_label)
    return np.asarray(D[np.ix_(idx, idx)]), [accessions[i] for i in idx], int(n - len(idx))


def tree_geometry(graph_dir: Path, hyper_samples: int, hyper_seed: int) -> dict:
    nodes = pd.read_csv(graph_dir / "nodes.csv", low_memory=False)
    adj = load_npz(graph_dir / "adj.npz").tocsr()
    accessions = nodes["accession"].astype(str).str.strip().tolist()
    D = geodesic_matrix(adj)
    D, acc, dropped = largest_component_subset(D, accessions)
    if len(acc) < 3:
        return {"n_accessions": 0, "n_nodes_dropped_for_nj": dropped}
    kind, tree = build_nj_tree(D, acc, prefer="numpy")
    T = patristic_distance_matrix(kind, tree, acc)
    score = distance_tree_metrics_all(D, T)
    gro = gromov_delta_hyperbolicity(D, hyper_samples, hyper_seed)
    iu = np.triu_indices_from(D, k=1)
    rho = float(spearmanr(np.asarray(D[iu]), np.asarray(T[iu])).statistic)
    return {
        "n_accessions": int(len(acc)),
        "n_nodes_dropped_for_nj": dropped,
        "tree_builder_backend": kind,
        **score,
        **gro,
        "spearman_rho_own_nj": rho,
    }


# ---------------------------------------------------------------- stored baselines


def load_stored_tree_geometry() -> tuple[pd.DataFrame, pd.DataFrame]:
    dist = pd.read_csv(TREE_ROOT / "all_graph_tree_distortion_metrics.csv", low_memory=False)
    gro = pd.read_csv(TREE_ROOT / "all_graph_gromov_hyperbolicity_metrics.csv", low_memory=False)
    return dist, gro


def load_stored_assortativity(panel_root: Path) -> pd.DataFrame:
    frames = []
    for seed_dir in sorted(panel_root.glob("seed_*")):
        p = seed_dir / "summaries" / "seed_cohort_id_assortativity_summary.csv"
        if p.exists():
            frames.append(pd.read_csv(p, low_memory=False))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def graph_state_cols(metric: str, seed: int, state: str) -> dict:
    return {"metric": metric, "seed": int(seed), "state": state}


def score_one(panel_root: Path, out_root: Path, seed: int, metric: str, hyper_samples: int, hyper_seed: int) -> list[dict]:
    graph_dir = rng_graph_dir(panel_root, seed, metric, "pool_n2000")
    pdir = purified_graph_dir(out_root, seed, metric)
    rows = []
    for state, gdir in [("original", graph_dir), ("purified", pdir)]:
        nodes, src, tgt = load_graph_edges_nodes(gdir)
        h_id = coleman_index(nodes, src, tgt, "cohort_id", {})
        h_name = coleman_index(nodes, src, tgt, "cohort_name", {}) if "cohort_name" in nodes.columns else {"coleman_h": np.nan}
        row = {
            **graph_state_cols(metric, seed, state),
            "n_nodes": int(len(nodes)),
            "n_edges": int(len(src)),
            "coleman_h_cohort_id": h_id["coleman_h"],
            "coleman_h_cohort_name": h_name["coleman_h"],
            "observed_same_fraction_cohort_id": h_id["observed_same_fraction"],
            "expected_same_fraction_cohort_id": h_id["expected_same_fraction"],
        }
        rows.append(row)
    # tree geometry for purified only (original read from stored)
    tg = tree_geometry(pdir, hyper_samples, hyper_seed + seed)
    rows[1].update({f"tree_{k}": v for k, v in tg.items()})
    return rows


def _worker(payload: dict) -> list[dict]:
    return score_one(
        Path(payload["panel_root"]),
        Path(payload["out_root"]),
        payload["seed"],
        payload["metric"],
        payload["hyper_samples"],
        payload["hyper_seed"],
    )


def summarize(values: np.ndarray) -> dict:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return {"mean": np.nan, "sd": np.nan, "se": np.nan, "ci95_halfwidth": np.nan, "min": np.nan, "max": np.nan, "n": 0}
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if v.size > 1 else 0.0
    se = sd / math.sqrt(v.size)
    return {
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci95_halfwidth": 1.96 * se,
        "min": float(v.min()),
        "max": float(v.max()),
        "n": int(v.size),
    }


def attach_stored_tree_geometry(frame: pd.DataFrame, dist: pd.DataFrame, gro: pd.DataFrame) -> pd.DataFrame:
    """Attach original-side tree geometry from the stored per-seed rows."""
    dist_map = {}
    for _, r in dist.iterrows():
        key = (str(r["graph_name"]), int(r["seed"]))
        dist_map[key] = {
            "tree_relative_square_deviation": float(r["relative_square_deviation"]),
            "tree_max_tree_distortion": float(r["max_tree_distortion"]),
            "tree_tree_distortion_log_delta": float(r.get("tree_distortion_log_delta", np.nan)),
        }
    gro_map = {}
    for _, r in gro.iterrows():
        key = (str(r["graph_name"]), int(r["seed"]))
        gro_map[key] = {
            "tree_gromov_delta": float(r["gromov_delta"]),
            "tree_gromov_delta_norm_iqr": float(r["gromov_delta_norm_iqr"]),
        }
    for idx, row in frame.iterrows():
        if row["state"] != "original":
            continue
        gname = stored_graph_name(str(row["metric"]))
        key = (gname, int(row["seed"]))
        if key in dist_map:
            for k, v in dist_map[key].items():
                frame.at[idx, k] = v
        if key in gro_map:
            for k, v in gro_map[key].items():
                frame.at[idx, k] = v
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel-root", type=Path, default=Path("analysis/cohort_validation/08_sampling_design_2k/random_full_dataset"))
    ap.add_argument("--out-root", type=Path, default=Path("analysis/cohort_validation/35_2k_rng_geodesic_purification"))
    ap.add_argument("--seeds", default="0-199")
    ap.add_argument("--metrics", default="cityblock,euclidean,hamming")
    ap.add_argument("--hyper-samples", type=int, default=50_000)
    ap.add_argument("--hyper-seed", type=int, default=12345)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--validate-tree-seeds", default=None, help="Comma seeds whose ORIGINAL tree geometry is recomputed with numpy-NJ")
    ap.add_argument("--skip-stored-tree", action="store_true")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    jobs = [{"panel_root": str(args.panel_root), "out_root": str(args.out_root), "seed": s, "metric": m, "hyper_samples": args.hyper_samples, "hyper_seed": args.hyper_seed} for s in seeds for m in metrics]
    rows: list[dict] = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for rlist in ex.map(_worker, jobs, chunksize=1):
                rows.extend(rlist)
    else:
        for job in jobs:
            rows.extend(_worker(job))

    frame = pd.DataFrame(rows)
    args.out_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_root / "rescore_per_seed.csv", index=False)

    # attach stored original-side tree geometry (skbio-NJ backend, full 2000 nodes)
    if not args.skip_stored_tree:
        dist, gro = load_stored_tree_geometry()
        frame = attach_stored_tree_geometry(frame, dist, gro)
        frame.to_csv(args.out_root / "rescore_per_seed.csv", index=False)

    # ---- original assortativity validation against stored per-seed summary ----
    stored = load_stored_assortativity(args.panel_root)
    validation_rows = []
    if not stored.empty:
        for _, r in frame[frame.state == "original"].iterrows():
            metric = r["metric"]
            seed = int(r["seed"])
            if metric == "hamming":
                sel = stored[(stored["seed"] == seed) & (stored["metric_space"].astype(str) == "Hamming") & (stored["Graph"].astype(str) == "RNG")]
            else:
                sel = stored[(stored["seed"] == seed) & (stored["embedding_metric"].astype(str) == metric) & (stored["Graph"].astype(str) == "RNG")]
            if len(sel):
                stored_h = float(sel.iloc[0]["Coleman h Index"])
                validation_rows.append({"seed": seed, "metric": metric, "recomputed_h": r["coleman_h_cohort_id"], "stored_h": stored_h, "delta": float(r["coleman_h_cohort_id"]) - stored_h})
    if validation_rows:
        vdf = pd.DataFrame(validation_rows)
        vdf.to_csv(args.out_root / "validation_coleman_h.csv", index=False)
        max_abs = vdf["delta"].abs().max()
        log(f"Coleman-h validation: n={len(vdf)} max|recomputed - stored| = {max_abs:.3e}")
        if max_abs > 1e-6:
            log("WARNING: recomputed Coleman h does not reproduce stored values!")

    # ---- tree geometry validation (numpy-NJ vs stored skbio-NJ on a few seeds) ----
    if args.validate_tree_seeds:
        dist, gro = load_stored_tree_geometry()
        tv = []
        for metric in metrics:
            gname = stored_graph_name(metric)
            for seed in parse_seeds(args.validate_tree_seeds):
                gdir = rng_graph_dir(args.panel_root, seed, metric, "pool_n2000")
                tg = tree_geometry(gdir, args.hyper_samples, args.hyper_seed + seed)
                drow = dist[(dist["graph_name"] == gname) & (dist["seed"] == seed)]
                grow = gro[(gro["graph_name"] == gname) & (gro["seed"] == seed)]
                rec = {"seed": seed, "metric": metric}
                if len(drow):
                    rec["rsd_recomputed"] = tg.get("relative_square_deviation")
                    rec["rsd_stored"] = float(drow.iloc[0]["relative_square_deviation"])
                    rec["distortion_recomputed"] = tg.get("max_tree_distortion")
                    rec["distortion_stored"] = float(drow.iloc[0]["max_tree_distortion"])
                if len(grow):
                    rec["gromov_recomputed"] = tg.get("gromov_delta")
                    rec["gromov_stored"] = float(grow.iloc[0]["gromov_delta"])
                tv.append(rec)
        tdf = pd.DataFrame(tv)
        tdf.to_csv(args.out_root / "validation_tree_geometry.csv", index=False)
        log(f"Tree-geometry validation rows written (see {args.out_root / 'validation_tree_geometry.csv'})")
        for _, r in tdf.iterrows():
            rsd = r.get("rsd_recomputed"); rsd_s = r.get("rsd_stored")
            dist_r = r.get("distortion_recomputed"); dist_s = r.get("distortion_stored")
            log(f"  seed {int(r['seed'])} {r['metric']}: RSD {rsd:.4f} vs {rsd_s:.4f} | distortion {dist_r:.2f} vs {dist_s:.2f}")

    # ---- aggregate CI summary: purified vs original ----
    agg_rows = []
    for metric in metrics:
        sub = frame[frame.metric == metric]
        for state in ["original", "purified"]:
            s = sub[sub.state == state]
            agg_rows.append(
                {
                    "metric": metric,
                    "state": state,
                    "coleman_h_cohort_id": summarize(s["coleman_h_cohort_id"].to_numpy()),
                    "coleman_h_cohort_name": summarize(s["coleman_h_cohort_name"].to_numpy()),
                    "tree_relative_square_deviation": summarize(s["tree_relative_square_deviation"].to_numpy()),
                    "tree_max_tree_distortion": summarize(s["tree_max_tree_distortion"].to_numpy()),
                    "tree_gromov_delta": summarize(s["tree_gromov_delta"].to_numpy()),
                    "tree_gromov_delta_norm_iqr": summarize(s["tree_gromov_delta_norm_iqr"].to_numpy()),
                    "tree_spearman_rho_own_nj": summarize(s["tree_spearman_rho_own_nj"].to_numpy()),
                }
            )

    flat = []
    for r in agg_rows:
        row = {"metric": r["metric"], "state": r["state"]}
        for key, val in r.items():
            if key in {"metric", "state"}:
                continue
            if isinstance(val, dict):
                for k2, v2 in val.items():
                    row[f"{key}_{k2}"] = v2
            else:
                row[key] = val
        flat.append(row)
    agg = pd.DataFrame(flat)
    agg.to_csv(args.out_root / "rescore_ci_summary.csv", index=False)
    log(f"Saved {args.out_root / 'rescore_per_seed.csv'} and {args.out_root / 'rescore_ci_summary.csv'}")

    # pretty print
    cols = ["metric", "state", "coleman_h_cohort_id_mean", "tree_relative_square_deviation_mean", "tree_max_tree_distortion_mean", "tree_gromov_delta_norm_iqr_mean"]
    show = agg[[c for c in cols if c in agg.columns]]
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
