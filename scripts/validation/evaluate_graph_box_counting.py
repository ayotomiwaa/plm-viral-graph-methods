#!/usr/bin/env python3
"""Compare multiscale graph growth with randomized ball-burning covers."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse import csr_matrix, load_npz  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class GraphSpec:
    key: str
    label: str
    metric_family: str
    construction: str
    relative_dir: str


GRAPH_SPECS = (
    GraphSpec("hamming_knn_k05", "Hamming k-NN (k=5)", "Hamming", "k-NN (k=5)", "hamming/{sample}/hamming_knn_k05"),
    GraphSpec("hamming_knn_k50", "Hamming k-NN (k=50)", "Hamming", "k-NN (k=50)", "hamming/{sample}/hamming_knn_k50"),
    GraphSpec("hamming_rng_exact", "Hamming RNG", "Hamming", "RNG", "hamming/{sample}/hamming_rng_exact"),
    GraphSpec(
        "embedding_knn_k05",
        "Embedding k-NN (k=5)",
        "Embedding cityblock",
        "k-NN (k=5)",
        "esm2_650M/cityblock/{sample}/embedding_knn_k05",
    ),
    GraphSpec(
        "embedding_knn_k50",
        "Embedding k-NN (k=50)",
        "Embedding cityblock",
        "k-NN (k=50)",
        "esm2_650M/cityblock/{sample}/embedding_knn_k50",
    ),
    GraphSpec(
        "embedding_rng_exact",
        "Embedding RNG",
        "Embedding cityblock",
        "RNG",
        "esm2_650M/cityblock/{sample}/embedding_rng_exact",
    ),
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_box_sizes(value: str) -> list[int]:
    box_sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not box_sizes:
        raise ValueError("At least one box size is required")
    if len(set(box_sizes)) != len(box_sizes):
        raise ValueError("Box sizes must be unique")
    if any(size < 3 or size % 2 == 0 for size in box_sizes):
        raise ValueError("Box sizes must be odd integers greater than or equal to 3")
    return sorted(box_sizes)


def graph_paths(source_root: Path, panel: str, seed: int, sample_label: str) -> dict[str, Path]:
    graph_root = source_root / panel / f"seed_{seed}" / "graphs"
    paths = {
        spec.key: graph_root / spec.relative_dir.format(sample=sample_label) / "adj.npz"
        for spec in GRAPH_SPECS
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required graph adjacency files:\n" + "\n".join(missing))
    return paths


def load_unweighted_graph(path: Path) -> csr_matrix:
    graph = load_npz(path).tocsr()
    if graph.shape[0] != graph.shape[1]:
        raise ValueError(f"Adjacency matrix is not square: {path} has shape {graph.shape}")
    graph.sum_duplicates()
    # A stored zero is still a real edge when two sequence representations are identical.
    graph.data = np.ones(graph.nnz, dtype=np.uint8)
    graph.setdiag(0)
    graph.eliminate_zeros()
    graph = graph.astype(bool).maximum(graph.T.astype(bool)).astype(np.uint8).tocsr()
    graph.sum_duplicates()
    graph.sort_indices()
    graph.data[:] = 1
    return graph


def largest_component_mask(graph: csr_matrix) -> np.ndarray:
    n_components, labels = connected_components(graph, directed=False, return_labels=True)
    if n_components == 1:
        return np.ones(graph.shape[0], dtype=bool)
    counts = np.bincount(labels, minlength=n_components)
    largest_label = int(np.flatnonzero(counts == counts.max())[0])
    return labels == largest_label


def stable_common_lcc(
    graphs: dict[str, csr_matrix],
    max_iterations: int = 20,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    sizes = {graph.shape[0] for graph in graphs.values()}
    if len(sizes) != 1:
        raise ValueError(f"Graphs do not have identical node counts: {sorted(sizes)}")
    common_ids = np.arange(next(iter(sizes)), dtype=np.int64)
    audit: list[dict[str, Any]] = []

    for iteration in range(1, max_iterations + 1):
        prior_ids = common_ids
        keep = np.ones(prior_ids.size, dtype=bool)
        graph_lcc_sizes: dict[str, int] = {}
        for key, graph in graphs.items():
            induced = graph[prior_ids][:, prior_ids].tocsr()
            component_mask = largest_component_mask(induced)
            keep &= component_mask
            graph_lcc_sizes[key] = int(component_mask.sum())
        common_ids = prior_ids[keep]
        audit.append(
            {
                "iteration": iteration,
                "n_input_nodes": int(prior_ids.size),
                "n_common_nodes": int(common_ids.size),
                "graph_lcc_sizes": graph_lcc_sizes,
            }
        )
        if common_ids.size == 0:
            raise ValueError("The iterative common-LCC intersection became empty")
        if common_ids.size == prior_ids.size:
            return common_ids, audit
    raise RuntimeError(f"Common-LCC policy did not stabilize within {max_iterations} iterations")


def induce_graphs(graphs: dict[str, csr_matrix], common_ids: np.ndarray) -> dict[str, csr_matrix]:
    induced: dict[str, csr_matrix] = {}
    for key, graph in graphs.items():
        subgraph = graph[common_ids][:, common_ids].tocsr()
        n_components = connected_components(subgraph, directed=False, return_labels=False)
        if n_components != 1:
            raise RuntimeError(f"Induced graph {key} remains disconnected ({n_components} components)")
        induced[key] = subgraph
    return induced


def matched_priority(n_nodes: int, random_seed: int, box_size: int, trial: int) -> tuple[np.ndarray, int]:
    seed_sequence = np.random.SeedSequence([random_seed, box_size, trial])
    priority_seed = int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
    priority = np.random.default_rng(priority_seed).permutation(n_nodes).astype(np.int64, copy=False)
    return priority, priority_seed


def ball_burning_cover(
    graph: csr_matrix,
    radius: int,
    priority: np.ndarray,
    return_labels: bool = False,
) -> tuple[int, np.ndarray | None]:
    """Greedily cover a graph with radius-r balls centered on uncovered nodes."""
    n_nodes = graph.shape[0]
    if priority.shape != (n_nodes,) or priority.min(initial=0) < 0 or priority.max(initial=-1) >= n_nodes:
        raise ValueError("priority must be a permutation of all local node IDs")
    if radius < 0:
        raise ValueError("radius must be nonnegative")

    uncovered = np.ones(n_nodes, dtype=bool)
    n_uncovered = n_nodes
    labels = np.full(n_nodes, -1, dtype=np.int64) if return_labels else None
    seen = np.zeros(n_nodes, dtype=np.int64)
    queue = np.empty(n_nodes, dtype=np.int64)
    marker = 0
    n_boxes = 0
    indices = graph.indices
    indptr = graph.indptr

    for center_value in priority:
        center = int(center_value)
        if not uncovered[center]:
            continue
        marker += 1
        queue[0] = center
        seen[center] = marker
        head = 0
        tail = 1
        level_end = 1

        for _ in range(radius):
            while head < level_end:
                node = int(queue[head])
                head += 1
                neighbors = indices[indptr[node] : indptr[node + 1]]
                fresh = neighbors[seen[neighbors] != marker]
                if fresh.size:
                    seen[fresh] = marker
                    queue[tail : tail + fresh.size] = fresh
                    tail += int(fresh.size)
            if tail == level_end:
                break
            level_end = tail

        ball_nodes = queue[:tail]
        newly_covered = ball_nodes[uncovered[ball_nodes]]
        uncovered[newly_covered] = False
        n_uncovered -= int(newly_covered.size)
        if labels is not None:
            labels[newly_covered] = n_boxes
        n_boxes += 1
        if n_uncovered == 0:
            break

    if uncovered.any() or (labels is not None and np.any(labels < 0)):
        raise RuntimeError("Ball-burning cover failed to assign every node")
    return n_boxes, labels


def fit_box_dimension(box_sizes: np.ndarray, n_boxes: np.ndarray) -> dict[str, Any]:
    if box_sizes.size < 2:
        raise ValueError("At least two box sizes are required for the dimension fit")
    if np.any(box_sizes <= 0) or np.any(n_boxes <= 0):
        raise ValueError("Box sizes and box counts must be positive")
    x = np.log(box_sizes.astype(np.float64))
    y = np.log(n_boxes.astype(np.float64))
    slope, intercept = np.polyfit(x, y, deg=1)
    fitted = slope * x + intercept
    residual_ss = float(np.sum((y - fitted) ** 2))
    total_ss = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - residual_ss / total_ss if total_ss > 0 else math.nan
    unit_indices = np.flatnonzero(n_boxes == 1)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "d_B": float(-slope),
        "r_squared": r_squared,
        "n_fit_points": int(box_sizes.size),
        "n_nontrivial_scales": int(np.count_nonzero(n_boxes > 1)),
        "first_unit_box_size": int(box_sizes[unit_indices[0]]) if unit_indices.size else None,
        "unit_box_fraction": float(np.mean(n_boxes == 1)),
        "fit_warning": "fewer than 3 scales have N_B > 1" if np.count_nonzero(n_boxes > 1) < 3 else "",
    }


def summarize_trials(trials: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (key, box_size), group in trials.groupby(["graph_key", "box_size"], sort=False):
        values = group["n_boxes"].to_numpy(dtype=np.float64)
        first = group.iloc[0]
        rows.append(
            {
                "graph_key": key,
                "graph_label": first["graph_label"],
                "metric_family": first["metric_family"],
                "construction": first["construction"],
                "box_size": int(box_size),
                "ball_radius": int(first["ball_radius"]),
                "n_trials": int(values.size),
                "n_boxes_min": int(values.min()),
                "n_boxes_q1": float(np.quantile(values, 0.25)),
                "n_boxes_median": float(np.median(values)),
                "n_boxes_mean": float(values.mean()),
                "n_boxes_q3": float(np.quantile(values, 0.75)),
                "n_boxes_max": int(values.max()),
                "n_boxes_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def dimension_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in summary.groupby("graph_key", sort=False):
        group = group.sort_values("box_size")
        first = group.iloc[0]
        fit = fit_box_dimension(
            group["box_size"].to_numpy(dtype=np.float64),
            group["n_boxes_min"].to_numpy(dtype=np.float64),
        )
        rows.append(
            {
                "graph_key": key,
                "graph_label": first["graph_label"],
                "metric_family": first["metric_family"],
                "construction": first["construction"],
                "n_nodes": int(first["n_nodes"]),
                "n_edges": int(first["n_edges"]),
                **fit,
            }
        )
    return pd.DataFrame(rows)


def plot_curves(summary: pd.DataFrame, dimensions: pd.DataFrame, out_path: Path, loglog: bool) -> None:
    colors = {"Hamming": "#2166AC", "Embedding cityblock": "#D6604D"}
    styles = {
        "k-NN (k=5)": ("o", ":"),
        "k-NN (k=50)": ("s", "--"),
        "RNG": ("D", "-"),
    }
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    for spec in GRAPH_SPECS:
        group = summary[summary["graph_key"] == spec.key].sort_values("box_size")
        if group.empty:
            continue
        marker, linestyle = styles[spec.construction]
        ax.plot(
            group["box_size"],
            group["n_boxes_min"],
            color=colors[spec.metric_family],
            marker=marker,
            linestyle=linestyle,
            linewidth=2.1,
            markersize=6,
            label=spec.label,
        )
    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("Random ball-burning box counts and fitted graph dimensions")
    else:
        ax.set_title("Random ball-burning box counts")
    ax.set_xlabel(r"Box size $\ell_B$ (unweighted graph hops; strict diameter $<\ell_B$)")
    ax.set_ylabel(r"Estimated minimum boxes $N_B(\ell_B)$")
    ax.set_xticks(sorted(summary["box_size"].unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid(True, which="major", color="#D9D9D9", linewidth=0.8)
    ax.grid(True, which="minor", color="#EEEEEE", linewidth=0.5)
    ax.legend(frameon=False, ncol=2, loc="best")
    if loglog:
        readable = dimensions[["graph_label", "d_B", "r_squared"]].copy()
        lines = [
            f"{row.graph_label}: dB={row.d_B:.3f}, R2={row.r_squared:.3f}"
            for row in readable.itertuples(index=False)
        ]
        ax.text(
            0.99,
            0.02,
            "\n".join(lines),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9, "pad": 5},
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    box_sizes = parse_box_sizes(args.box_sizes)
    if args.trials < 1:
        raise ValueError("--trials must be at least 1")
    if len(box_sizes) < 2:
        raise ValueError("At least two box sizes are required by the fixed OLS fitting rule")

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    paths = graph_paths(Path(args.source_root), args.panel, args.seed, args.sample_label)

    graphs: dict[str, csr_matrix] = {}
    original_stats: dict[str, dict[str, int]] = {}
    for spec in GRAPH_SPECS:
        log(f"Loading {spec.label}: {paths[spec.key]}")
        graph = load_unweighted_graph(paths[spec.key])
        graphs[spec.key] = graph
        original_stats[spec.key] = {
            "n_nodes": int(graph.shape[0]),
            "n_edges": int(graph.nnz // 2),
        }

    log("Applying iterative common largest-connected-component policy")
    common_ids, lcc_audit = stable_common_lcc(graphs, max_iterations=args.max_lcc_iterations)
    graphs = induce_graphs(graphs, common_ids)
    pd.DataFrame({"node_id": common_ids}).to_csv(workspace / "common_node_ids.csv", index=False)
    log(f"Stable common connected node set: {common_ids.size:,} nodes")

    trial_rows: list[dict[str, Any]] = []
    for box_size in box_sizes:
        radius = (box_size - 1) // 2
        log(f"Box size {box_size} (ball radius {radius}): {args.trials} matched trials")
        priorities = [matched_priority(common_ids.size, args.random_seed, box_size, trial) for trial in range(args.trials)]
        for spec in GRAPH_SPECS:
            graph = graphs[spec.key]
            graph_start = time.perf_counter()
            for trial, (priority, priority_seed) in enumerate(priorities):
                start = time.perf_counter()
                n_boxes, _ = ball_burning_cover(graph, radius=radius, priority=priority)
                trial_rows.append(
                    {
                        "graph_key": spec.key,
                        "graph_label": spec.label,
                        "metric_family": spec.metric_family,
                        "construction": spec.construction,
                        "box_size": box_size,
                        "ball_radius": radius,
                        "trial": trial,
                        "priority_seed": priority_seed,
                        "n_boxes": n_boxes,
                        "elapsed_seconds": time.perf_counter() - start,
                    }
                )
            log(f"  {spec.label}: completed in {time.perf_counter() - graph_start:.1f}s")

    trials = pd.DataFrame(trial_rows)
    trials.to_csv(workspace / "graph_box_counting_trials.csv", index=False)
    summary = summarize_trials(trials)
    graph_size_rows = []
    for spec in GRAPH_SPECS:
        graph = graphs[spec.key]
        graph_size_rows.append({"graph_key": spec.key, "n_nodes": graph.shape[0], "n_edges": graph.nnz // 2})
    summary = summary.merge(pd.DataFrame(graph_size_rows), on="graph_key", validate="many_to_one")
    summary.to_csv(workspace / "graph_box_counting_summary.csv", index=False)
    dimensions = dimension_table(summary)
    dimensions.to_csv(workspace / "graph_box_dimension_summary.csv", index=False)

    plot_curves(summary, dimensions, workspace / "graph_box_counting_curves_linear.png", loglog=False)
    plot_curves(summary, dimensions, workspace / "graph_box_counting_curves_loglog.png", loglog=True)

    manifest = {
        "method": "randomized greedy radius-ball burning",
        "box_constraint": "each assigned box is contained in a radius-r ball, so pairwise distance <= 2r = lB-1 < lB",
        "distance": "unweighted shortest-path hops",
        "box_sizes": box_sizes,
        "ball_radii": [(size - 1) // 2 for size in box_sizes],
        "trials": args.trials,
        "random_seed": args.random_seed,
        "matched_trial_priorities_across_graphs": True,
        "cover_estimator": "minimum n_boxes across trials",
        "dimension_fit": "ordinary least squares of log(min n_boxes) on log(box_size), using every requested box size",
        "common_node_policy": "iterate: induce each graph on current IDs, take each LCC, intersect, repeat until stable",
        "n_common_nodes": int(common_ids.size),
        "common_lcc_audit": lcc_audit,
        "graphs": [
            {
                "key": spec.key,
                "label": spec.label,
                "metric_family": spec.metric_family,
                "construction": spec.construction,
                "adjacency_path": str(paths[spec.key]),
                "original": original_stats[spec.key],
                "induced": {"n_nodes": int(graphs[spec.key].shape[0]), "n_edges": int(graphs[spec.key].nnz // 2)},
            }
            for spec in GRAPH_SPECS
        ],
    }
    (workspace / "graph_box_counting_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"Wrote box-counting outputs to {workspace}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        default="analysis/cohort_validation/15_seed42_20k/graph_box_counting/hamming_embedding_knn05_knn50_rng",
    )
    parser.add_argument("--source-root", default="analysis/cohort_validation/07_sampling_design_20k")
    parser.add_argument("--panel", default="random_full_dataset_seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-label", default="pool_n20000")
    parser.add_argument("--box-sizes", default="3,5,7,9,11,13,15,17,19,21")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-lcc-iterations", type=int, default=20)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
