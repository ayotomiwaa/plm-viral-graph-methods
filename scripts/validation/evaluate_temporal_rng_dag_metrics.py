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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.graph_construction.build_panel_nj_distance_reference_trees import (  # noqa: E402
    parse_seed_list,
)
from scripts.validation.nextstrain_spike_tree_validation import (  # noqa: E402
    graph_paths,
)


DEFAULT_SUMMARY_METRICS = [
    "directed_edge_fraction",
    "source_fraction_dated",
    "sink_fraction_dated",
    "temporal_bridge_fraction_dated",
    "mean_temporal_in_degree",
    "mean_temporal_out_degree",
    "max_temporal_in_degree",
    "max_temporal_out_degree",
    "mean_temporal_lag_days",
    "median_temporal_lag_days",
]


NULL_SUMMARY_METRICS = [
    "observed_source_count",
    "shuffle_source_count_mean",
    "source_count_delta",
    "z_source",
    "p_source_empirical_two_sided",
    "observed_sink_count",
    "shuffle_sink_count_mean",
    "sink_count_delta",
    "z_sink",
    "p_sink_empirical_two_sided",
    "observed_bridge_count",
    "shuffle_bridge_count_mean",
    "bridge_count_delta",
    "z_bridge",
    "p_bridge_empirical_two_sided",
]


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def wanted_rng_graphs(panel_root: Path, sample_label: str, graph_names: str) -> list[dict[str, Any]]:
    specs = [spec for spec in graph_paths(panel_root, sample_label) if spec["graph_family"] == "rng_exact"]
    wanted = {item.strip() for item in graph_names.split(",") if item.strip()}
    if wanted:
        specs = [spec for spec in specs if str(spec["graph_name"]) in wanted]
    return specs


def read_graph_tables(graph_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    nodes_path = graph_dir / "nodes.csv"
    edges_path = graph_dir / "edges.csv"
    stats_path = graph_dir / "stats.json"
    if not nodes_path.exists() or not edges_path.exists():
        raise FileNotFoundError(f"Missing RNG graph nodes/edges in {graph_dir}")

    nodes = pd.read_csv(nodes_path, low_memory=False)
    edges = pd.read_csv(edges_path, low_memory=False)
    if "status" in edges.columns:
        edges = edges[edges["status"].astype(str) == "kept"].copy()
    nodes["node_id"] = pd.to_numeric(nodes["node_id"], errors="raise").astype(int)
    edges["source"] = pd.to_numeric(edges["source"], errors="raise").astype(int)
    edges["target"] = pd.to_numeric(edges["target"], errors="raise").astype(int)
    return nodes, edges, load_json(stats_path)


def date_arrays(nodes: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parsed = pd.to_datetime(nodes["collection_date"], errors="coerce")
    node_ids = nodes["node_id"].astype(int).to_numpy()
    max_node_id = int(node_ids.max()) if len(node_ids) else -1
    date_days = np.zeros(max_node_id + 1, dtype=np.int64)
    valid_dates = np.zeros(max_node_id + 1, dtype=bool)
    valid_node_mask = parsed.notna().to_numpy()
    if valid_node_mask.any():
        valid_nodes = node_ids[valid_node_mask]
        valid_values = parsed[valid_node_mask].to_numpy(dtype="datetime64[D]").astype(np.int64)
        date_days[valid_nodes] = valid_values
        valid_dates[valid_nodes] = True
    return node_ids, date_days, valid_dates


def temporal_counts_from_date_arrays(
    sources: np.ndarray,
    targets: np.ndarray,
    node_ids: np.ndarray,
    date_days: np.ndarray,
    valid_dates: np.ndarray,
) -> dict[str, Any]:
    source_date = date_days[sources]
    target_date = date_days[targets]
    valid_edge = valid_dates[sources] & valid_dates[targets]
    source_older = valid_edge & (source_date < target_date)
    target_older = valid_edge & (target_date < source_date)
    directed_mask = source_older | target_older
    equal_date = valid_edge & (source_date == target_date)
    missing_date = ~valid_edge

    n = len(date_days)
    in_degree = np.zeros(n, dtype=np.int64)
    out_degree = np.zeros(n, dtype=np.int64)
    if directed_mask.any():
        older = np.where(source_older[directed_mask], sources[directed_mask], targets[directed_mask])
        newer = np.where(source_older[directed_mask], targets[directed_mask], sources[directed_mask])
        in_degree += np.bincount(newer, minlength=n)
        out_degree += np.bincount(older, minlength=n)

    dated_node_ids = node_ids[valid_dates[node_ids]]
    dated_in = in_degree[dated_node_ids]
    dated_out = out_degree[dated_node_ids]
    return {
        "n_directed_edges": int(directed_mask.sum()),
        "n_equal_date_edges": int(equal_date.sum()),
        "n_missing_date_edges": int(missing_date.sum()),
        "n_source_indegree0_dated": int((dated_in == 0).sum()),
        "n_sink_outdegree0_dated": int((dated_out == 0).sum()),
        "n_temporal_bridge_nodes": int(((dated_in > 0) & (dated_out > 0)).sum()),
    }


def orient_edges_by_collection_date(nodes: pd.DataFrame, edges: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    date_table = nodes[["node_id", "collection_date"]].copy()
    date_table["collection_date_parsed"] = pd.to_datetime(date_table["collection_date"], errors="coerce")

    merged = edges.merge(
        date_table.rename(
            columns={
                "node_id": "source",
                "collection_date": "source_collection_date",
                "collection_date_parsed": "source_collection_date_parsed",
            }
        ),
        on="source",
        how="left",
    )
    merged = merged.merge(
        date_table.rename(
            columns={
                "node_id": "target",
                "collection_date": "target_collection_date",
                "collection_date_parsed": "target_collection_date_parsed",
            }
        ),
        on="target",
        how="left",
    )

    source_date = merged["source_collection_date_parsed"]
    target_date = merged["target_collection_date_parsed"]
    valid = source_date.notna() & target_date.notna()
    source_older = valid & (source_date < target_date)
    target_older = valid & (target_date < source_date)
    equal_date = valid & (source_date == target_date)
    missing_date = ~valid

    older_node = np.where(source_older, merged["source"], merged["target"])
    newer_node = np.where(source_older, merged["target"], merged["source"])
    older_date = np.where(source_older, source_date, target_date)
    newer_date = np.where(source_older, target_date, source_date)
    directed_mask = source_older | target_older

    directed = pd.DataFrame(
        {
            "undirected_source": merged.loc[directed_mask, "source"].astype(int).to_numpy(),
            "undirected_target": merged.loc[directed_mask, "target"].astype(int).to_numpy(),
            "older_node_id": older_node[directed_mask],
            "newer_node_id": newer_node[directed_mask],
            "older_collection_date": pd.to_datetime(older_date[directed_mask]).date.astype(str),
            "newer_collection_date": pd.to_datetime(newer_date[directed_mask]).date.astype(str),
        }
    )
    if "weight" in merged.columns:
        directed["weight"] = merged.loc[directed_mask, "weight"].to_numpy()
    directed["temporal_lag_days"] = (
        pd.to_datetime(directed["newer_collection_date"]) - pd.to_datetime(directed["older_collection_date"])
    ).dt.days.astype(int)

    qc = {
        "n_undirected_edges": int(len(edges)),
        "n_directed_edges": int(directed.shape[0]),
        "n_equal_date_edges": int(equal_date.sum()),
        "n_missing_date_edges": int(missing_date.sum()),
        "orientation_rule": "strictly older collection_date -> newer collection_date; equal-date and missing-date edges are not directed",
        "is_dag_by_construction": True,
    }
    return directed, qc


def z_score(observed: float, null_values: np.ndarray) -> float:
    if null_values.size < 2:
        return math.nan
    sd = float(np.std(null_values, ddof=1))
    if sd == 0.0:
        return math.nan
    return float((observed - float(np.mean(null_values))) / sd)


def empirical_p_values(observed: int, null_values: np.ndarray) -> dict[str, float]:
    n = int(null_values.size)
    if n == 0:
        return {
            "empirical_p_ge": math.nan,
            "empirical_p_le": math.nan,
            "empirical_p_two_sided": math.nan,
        }
    mean = float(np.mean(null_values))
    return {
        "empirical_p_ge": float((1 + np.sum(null_values >= observed)) / (n + 1)),
        "empirical_p_le": float((1 + np.sum(null_values <= observed)) / (n + 1)),
        "empirical_p_two_sided": float((1 + np.sum(np.abs(null_values - mean) >= abs(observed - mean))) / (n + 1)),
    }


def shuffle_null_metrics(
    panel: str,
    seed: int,
    spec: dict[str, Any],
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    observed_row: dict[str, Any],
    graph_index: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    node_ids, date_days, valid_dates = date_arrays(nodes)
    valid_node_ids = node_ids[valid_dates[node_ids]]
    valid_node_dates = date_days[valid_node_ids].copy()
    sources = edges["source"].astype(int).to_numpy()
    targets = edges["target"].astype(int).to_numpy()

    n_shuffles = int(args.shuffle_permutations)
    source_counts = np.zeros(n_shuffles, dtype=np.int64)
    sink_counts = np.zeros(n_shuffles, dtype=np.int64)
    bridge_counts = np.zeros(n_shuffles, dtype=np.int64)
    directed_counts = np.zeros(n_shuffles, dtype=np.int64)
    equal_date_counts = np.zeros(n_shuffles, dtype=np.int64)

    rng = np.random.default_rng(int(args.shuffle_seed) + int(seed) * 1009 + int(graph_index) * 1_000_003)
    shuffled_dates = date_days.copy()
    for i in range(n_shuffles):
        shuffled_dates[valid_node_ids] = rng.permutation(valid_node_dates)
        counts = temporal_counts_from_date_arrays(sources, targets, node_ids, shuffled_dates, valid_dates)
        source_counts[i] = counts["n_source_indegree0_dated"]
        sink_counts[i] = counts["n_sink_outdegree0_dated"]
        bridge_counts[i] = counts["n_temporal_bridge_nodes"]
        directed_counts[i] = counts["n_directed_edges"]
        equal_date_counts[i] = counts["n_equal_date_edges"]

    observed_sources = int(observed_row["n_source_indegree0_dated"])
    observed_sinks = int(observed_row["n_sink_outdegree0_dated"])
    observed_bridges = int(observed_row["n_temporal_bridge_nodes"])
    p_source = empirical_p_values(observed_sources, source_counts)
    p_sink = empirical_p_values(observed_sinks, sink_counts)
    p_bridge = empirical_p_values(observed_bridges, bridge_counts)

    source_mean = float(np.mean(source_counts)) if n_shuffles else math.nan
    sink_mean = float(np.mean(sink_counts)) if n_shuffles else math.nan
    bridge_mean = float(np.mean(bridge_counts)) if n_shuffles else math.nan
    return {
        "panel": panel,
        "seed": int(seed),
        "graph_name": spec["graph_name"],
        "metric_family": spec["metric_family"],
        "embedding_metric": spec["embedding_metric"],
        "graph_family": spec["graph_family"],
        "null_model": "fixed_rng_topology_permute_valid_collection_dates_across_nodes",
        "shuffle_permutations": n_shuffles,
        "shuffle_seed": int(args.shuffle_seed),
        "observed_source_count": observed_sources,
        "shuffle_source_count_mean": source_mean,
        "shuffle_source_count_sd": float(np.std(source_counts, ddof=1)) if n_shuffles > 1 else 0.0 if n_shuffles == 1 else math.nan,
        "source_count_delta": float(observed_sources - source_mean) if np.isfinite(source_mean) else math.nan,
        "z_source": z_score(observed_sources, source_counts),
        "p_source_empirical_ge": p_source["empirical_p_ge"],
        "p_source_empirical_le": p_source["empirical_p_le"],
        "p_source_empirical_two_sided": p_source["empirical_p_two_sided"],
        "observed_sink_count": observed_sinks,
        "shuffle_sink_count_mean": sink_mean,
        "shuffle_sink_count_sd": float(np.std(sink_counts, ddof=1)) if n_shuffles > 1 else 0.0 if n_shuffles == 1 else math.nan,
        "sink_count_delta": float(observed_sinks - sink_mean) if np.isfinite(sink_mean) else math.nan,
        "z_sink": z_score(observed_sinks, sink_counts),
        "p_sink_empirical_ge": p_sink["empirical_p_ge"],
        "p_sink_empirical_le": p_sink["empirical_p_le"],
        "p_sink_empirical_two_sided": p_sink["empirical_p_two_sided"],
        "observed_bridge_count": observed_bridges,
        "shuffle_bridge_count_mean": bridge_mean,
        "shuffle_bridge_count_sd": float(np.std(bridge_counts, ddof=1)) if n_shuffles > 1 else 0.0 if n_shuffles == 1 else math.nan,
        "bridge_count_delta": float(observed_bridges - bridge_mean) if np.isfinite(bridge_mean) else math.nan,
        "z_bridge": z_score(observed_bridges, bridge_counts),
        "p_bridge_empirical_ge": p_bridge["empirical_p_ge"],
        "p_bridge_empirical_le": p_bridge["empirical_p_le"],
        "p_bridge_empirical_two_sided": p_bridge["empirical_p_two_sided"],
        "observed_directed_edge_count": int(observed_row["n_directed_edges"]),
        "shuffle_directed_edge_count_mean": float(np.mean(directed_counts)) if n_shuffles else math.nan,
        "shuffle_directed_edge_count_sd": float(np.std(directed_counts, ddof=1)) if n_shuffles > 1 else 0.0 if n_shuffles == 1 else math.nan,
        "observed_equal_date_edge_count": int(observed_row["n_equal_date_edges"]),
        "shuffle_equal_date_edge_count_mean": float(np.mean(equal_date_counts)) if n_shuffles else math.nan,
        "shuffle_equal_date_edge_count_sd": float(np.std(equal_date_counts, ddof=1)) if n_shuffles > 1 else 0.0 if n_shuffles == 1 else math.nan,
    }


def add_temporal_node_metrics(nodes: pd.DataFrame, directed_edges: pd.DataFrame) -> pd.DataFrame:
    out = nodes.copy()
    out["collection_date_parsed"] = pd.to_datetime(out["collection_date"], errors="coerce")
    out["has_valid_collection_date"] = out["collection_date_parsed"].notna()
    max_node_id = int(out["node_id"].max()) if not out.empty else -1
    n = max_node_id + 1

    in_degree = np.zeros(n, dtype=np.int64)
    out_degree = np.zeros(n, dtype=np.int64)
    if not directed_edges.empty:
        in_degree += np.bincount(directed_edges["newer_node_id"].astype(int), minlength=n)
        out_degree += np.bincount(directed_edges["older_node_id"].astype(int), minlength=n)

    node_ids = out["node_id"].astype(int).to_numpy()
    out["temporal_in_degree"] = in_degree[node_ids]
    out["temporal_out_degree"] = out_degree[node_ids]
    out["temporal_total_directed_degree"] = out["temporal_in_degree"] + out["temporal_out_degree"]
    out["is_temporal_source_indegree0"] = out["has_valid_collection_date"] & (out["temporal_in_degree"] == 0)
    out["is_temporal_sink_outdegree0"] = out["has_valid_collection_date"] & (out["temporal_out_degree"] == 0)
    out["is_temporal_bridge_node"] = (out["temporal_in_degree"] > 0) & (out["temporal_out_degree"] > 0)
    out["temporal_bridge_score"] = np.minimum(out["temporal_in_degree"], out["temporal_out_degree"])
    out["temporal_flow_score"] = out["temporal_in_degree"] * out["temporal_out_degree"]
    out = out.drop(columns=["collection_date_parsed"])
    return out


def graph_metric_row(
    panel: str,
    seed: int,
    spec: dict[str, Any],
    nodes: pd.DataFrame,
    directed_edges: pd.DataFrame,
    edge_qc: dict[str, Any],
    graph_stats: dict[str, Any],
) -> dict[str, Any]:
    dated = nodes[nodes["has_valid_collection_date"]]
    lag = directed_edges["temporal_lag_days"] if "temporal_lag_days" in directed_edges else pd.Series(dtype=float)
    n_dated = int(dated.shape[0])
    n_edges = int(edge_qc["n_undirected_edges"])
    n_directed = int(edge_qc["n_directed_edges"])
    return {
        "panel": panel,
        "seed": int(seed),
        "graph_name": spec["graph_name"],
        "metric_family": spec["metric_family"],
        "embedding_metric": spec["embedding_metric"],
        "graph_family": spec["graph_family"],
        "graph_dir": str(spec["graph_dir"]),
        "n_nodes": int(nodes.shape[0]),
        "n_dated_nodes": n_dated,
        "n_missing_date_nodes": int((~nodes["has_valid_collection_date"]).sum()),
        "n_undirected_edges": n_edges,
        "n_directed_edges": n_directed,
        "n_equal_date_edges": int(edge_qc["n_equal_date_edges"]),
        "n_missing_date_edges": int(edge_qc["n_missing_date_edges"]),
        "directed_edge_fraction": float(n_directed / n_edges) if n_edges else math.nan,
        "n_source_indegree0_dated": int(dated["is_temporal_source_indegree0"].sum()),
        "source_fraction_dated": float(dated["is_temporal_source_indegree0"].mean()) if n_dated else math.nan,
        "n_sink_outdegree0_dated": int(dated["is_temporal_sink_outdegree0"].sum()),
        "sink_fraction_dated": float(dated["is_temporal_sink_outdegree0"].mean()) if n_dated else math.nan,
        "n_temporal_bridge_nodes": int(dated["is_temporal_bridge_node"].sum()),
        "temporal_bridge_fraction_dated": float(dated["is_temporal_bridge_node"].mean()) if n_dated else math.nan,
        "mean_temporal_in_degree": float(dated["temporal_in_degree"].mean()) if n_dated else math.nan,
        "mean_temporal_out_degree": float(dated["temporal_out_degree"].mean()) if n_dated else math.nan,
        "max_temporal_in_degree": int(dated["temporal_in_degree"].max()) if n_dated else 0,
        "max_temporal_out_degree": int(dated["temporal_out_degree"].max()) if n_dated else 0,
        "max_temporal_bridge_score": int(dated["temporal_bridge_score"].max()) if n_dated else 0,
        "max_temporal_flow_score": int(dated["temporal_flow_score"].max()) if n_dated else 0,
        "mean_temporal_lag_days": float(lag.mean()) if len(lag) else math.nan,
        "median_temporal_lag_days": float(lag.median()) if len(lag) else math.nan,
        "max_temporal_lag_days": int(lag.max()) if len(lag) else 0,
        "source_graph_n_components": graph_stats.get("n_components", ""),
        "source_graph_giant_component_size": graph_stats.get("giant_component_size", ""),
        "orientation_rule": edge_qc["orientation_rule"],
        "is_dag_by_construction": edge_qc["is_dag_by_construction"],
    }


def group_metrics(
    panel: str,
    seed: int,
    spec: dict[str, Any],
    nodes: pd.DataFrame,
    group_cols: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dated = nodes[nodes["has_valid_collection_date"]].copy()
    for col in group_cols:
        if col not in dated.columns:
            continue
        values = dated[col].fillna("").astype(str).str.strip()
        subframe = dated.assign(_group_value=values)
        subframe = subframe[subframe["_group_value"] != ""]
        for group_value, group in subframe.groupby("_group_value", dropna=False):
            n = int(group.shape[0])
            rows.append(
                {
                    "panel": panel,
                    "seed": int(seed),
                    "graph_name": spec["graph_name"],
                    "metric_family": spec["metric_family"],
                    "embedding_metric": spec["embedding_metric"],
                    "graph_family": spec["graph_family"],
                    "group_type": col,
                    "group_value": group_value,
                    "n_nodes": n,
                    "mean_temporal_in_degree": float(group["temporal_in_degree"].mean()) if n else math.nan,
                    "mean_temporal_out_degree": float(group["temporal_out_degree"].mean()) if n else math.nan,
                    "source_fraction_dated": float(group["is_temporal_source_indegree0"].mean()) if n else math.nan,
                    "sink_fraction_dated": float(group["is_temporal_sink_outdegree0"].mean()) if n else math.nan,
                    "temporal_bridge_fraction_dated": float(group["is_temporal_bridge_node"].mean()) if n else math.nan,
                    "max_temporal_bridge_score": int(group["temporal_bridge_score"].max()) if n else 0,
                    "max_temporal_flow_score": int(group["temporal_flow_score"].max()) if n else 0,
                }
            )
    return rows


def prepend_metadata(frame: pd.DataFrame, metadata: list[tuple[str, Any]]) -> pd.DataFrame:
    out = frame.copy()
    for col, _ in metadata:
        if col in out.columns:
            out = out.drop(columns=[col])
    for pos, (col, value) in enumerate(metadata):
        out.insert(pos, col, value)
    return out


def evaluate_seed(
    panel: str,
    seed: int,
    panel_root: Path,
    seed_out: Path,
    args: argparse.Namespace,
) -> None:
    graph_rows: list[dict[str, Any]] = []
    node_frames: list[pd.DataFrame] = []
    group_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    directed_edge_frames: list[pd.DataFrame] = []

    specs = wanted_rng_graphs(panel_root, args.sample_label, args.graph_names)
    for graph_index, spec in enumerate(specs):
        graph_dir = Path(spec["graph_dir"])
        if not graph_dir.exists():
            log(f"Skipping missing RNG graph: {graph_dir}")
            continue
        log(f"Orienting temporal RNG DAG: {panel}/seed_{seed} {spec['graph_name']}")
        nodes, edges, graph_stats = read_graph_tables(graph_dir)
        directed_edges, edge_qc = orient_edges_by_collection_date(nodes, edges)
        node_metrics = add_temporal_node_metrics(nodes, directed_edges)
        metadata = [
            ("panel", panel),
            ("seed", int(seed)),
            ("graph_name", spec["graph_name"]),
            ("metric_family", spec["metric_family"]),
            ("embedding_metric", spec["embedding_metric"]),
            ("graph_family", spec["graph_family"]),
        ]
        node_metrics = prepend_metadata(node_metrics, metadata)
        node_frames.append(node_metrics)

        observed_row = graph_metric_row(panel, seed, spec, node_metrics, directed_edges, edge_qc, graph_stats)
        graph_rows.append(observed_row)
        group_rows.extend(group_metrics(panel, seed, spec, node_metrics, args.group_cols))

        if args.shuffle_permutations > 0:
            log(
                "Scoring timestamp-shuffle null: "
                f"{panel}/seed_{seed} {spec['graph_name']} permutations={args.shuffle_permutations}"
            )
            null_rows.append(shuffle_null_metrics(panel, seed, spec, nodes, edges, observed_row, graph_index, args))

        if args.write_directed_edges:
            directed = prepend_metadata(directed_edges, metadata)
            directed_edge_frames.append(directed)

    seed_out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(graph_rows).to_csv(seed_out / "temporal_rng_dag_graph_metrics.csv", index=False)
    node_frame = pd.concat(node_frames, ignore_index=True) if node_frames else pd.DataFrame()
    node_frame.to_csv(seed_out / "temporal_rng_dag_node_metrics.csv", index=False)
    pd.DataFrame(group_rows).to_csv(seed_out / "temporal_rng_dag_group_metrics.csv", index=False)
    pd.DataFrame(null_rows).to_csv(seed_out / "temporal_rng_dag_shuffle_null_metrics.csv", index=False)
    if args.write_directed_edges:
        edge_frame = pd.concat(directed_edge_frames, ignore_index=True) if directed_edge_frames else pd.DataFrame()
        edge_frame.to_csv(
            seed_out / "temporal_rng_dag_directed_edges.csv.gz",
            index=False,
            compression="gzip",
        )


def summarize(frame: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        out = {col: key for col, key in zip(group_cols, keys)}
        out["n_seeds"] = int(group["seed"].nunique()) if "seed" in group.columns else int(group.shape[0])
        for value_col in value_cols:
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
    return pd.DataFrame(rows)


def aggregate_workspace(workspace: Path) -> None:
    graph_frames = [pd.read_csv(path) for path in workspace.glob("*/seed_*/temporal_rng_dag_graph_metrics.csv")]
    node_frames = [pd.read_csv(path) for path in workspace.glob("*/seed_*/temporal_rng_dag_node_metrics.csv")]
    group_frames = [pd.read_csv(path) for path in workspace.glob("*/seed_*/temporal_rng_dag_group_metrics.csv")]
    null_frames = [pd.read_csv(path) for path in workspace.glob("*/seed_*/temporal_rng_dag_shuffle_null_metrics.csv")]

    if graph_frames:
        graph = pd.concat(graph_frames, ignore_index=True)
        graph.to_csv(workspace / "all_temporal_rng_dag_graph_metrics.csv", index=False)
        summary = summarize(
            graph,
            ["panel", "graph_name", "metric_family", "embedding_metric", "graph_family"],
            DEFAULT_SUMMARY_METRICS,
        )
        if not summary.empty:
            summary = summary.sort_values(["panel", "source_fraction_dated_mean", "graph_name"]).reset_index(drop=True)
        summary.to_csv(workspace / "temporal_rng_dag_graph_seed_summary.csv", index=False)

    if node_frames:
        node = pd.concat(node_frames, ignore_index=True)
        node.to_csv(workspace / "all_temporal_rng_dag_node_metrics.csv", index=False)

    if group_frames:
        grouped = pd.concat(group_frames, ignore_index=True)
        grouped.to_csv(workspace / "all_temporal_rng_dag_group_metrics.csv", index=False)
        summary = summarize(
            grouped,
            ["panel", "graph_name", "metric_family", "embedding_metric", "graph_family", "group_type", "group_value"],
            [
                "n_nodes",
                "mean_temporal_in_degree",
                "mean_temporal_out_degree",
                "source_fraction_dated",
                "sink_fraction_dated",
                "temporal_bridge_fraction_dated",
                "max_temporal_bridge_score",
                "max_temporal_flow_score",
            ],
        )
        if not summary.empty:
            summary = summary.sort_values(
                ["panel", "graph_name", "group_type", "source_fraction_dated_mean", "group_value"]
            ).reset_index(drop=True)
        summary.to_csv(workspace / "temporal_rng_dag_group_seed_summary.csv", index=False)

    if null_frames:
        null = pd.concat(null_frames, ignore_index=True)
        null.to_csv(workspace / "all_temporal_rng_dag_shuffle_null_metrics.csv", index=False)
        summary = summarize(
            null,
            ["panel", "graph_name", "metric_family", "embedding_metric", "graph_family"],
            NULL_SUMMARY_METRICS,
        )
        if not summary.empty:
            summary = summary.sort_values(["panel", "z_source_mean", "graph_name"]).reset_index(drop=True)
        summary.to_csv(workspace / "temporal_rng_dag_shuffle_null_seed_summary.csv", index=False)


def infer_workspace_and_panel(args: argparse.Namespace) -> tuple[Path, list[str]]:
    if args.panel_workspace:
        panel_path = args.panel_workspace
        return panel_path.parent, [panel_path.name]
    panels = [item.strip() for item in args.panels.split(",") if item.strip()]
    return args.workspace, panels


def main() -> None:
    ap = argparse.ArgumentParser(description="Orient RNG graphs from older to newer collection dates and score temporal DAG structure.")
    ap.add_argument("--workspace", type=Path, default=Path("analysis/cohort_validation/13_random_full_dataset_2k_nj_tree_validation"))
    ap.add_argument("--panel-workspace", type=Path, default=None)
    ap.add_argument("--source-root", type=Path, default=None)
    ap.add_argument("--panels", default="random_full_dataset_2k")
    ap.add_argument("--seeds", default="0-199")
    ap.add_argument("--sample-label", default="pool_n2000")
    ap.add_argument("--graph-names", default="", help="Comma-separated RNG graph names; empty means all RNG graphs.")
    ap.add_argument(
        "--group-cols",
        nargs="*",
        default=["cohort_id", "cohort_name", "within_lineage_label", "lineage", "collection_month"],
        help="Node metadata columns to summarize for emerging-variant behavior.",
    )
    ap.add_argument("--write-directed-edges", action="store_true", help="Write per-seed directed edge lists as CSV.GZ.")
    ap.add_argument("--shuffle-permutations", type=int, default=0, help="Timestamp shuffles per graph. Set >0 to run the null model.")
    ap.add_argument("--shuffle-seed", type=int, default=20260608)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

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
                evaluate_seed(panel, seed, panel_root, seed_out, args)

    aggregate_workspace(workspace)


if __name__ == "__main__":
    main()
