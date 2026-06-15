#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy.sparse import csr_matrix, save_npz
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial.distance import cdist


MISSING_LABELS = {"", "nan", "none", "null", "nat"}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_ids(ids_path: Path) -> pd.DataFrame:
    rows = []
    with ids_path.open("r", encoding="utf-8", errors="replace") as handle:
        for row_idx, line in enumerate(handle):
            raw = line.strip()
            if not raw:
                continue
            parts = raw.split("|")
            rows.append(
                {
                    "embedding_row": row_idx,
                    "lineage_from_ids": parts[0] if len(parts) > 0 else "",
                    "collection_date_from_ids": parts[1] if len(parts) > 1 else "",
                    "accession": parts[2] if len(parts) > 2 else raw,
                }
            )
    return pd.DataFrame(rows)


def fix_partial_date(value: object) -> str:
    s = str(value)
    if s.endswith("-00-00"):
        return s[:4] + "-06-15"
    if s.endswith("-00"):
        return s[:7] + "-15"
    return s


def parse_label_groups(group_spec: str) -> dict[str, set[str]]:
    if group_spec.strip().lower() in {"", "none"}:
        return {}
    groups: dict[str, set[str]] = {}
    for spec in group_spec.split(";"):
        spec = spec.strip()
        if not spec:
            continue
        if "=" not in spec:
            raise ValueError(f"Invalid label group spec {spec!r}. Expected NewLabel=a,b,c")
        label, members = spec.split("=", 1)
        member_set = {m.strip() for m in members.split(",") if m.strip()}
        if not label.strip() or not member_set:
            raise ValueError(f"Invalid label group spec {spec!r}. Expected NewLabel=a,b,c")
        groups[label.strip()] = member_set
    return groups


def apply_label_group(value: object, groups: dict[str, set[str]]) -> str:
    label = str(value).strip()
    for new_label, members in groups.items():
        if label in members:
            return new_label
    return label


def canonical_fasta_id(header: str) -> str:
    token = header.split()[0]
    parts = [p.strip() for p in token.split("|") if p.strip()]
    for part in parts:
        if part.startswith(("NC_", "EPI_ISL_", "OP", "ON", "OM", "MZ", "MW", "MT")):
            return part
    return token


def read_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    cur_id: str | None = None
    cur: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    seqs[cur_id] = "".join(cur).upper()
                cur_id = canonical_fasta_id(line[1:])
                cur = []
            else:
                cur.append(line)
    if cur_id is not None:
        seqs[cur_id] = "".join(cur).upper()
    return seqs


def load_metadata(metadata_path: Path, label_col: str, exclude_labels: set[str]) -> pd.DataFrame:
    meta = pd.read_csv(metadata_path, low_memory=False)
    if "accession" not in meta.columns:
        raise ValueError(f"{metadata_path} must contain accession")
    if label_col not in meta.columns:
        raise ValueError(f"{metadata_path} must contain {label_col}")
    meta = meta.copy()
    meta["accession"] = meta["accession"].astype(str).str.strip()
    labels = meta[label_col].astype(str).str.strip()
    valid = ~labels.str.casefold().isin(MISSING_LABELS)
    if exclude_labels:
        valid &= ~labels.isin(exclude_labels)
    return meta.loc[valid].copy()


def load_embedding_panel(
    embeddings_path: Path,
    ids_path: Path,
    metadata_path: Path,
    label_col: str,
    exclude_labels: set[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    X_all = np.load(embeddings_path, mmap_mode="r")
    ids = parse_ids(ids_path)
    if len(ids) != X_all.shape[0]:
        raise ValueError(f"ids rows {len(ids):,} != embeddings rows {X_all.shape[0]:,}")
    ids["accession"] = ids["accession"].astype(str).str.strip()
    meta = load_metadata(metadata_path, label_col, exclude_labels)
    merged = ids.merge(meta, on="accession", how="inner", suffixes=("", "_meta"))
    row_idx = merged["embedding_row"].astype(int).to_numpy()
    X = np.asarray(X_all[row_idx], dtype=np.float32)
    dates = pd.to_datetime(merged.get("collection_date", merged["collection_date_from_ids"]).map(fix_partial_date), errors="coerce")
    lineage = merged["lineage"].astype(str) if "lineage" in merged.columns else merged["lineage_from_ids"].astype(str)
    nodes = pd.DataFrame(
        {
            "node_id": np.arange(len(merged), dtype=int),
            "accession": merged["accession"].astype(str).to_numpy(),
            "collection_date": dates.dt.strftime("%Y-%m-%d").fillna("").to_numpy(),
            "lineage": lineage.to_numpy(),
            label_col: merged[label_col].astype(str).to_numpy(),
            "embedding_row": row_idx,
        }
    )
    if "location" in merged.columns:
        nodes["location"] = merged["location"].astype(str).to_numpy()
    log(f"Loaded embedding panel: n={X.shape[0]:,}, d={X.shape[1]:,}")
    return X, nodes


def load_hamming_panel(
    aligned_fasta: Path,
    metadata_path: Path,
    label_col: str,
    exclude_labels: set[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    seqs_by_id = read_fasta(aligned_fasta)
    meta = load_metadata(metadata_path, label_col, exclude_labels)
    meta = meta[meta["accession"].isin(seqs_by_id)].copy()
    if meta.empty:
        raise ValueError(f"No metadata accessions from {metadata_path} were present in {aligned_fasta}")
    seqs = [seqs_by_id[acc] for acc in meta["accession"].astype(str)]
    lengths = {len(seq) for seq in seqs}
    if len(lengths) != 1:
        raise ValueError(f"Selected FASTA records are not aligned to one length. Lengths include: {sorted(lengths)[:10]}")
    dates = pd.to_datetime(meta["collection_date"].map(fix_partial_date), errors="coerce")
    nodes = pd.DataFrame(
        {
            "node_id": np.arange(len(meta), dtype=int),
            "accession": meta["accession"].astype(str).to_numpy(),
            "collection_date": dates.dt.strftime("%Y-%m-%d").fillna("").to_numpy(),
            "lineage": meta["lineage"].astype(str).to_numpy() if "lineage" in meta.columns else "",
            label_col: meta[label_col].astype(str).to_numpy(),
            "alignment_row": np.arange(len(meta), dtype=int),
        }
    )
    if "location" in meta.columns:
        nodes["location"] = meta["location"].astype(str).to_numpy()
    arr = np.frombuffer("".join(seqs).encode("ascii"), dtype=np.uint8).reshape(len(seqs), len(seqs[0]))
    log(f"Loaded Hamming panel: n={arr.shape[0]:,}, alignment_length={arr.shape[1]:,}")
    return arr, nodes


def compute_embedding_distance(X: np.ndarray, metric: str) -> np.ndarray:
    log(f"Computing embedding pairwise {metric}: n={X.shape[0]:,}, d={X.shape[1]:,}")
    D = cdist(X, X, metric=metric).astype(np.float32, copy=False)
    np.fill_diagonal(D, np.inf)
    log(f"Embedding distance matrix ready: {D.nbytes / 1e9:.2f} GB")
    return D


def compute_hamming_distance(
    seqs: np.ndarray,
    block_size: int,
    gap_policy: str,
    ignore_ambiguous: bool,
) -> np.ndarray:
    n = seqs.shape[0]
    D = np.zeros((n, n), dtype=np.uint16)
    gap = ord("-")
    ambiguous = np.frombuffer(b"XBZJUO?", dtype=np.uint8)
    valid = seqs != gap if gap_policy == "ignore-any-gap" else np.ones_like(seqs, dtype=bool)
    if ignore_ambiguous:
        valid &= ~np.isin(seqs, ambiguous)

    log(f"Computing Hamming pairwise distances: n={n:,}, block_size={block_size}")
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        diff = seqs[start:stop, None, :] != seqs[None, :, :]
        if gap_policy == "ignore-any-gap" or ignore_ambiguous:
            diff &= valid[start:stop, None, :] & valid[None, :, :]
        D[start:stop, :] = diff.sum(axis=2, dtype=np.int32).astype(np.uint16)
        log(f"Hamming rows {start:,}-{stop - 1:,}/{n:,}")
    np.fill_diagonal(D, np.iinfo(np.uint16).max)
    log(f"Hamming distance matrix ready: {D.nbytes / 1e9:.2f} GB")
    return D


def symmetric_knn_edges(D: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log(f"Building symmetric kNN edges k={k}")
    n = D.shape[0]
    idx = np.argpartition(D, kth=k - 1, axis=1)[:, :k]
    row_ids = np.arange(n)[:, None]
    dist = D[row_ids, idx]
    order = np.argsort(dist, axis=1, kind="stable")
    idx = np.take_along_axis(idx, order, axis=1)
    dist = np.take_along_axis(dist, order, axis=1)

    edges: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j, dij in zip(idx[i], dist[i]):
            j = int(j)
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            value = float(dij)
            previous = edges.get((a, b))
            if previous is None or value < previous:
                edges[(a, b)] = value
    return edge_dict_to_arrays(edges)


def mst_edges(D: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log("Building MST edges")
    D_mst = np.asarray(D, dtype=np.float32).copy()
    np.fill_diagonal(D_mst, 0.0)
    mst = minimum_spanning_tree(D_mst).tocoo()
    sources = mst.row.astype(np.int32)
    targets = mst.col.astype(np.int32)
    weights = mst.data.astype(np.float32)
    mask = sources != targets
    return sources[mask], targets[mask], weights[mask]


def edge_dict_to_arrays(edges: dict[tuple[int, int], float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not edges:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float32)
    items = sorted(edges.items())
    sources = np.array([i for (i, _), _ in items], dtype=np.int32)
    targets = np.array([j for (_, j), _ in items], dtype=np.int32)
    weights = np.array([w for _, w in items], dtype=np.float32)
    return sources, targets, weights


@njit(cache=True)
def exact_rng_rows(
    D: np.ndarray,
    order: np.ndarray,
    row_start: int,
    row_end: int,
    max_edges: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    n = D.shape[0]
    sources = np.empty(max_edges, dtype=np.int32)
    targets = np.empty(max_edges, dtype=np.int32)
    weights = np.empty(max_edges, dtype=np.float32)
    kept = 0
    pruned = 0
    checked = 0
    for i in range(row_start, row_end):
        oi = order[i]
        for j in range(i + 1, n):
            dij = D[i, j]
            witness = False
            for pos in range(n):
                k = oi[pos]
                if D[i, k] >= dij:
                    break
                if D[j, k] < dij:
                    witness = True
                    break
            checked += 1
            if witness:
                pruned += 1
            else:
                if kept >= max_edges:
                    return sources, targets, weights, kept, pruned, checked
                sources[kept] = i
                targets[kept] = j
                weights[kept] = float(dij)
                kept += 1
    return sources, targets, weights, kept, pruned, checked


def exact_rng_edges(
    D: np.ndarray,
    row_block_size: int,
    max_block_edges: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    log("Sorting distance rows for exact RNG witness scan")
    order = np.argsort(D, axis=1).astype(np.int32, copy=False)
    n = D.shape[0]
    candidate_edges = n * (n - 1) // 2
    sources_all = []
    targets_all = []
    weights_all = []
    checked_total = 0
    pruned_total = 0
    kept_total = 0
    t0 = time.time()
    for row_start in range(0, n - 1, row_block_size):
        row_end = min(row_start + row_block_size, n - 1)
        sources, targets, weights, kept, pruned, checked = exact_rng_rows(
            D, order, row_start, row_end, max_block_edges
        )
        if kept >= max_block_edges:
            raise RuntimeError(f"RNG block {row_start}:{row_end} reached max_block_edges={max_block_edges}")
        if kept:
            sources_all.append(sources[:kept].copy())
            targets_all.append(targets[:kept].copy())
            weights_all.append(weights[:kept].copy())
        checked_total += int(checked)
        pruned_total += int(pruned)
        kept_total += int(kept)
        elapsed = time.time() - t0
        rate = checked_total / elapsed if elapsed else 0.0
        eta = (candidate_edges - checked_total) / rate if rate else 0.0
        log(
            f"RNG rows {row_start:,}-{row_end - 1:,}: checked={checked_total:,}/{candidate_edges:,} "
            f"kept={kept_total:,} pruned={pruned_total:,} rate={rate:,.0f}/s eta={eta / 3600:.2f}h"
        )
    sources_out = np.concatenate(sources_all) if sources_all else np.array([], dtype=np.int32)
    targets_out = np.concatenate(targets_all) if targets_all else np.array([], dtype=np.int32)
    weights_out = np.concatenate(weights_all) if weights_all else np.array([], dtype=np.float32)
    return sources_out, targets_out, weights_out, checked_total, pruned_total


def load_edges_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = pd.read_csv(path)
    if "status" in edges.columns:
        edges = edges[edges["status"].astype(str) == "kept"]
    if edges.empty:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float32)
    return (
        edges["source"].astype(np.int32).to_numpy(),
        edges["target"].astype(np.int32).to_numpy(),
        edges["weight"].astype(np.float32).to_numpy(),
    )


def edges_to_csr(n_nodes: int, sources: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> csr_matrix:
    rows = np.concatenate([sources, targets]).astype(np.int32, copy=False)
    cols = np.concatenate([targets, sources]).astype(np.int32, copy=False)
    data = np.concatenate([weights, weights]).astype(np.float32, copy=False)
    return csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float32)


def graph_stats(adj: csr_matrix) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    n_comp, labels = connected_components(adj, directed=False, return_labels=True)
    sizes = np.bincount(labels, minlength=n_comp) if len(labels) else np.array([], dtype=int)
    giant_label = int(sizes.argmax()) if len(sizes) else -1
    giant_size = int(sizes[giant_label]) if giant_label >= 0 else 0
    degree = np.asarray(adj.getnnz(axis=1)).ravel().astype(int)
    return labels, degree, {
        "n_nodes": int(adj.shape[0]),
        "n_edges": int(adj.nnz // 2),
        "n_components": int(n_comp),
        "giant_label": giant_label,
        "giant_component_size": giant_size,
        "giant_component_frac": float(giant_size / adj.shape[0]) if adj.shape[0] else 0.0,
        "mean_degree": float(degree.mean()) if len(degree) else 0.0,
        "median_degree": float(np.median(degree)) if len(degree) else 0.0,
        "max_degree": int(degree.max()) if len(degree) else 0,
    }


def write_graph(
    out_dir: Path,
    nodes: pd.DataFrame,
    sources: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    graph_type: str,
    metric_family: str,
    metric: str,
    k: int | None = None,
    candidate_edges: int | None = None,
    pruned_edges: int | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    adj = edges_to_csr(len(nodes), sources, targets, weights)
    comp_labels, degree, stats = graph_stats(adj)
    save_npz(out_dir / "adj.npz", adj)
    nodes_out = nodes.copy()
    nodes_out["graph_type"] = graph_type
    nodes_out["metric_family"] = metric_family
    nodes_out["metric"] = metric
    nodes_out["k"] = "" if k is None else int(k)
    nodes_out["component_id"] = comp_labels
    nodes_out["degree"] = degree
    nodes_out.to_csv(out_dir / "nodes.csv", index=False)
    with (out_dir / "edges.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "target", "weight", "status"])
        for i, j, w in zip(sources, targets, weights):
            writer.writerow([int(i), int(j), f"{float(w):.12g}", "kept"])

    meta = {
        "graph_type": graph_type,
        "metric_family": metric_family,
        "metric": metric,
        "k": None if k is None else int(k),
        **stats,
        "candidate_edges": None if candidate_edges is None else int(candidate_edges),
        "n_pruned_edges": 0 if pruned_edges is None else int(pruned_edges),
    }
    if extra:
        meta.update(extra)
    with (out_dir / "stats.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    log(f"Wrote {out_dir}: edges={stats['n_edges']:,}, components={stats['n_components']}, giant={stats['giant_component_frac']:.3f}")
    return meta


def coleman_index(
    nodes: pd.DataFrame,
    sources: np.ndarray,
    targets: np.ndarray,
    label_col: str,
    label_groups: dict[str, set[str]],
) -> dict[str, object]:
    labels = [apply_label_group(value, label_groups) for value in nodes[label_col].astype(str)]
    valid = np.array([label.casefold() not in MISSING_LABELS for label in labels], dtype=bool)
    edge_mask = valid[sources] & valid[targets]
    sources = sources[edge_mask]
    targets = targets[edge_mask]
    m = int(len(sources))
    if m == 0:
        return {
            "n_edges_used": 0,
            "observed_same_fraction": np.nan,
            "expected_same_fraction": np.nan,
            "coleman_h": np.nan,
            "nodepair_observed_same_fraction": np.nan,
            "nodepair_expected_same_fraction": np.nan,
            "nodepair_enrichment_ratio": np.nan,
            "global_observed_same_fraction": np.nan,
            "global_expected_same_fraction": np.nan,
            "global_assortativity_coefficient": np.nan,
            "label_counts": "{}",
            "stub_fractions": "{}",
        }

    same = sum(labels[int(i)] == labels[int(j)] for i, j in zip(sources, targets))
    observed = same / m
    stubs: Counter[str] = Counter()
    valid_labels = [label for label, is_valid in zip(labels, valid) if is_valid]
    label_counts: Counter[str] = Counter(valid_labels)
    for i, j in zip(sources, targets):
        stubs[labels[int(i)]] += 1
        stubs[labels[int(j)]] += 1
    stub_fractions = {label: count / (2 * m) for label, count in sorted(stubs.items())}
    expected = sum(value * value for value in stub_fractions.values())
    denom = 1.0 - expected
    h = (observed - expected) / denom if denom != 0 else np.nan
    n_valid = sum(label_counts.values())
    total_possible_pairs = n_valid * (n_valid - 1) / 2
    same_possible_pairs = sum(count * (count - 1) / 2 for count in label_counts.values())
    nodepair_expected = same_possible_pairs / total_possible_pairs if total_possible_pairs else np.nan
    nodepair_enrichment = observed / nodepair_expected if nodepair_expected else np.nan
    global_observed = observed
    global_expected = nodepair_expected
    global_assortativity = global_observed / global_expected if global_expected else np.nan
    return {
        "n_edges_used": m,
        "observed_same_fraction": observed,
        "expected_same_fraction": expected,
        "coleman_h": h,
        "nodepair_observed_same_fraction": observed,
        "nodepair_expected_same_fraction": nodepair_expected,
        "nodepair_enrichment_ratio": nodepair_enrichment,
        "global_observed_same_fraction": global_observed,
        "global_expected_same_fraction": global_expected,
        "global_assortativity_coefficient": global_assortativity,
        "label_counts": json.dumps(dict(sorted(label_counts.items())), sort_keys=True),
        "stub_fractions": json.dumps(stub_fractions, sort_keys=True),
    }


def evaluate_existing_graph(
    name: str,
    run_dir: Path,
    label_groups: dict[str, set[str]],
) -> dict[str, object]:
    nodes = pd.read_csv(run_dir / "nodes.csv", low_memory=False)
    sources, targets, weights = load_edges_csv(run_dir / "edges.csv")
    stats = json.loads((run_dir / "stats.json").read_text()) if (run_dir / "stats.json").exists() else {}
    coleman = coleman_index(nodes, sources, targets, "variant_bucket", label_groups)
    return {
        "graph_name": name,
        "metric_family": stats.get("metric_family", "embedding"),
        "graph_type": stats.get("graph_type", run_dir.name),
        "metric": stats.get("metric", ""),
        "k": stats.get("k", ""),
        "run_dir": str(run_dir),
        "n_nodes": int(stats.get("n_nodes", len(nodes))),
        "n_edges": int(stats.get("n_edges", len(sources))),
        "n_components": stats.get("n_components", ""),
        "giant_component_frac": stats.get("giant_component_frac", ""),
        **coleman,
    }


def build_family(
    metric_family: str,
    D: np.ndarray,
    nodes: pd.DataFrame,
    out_root: Path,
    metric: str,
    label_groups: dict[str, set[str]],
    build_rng: bool,
    row_block_size: int,
    max_block_edges: int,
) -> list[dict[str, object]]:
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []

    specs: list[tuple[str, str, int | None, tuple[np.ndarray, np.ndarray, np.ndarray], int | None, int | None]] = []
    specs.append(("mst", f"{metric_family}_mst", None, mst_edges(D), None, None))
    for k in [5, 50]:
        specs.append((f"knn_k{k:02d}", f"{metric_family}_knn", k, symmetric_knn_edges(D, k), None, None))
    if build_rng:
        sources, targets, weights, checked, pruned = exact_rng_edges(D, row_block_size, max_block_edges)
        specs.append(("rng_exact", f"{metric_family}_rng_exact", None, (sources, targets, weights), checked, pruned))

    for dirname, graph_type, k, edge_arrays, candidates, pruned in specs:
        sources, targets, weights = edge_arrays
        run_dir = out_root / dirname
        stats = write_graph(
            run_dir,
            nodes,
            sources,
            targets,
            weights,
            graph_type=graph_type,
            metric_family=metric_family,
            metric=metric,
            k=k,
            candidate_edges=candidates,
            pruned_edges=pruned,
        )
        coleman = coleman_index(nodes, sources, targets, "variant_bucket", label_groups)
        rows.append(
            {
                "graph_name": f"{metric_family}_{dirname}",
                "metric_family": metric_family,
                "graph_type": graph_type,
                "metric": metric,
                "k": "" if k is None else k,
                "run_dir": str(run_dir),
                "n_nodes": stats["n_nodes"],
                "n_edges": stats["n_edges"],
                "n_components": stats["n_components"],
                "giant_component_frac": stats["giant_component_frac"],
                **coleman,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Build MST/RNG/kNN graph families and compute Coleman homophily index")
    ap.add_argument("--embeddings", required=True, type=Path)
    ap.add_argument("--ids", required=True, type=Path)
    ap.add_argument("--aligned-fasta", required=True, type=Path)
    ap.add_argument("--metadata", required=True, type=Path)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--existing-embedding-rng", type=Path, default=None)
    ap.add_argument("--label-col", default="variant_bucket")
    ap.add_argument("--label-groups", default="Omicron=BA.1,BA.2,BA.2.86,BA.4,BA.5,JN.1,XBB")
    ap.add_argument("--exclude-label", action="append", default=["Reference"])
    ap.add_argument("--include-reference", action="store_true", help="Do not exclude the default Reference/Wuhan root label")
    ap.add_argument("--embedding-metric", choices=["cityblock", "manhattan", "euclidean"], default="cityblock")
    ap.add_argument("--hamming-block-size", type=int, default=24)
    ap.add_argument("--gap-policy", choices=["count-gap-state", "ignore-any-gap"], default="count-gap-state")
    ap.add_argument("--ignore-ambiguous", action="store_true")
    ap.add_argument("--rng-row-block-size", type=int, default=100)
    ap.add_argument("--max-block-edges", type=int, default=2_000_000)
    ap.add_argument("--build-embedding-rng", action="store_true", help="Build embedding exact RNG instead of only MST/kNN")
    ap.add_argument("--skip-hamming-rng", action="store_true")
    ap.add_argument("--skip-embedding-build", action="store_true")
    ap.add_argument("--skip-hamming-build", action="store_true")
    args = ap.parse_args()
    if args.build_embedding_rng and args.existing_embedding_rng is not None:
        raise ValueError("Use either --build-embedding-rng or --existing-embedding-rng, not both")

    label_groups = parse_label_groups(args.label_groups)
    exclude_labels = set(args.exclude_label)
    if args.include_reference:
        exclude_labels.discard("Reference")
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    embedding_metric = "cityblock" if args.embedding_metric == "manhattan" else args.embedding_metric
    if not args.skip_embedding_build:
        X, embedding_nodes = load_embedding_panel(args.embeddings, args.ids, args.metadata, args.label_col, exclude_labels)
        D_embedding = compute_embedding_distance(X, embedding_metric)
        summary_rows.extend(
            build_family(
                metric_family="embedding",
                D=D_embedding,
                nodes=embedding_nodes,
                out_root=out_root / "embedding",
                metric=embedding_metric,
                label_groups=label_groups,
                build_rng=args.build_embedding_rng,
                row_block_size=args.rng_row_block_size,
                max_block_edges=args.max_block_edges,
            )
        )
        del D_embedding
    if args.existing_embedding_rng is not None:
        summary_rows.append(evaluate_existing_graph("embedding_rng_exact", args.existing_embedding_rng, label_groups))

    if not args.skip_hamming_build:
        seqs, hamming_nodes = load_hamming_panel(args.aligned_fasta, args.metadata, args.label_col, exclude_labels)
        D_hamming = compute_hamming_distance(
            seqs,
            block_size=args.hamming_block_size,
            gap_policy=args.gap_policy,
            ignore_ambiguous=args.ignore_ambiguous,
        )
        summary_rows.extend(
            build_family(
                metric_family="hamming",
                D=D_hamming,
                nodes=hamming_nodes,
                out_root=out_root / "hamming",
                metric=f"hamming:{args.gap_policy}",
                label_groups=label_groups,
                build_rng=not args.skip_hamming_rng,
                row_block_size=args.rng_row_block_size,
                max_block_edges=args.max_block_edges,
            )
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = out_root / "coleman_summary.csv"
    summary.to_csv(summary_path, index=False)
    log(f"Saved Coleman summary: {summary_path}")
    if not summary.empty:
        cols = [
            "graph_name",
            "n_edges",
            "observed_same_fraction",
            "expected_same_fraction",
            "coleman_h",
            "global_observed_same_fraction",
            "global_expected_same_fraction",
            "global_assortativity_coefficient",
            "nodepair_expected_same_fraction",
            "nodepair_enrichment_ratio",
            "n_components",
            "giant_component_frac",
        ]
        print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
