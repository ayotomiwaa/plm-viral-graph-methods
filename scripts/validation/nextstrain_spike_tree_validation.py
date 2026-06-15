#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, load_npz
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.stats import rankdata, spearmanr, t as student_t


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_TREE_URL = "https://nextstrain.org/groups/neherlab/ncov/spike-only"
ACCESSION_RE = re.compile(r"EPI_ISL_\d+")
DATE_STAMP = time.strftime("%Y%m%d")


@dataclass
class ParsedTree:
    nodes: pd.DataFrame
    tips: pd.DataFrame
    edges: pd.DataFrame
    parse_qc: dict[str, Any]


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


def checksum_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def candidate_nextstrain_urls(url: str) -> list[str]:
    url = url.strip()
    out = [url]
    if not url.endswith(".json"):
        out.append(url + ".json")
    if url.startswith("https://nextstrain.org/"):
        path = url.removeprefix("https://nextstrain.org/").strip("/")
        data_url = f"https://data.nextstrain.org/{path}.json"
        out.append(data_url)
        out.append(f"https://nextstrain.org/charon/getDataset?prefix=/{path}")
    return list(dict.fromkeys(out))


def download_json(url: str, out_path: Path, force: bool = False) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_path.with_suffix(out_path.suffix + ".download.json")
    if out_path.exists() and not force:
        log(f"Using existing Nextstrain JSON: {out_path}")
        return {
            "path": str(out_path),
            "sha256": checksum_sha256(out_path),
            "downloaded": False,
        }

    last_error = ""
    headers = {"User-Agent": "Protein-embeddings-Nextstrain-validation/1.0"}
    for candidate in candidate_nextstrain_urls(url):
        log(f"Trying Nextstrain download: {candidate}")
        request = urllib.request.Request(candidate, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = response.read()
            parsed = json.loads(payload.decode("utf-8"))
            with out_path.open("wb") as handle:
                handle.write(payload)
            meta = {
                "source_url_requested": url,
                "source_url_used": candidate,
                "download_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "n_bytes": len(payload),
                "sha256": checksum_sha256(out_path),
                "downloaded": True,
            }
            meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            log(f"Downloaded Nextstrain JSON: {out_path}")
            return meta
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            log(f"Download candidate failed: {last_error}")
    raise RuntimeError(f"Could not download Nextstrain Auspice JSON. Last error: {last_error}")


def nested_values(obj: Any) -> list[str]:
    values: list[str] = []
    if isinstance(obj, dict):
        for value in obj.values():
            values.extend(nested_values(value))
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            values.extend(nested_values(value))
    elif obj is not None:
        values.append(str(obj))
    return values


def first_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, dict) and "value" in value:
        return first_number(value["value"])
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(number):
        return number
    return None


def get_node_attr(node_attrs: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in node_attrs:
            value = node_attrs[key]
            if isinstance(value, dict) and "value" in value:
                return value["value"]
            return value
    return None


def extract_accession_candidates(name: str, node_attrs: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    for text in [name, *nested_values(node_attrs)]:
        candidates.extend(ACCESSION_RE.findall(text))
    return list(dict.fromkeys(candidates))


def parse_auspice_tree(json_path: Path) -> ParsedTree:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    root = data.get("tree", data)
    if not isinstance(root, dict) or "children" not in root:
        raise ValueError(f"Could not find an Auspice v2 tree in {json_path}")

    rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    direct_div_values: list[float | None] = []
    branch_length_values: list[float | None] = []

    def walk(node: dict[str, Any], parent_id: int, depth: int, parent_div: float) -> int:
        node_id = len(rows)
        name = str(node.get("name", ""))
        node_attrs = node.get("node_attrs", {}) or {}
        branch_attrs = node.get("branch_attrs", {}) or {}

        direct_div = first_number(get_node_attr(node_attrs, ["div", "divergence", "num_substitutions"]))
        branch_len = first_number(node.get("branch_length"))
        if branch_len is None:
            branch_len = first_number(get_node_attr(branch_attrs, ["div", "length", "branch_length"]))
        if branch_len is None and direct_div is not None and parent_id >= 0:
            branch_len = max(0.0, float(direct_div) - float(parent_div))
        if branch_len is None:
            branch_len = 0.0

        root_dist = float(direct_div) if direct_div is not None else float(parent_div) + float(branch_len)
        accession_candidates = extract_accession_candidates(name, node_attrs)
        rows.append(
            {
                "node_index": node_id,
                "parent_index": parent_id,
                "depth": depth,
                "name": name,
                "is_tip": not bool(node.get("children")),
                "accession": accession_candidates[0] if accession_candidates else "",
                "accession_candidates": ";".join(accession_candidates),
                "strain": str(get_node_attr(node_attrs, ["strain"]) or name),
                "date": str(get_node_attr(node_attrs, ["date", "num_date"]) or ""),
                "clade": str(get_node_attr(node_attrs, ["clade_membership", "Nextstrain_clade", "clade"]) or ""),
                "direct_div": np.nan if direct_div is None else float(direct_div),
                "branch_length": float(branch_len),
                "root_dist": float(root_dist),
            }
        )
        direct_div_values.append(direct_div)
        branch_length_values.append(branch_len)

        if parent_id >= 0:
            edge_rows.append({"parent_index": parent_id, "child_index": node_id, "branch_length": float(branch_len)})

        for child in node.get("children", []) or []:
            walk(child, node_id, depth + 1, root_dist)
        return node_id

    walk(root, -1, 0, 0.0)

    nodes = pd.DataFrame(rows)
    tips = nodes[nodes["is_tip"]].copy().reset_index(drop=True)
    edges = pd.DataFrame(edge_rows)
    n_direct = sum(value is not None for value in direct_div_values)
    n_branch = sum(value is not None for value in branch_length_values)
    parse_qc = {
        "json_path": str(json_path),
        "n_nodes": int(len(nodes)),
        "n_tips": int(len(tips)),
        "n_tips_with_epi_isl_accession": int((tips["accession"] != "").sum()),
        "n_nodes_with_direct_div": int(n_direct),
        "n_nodes_with_branch_length_or_derived": int(n_branch),
        "distance_basis": "direct cumulative node div where available; otherwise accumulated branch lengths",
        "root_dist_min": float(nodes["root_dist"].min()) if len(nodes) else 0.0,
        "root_dist_max": float(nodes["root_dist"].max()) if len(nodes) else 0.0,
    }
    return ParsedTree(nodes=nodes, tips=tips, edges=edges, parse_qc=parse_qc)


def save_parsed_tree(json_path: Path, tree_dir: Path) -> ParsedTree:
    tree = parse_auspice_tree(json_path)
    tree_dir.mkdir(parents=True, exist_ok=True)
    tree.nodes.to_csv(tree_dir / "nextstrain_spike_only_nodes.csv", index=False)
    tree.tips.to_csv(tree_dir / "nextstrain_spike_only_tip_table.csv", index=False)
    tree.edges.to_csv(tree_dir / "nextstrain_spike_only_edges.csv", index=False)
    (tree_dir / "nextstrain_spike_only_parse_qc.json").write_text(
        json.dumps(tree.parse_qc, indent=2) + "\n",
        encoding="utf-8",
    )
    log(
        "Parsed Nextstrain tree: "
        f"nodes={tree.parse_qc['n_nodes']:,}, tips={tree.parse_qc['n_tips']:,}, "
        f"tips_with_accessions={tree.parse_qc['n_tips_with_epi_isl_accession']:,}"
    )
    return tree


def load_parsed_tree(tree_dir: Path) -> ParsedTree:
    nodes_path = tree_dir / "nextstrain_spike_only_nodes.csv"
    tips_path = tree_dir / "nextstrain_spike_only_tip_table.csv"
    edges_path = tree_dir / "nextstrain_spike_only_edges.csv"
    qc_path = tree_dir / "nextstrain_spike_only_parse_qc.json"
    if not nodes_path.exists() or not tips_path.exists() or not edges_path.exists():
        raise FileNotFoundError(f"Parsed tree files are missing in {tree_dir}; run parse stage first")
    return ParsedTree(
        nodes=pd.read_csv(nodes_path),
        tips=pd.read_csv(tips_path).fillna(""),
        edges=pd.read_csv(edges_path),
        parse_qc=json.loads(qc_path.read_text(encoding="utf-8")) if qc_path.exists() else {},
    )


def read_accession_list(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.open("r", encoding="utf-8", errors="replace") if line.strip()}


def read_fasta_accessions(path: Path) -> set[str]:
    accessions: set[str] = set()
    if not path.exists():
        return accessions
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(">"):
                match = ACCESSION_RE.search(line)
                accessions.add(match.group(0) if match else line[1:].split()[0].strip())
    return accessions


def read_node_accessions(path: Path) -> set[str]:
    if not path.exists():
        return set()
    nodes = pd.read_csv(path, usecols=["accession"], low_memory=False)
    return set(nodes["accession"].astype(str).str.strip())


def load_panel_sets(panel_root: Path, sample_label: str) -> dict[str, set[str]]:
    input_dir = panel_root / "inputs" / sample_label
    embedding_dir = panel_root / "embeddings" / "esm2_650M" / sample_label
    metadata_path = input_dir / "metadata.csv"
    aligned_path = input_dir / "spike_sequences_aligned_mafft.fasta"
    unaligned_path = input_dir / "spike_sequences.fasta"
    ids_path = embedding_dir / "ids.txt"

    metadata = pd.read_csv(metadata_path, usecols=["accession"], low_memory=False)
    metadata_set = set(metadata["accession"].astype(str).str.strip())
    embedding_nodes_path = panel_root / "graphs" / "esm2_650M" / "cityblock" / sample_label / "canonical_nodes.csv"
    ids_set = read_node_accessions(embedding_nodes_path)
    if not ids_set:
        ids_set = read_accession_list(ids_path)
    selected_set = read_accession_list(input_dir / "accessions.txt")
    alignment_set = read_fasta_accessions(aligned_path)
    if not alignment_set:
        alignment_set = read_fasta_accessions(unaligned_path)
    return {
        "selected": selected_set,
        "metadata": metadata_set,
        "embeddings": ids_set,
        "spike_alignment": alignment_set,
    }


def scoring_node_sets(panel_root: Path, sample_label: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for spec in raw_distance_paths(panel_root, sample_label):
        out[f"raw_nodes:{spec['baseline']}"] = read_node_accessions(spec["nodes"])
    for spec in graph_paths(panel_root, sample_label):
        out[f"graph_nodes:{spec['graph_name']}"] = read_node_accessions(spec["graph_dir"] / "nodes.csv")
    return {key: value for key, value in out.items() if value}


def build_accession_mapping(
    panel: str,
    seed: int,
    panel_root: Path,
    sample_label: str,
    tree: ParsedTree,
    out_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    sets = load_panel_sets(panel_root, sample_label)
    tree_tips = tree.tips.copy()
    tree_tips["accession"] = tree_tips["accession"].astype(str).str.strip()
    tree_tips = tree_tips[tree_tips["accession"] != ""].copy()
    duplicate_tree_accessions = sorted(tree_tips.loc[tree_tips["accession"].duplicated(), "accession"].unique())
    tree_first = tree_tips.drop_duplicates("accession", keep="first").copy()
    tree_accessions = set(tree_first["accession"])

    selected = sets["selected"] or sets["metadata"]
    all_accessions = sorted(selected | sets["metadata"] | sets["embeddings"] | sets["spike_alignment"] | tree_accessions)
    rows = []
    tip_lookup = tree_first.set_index("accession")
    for accession in all_accessions:
        in_selected = accession in selected
        in_metadata = accession in sets["metadata"]
        in_embeddings = accession in sets["embeddings"]
        in_alignment = accession in sets["spike_alignment"]
        in_tree = accession in tree_accessions
        matched = in_selected and in_metadata and in_embeddings and in_alignment and in_tree
        row = {
            "accession": accession,
            "in_selected_ids": in_selected,
            "in_metadata": in_metadata,
            "in_embeddings": in_embeddings,
            "in_spike_alignment": in_alignment,
            "in_nextstrain_spike_tree": in_tree,
            "matched_all": matched,
            "nextstrain_tip_name": "",
            "nextstrain_node_index": "",
        }
        if in_tree:
            tip = tip_lookup.loc[accession]
            row["nextstrain_tip_name"] = str(tip["name"])
            row["nextstrain_node_index"] = int(tip["node_index"])
        rows.append(row)

    mapping = pd.DataFrame(rows)
    matched = mapping[mapping["matched_all"]].copy().sort_values("accession").reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(out_dir / "accession_mapping.csv", index=False)
    matched["accession"].to_csv(out_dir / "matched_accessions.txt", index=False, header=False)

    qc = {
        "panel": panel,
        "seed": int(seed),
        "panel_root": str(panel_root),
        "sample_label": sample_label,
        "n_selected_ids": int(len(selected)),
        "n_metadata": int(len(sets["metadata"])),
        "n_embeddings": int(len(sets["embeddings"])),
        "n_spike_alignment": int(len(sets["spike_alignment"])),
        "n_nextstrain_tree_accessions": int(len(tree_accessions)),
        "n_duplicate_tree_accessions": int(len(duplicate_tree_accessions)),
        "duplicate_tree_accession_examples": duplicate_tree_accessions[:10],
        "n_matched_all": int(len(matched)),
        "n_selected_missing_tree": int(len(selected - tree_accessions)),
        "selected_missing_tree_examples": sorted(selected - tree_accessions)[:10],
        "n_selected_missing_alignment": int(len(selected - sets["spike_alignment"])),
        "n_selected_missing_embeddings": int(len(selected - sets["embeddings"])),
        "n_selected_missing_metadata": int(len(selected - sets["metadata"])),
    }
    (out_dir / "match_qc.json").write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
    log(f"{panel}/seed_{seed}: matched {qc['n_matched_all']:,}/{qc['n_selected_ids']:,} selected accessions")
    return matched, qc


def load_matched_mapping(out_dir: Path) -> pd.DataFrame:
    path = out_dir / "accession_mapping.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing accession mapping: {path}")
    mapping = pd.read_csv(path)
    return mapping[mapping["matched_all"]].copy().sort_values("accession").reset_index(drop=True)


def compute_patristic_matrix(tree: ParsedTree, matched: pd.DataFrame, out_dir: Path) -> np.ndarray:
    node_indices = matched["nextstrain_node_index"].astype(int).to_numpy()
    n_total = int(tree.nodes["node_index"].max()) + 1
    if tree.edges.empty:
        raise ValueError("Parsed tree has no edges")

    rows = pd.concat([tree.edges["parent_index"], tree.edges["child_index"]], ignore_index=True).astype(int).to_numpy()
    cols = pd.concat([tree.edges["child_index"], tree.edges["parent_index"]], ignore_index=True).astype(int).to_numpy()
    weights = pd.concat([tree.edges["branch_length"], tree.edges["branch_length"]], ignore_index=True).astype(float).to_numpy()
    graph = coo_matrix((weights, (rows, cols)), shape=(n_total, n_total)).tocsr()
    log(f"Computing Nextstrain patristic distances for n_matched={len(node_indices):,}")
    D = shortest_path(graph, directed=False, indices=node_indices, unweighted=False)
    D = np.asarray(D[:, node_indices], dtype=np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "D_NS_spike_float32.npy", D)
    matched[["accession", "nextstrain_tip_name", "nextstrain_node_index"]].to_csv(
        out_dir / "D_NS_spike_nodes.csv",
        index=False,
    )
    qc = {
        "n_matched": int(D.shape[0]),
        "dtype": str(D.dtype),
        "shape": [int(D.shape[0]), int(D.shape[1])],
        "size_gb": float(D.nbytes / 1e9),
        "n_nonfinite": int((~np.isfinite(D)).sum()),
        "min": float(np.nanmin(D)) if D.size else 0.0,
        "max": float(np.nanmax(D)) if D.size else 0.0,
    }
    (out_dir / "D_NS_spike_qc.json").write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
    log(f"Wrote Nextstrain patristic matrix: shape={D.shape}, size={D.nbytes / 1e9:.2f} GB")
    return D


def load_patristic(out_dir: Path) -> np.ndarray:
    path = out_dir / "D_NS_spike_float32.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing patristic matrix: {path}")
    return np.load(path, mmap_mode="r")


def upper_pair_values(D_ref: np.ndarray, D_other: np.ndarray, pair_mode: str, sample_size: int, seed: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    n = D_ref.shape[0]
    if D_other.shape != (n, n):
        raise ValueError(f"Matrix shape mismatch: ref={D_ref.shape}, other={D_other.shape}")
    if n < 2:
        return np.array([]), np.array([]), 0, 0

    if pair_mode == "all":
        i, j = np.triu_indices(n, k=1)
    else:
        total_pairs = n * (n - 1) // 2
        m = min(int(sample_size), total_pairs)
        rng = np.random.default_rng(seed)
        i = rng.integers(0, n, size=m, dtype=np.int64)
        j = rng.integers(0, n - 1, size=m, dtype=np.int64)
        j = j + (j >= i)

    x = np.asarray(D_ref[i, j])
    y = np.asarray(D_other[i, j])
    raw_pairs = int(len(x))
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask], raw_pairs, int(mask.sum())


def make_pair_indices(n: int, pair_mode: str, sample_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n < 2:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    if pair_mode == "all":
        return np.triu_indices(n, k=1)
    total_pairs = n * (n - 1) // 2
    m = min(int(sample_size), total_pairs)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=m, dtype=np.int64)
    j = rng.integers(0, n - 1, size=m, dtype=np.int64)
    j = j + (j >= i)
    return i, j


def spearman_score(
    D_ref: np.ndarray,
    D_other: np.ndarray,
    pair_mode: str,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    x, y, raw_pairs, finite_pairs = upper_pair_values(D_ref, D_other, pair_mode, sample_size, seed)
    if finite_pairs < 3:
        rho = np.nan
        pvalue = np.nan
    else:
        rho, pvalue = spearmanr(x, y)
    return {
        "pair_mode": pair_mode,
        "n_pairs_raw": raw_pairs,
        "n_pairs_used": finite_pairs,
        "finite_pair_fraction": float(finite_pairs / raw_pairs) if raw_pairs else np.nan,
        "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
        "spearman_pvalue": float(pvalue) if np.isfinite(pvalue) else np.nan,
    }


def upper_three_values(
    D_ref: np.ndarray,
    D_other: np.ndarray,
    D_control: np.ndarray,
    pair_mode: str,
    sample_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    n = D_ref.shape[0]
    if D_other.shape != (n, n) or D_control.shape != (n, n):
        raise ValueError(f"Matrix shape mismatch: ref={D_ref.shape}, other={D_other.shape}, control={D_control.shape}")
    if n < 2:
        return np.array([]), np.array([]), np.array([]), 0, 0

    i, j = make_pair_indices(n, pair_mode, sample_size, seed)

    x = np.asarray(D_ref[i, j])
    y = np.asarray(D_other[i, j])
    z = np.asarray(D_control[i, j])
    raw_pairs = int(len(x))
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    return x[mask], y[mask], z[mask], raw_pairs, int(mask.sum())


def corr_from_rank_vectors(a: np.ndarray, b: np.ndarray) -> float:
    n = a.size
    if n < 3:
        return np.nan
    sum_a = float(a.sum())
    sum_b = float(b.sum())
    sum_ab = float(np.dot(a, b))
    sum_a2 = float(np.dot(a, a))
    sum_b2 = float(np.dot(b, b))
    cov = sum_ab - (sum_a * sum_b / n)
    var_a = sum_a2 - (sum_a * sum_a / n)
    var_b = sum_b2 - (sum_b * sum_b / n)
    denom = math.sqrt(var_a * var_b) if var_a > 0 and var_b > 0 else np.nan
    return cov / denom if denom and np.isfinite(denom) else np.nan


def partial_spearman_score(
    D_ref: np.ndarray,
    D_other: np.ndarray,
    D_control: np.ndarray,
    pair_mode: str,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    x, y, z, raw_pairs, finite_pairs = upper_three_values(D_ref, D_other, D_control, pair_mode, sample_size, seed)
    if finite_pairs < 4:
        rho = np.nan
        pvalue = np.nan
        rho_ref_other = np.nan
        rho_ref_control = np.nan
        rho_other_control = np.nan
    else:
        log(f"Ranking {finite_pairs:,} finite pairs for partial Spearman")
        rx = rankdata(x, method="average")
        ry = rankdata(y, method="average")
        rz = rankdata(z, method="average")
        rho_ref_other = corr_from_rank_vectors(rx, ry)
        rho_ref_control = corr_from_rank_vectors(rx, rz)
        rho_other_control = corr_from_rank_vectors(ry, rz)
        denom = math.sqrt((1.0 - rho_ref_control**2) * (1.0 - rho_other_control**2))
        rho = (rho_ref_other - rho_ref_control * rho_other_control) / denom if denom > 0 else np.nan
        if np.isfinite(rho) and finite_pairs > 3 and abs(rho) < 1:
            stat = rho * math.sqrt((finite_pairs - 3) / max(1e-300, 1.0 - rho**2))
            pvalue = float(2.0 * student_t.sf(abs(stat), df=finite_pairs - 3))
        else:
            pvalue = 0.0 if np.isfinite(rho) and abs(rho) == 1 else np.nan
        del rx, ry, rz
    return {
        "pair_mode": pair_mode,
        "n_pairs_raw": raw_pairs,
        "n_pairs_used": finite_pairs,
        "finite_pair_fraction": float(finite_pairs / raw_pairs) if raw_pairs else np.nan,
        "partial_spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
        "partial_spearman_pvalue": float(pvalue) if np.isfinite(pvalue) else np.nan,
        "marginal_spearman_rho": float(rho_ref_other) if np.isfinite(rho_ref_other) else np.nan,
        "reference_control_spearman_rho": float(rho_ref_control) if np.isfinite(rho_ref_control) else np.nan,
        "candidate_control_spearman_rho": float(rho_other_control) if np.isfinite(rho_other_control) else np.nan,
    }


def partial_spearman_from_vectors(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    raw_pairs: int,
    pair_mode: str,
) -> dict[str, Any]:
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x = x[mask]
    y = y[mask]
    z = z[mask]
    finite_pairs = int(mask.sum())
    if finite_pairs < 4:
        rho = np.nan
        pvalue = np.nan
        rho_ref_other = np.nan
        rho_ref_control = np.nan
        rho_other_control = np.nan
    else:
        log(f"Ranking {finite_pairs:,} finite pairs for partial Spearman")
        rx = rankdata(x, method="average")
        ry = rankdata(y, method="average")
        rz = rankdata(z, method="average")
        rho_ref_other = corr_from_rank_vectors(rx, ry)
        rho_ref_control = corr_from_rank_vectors(rx, rz)
        rho_other_control = corr_from_rank_vectors(ry, rz)
        denom = math.sqrt((1.0 - rho_ref_control**2) * (1.0 - rho_other_control**2))
        rho = (rho_ref_other - rho_ref_control * rho_other_control) / denom if denom > 0 else np.nan
        if np.isfinite(rho) and finite_pairs > 3 and abs(rho) < 1:
            stat = rho * math.sqrt((finite_pairs - 3) / max(1e-300, 1.0 - rho**2))
            pvalue = float(2.0 * student_t.sf(abs(stat), df=finite_pairs - 3))
        else:
            pvalue = 0.0 if np.isfinite(rho) and abs(rho) == 1 else np.nan
        del rx, ry, rz
    return {
        "pair_mode": pair_mode,
        "n_pairs_raw": int(raw_pairs),
        "n_pairs_used": finite_pairs,
        "finite_pair_fraction": float(finite_pairs / raw_pairs) if raw_pairs else np.nan,
        "partial_spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
        "partial_spearman_pvalue": float(pvalue) if np.isfinite(pvalue) else np.nan,
        "marginal_spearman_rho": float(rho_ref_other) if np.isfinite(rho_ref_other) else np.nan,
        "reference_control_spearman_rho": float(rho_ref_control) if np.isfinite(rho_ref_control) else np.nan,
        "candidate_control_spearman_rho": float(rho_other_control) if np.isfinite(rho_other_control) else np.nan,
    }


def accession_index(nodes_path: Path) -> dict[str, int]:
    nodes = pd.read_csv(nodes_path, usecols=["node_id", "accession"])
    nodes["accession"] = nodes["accession"].astype(str).str.strip()
    return dict(zip(nodes["accession"], nodes["node_id"].astype(int)))


def subset_dense_matrix(matrix_path: Path, nodes_path: Path, accessions: list[str]) -> np.ndarray:
    idx_map = accession_index(nodes_path)
    missing = [acc for acc in accessions if acc not in idx_map]
    if missing:
        raise ValueError(f"{nodes_path}: missing {len(missing):,} matched accessions, examples={missing[:5]}")
    idx = np.array([idx_map[acc] for acc in accessions], dtype=int)
    D = np.load(matrix_path, mmap_mode="r")
    return np.asarray(D[np.ix_(idx, idx)])


def graph_shortest_path_matrix(graph_dir: Path, accessions: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    nodes_path = graph_dir / "nodes.csv"
    adj_path = graph_dir / "adj.npz"
    stats_path = graph_dir / "stats.json"
    if not nodes_path.exists() or not adj_path.exists():
        raise FileNotFoundError(f"Missing graph nodes/adj in {graph_dir}")
    idx_map = accession_index(nodes_path)
    missing = [acc for acc in accessions if acc not in idx_map]
    if missing:
        raise ValueError(f"{graph_dir}: missing {len(missing):,} matched accessions, examples={missing[:5]}")
    idx = np.array([idx_map[acc] for acc in accessions], dtype=int)
    adj = load_npz(adj_path).tocsr()
    sub = adj[idx, :][:, idx].tocsr()
    n_components, labels = connected_components(sub, directed=False, return_labels=True)
    component_sizes = np.bincount(labels, minlength=n_components) if len(labels) else np.array([], dtype=int)
    D = shortest_path(sub, directed=False, unweighted=False)
    D = np.asarray(D, dtype=np.float32)
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    qc = {
        "graph_dir": str(graph_dir),
        "n_nodes_matched": int(sub.shape[0]),
        "n_edges_matched": int(sub.nnz // 2),
        "n_components_matched": int(n_components),
        "giant_component_size_matched": int(component_sizes.max()) if len(component_sizes) else 0,
        "source_graph_n_components": stats.get("n_components", ""),
        "source_graph_giant_component_size": stats.get("giant_component_size", ""),
    }
    return D, qc


def graph_shortest_path_pair_values(
    graph_dir: Path,
    accessions: list[str],
    pair_i: np.ndarray,
    pair_j: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    nodes_path = graph_dir / "nodes.csv"
    adj_path = graph_dir / "adj.npz"
    stats_path = graph_dir / "stats.json"
    if not nodes_path.exists() or not adj_path.exists():
        raise FileNotFoundError(f"Missing graph nodes/adj in {graph_dir}")
    idx_map = accession_index(nodes_path)
    missing = [acc for acc in accessions if acc not in idx_map]
    if missing:
        raise ValueError(f"{graph_dir}: missing {len(missing):,} matched accessions, examples={missing[:5]}")
    idx = np.array([idx_map[acc] for acc in accessions], dtype=int)
    adj = load_npz(adj_path).tocsr()
    sub = adj[idx, :][:, idx].tocsr()
    n_components, labels = connected_components(sub, directed=False, return_labels=True)
    component_sizes = np.bincount(labels, minlength=n_components) if len(labels) else np.array([], dtype=int)
    unique_sources, inverse = np.unique(pair_i, return_inverse=True)
    log(f"Computing sampled graph shortest paths: sources={len(unique_sources):,}, pairs={len(pair_i):,}")
    dist_rows = shortest_path(sub, directed=False, unweighted=False, indices=unique_sources)
    values = np.asarray(dist_rows[inverse, pair_j], dtype=np.float32)
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    qc = {
        "graph_dir": str(graph_dir),
        "n_nodes_matched": int(sub.shape[0]),
        "n_edges_matched": int(sub.nnz // 2),
        "n_components_matched": int(n_components),
        "giant_component_size_matched": int(component_sizes.max()) if len(component_sizes) else 0,
        "source_graph_n_components": stats.get("n_components", ""),
        "source_graph_giant_component_size": stats.get("giant_component_size", ""),
        "shortest_path_mode": "sampled_sources",
        "n_sampled_sources": int(len(unique_sources)),
    }
    return values, qc


def raw_distance_paths(panel_root: Path, sample_label: str) -> list[dict[str, Any]]:
    graph_root = panel_root / "graphs"
    return [
        {
            "baseline": "raw_hamming",
            "metric_family": "hamming",
            "metric": "hamming",
            "matrix": graph_root / "hamming" / sample_label / "distance_matrices" / "hamming_count-gap-state_all_states_uint16.npy",
            "nodes": graph_root / "hamming" / sample_label / "canonical_nodes.csv",
        },
        {
            "baseline": "raw_esm2_cityblock",
            "metric_family": "embedding",
            "metric": "cityblock",
            "matrix": graph_root / "esm2_650M" / "cityblock" / sample_label / "distance_matrices" / "embedding_cityblock_float32.npy",
            "nodes": graph_root / "esm2_650M" / "cityblock" / sample_label / "canonical_nodes.csv",
        },
        {
            "baseline": "raw_esm2_euclidean",
            "metric_family": "embedding",
            "metric": "euclidean",
            "matrix": graph_root / "esm2_650M" / "euclidean" / sample_label / "distance_matrices" / "embedding_euclidean_float32.npy",
            "nodes": graph_root / "esm2_650M" / "euclidean" / sample_label / "canonical_nodes.csv",
        },
    ]


def graph_paths(panel_root: Path, sample_label: str) -> list[dict[str, Any]]:
    graph_root = panel_root / "graphs"
    rows: list[dict[str, Any]] = []
    for family, dirname in [
        ("mst", "hamming_mst"),
        ("rng_exact", "hamming_rng_exact"),
        ("knn_k05", "hamming_knn_k05"),
        ("knn_k50", "hamming_knn_k50"),
    ]:
        rows.append(
            {
                "graph_name": dirname,
                "metric_family": "hamming",
                "embedding_metric": "",
                "graph_family": family,
                "graph_dir": graph_root / "hamming" / sample_label / dirname,
            }
        )
    for metric in ["cityblock", "euclidean"]:
        for family, dirname in [
            ("mst", "embedding_mst"),
            ("rng_exact", "embedding_rng_exact"),
            ("knn_k05", "embedding_knn_k05"),
            ("knn_k50", "embedding_knn_k50"),
        ]:
            rows.append(
                {
                    "graph_name": f"embedding_{metric}_{family}",
                    "metric_family": "embedding",
                    "embedding_metric": metric,
                    "graph_family": family,
                    "graph_dir": graph_root / "esm2_650M" / metric / sample_label / dirname,
                }
            )
    return rows


def score_raw_distances(
    panel: str,
    seed: int,
    panel_root: Path,
    sample_label: str,
    matched: pd.DataFrame,
    D_ref: np.ndarray,
    out_dir: Path,
    pair_mode: str,
    sample_size: int,
    pair_seed: int,
) -> pd.DataFrame:
    accessions = matched["accession"].astype(str).tolist()
    rows = []
    for spec in raw_distance_paths(panel_root, sample_label):
        if not spec["matrix"].exists():
            log(f"Skipping missing raw distance matrix: {spec['matrix']}")
            continue
        log(f"Scoring raw distance baseline: {spec['baseline']}")
        D = subset_dense_matrix(spec["matrix"], spec["nodes"], accessions)
        score = spearman_score(D_ref, D, pair_mode, sample_size, pair_seed)
        rows.append(
            {
                "panel": panel,
                "seed": int(seed),
                "baseline": spec["baseline"],
                "metric_family": spec["metric_family"],
                "metric": spec["metric"],
                "n_matched": int(len(accessions)),
                **score,
            }
        )
        del D
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "raw_distance_nextstrain_correlations.csv", index=False)
    return frame


def score_graph_distances(
    panel: str,
    seed: int,
    panel_root: Path,
    sample_label: str,
    matched: pd.DataFrame,
    D_ref: np.ndarray,
    out_dir: Path,
    pair_mode: str,
    sample_size: int,
    pair_seed: int,
) -> pd.DataFrame:
    accessions = matched["accession"].astype(str).tolist()
    rows = []
    for spec in graph_paths(panel_root, sample_label):
        if not spec["graph_dir"].exists():
            log(f"Skipping missing graph: {spec['graph_dir']}")
            continue
        log(f"Scoring graph geodesic distances: {spec['graph_name']}")
        D, qc = graph_shortest_path_matrix(spec["graph_dir"], accessions)
        score = spearman_score(D_ref, D, pair_mode, sample_size, pair_seed)
        rows.append(
            {
                "panel": panel,
                "seed": int(seed),
                "graph_name": spec["graph_name"],
                "metric_family": spec["metric_family"],
                "embedding_metric": spec["embedding_metric"],
                "graph_family": spec["graph_family"],
                "n_matched": int(len(accessions)),
                **qc,
                **score,
            }
        )
        del D
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "graph_geodesic_nextstrain_correlations.csv", index=False)
    return frame


def load_raw_distance_by_name(panel_root: Path, sample_label: str, matched: pd.DataFrame) -> dict[str, np.ndarray]:
    accessions = matched["accession"].astype(str).tolist()
    out: dict[str, np.ndarray] = {}
    for spec in raw_distance_paths(panel_root, sample_label):
        if not spec["matrix"].exists():
            continue
        out[spec["baseline"]] = subset_dense_matrix(spec["matrix"], spec["nodes"], accessions)
    return out


def score_partial_correlations(
    panel: str,
    seed: int,
    panel_root: Path,
    sample_label: str,
    matched: pd.DataFrame,
    D_ref: np.ndarray,
    out_dir: Path,
    pair_mode: str,
    sample_size: int,
    pair_seed: int,
) -> pd.DataFrame:
    accessions = matched["accession"].astype(str).tolist()
    raw = load_raw_distance_by_name(panel_root, sample_label, matched)
    pair_i, pair_j = make_pair_indices(len(accessions), pair_mode, sample_size, pair_seed)
    raw_pair_count = int(len(pair_i))
    rows: list[dict[str, Any]] = []
    if "raw_hamming" not in raw:
        raise FileNotFoundError("Partial correlation requires raw_hamming distance matrix")

    for embedding_baseline in ["raw_esm2_cityblock", "raw_esm2_euclidean"]:
        if embedding_baseline not in raw:
            continue
        metric = embedding_baseline.removeprefix("raw_esm2_")
        for candidate_name, candidate_matrix, control_name, control_matrix, direction in [
            (embedding_baseline, raw[embedding_baseline], "raw_hamming", raw["raw_hamming"], "embedding_given_raw_hamming"),
            ("raw_hamming", raw["raw_hamming"], embedding_baseline, raw[embedding_baseline], "hamming_given_raw_embedding"),
        ]:
            log(f"Partial Spearman: {candidate_name} vs reference | {control_name}")
            score = partial_spearman_score(D_ref, candidate_matrix, control_matrix, pair_mode, sample_size, pair_seed)
            rows.append(
                {
                    "panel": panel,
                    "seed": int(seed),
                    "comparison_type": "raw_distance_partial",
                    "direction": direction,
                    "candidate": candidate_name,
                    "candidate_metric_family": "embedding" if candidate_name.startswith("raw_esm2") else "hamming",
                    "embedding_metric": metric,
                    "graph_family": "raw",
                    "control": control_name,
                    "control_metric_family": "hamming" if control_name == "raw_hamming" else "embedding",
                    "n_matched": int(len(accessions)),
                    **score,
                }
            )

    graph_specs = graph_paths(panel_root, sample_label)
    for spec in graph_specs:
        if not spec["graph_dir"].exists():
            log(f"Skipping missing graph for partial correlation: {spec['graph_dir']}")
            continue
        if spec["metric_family"] == "embedding":
            control_name = "raw_hamming"
            control_matrix = raw["raw_hamming"]
            direction = "embedding_graph_given_raw_hamming"
            embedding_metric = spec["embedding_metric"]
        else:
            # Symmetric question: does the Hamming graph carry tree signal after
            # controlling for raw ESM-2 distance?  Report once per embedding metric.
            if pair_mode == "sample":
                graph_values, qc = graph_shortest_path_pair_values(spec["graph_dir"], accessions, pair_i, pair_j)
            else:
                D_graph, qc = graph_shortest_path_matrix(spec["graph_dir"], accessions)
            for embedding_metric, control_name in [
                ("cityblock", "raw_esm2_cityblock"),
                ("euclidean", "raw_esm2_euclidean"),
            ]:
                if control_name not in raw:
                    continue
                log(f"Partial Spearman: {spec['graph_name']} vs reference | {control_name}")
                if pair_mode == "sample":
                    score = partial_spearman_from_vectors(
                        np.asarray(D_ref[pair_i, pair_j]),
                        graph_values,
                        np.asarray(raw[control_name][pair_i, pair_j]),
                        raw_pair_count,
                        pair_mode,
                    )
                else:
                    score = partial_spearman_score(D_ref, D_graph, raw[control_name], pair_mode, sample_size, pair_seed)
                rows.append(
                    {
                        "panel": panel,
                        "seed": int(seed),
                        "comparison_type": "graph_geodesic_partial",
                        "direction": "hamming_graph_given_raw_embedding",
                        "candidate": spec["graph_name"],
                        "candidate_metric_family": spec["metric_family"],
                        "embedding_metric": embedding_metric,
                        "graph_family": spec["graph_family"],
                        "control": control_name,
                        "control_metric_family": "embedding",
                        "n_matched": int(len(accessions)),
                        **qc,
                        **score,
                    }
                )
            if pair_mode == "sample":
                del graph_values
            else:
                del D_graph
            continue

        log(f"Partial Spearman: {spec['graph_name']} vs reference | {control_name}")
        if pair_mode == "sample":
            graph_values, qc = graph_shortest_path_pair_values(spec["graph_dir"], accessions, pair_i, pair_j)
            score = partial_spearman_from_vectors(
                np.asarray(D_ref[pair_i, pair_j]),
                graph_values,
                np.asarray(control_matrix[pair_i, pair_j]),
                raw_pair_count,
                pair_mode,
            )
            del graph_values
        else:
            D_graph, qc = graph_shortest_path_matrix(spec["graph_dir"], accessions)
            score = partial_spearman_score(D_ref, D_graph, control_matrix, pair_mode, sample_size, pair_seed)
            del D_graph
        rows.append(
            {
                "panel": panel,
                "seed": int(seed),
                "comparison_type": "graph_geodesic_partial",
                "direction": direction,
                "candidate": spec["graph_name"],
                "candidate_metric_family": spec["metric_family"],
                "embedding_metric": embedding_metric,
                "graph_family": spec["graph_family"],
                "control": control_name,
                "control_metric_family": "hamming",
                "n_matched": int(len(accessions)),
                **qc,
                **score,
            }
        )

    for matrix in raw.values():
        del matrix
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "partial_nextstrain_correlations.csv", index=False)
    return frame


def write_delta_summary(out_dir: Path) -> pd.DataFrame:
    graph_path = out_dir / "graph_geodesic_nextstrain_correlations.csv"
    raw_path = out_dir / "raw_distance_nextstrain_correlations.csv"
    rows: list[dict[str, Any]] = []
    if graph_path.exists():
        graph = pd.read_csv(graph_path)
        hamming = graph[graph["metric_family"] == "hamming"].copy()
        embedding = graph[graph["metric_family"] == "embedding"].copy()
        for _, emb in embedding.iterrows():
            base = hamming[hamming["graph_family"] == emb["graph_family"]]
            if base.empty:
                continue
            base_row = base.iloc[0]
            rows.append(
                {
                    "panel": emb["panel"],
                    "seed": int(emb["seed"]),
                    "comparison_type": "matched_graph_family",
                    "embedding_metric": emb["embedding_metric"],
                    "graph_family": emb["graph_family"],
                    "embedding_graph": emb["graph_name"],
                    "hamming_graph": base_row["graph_name"],
                    "rho_embedding": emb["spearman_rho"],
                    "rho_hamming": base_row["spearman_rho"],
                    "delta_rho_embedding_minus_hamming": emb["spearman_rho"] - base_row["spearman_rho"],
                    "embedding_n_pairs_used": emb["n_pairs_used"],
                    "hamming_n_pairs_used": base_row["n_pairs_used"],
                }
            )
    if raw_path.exists():
        raw = pd.read_csv(raw_path)
        hamming = raw[raw["baseline"] == "raw_hamming"]
        if not hamming.empty:
            hamming_rho = float(hamming.iloc[0]["spearman_rho"])
            for _, row in raw[raw["baseline"] != "raw_hamming"].iterrows():
                rows.append(
                    {
                        "panel": row["panel"],
                        "seed": int(row["seed"]),
                        "comparison_type": "raw_distance_baseline",
                        "embedding_metric": row["metric"],
                        "graph_family": "raw",
                        "embedding_graph": row["baseline"],
                        "hamming_graph": "raw_hamming",
                        "rho_embedding": row["spearman_rho"],
                        "rho_hamming": hamming_rho,
                        "delta_rho_embedding_minus_hamming": row["spearman_rho"] - hamming_rho,
                        "embedding_n_pairs_used": row["n_pairs_used"],
                        "hamming_n_pairs_used": hamming.iloc[0]["n_pairs_used"],
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "paired_delta_rho_summary.csv", index=False)
    return frame


def load_panel_tree_reference(workspace: Path, panel: str, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    ref_dir = workspace / panel / f"seed_{seed}" / "reference_tree"
    nodes_path = ref_dir / "D_reference_spike_nodes.csv"
    matrix_path = ref_dir / "D_reference_spike_float32.npy"
    if not nodes_path.exists() or not matrix_path.exists():
        raise FileNotFoundError(f"Missing panel-tree reference outputs under {ref_dir}")
    nodes = pd.read_csv(nodes_path)
    if "accession" not in nodes.columns:
        raise ValueError(f"{nodes_path} must contain accession")
    nodes = nodes.copy()
    nodes["accession"] = nodes["accession"].astype(str).str.strip()
    D_ref = np.load(matrix_path, mmap_mode="r")
    if D_ref.shape != (len(nodes), len(nodes)):
        raise ValueError(f"Reference matrix shape {D_ref.shape} does not match nodes rows {len(nodes):,}")
    return nodes, D_ref


def filter_panel_tree_reference_to_scored_accessions(
    panel_root: Path,
    sample_label: str,
    matched: pd.DataFrame,
    D_ref: np.ndarray,
    out_dir: Path,
) -> tuple[pd.DataFrame, np.ndarray]:
    required_sets = load_panel_sets(panel_root, sample_label)
    required_sets.update(scoring_node_sets(panel_root, sample_label))
    nonempty_sets = {key: value for key, value in required_sets.items() if value}
    if not nonempty_sets:
        return matched, D_ref

    common = set(matched["accession"].astype(str).str.strip())
    for values in nonempty_sets.values():
        common &= values

    keep = matched["accession"].astype(str).str.strip().isin(common).to_numpy()
    qc = {
        "n_reference_tree_accessions": int(len(matched)),
        "n_scoring_intersection": int(keep.sum()),
        "n_dropped_from_reference_tree": int((~keep).sum()),
        "required_sets": {key: int(len(value)) for key, value in nonempty_sets.items()},
    }
    (out_dir / "panel_tree_scoring_intersection_qc.json").write_text(
        json.dumps(qc, indent=2) + "\n", encoding="utf-8"
    )
    if keep.all():
        return matched, D_ref
    if not keep.any():
        raise ValueError(f"No panel-tree accessions remain after scoring intersection; see {out_dir}")
    idx = np.flatnonzero(keep)
    filtered = matched.loc[keep].copy().reset_index(drop=True)
    filtered["accession"].to_csv(out_dir / "matched_accessions.txt", index=False, header=False)
    log(
        f"Panel-tree scoring intersection: kept {len(filtered):,}/{len(matched):,} accessions "
        f"for matrices/graphs"
    )
    return filtered, np.asarray(D_ref[np.ix_(idx, idx)])


def run_panel_seed(
    panel: str,
    seed: int,
    sample_label: str,
    source_root: Path,
    workspace: Path,
    tree: ParsedTree | None,
    reference_source: str,
    stages: set[str],
    pair_mode: str,
    pair_sample_size: int,
    pair_seed: int,
) -> None:
    panel_root = source_root / panel / f"seed_{seed}"
    out_dir = workspace / panel / f"seed_{seed}"
    if not panel_root.exists():
        log(f"Skipping missing panel seed root: {panel_root}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    if reference_source == "panel_tree":
        matched, D_ref = load_panel_tree_reference(workspace, panel, seed)
        matched, D_ref = filter_panel_tree_reference_to_scored_accessions(
            panel_root=panel_root,
            sample_label=sample_label,
            matched=matched,
            D_ref=D_ref,
            out_dir=out_dir,
        )
    else:
        if tree is None:
            raise ValueError("Nextstrain reference source requires a parsed tree")
        if "match" in stages:
            matched, _qc = build_accession_mapping(panel, seed, panel_root, sample_label, tree, out_dir)
        else:
            matched = load_matched_mapping(out_dir)

        if matched.empty:
            log(f"{panel}/seed_{seed}: no matched accessions; skipping distance stages")
            return

        if "patristic" in stages:
            D_ref = compute_patristic_matrix(tree, matched, out_dir)
        else:
            D_ref = load_patristic(out_dir)

    if "raw" in stages:
        score_raw_distances(panel, seed, panel_root, sample_label, matched, D_ref, out_dir, pair_mode, pair_sample_size, pair_seed)
    if "graphs" in stages:
        score_graph_distances(panel, seed, panel_root, sample_label, matched, D_ref, out_dir, pair_mode, pair_sample_size, pair_seed)
    if "partial" in stages:
        score_partial_correlations(panel, seed, panel_root, sample_label, matched, D_ref, out_dir, pair_mode, pair_sample_size, pair_seed)
    if "summary" in stages:
        write_delta_summary(out_dir)


def aggregate_workspace(workspace: Path) -> None:
    raw_frames = []
    graph_frames = []
    delta_frames = []
    partial_frames = []
    for path in workspace.glob("*/seed_*/*nextstrain_correlations.csv"):
        if path.name.startswith("raw_"):
            raw_frames.append(pd.read_csv(path))
        elif path.name.startswith("graph_"):
            graph_frames.append(pd.read_csv(path))
    for path in workspace.glob("*/seed_*/paired_delta_rho_summary.csv"):
        delta_frames.append(pd.read_csv(path))
    for path in workspace.glob("*/seed_*/partial_nextstrain_correlations.csv"):
        partial_frames.append(pd.read_csv(path))
    if raw_frames:
        pd.concat(raw_frames, ignore_index=True).to_csv(workspace / "all_raw_distance_nextstrain_correlations.csv", index=False)
    if graph_frames:
        pd.concat(graph_frames, ignore_index=True).to_csv(workspace / "all_graph_geodesic_nextstrain_correlations.csv", index=False)
    if delta_frames:
        pd.concat(delta_frames, ignore_index=True).to_csv(workspace / "all_paired_delta_rho_summary.csv", index=False)
    if partial_frames:
        pd.concat(partial_frames, ignore_index=True).to_csv(workspace / "all_partial_nextstrain_correlations.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate 20k graph distances against the spike-only Nextstrain tree.")
    ap.add_argument("--workspace", type=Path, default=Path("analysis/cohort_validation/09_nextstrain_spike_tree_validation"))
    ap.add_argument("--source-root", type=Path, default=Path("analysis/cohort_validation/07_sampling_design_20k"))
    ap.add_argument("--nextstrain-url", default=DEFAULT_TREE_URL)
    ap.add_argument("--nextstrain-json", type=Path, default=None)
    ap.add_argument("--reference-source", choices=["nextstrain", "panel_tree"], default="nextstrain")
    ap.add_argument("--panels", default="random_full_dataset_seed42,monthly_stratified_full_dataset_seed42")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--sample-label", default="pool_n20000")
    ap.add_argument("--stages", default="download,parse,match,patristic,raw,graphs,partial,summary,aggregate")
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--pair-mode", choices=["all", "sample"], default="all")
    ap.add_argument("--pair-sample-size", type=int, default=5_000_000)
    ap.add_argument("--pair-seed", type=int, default=12345)
    args = ap.parse_args()

    args.workspace.mkdir(parents=True, exist_ok=True)
    tree_dir = args.workspace / "nextstrain_spike_only"
    tree_dir.mkdir(parents=True, exist_ok=True)
    stages = {stage.strip() for stage in args.stages.split(",") if stage.strip()}

    tree: ParsedTree | None = None
    if args.reference_source == "nextstrain":
        json_path = args.nextstrain_json or (tree_dir / f"nextstrain_neherlab_ncov_spike_only_{DATE_STAMP}.json")
        if "download" in stages and args.nextstrain_json is None:
            download_json(args.nextstrain_url, json_path, force=args.force_download)
        elif not json_path.exists():
            raise FileNotFoundError(f"Missing Nextstrain JSON: {json_path}")

        if "parse" in stages:
            tree = save_parsed_tree(json_path, tree_dir)
        else:
            tree = load_parsed_tree(tree_dir)

    panel_stages = {"match", "patristic", "raw", "graphs", "partial", "summary"}
    if stages & panel_stages:
        panels = [panel.strip() for panel in args.panels.split(",") if panel.strip()]
        seeds = parse_seed_list(args.seeds)
        for panel in panels:
            for seed in seeds:
                run_panel_seed(
                    panel=panel,
                    seed=seed,
                    sample_label=args.sample_label,
                    source_root=args.source_root,
                    workspace=args.workspace,
                    tree=tree,
                    reference_source=args.reference_source,
                    stages=stages,
                    pair_mode=args.pair_mode,
                    pair_sample_size=args.pair_sample_size,
                    pair_seed=args.pair_seed,
                )

    if "aggregate" in stages:
        aggregate_workspace(args.workspace)
        log(f"Wrote aggregate summaries under {args.workspace}")


if __name__ == "__main__":
    main()
