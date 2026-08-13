#!/usr/bin/env python3
"""Paired 2k NJ/tree-likeness evaluation for seed-42 20k distances.

This workflow is sequence-free.  It consumes existing distance matrices and
node metadata only:

1. validate-inputs
2. prepare-design
3. prepare-matrices
4. evaluate
5. summarize

The frozen design is a cohort-balanced subset from the common connected
19,057-node population.  Every representation is evaluated on the exact same
2,000 tips.  Graph distances come from full-graph shortest paths that were
already computed; this script only slices the shared 2k submatrices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
from scripts.graph_construction.build_panel_nj_distance_reference_trees import (  # noqa: E402
    compute_patristic_matrix,
)


DEFAULT_PANEL_ROOT = Path(
    "analysis/cohort_validation/07_sampling_design_20k/random_full_dataset_seed42/seed_42"
)
DEFAULT_DIRECTIONAL_ROOT = Path(
    "analysis/cohort_validation/24_seed42_20k_directional_intrinsic_distances/"
    "random_full_dataset_seed42/seed_42"
)
DEFAULT_COMMON_NODE_IDS = Path(
    "analysis/cohort_validation/15_seed42_20k/graph_box_counting/"
    "hamming_embedding_knn05_knn50_rng/common_node_ids.csv"
)
DEFAULT_ZERO_SKEW_ROOT = Path(
    "analysis/cohort_validation/15_seed42_20k/zero_skew_constructive_lower_bound/"
    "hamming_embedding_knn05_knn50_rng"
)
DEFAULT_KMEDOIDS_ROOT = Path(
    "analysis/cohort_validation/16_seed42_20k_kmedoids/"
    "random_full_dataset_seed42/seed_42"
)
DEFAULT_OUT_ROOT = Path(
    "analysis/cohort_validation/25_seed42_2k_paired_tree_geometry/"
    "random_full_dataset_seed42/seed_42"
)


@dataclass(frozen=True)
class MatrixSpec:
    key: str
    display_name: str
    representation_family: str
    graph: str
    fj_refinement: bool
    matrix_path: Path
    nodes_path: Path
    row_space: str
    source_detail: str


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def stable_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_csv_choices(value: str, allowed: Iterable[str]) -> list[str]:
    allowed_list = list(allowed)
    if value.strip().lower() == "all":
        return allowed_list
    selected = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(selected).difference(allowed_list))
    if unknown:
        raise ValueError(f"Unknown representation(s) {unknown}; allowed={allowed_list}")
    if not selected:
        raise ValueError("At least one representation is required")
    return selected


def matrix_specs(args: argparse.Namespace) -> dict[str, MatrixSpec]:
    panel = args.panel_root
    zero = args.zero_skew_root / "distance_caches"
    direction = args.directional_root / "distance_matrices/candidate_0p1_delta_0p01"
    hamming_nodes = panel / "graphs/hamming/pool_n20000/canonical_nodes.csv"
    embedding_nodes = panel / "graphs/esm2_650M/cityblock/pool_n20000/canonical_nodes.csv"
    common_nodes = args.common_node_ids
    return {
        "raw_hamming": MatrixSpec(
            key="raw_hamming",
            display_name="Hamming raw",
            representation_family="hamming",
            graph="none",
            fj_refinement=False,
            matrix_path=panel
            / "graphs/hamming/pool_n20000/distance_matrices/"
            "hamming_count-gap-state_all_states_uint16.npy",
            nodes_path=hamming_nodes,
            row_space="full20k",
            source_detail="raw pairwise Hamming matrix; diagonal is reset to zero after slicing",
        ),
        "hamming_knn5": MatrixSpec(
            key="hamming_knn5",
            display_name="Hamming kNN-5 geodesic",
            representation_family="hamming",
            graph="knn5",
            fj_refinement=False,
            matrix_path=zero / "hamming_knn_k05_weighted_shortest_path_common_float32.npy",
            nodes_path=common_nodes,
            row_space="common19057",
            source_detail="weighted shortest paths on full 20k Hamming kNN-5 graph, stored on common nodes",
        ),
        "hamming_knn50": MatrixSpec(
            key="hamming_knn50",
            display_name="Hamming kNN-50 geodesic",
            representation_family="hamming",
            graph="knn50",
            fj_refinement=False,
            matrix_path=zero / "hamming_knn_k50_weighted_shortest_path_common_float32.npy",
            nodes_path=common_nodes,
            row_space="common19057",
            source_detail="weighted shortest paths on full 20k Hamming kNN-50 graph, stored on common nodes",
        ),
        "hamming_rng": MatrixSpec(
            key="hamming_rng",
            display_name="Hamming RNG geodesic",
            representation_family="hamming",
            graph="rng",
            fj_refinement=False,
            matrix_path=args.kmedoids_root / "distance_rows/hamming_rng_candidate_to_all_float32.npy",
            nodes_path=hamming_nodes,
            row_space="full20k",
            source_detail="weighted shortest paths on full 20k Hamming RNG graph",
        ),
        "raw_embedding_cityblock": MatrixSpec(
            key="raw_embedding_cityblock",
            display_name="ESM-2 cityblock raw",
            representation_family="embedding",
            graph="none",
            fj_refinement=False,
            matrix_path=panel
            / "graphs/esm2_650M/cityblock/pool_n20000/distance_matrices/"
            "embedding_cityblock_float32.npy",
            nodes_path=embedding_nodes,
            row_space="full20k",
            source_detail="raw pairwise ESM-2 cityblock matrix; diagonal is reset to zero after slicing",
        ),
        "embedding_knn5": MatrixSpec(
            key="embedding_knn5",
            display_name="ESM-2 kNN-5 geodesic",
            representation_family="embedding",
            graph="knn5",
            fj_refinement=False,
            matrix_path=direction / "baseline/knn5_weighted_shortest_path_float32.npy",
            nodes_path=embedding_nodes,
            row_space="full20k",
            source_detail="weighted shortest paths on full 20k unrefined ESM-2 kNN-5 graph",
        ),
        "embedding_knn50": MatrixSpec(
            key="embedding_knn50",
            display_name="ESM-2 kNN-50 geodesic",
            representation_family="embedding",
            graph="knn50",
            fj_refinement=False,
            matrix_path=direction / "baseline/knn50_weighted_shortest_path_float32.npy",
            nodes_path=embedding_nodes,
            row_space="full20k",
            source_detail="weighted shortest paths on full 20k unrefined ESM-2 kNN-50 graph",
        ),
        "embedding_rng": MatrixSpec(
            key="embedding_rng",
            display_name="ESM-2 RNG geodesic",
            representation_family="embedding",
            graph="rng",
            fj_refinement=False,
            matrix_path=direction / "baseline/rng_weighted_shortest_path_float32.npy",
            nodes_path=embedding_nodes,
            row_space="full20k",
            source_detail="weighted shortest paths on full 20k unrefined ESM-2 RNG graph",
        ),
        "refined_embedding_knn5": MatrixSpec(
            key="refined_embedding_knn5",
            display_name="ESM-2 refined kNN-5 geodesic",
            representation_family="embedding",
            graph="knn5",
            fj_refinement=True,
            matrix_path=direction / "refined/knn5_weighted_shortest_path_float32.npy",
            nodes_path=embedding_nodes,
            row_space="full20k",
            source_detail="weighted shortest paths on full 20k directionally refined ESM-2 kNN-5 graph",
        ),
        "refined_embedding_knn50": MatrixSpec(
            key="refined_embedding_knn50",
            display_name="ESM-2 refined kNN-50 geodesic",
            representation_family="embedding",
            graph="knn50",
            fj_refinement=True,
            matrix_path=direction / "refined/knn50_weighted_shortest_path_float32.npy",
            nodes_path=embedding_nodes,
            row_space="full20k",
            source_detail="weighted shortest paths on full 20k directionally refined ESM-2 kNN-50 graph",
        ),
        "refined_embedding_rng": MatrixSpec(
            key="refined_embedding_rng",
            display_name="ESM-2 refined RNG geodesic",
            representation_family="embedding",
            graph="rng",
            fj_refinement=True,
            matrix_path=direction / "refined/rng_weighted_shortest_path_float32.npy",
            nodes_path=embedding_nodes,
            row_space="full20k",
            source_detail="weighted shortest paths on full 20k directionally refined ESM-2 RNG graph",
        ),
    }


def load_nodes(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if columns is None:
        nodes = pd.read_csv(path, low_memory=False)
    else:
        nodes = pd.read_csv(path, usecols=columns, low_memory=False)
    if "node_id" not in nodes.columns:
        raise ValueError(f"{path} must contain node_id")
    nodes = nodes.sort_values("node_id").reset_index(drop=True)
    expected = np.arange(len(nodes), dtype=np.int64)
    observed = nodes["node_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(observed, expected):
        raise ValueError(f"{path}: node_id is not row-aligned 0..n-1")
    return nodes


def load_canonical_nodes(args: argparse.Namespace) -> pd.DataFrame:
    columns = ["node_id", "accession", "cohort_id", "cohort_name"]
    hamming = load_nodes(
        args.panel_root / "graphs/hamming/pool_n20000/canonical_nodes.csv", columns=columns
    )
    embedding = load_nodes(
        args.panel_root / "graphs/esm2_650M/cityblock/pool_n20000/canonical_nodes.csv",
        columns=columns,
    )
    for col in columns:
        if not hamming[col].astype(str).equals(embedding[col].astype(str)):
            raise ValueError(f"Hamming and embedding canonical nodes differ in column {col}")
    hamming["accession"] = hamming["accession"].astype(str).str.strip()
    hamming["cohort_id"] = hamming["cohort_id"].astype(str).str.strip()
    hamming["cohort_name"] = hamming["cohort_name"].astype(str).str.strip()
    if hamming["accession"].duplicated().any():
        raise ValueError("canonical accession values are not unique")
    return hamming


def load_common_nodes(args: argparse.Namespace, canonical: pd.DataFrame) -> pd.DataFrame:
    common = pd.read_csv(args.common_node_ids, usecols=["node_id"])
    common["node_id"] = common["node_id"].astype(int)
    if common["node_id"].duplicated().any():
        raise ValueError(f"{args.common_node_ids}: duplicate node_id values")
    if common["node_id"].min() < 0 or common["node_id"].max() >= len(canonical):
        raise ValueError(f"{args.common_node_ids}: node_id outside canonical range")
    common = common.merge(canonical, on="node_id", how="left", validate="one_to_one")
    if common["accession"].isna().any():
        raise ValueError("common node table did not fully map to canonical nodes")
    common = common.sort_values("node_id").reset_index(drop=True)
    common["common_row"] = np.arange(len(common), dtype=int)
    return common


def matrix_basic_report(spec: MatrixSpec) -> dict[str, Any]:
    if not spec.matrix_path.exists():
        raise FileNotFoundError(f"Missing matrix for {spec.key}: {spec.matrix_path}")
    if not spec.nodes_path.exists():
        raise FileNotFoundError(f"Missing nodes for {spec.key}: {spec.nodes_path}")
    D = np.load(spec.matrix_path, mmap_mode="r")
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"{spec.matrix_path}: expected square matrix, observed {D.shape}")
    return {
        "key": spec.key,
        "display_name": spec.display_name,
        "representation_family": spec.representation_family,
        "graph": spec.graph,
        "fj_refinement": bool(spec.fj_refinement),
        "row_space": spec.row_space,
        "source_detail": spec.source_detail,
        "matrix_shape": [int(D.shape[0]), int(D.shape[1])],
        "matrix_dtype": str(D.dtype),
        "matrix_signature": file_signature(spec.matrix_path),
        "nodes_signature": file_signature(spec.nodes_path),
    }


def validate_inputs(args: argparse.Namespace, selected_keys: list[str]) -> dict[str, Any]:
    canonical = load_canonical_nodes(args)
    common = load_common_nodes(args, canonical)
    specs = matrix_specs(args)
    reports = {key: matrix_basic_report(specs[key]) for key in selected_keys}
    full_n = len(canonical)
    common_n = len(common)
    for key in selected_keys:
        report = reports[key]
        expected_n = full_n if report["row_space"] == "full20k" else common_n
        if report["matrix_shape"] != [expected_n, expected_n]:
            raise ValueError(
                f"{key}: expected matrix shape {(expected_n, expected_n)} for "
                f"{report['row_space']}, observed {report['matrix_shape']}"
            )
    cohort_counts = (
        common.groupby(["cohort_id", "cohort_name"], dropna=False)
        .size()
        .reset_index(name="n_common_nodes")
        .sort_values("cohort_id")
    )
    if cohort_counts["n_common_nodes"].min() < args.n_tips // len(cohort_counts):
        raise ValueError("At least one cohort is too small for balanced sampling")
    payload = {
        "validated_at_unix": time.time(),
        "sequence_free_validation": True,
        "panel_root": str(args.panel_root),
        "directional_root": str(args.directional_root),
        "zero_skew_root": str(args.zero_skew_root),
        "kmedoids_root": str(args.kmedoids_root),
        "common_node_ids": str(args.common_node_ids),
        "n_full_nodes": int(full_n),
        "n_common_nodes": int(common_n),
        "n_requested_tips": int(args.n_tips),
        "selected_representations": selected_keys,
        "cohort_counts": cohort_counts.to_dict(orient="records"),
        "representations": reports,
    }
    write_json(args.out_root / "input_validation.json", payload)
    log(f"Wrote input validation: {args.out_root / 'input_validation.json'}")
    return payload


def per_cohort_targets(cohorts: list[str], n_tips: int) -> dict[str, int]:
    if not cohorts:
        raise ValueError("No cohorts available for sampling")
    base = n_tips // len(cohorts)
    remainder = n_tips % len(cohorts)
    return {cohort: base + (1 if index < remainder else 0) for index, cohort in enumerate(cohorts)}


def cohort_seed(base_seed: int, cohort_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{cohort_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def select_balanced_tips(common: pd.DataFrame, n_tips: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohorts = sorted(common["cohort_id"].astype(str).unique().tolist())
    targets = per_cohort_targets(cohorts, n_tips)
    selected_parts: list[pd.DataFrame] = []
    allocation_rows: list[dict[str, Any]] = []
    for cohort in cohorts:
        group = common[common["cohort_id"].astype(str) == cohort].sort_values("node_id").reset_index(drop=True)
        target = int(targets[cohort])
        if len(group) < target:
            raise ValueError(f"Cohort {cohort} has {len(group)} common nodes but target is {target}")
        rng = np.random.default_rng(cohort_seed(seed, cohort))
        chosen = np.sort(rng.choice(np.arange(len(group)), size=target, replace=False))
        chosen_group = group.iloc[chosen].copy()
        selected_parts.append(chosen_group)
        allocation_rows.append(
            {
                "cohort_id": cohort,
                "cohort_name": str(group["cohort_name"].iloc[0]),
                "n_common_nodes": int(len(group)),
                "n_selected": target,
                "sampling_seed": int(cohort_seed(seed, cohort) % (2**63 - 1)),
            }
        )
    selected = pd.concat(selected_parts, ignore_index=True).sort_values("node_id").reset_index(drop=True)
    selected.insert(0, "tip_row", np.arange(len(selected), dtype=int))
    return selected, pd.DataFrame(allocation_rows)


def prepare_design(args: argparse.Namespace, selected_keys: list[str]) -> dict[str, Any]:
    validate_inputs(args, selected_keys)
    canonical = load_canonical_nodes(args)
    common = load_common_nodes(args, canonical)
    selected, allocation = select_balanced_tips(common, args.n_tips, args.selection_seed)
    design_dir = args.out_root / "design"
    design_dir.mkdir(parents=True, exist_ok=True)
    selected_path = design_dir / "selected_tips.csv"
    allocation_path = design_dir / "cohort_allocation.csv"
    selected.to_csv(selected_path, index=False)
    allocation.to_csv(allocation_path, index=False)
    quadruples = make_gromov_quadruples(args.n_tips, args.gromov_samples, args.gromov_seed)
    quadruple_path = design_dir / "gromov_quadruples_int32.npy"
    np.save(quadruple_path, quadruples)
    fingerprint_payload = {
        "n_tips": int(args.n_tips),
        "selection_seed": int(args.selection_seed),
        "gromov_samples": int(args.gromov_samples),
        "gromov_seed": int(args.gromov_seed),
        "selected_node_ids": selected["node_id"].astype(int).tolist(),
        "selected_representations": selected_keys,
    }
    manifest = {
        "prepared_at_unix": time.time(),
        "sequence_free_design": True,
        "sample_source": "common connected 19,057-node population",
        "selection_policy": "balanced by cohort_id; remainder assigned to sorted cohort_id order",
        "within_cohort_policy": "deterministic random sample without replacement using sha256-derived per-cohort seeds",
        "final_tip_order_policy": "ascending original node_id",
        "n_tips": int(len(selected)),
        "selection_seed": int(args.selection_seed),
        "gromov_samples": int(args.gromov_samples),
        "gromov_seed": int(args.gromov_seed),
        "selected_tips": str(selected_path),
        "cohort_allocation": str(allocation_path),
        "gromov_quadruples": str(quadruple_path),
        "design_fingerprint": stable_fingerprint(fingerprint_payload),
    }
    write_json(design_dir / "design_manifest.json", manifest)
    log(f"Wrote frozen 2k design: {selected_path}")
    return manifest


def load_design(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    design_dir = args.out_root / "design"
    selected_path = design_dir / "selected_tips.csv"
    manifest_path = design_dir / "design_manifest.json"
    if not selected_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("Missing design; run --stages prepare-design first")
    selected = pd.read_csv(selected_path)
    manifest = read_json(manifest_path)
    if len(selected) != int(manifest["n_tips"]):
        raise ValueError("selected_tips.csv row count does not match design manifest")
    if not np.array_equal(selected["tip_row"].to_numpy(dtype=int), np.arange(len(selected), dtype=int)):
        raise ValueError("selected_tips.csv tip_row is not row-aligned")
    return selected, manifest


def source_indices_for_spec(spec: MatrixSpec, selected: pd.DataFrame) -> np.ndarray:
    if spec.row_space == "full20k":
        return selected["node_id"].to_numpy(dtype=np.int64)
    if spec.row_space == "common19057":
        if "common_row" not in selected.columns:
            raise ValueError("selected tips must contain common_row for common-row matrices")
        return selected["common_row"].to_numpy(dtype=np.int64)
    raise ValueError(f"Unknown row_space: {spec.row_space}")


def slice_distance_matrix(spec: MatrixSpec, selected: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.load(spec.matrix_path, mmap_mode="r")
    idx = source_indices_for_spec(spec, selected)
    if idx.min() < 0 or idx.max() >= source.shape[0]:
        raise ValueError(f"{spec.key}: selected index outside matrix shape {source.shape}")
    D = np.asarray(source[np.ix_(idx, idx)], dtype=np.float64)
    if D.shape[0] != D.shape[1]:
        raise ValueError(f"{spec.key}: sliced matrix is not square")
    diag_before = np.diag(D).copy()
    np.fill_diagonal(D, 0.0)
    asym = D - D.T
    max_asym = float(np.max(np.abs(asym))) if D.size else 0.0
    if max_asym > 1e-5:
        raise ValueError(f"{spec.key}: 2k submatrix is asymmetric by {max_asym}")
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    offdiag = ~np.eye(D.shape[0], dtype=bool)
    if not np.isfinite(D[offdiag]).all():
        raise ValueError(f"{spec.key}: non-finite off-diagonal distance in selected 2k matrix")
    if np.any(D < 0):
        raise ValueError(f"{spec.key}: negative distance detected")
    qc = {
        "source_matrix": str(spec.matrix_path),
        "source_matrix_signature": file_signature(spec.matrix_path),
        "source_nodes": str(spec.nodes_path),
        "source_nodes_signature": file_signature(spec.nodes_path),
        "source_row_space": spec.row_space,
        "source_dtype": str(source.dtype),
        "source_shape": [int(source.shape[0]), int(source.shape[1])],
        "n_tips": int(D.shape[0]),
        "max_absolute_asymmetry_before_symmetrize": max_asym,
        "diagonal_min_before_reset": float(np.nanmin(diag_before)) if diag_before.size else math.nan,
        "diagonal_max_before_reset": float(np.nanmax(diag_before)) if diag_before.size else math.nan,
        "distance_min": float(np.min(D)) if D.size else math.nan,
        "distance_max": float(np.max(D)) if D.size else math.nan,
        "offdiag_zero_pairs": int(np.count_nonzero(D[offdiag] == 0.0)),
    }
    return D.astype(np.float32, copy=False), qc


def prepare_matrices(args: argparse.Namespace, selected_keys: list[str]) -> None:
    selected, design_manifest = load_design(args)
    specs = matrix_specs(args)
    rows: list[dict[str, Any]] = []
    for key in selected_keys:
        spec = specs[key]
        out_dir = args.out_root / "matrices" / key
        matrix_path = out_dir / "D_input_float32.npy"
        manifest_path = out_dir / "matrix_manifest.json"
        fingerprint_payload = {
            "design_fingerprint": design_manifest["design_fingerprint"],
            "representation": key,
            "source_matrix_signature": file_signature(spec.matrix_path),
            "source_nodes_signature": file_signature(spec.nodes_path),
        }
        fingerprint = stable_fingerprint(fingerprint_payload)
        if matrix_path.exists() and manifest_path.exists() and not args.overwrite_matrices:
            manifest = read_json(manifest_path)
            if manifest.get("matrix_fingerprint") == fingerprint:
                log(f"Using existing 2k matrix: {matrix_path}")
                rows.append(summary_row_from_matrix_manifest(key, spec, manifest))
                continue
        log(f"Slicing 2k matrix: {key}")
        D, qc = slice_distance_matrix(spec, selected)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(matrix_path, D)
        manifest = {
            "prepared_at_unix": time.time(),
            "sequence_free_matrix_slice": True,
            "representation": key,
            "display_name": spec.display_name,
            "representation_family": spec.representation_family,
            "graph": spec.graph,
            "fj_refinement": bool(spec.fj_refinement),
            "source_detail": spec.source_detail,
            "design_fingerprint": design_manifest["design_fingerprint"],
            "matrix_fingerprint": fingerprint,
            "matrix_path": str(matrix_path),
            "matrix_dtype": "float32",
            **qc,
        }
        write_json(manifest_path, manifest)
        rows.append(summary_row_from_matrix_manifest(key, spec, manifest))
    pd.DataFrame(rows).to_csv(args.out_root / "matrices" / "matrix_summary.csv", index=False)
    log(f"Wrote matrix summary: {args.out_root / 'matrices' / 'matrix_summary.csv'}")


def summary_row_from_matrix_manifest(key: str, spec: MatrixSpec, manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "representation": key,
        "display_name": spec.display_name,
        "representation_family": spec.representation_family,
        "graph": spec.graph,
        "fj_refinement": bool(spec.fj_refinement),
        "matrix_path": manifest["matrix_path"],
        "n_tips": manifest["n_tips"],
        "source_dtype": manifest["source_dtype"],
        "distance_max": manifest["distance_max"],
        "offdiag_zero_pairs": manifest["offdiag_zero_pairs"],
    }


def make_gromov_quadruples(n: int, n_samples: int, seed: int) -> np.ndarray:
    if n < 4:
        return np.empty((0, 4), dtype=np.int32)
    rng = np.random.default_rng(seed)
    out = np.empty((n_samples, 4), dtype=np.int32)
    for row in range(n_samples):
        out[row] = rng.choice(n, size=4, replace=False)
    return out


def upper_triangle_vectors(D: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if D.shape != T.shape:
        raise ValueError(f"Matrix shapes differ: D={D.shape}, T={T.shape}")
    i, j = np.triu_indices(D.shape[0], k=1)
    return np.asarray(D[i, j], dtype=np.float64), np.asarray(T[i, j], dtype=np.float64)


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


def gromov_from_quadruples(D: np.ndarray, quadruples: np.ndarray) -> dict[str, Any]:
    if D.shape[0] < 4 or quadruples.size == 0:
        return {
            "gromov_delta": math.nan,
            "gromov_delta_norm_iqr": math.nan,
            "gromov_samples_requested": int(len(quadruples)),
            "gromov_samples_used": 0,
            "distance_iqr": math.nan,
        }
    vals = np.column_stack(
        [
            D[quadruples[:, 0], quadruples[:, 1]] + D[quadruples[:, 2], quadruples[:, 3]],
            D[quadruples[:, 0], quadruples[:, 2]] + D[quadruples[:, 1], quadruples[:, 3]],
            D[quadruples[:, 0], quadruples[:, 3]] + D[quadruples[:, 1], quadruples[:, 2]],
        ]
    ).astype(np.float64, copy=False)
    finite = np.isfinite(vals).all(axis=1)
    if not finite.any():
        delta = math.nan
        used = 0
    else:
        vals = vals[finite]
        vals.sort(axis=1)
        delta = float(np.max(0.5 * (vals[:, 2] - vals[:, 1])))
        used = int(vals.shape[0])
    d_values, _ = upper_triangle_vectors(D, D)
    iqr = robust_iqr(d_values)
    return {
        "gromov_delta": delta,
        "gromov_delta_norm_iqr": float(delta / iqr) if used and np.isfinite(iqr) and iqr != 0.0 else math.nan,
        "gromov_samples_requested": int(len(quadruples)),
        "gromov_samples_used": used,
        "distance_iqr": float(iqr) if np.isfinite(iqr) else math.nan,
    }


def tree_branch_lengths(tree_kind: str, tree: Any) -> list[float]:
    lengths: list[float] = []
    if tree_kind == "skbio":
        for node in tree.postorder():
            if node.length is not None:
                lengths.append(float(node.length))
    else:
        for clade in tree.find_clades():
            if clade.branch_length is not None:
                lengths.append(float(clade.branch_length))
    return lengths


def branch_length_stats(lengths: list[float]) -> dict[str, Any]:
    negatives = [value for value in lengths if value < 0.0]
    return {
        "n_branches": int(len(lengths)),
        "n_negative_branches": int(len(negatives)),
        "fraction_negative_branches": float(len(negatives) / len(lengths)) if lengths else math.nan,
        "min_branch_length": float(min(lengths)) if lengths else math.nan,
        "sum_abs_negative_branch_lengths": float(sum(abs(value) for value in negatives)),
        "min_negative_branch_length": float(min(negatives)) if negatives else 0.0,
    }


def tree_fit_metrics(D: np.ndarray, T: np.ndarray) -> dict[str, Any]:
    d, t = upper_triangle_vectors(D, T)
    finite = np.isfinite(d) & np.isfinite(t)
    if not finite.all():
        raise ValueError("Tree-fit metrics require finite input and tree distances")
    total_pairs = int(d.size)
    d_zero = d == 0.0
    t_zero = t == 0.0
    d_pos = d > 0.0
    t_pos = t > 0.0
    denom = float(np.sum(d * d))
    rsd = float(np.sqrt(np.sum((d - t) ** 2) / denom)) if denom > 0.0 else math.nan
    rho = spearmanr(d, t).statistic
    scale_denom = float(np.sum(t * t))
    s_star = float(np.sum(d * t) / scale_denom) if scale_denom > 0.0 else math.nan
    positive = d_pos & t_pos & np.isfinite(s_star) & (s_star > 0.0)
    if positive.any():
        ratio = d[positive] / (s_star * t[positive])
        delta_max = float(np.maximum(ratio, 1.0 / ratio).max())
        log_delta = float(np.abs(np.log(ratio)).max())
    else:
        delta_max = math.nan
        log_delta = math.nan
    zero_mismatch = int(np.count_nonzero((d_zero & t_pos) | (d_pos & t_zero)))
    return {
        "n_pairs": total_pairs,
        "spearman_rho": float(rho) if np.isfinite(rho) else math.nan,
        "relative_square_deviation": rsd,
        "tree_dist_scale_s_star": s_star,
        "max_multiplicative_distortion_positive_pairs": delta_max,
        "log_max_multiplicative_distortion_positive_pairs": log_delta,
        "positive_pair_count_for_multiplicative_distortion": int(np.count_nonzero(positive)),
        "zero_pairs_d0_t0": int(np.count_nonzero(d_zero & t_zero)),
        "zero_pairs_d0_t_positive": int(np.count_nonzero(d_zero & t_pos)),
        "zero_pairs_d_positive_t0": int(np.count_nonzero(d_pos & t_zero)),
        "positive_pairs_d_positive_t_positive": int(np.count_nonzero(d_pos & t_pos)),
        "multiplicative_zero_mismatch_pairs": zero_mismatch,
        "multiplicative_distortion_status": "zero_mismatch_present" if zero_mismatch else "finite_on_positive_pairs",
    }


def evaluate_one(args: argparse.Namespace, key: str, spec: MatrixSpec) -> dict[str, Any]:
    selected, design_manifest = load_design(args)
    matrix_dir = args.out_root / "matrices" / key
    matrix_path = matrix_dir / "D_input_float32.npy"
    matrix_manifest_path = matrix_dir / "matrix_manifest.json"
    if not matrix_path.exists() or not matrix_manifest_path.exists():
        raise FileNotFoundError(f"Missing prepared matrix for {key}; run --stages prepare-matrices")
    matrix_manifest = read_json(matrix_manifest_path)
    out_dir = args.out_root / "tree_evaluation" / key
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists() and not args.overwrite_evaluation:
        metrics = read_json(metrics_path)
        log(f"Using existing tree metrics: {metrics_path}")
        return metrics
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Evaluating NJ/tree fit: {key}")
    D = np.load(matrix_path, mmap_mode="r")
    D64 = np.asarray(D, dtype=np.float64)
    labels = selected["accession"].astype(str).tolist()
    tree_kind, tree = build_nj_tree(D64, labels, prefer=args.prefer_tree_builder)
    before_stats = branch_length_stats(tree_branch_lengths(tree_kind, tree))
    unclipped_newick = out_dir / "tree_unclipped.nwk"
    save_newick(tree_kind, tree, str(unclipped_newick))
    clip_negative_branch_lengths(tree_kind, tree)
    after_stats = branch_length_stats(tree_branch_lengths(tree_kind, tree))
    clipped_newick = out_dir / "tree_clipped.nwk"
    save_newick(tree_kind, tree, str(clipped_newick))
    patristic_path, patristic_nodes, patristic_qc = compute_patristic_matrix(
        newick_path=clipped_newick,
        accessions=labels,
        out_dir=out_dir,
        matrix_name="D_tree_clipped_float32.npy",
        nodes_name="D_tree_clipped_nodes.csv",
        qc_name="D_tree_clipped_qc.json",
        block_size=args.patristic_block_size,
        overwrite=args.overwrite_evaluation,
    )
    T = np.load(patristic_path, mmap_mode="r")
    fit = tree_fit_metrics(D64, np.asarray(T, dtype=np.float64))
    quadruples = np.load(args.out_root / "design" / "gromov_quadruples_int32.npy", mmap_mode="r")
    gromov = gromov_from_quadruples(D64, np.asarray(quadruples, dtype=np.int32))
    metrics = {
        "evaluated_at_unix": time.time(),
        "sequence_free_evaluation": True,
        "representation": key,
        "display_name": spec.display_name,
        "representation_family": spec.representation_family,
        "graph": spec.graph,
        "fj_refinement": bool(spec.fj_refinement),
        "source_detail": spec.source_detail,
        "n_tips": int(D.shape[0]),
        "design_fingerprint": design_manifest["design_fingerprint"],
        "matrix_fingerprint": matrix_manifest["matrix_fingerprint"],
        "input_matrix": str(matrix_path),
        "tree_builder_backend": tree_kind,
        "unclipped_newick": str(unclipped_newick),
        "clipped_newick": str(clipped_newick),
        "patristic_matrix": str(patristic_path),
        "patristic_nodes": str(patristic_nodes),
        "patristic_qc": patristic_qc,
        "branch_lengths_before_clip": before_stats,
        "branch_lengths_after_clip": after_stats,
        **fit,
        **gromov,
    }
    write_json(metrics_path, metrics)
    return metrics


def evaluate_all(args: argparse.Namespace, selected_keys: list[str]) -> None:
    specs = matrix_specs(args)
    rows = [evaluate_one(args, key, specs[key]) for key in selected_keys]
    write_metrics_table(args, rows)


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    before = metrics.get("branch_lengths_before_clip", {})
    after = metrics.get("branch_lengths_after_clip", {})
    row = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
    for prefix, stats in [("before_clip", before), ("after_clip", after)]:
        for key, value in stats.items():
            row[f"{prefix}_{key}"] = value
    return row


def write_metrics_table(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    summary_dir = args.out_root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([flatten_metrics(row) for row in rows])
    metrics_path = summary_dir / "tree_geometry_metrics.csv"
    frame.to_csv(metrics_path, index=False)
    contrasts = build_contrasts(frame)
    contrasts_path = summary_dir / "planned_contrasts.csv"
    contrasts.to_csv(contrasts_path, index=False)
    write_json(
        summary_dir / "summary_manifest.json",
        {
            "summarized_at_unix": time.time(),
            "metrics_path": str(metrics_path),
            "planned_contrasts_path": str(contrasts_path),
            "n_representations": int(len(frame)),
            "lower_is_better_metrics": [
                "relative_square_deviation",
                "max_multiplicative_distortion_positive_pairs",
                "gromov_delta",
                "gromov_delta_norm_iqr",
                "before_clip_n_negative_branches",
                "before_clip_sum_abs_negative_branch_lengths",
            ],
            "higher_is_better_metrics": ["spearman_rho"],
        },
    )
    log(f"Wrote metrics table: {metrics_path}")
    log(f"Wrote planned contrasts: {contrasts_path}")


def build_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    by_key = {str(row["representation"]): row for _, row in frame.iterrows()}
    planned = [
        ("hamming_rng_vs_raw", "raw_hamming", "hamming_rng"),
        ("embedding_rng_vs_raw", "raw_embedding_cityblock", "embedding_rng"),
        ("hamming_knn5_vs_raw", "raw_hamming", "hamming_knn5"),
        ("hamming_knn50_vs_raw", "raw_hamming", "hamming_knn50"),
        ("embedding_knn5_vs_raw", "raw_embedding_cityblock", "embedding_knn5"),
        ("embedding_knn50_vs_raw", "raw_embedding_cityblock", "embedding_knn50"),
        ("refined_knn5_vs_baseline_knn5", "embedding_knn5", "refined_embedding_knn5"),
        ("refined_knn50_vs_baseline_knn50", "embedding_knn50", "refined_embedding_knn50"),
        ("refined_rng_vs_baseline_rng", "embedding_rng", "refined_embedding_rng"),
    ]
    metrics = [
        "spearman_rho",
        "relative_square_deviation",
        "max_multiplicative_distortion_positive_pairs",
        "gromov_delta",
        "gromov_delta_norm_iqr",
        "before_clip_n_negative_branches",
        "before_clip_sum_abs_negative_branch_lengths",
        "multiplicative_zero_mismatch_pairs",
    ]
    rows: list[dict[str, Any]] = []
    for name, baseline, candidate in planned:
        if baseline not in by_key or candidate not in by_key:
            continue
        out: dict[str, Any] = {
            "contrast": name,
            "baseline": baseline,
            "candidate": candidate,
        }
        for metric in metrics:
            a = pd.to_numeric(pd.Series([by_key[baseline].get(metric)]), errors="coerce").iloc[0]
            b = pd.to_numeric(pd.Series([by_key[candidate].get(metric)]), errors="coerce").iloc[0]
            out[f"{metric}_baseline"] = a
            out[f"{metric}_candidate"] = b
            out[f"{metric}_delta_candidate_minus_baseline"] = b - a
            out[f"{metric}_relative_change"] = ((b - a) / abs(a)) if pd.notna(a) and a != 0 else math.nan
        rows.append(out)
    return pd.DataFrame(rows)


def summarize_existing(args: argparse.Namespace, selected_keys: list[str]) -> None:
    rows: list[dict[str, Any]] = []
    for key in selected_keys:
        metrics_path = args.out_root / "tree_evaluation" / key / "metrics.json"
        if metrics_path.exists():
            rows.append(read_json(metrics_path))
    if not rows:
        raise FileNotFoundError("No metrics.json files found; run --stages evaluate first")
    write_metrics_table(args, rows)


def run_stages(args: argparse.Namespace) -> None:
    specs = matrix_specs(args)
    selected_keys = parse_csv_choices(args.representations, specs.keys())
    stages = {stage.strip() for stage in args.stages.split(",") if stage.strip()}
    if "all" in stages:
        stages = {"validate-inputs", "prepare-design", "prepare-matrices", "evaluate", "summarize"}
    valid = {"validate-inputs", "prepare-design", "prepare-matrices", "evaluate", "summarize"}
    unknown = stages.difference(valid)
    if unknown:
        raise ValueError(f"Unknown stage(s): {sorted(unknown)}")
    args.out_root.mkdir(parents=True, exist_ok=True)
    if "validate-inputs" in stages:
        validate_inputs(args, selected_keys)
    if "prepare-design" in stages:
        prepare_design(args, selected_keys)
    if "prepare-matrices" in stages:
        prepare_matrices(args, selected_keys)
    if "evaluate" in stages:
        evaluate_all(args, selected_keys)
    if "summarize" in stages and "evaluate" not in stages:
        summarize_existing(args, selected_keys)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel-root", type=Path, default=DEFAULT_PANEL_ROOT)
    ap.add_argument("--directional-root", type=Path, default=DEFAULT_DIRECTIONAL_ROOT)
    ap.add_argument("--common-node-ids", type=Path, default=DEFAULT_COMMON_NODE_IDS)
    ap.add_argument("--zero-skew-root", type=Path, default=DEFAULT_ZERO_SKEW_ROOT)
    ap.add_argument("--kmedoids-root", type=Path, default=DEFAULT_KMEDOIDS_ROOT)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--stages", default="validate-inputs,prepare-design")
    ap.add_argument("--representations", default="all")
    ap.add_argument("--n-tips", type=int, default=2000)
    ap.add_argument("--selection-seed", type=int, default=42)
    ap.add_argument("--gromov-samples", type=int, default=50000)
    ap.add_argument("--gromov-seed", type=int, default=42)
    ap.add_argument("--prefer-tree-builder", default="skbio", choices=["auto", "skbio", "biopython"])
    ap.add_argument("--patristic-block-size", type=int, default=128)
    ap.add_argument("--overwrite-matrices", action="store_true")
    ap.add_argument("--overwrite-evaluation", action="store_true")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    run_stages(args)


if __name__ == "__main__":
    main()
