#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


COHORT_LABELS = {
    "A_early_US_ancestral_B1x": "A early ancestral/B.1.x",
    "B_alpha_clean": "B Alpha clean",
    "C_delta_dominant": "C Delta dominant",
    "D_early_omicron_BA1_BA2_BA2121": "D early Omicron",
    "E1_BA4_BA5_era": "E1 BA.4/BA.5",
    "E2_BQ_XBB_transition": "E2 BQ/XBB transition",
    "E3_XBB_JN1_era": "E3 XBB/JN.1",
    "F_beta_focused": "F Beta focused",
    "G_gamma_focused": "G Gamma focused",
}

GRAPH_LABELS = {
    "mst": "MST",
    "knn_5": "kNN-5",
    "knn_50": "kNN-50",
    "rng_exact": "RNG",
}

GRAPH_ORDER = {
    "MST": 0,
    "kNN-5": 1,
    "kNN-50": 2,
    "RNG": 3,
}

LABEL_PREFIXES = [
    "within_lineage_label",
    "lineage",
    "collection_month",
    "region",
]


def normalize_graph(row: pd.Series) -> str:
    graph_type = str(row["graph_type"])
    k = row.get("k")
    if graph_type.endswith("_mst"):
        return "MST"
    if graph_type.endswith("_rng_exact"):
        return "RNG"
    if graph_type.endswith("_knn") or graph_type.endswith("knn"):
        if pd.isna(k):
            return "kNN"
        return f"kNN-{int(float(k))}"
    return graph_type


def add_derived_columns(df: pd.DataFrame, metric_space: str) -> pd.DataFrame:
    out = df.copy()
    out["metric_space"] = metric_space
    out["cohort"] = out["cohort_dir"].map(COHORT_LABELS).fillna(out["cohort_dir"])
    out["graph_family"] = out.apply(normalize_graph, axis=1)
    out["graph_order"] = out["graph_family"].map(GRAPH_ORDER).fillna(99).astype(int)

    possible_edges = out["n_nodes"] * (out["n_nodes"] - 1) / 2
    out["edge_density"] = out["n_edges"] / possible_edges.replace(0, np.nan)

    for prefix in LABEL_PREFIXES:
        observed_col = f"{prefix}_observed_same_fraction"
        expected_col = f"{prefix}_nodepair_expected_same_fraction"
        if observed_col in out.columns and expected_col in out.columns:
            denom = 1 - out[expected_col]
            h = (out[observed_col] - out[expected_col]) / denom.replace(0, np.nan)
            out[f"{prefix}_coleman_h_nodepair"] = h
    return out


def fmt_int(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{int(round(float(value))):,}"


def fmt_float(value: object, digits: int = 3, signed: bool = False) -> str:
    if pd.isna(value):
        return ""
    spec = f"+.{digits}f" if signed else f".{digits}f"
    return format(float(value), spec)


def fmt_ratio(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2f}x"


def build_pairwise_table(combined: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["cohort_dir", "cohort", "graph_family", "graph_order"]
    metric_cols = [
        "n_nodes",
        "n_edges",
        "edge_density",
        "n_components",
        "giant_component_frac",
        "within_lineage_label_observed_same_fraction",
        "within_lineage_label_nodepair_expected_same_fraction",
        "within_lineage_label_nodepair_enrichment_ratio",
        "within_lineage_label_coleman_h_nodepair",
    ]
    slim = combined[key_cols + ["metric_space"] + metric_cols].copy()
    paired = slim.pivot_table(
        index=key_cols,
        columns="metric_space",
        values=metric_cols,
        aggfunc="first",
    )
    paired.columns = [f"{metric}_{space}" for metric, space in paired.columns]
    paired = paired.reset_index()

    paired["edge_ratio_embedding_over_hamming"] = (
        paired["n_edges_embedding"] / paired["n_edges_hamming"].replace(0, np.nan)
    )
    paired["density_delta_embedding_minus_hamming"] = paired["edge_density_embedding"] - paired["edge_density_hamming"]
    paired["within_lineage_h_delta_embedding_minus_hamming"] = (
        paired["within_lineage_label_coleman_h_nodepair_embedding"]
        - paired["within_lineage_label_coleman_h_nodepair_hamming"]
    )
    paired["within_lineage_enrichment_delta_embedding_minus_hamming"] = (
        paired["within_lineage_label_nodepair_enrichment_ratio_embedding"]
        - paired["within_lineage_label_nodepair_enrichment_ratio_hamming"]
    )
    paired = paired.sort_values(["cohort_dir", "graph_order"]).reset_index(drop=True)
    return paired


def build_presentation_table(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in paired.iterrows():
        rows.append(
            {
                "Cohort": row["cohort"],
                "Graph": row["graph_family"],
                "n": fmt_int(row["n_nodes_embedding"]),
                "ESM-2 edges": fmt_int(row["n_edges_embedding"]),
                "Hamming edges": fmt_int(row["n_edges_hamming"]),
                "Edge ratio ESM/Ham": fmt_ratio(row["edge_ratio_embedding_over_hamming"]),
                "ESM-2 same-lineage h": fmt_float(row["within_lineage_label_coleman_h_nodepair_embedding"]),
                "Hamming same-lineage h": fmt_float(row["within_lineage_label_coleman_h_nodepair_hamming"]),
                "Delta h": fmt_float(row["within_lineage_h_delta_embedding_minus_hamming"], signed=True),
                "ESM/Ham giant": (
                    f"{fmt_float(row['giant_component_frac_embedding'])}/"
                    f"{fmt_float(row['giant_component_frac_hamming'])}"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_graph_family_overview(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for graph in ["MST", "kNN-5", "kNN-50", "RNG"]:
        sub = paired[paired["graph_family"] == graph]
        if sub.empty:
            continue
        higher = int((sub["within_lineage_h_delta_embedding_minus_hamming"] > 0).sum())
        total = int(len(sub))
        rows.append(
            {
                "Graph": graph,
                "Mean ESM/Ham edge ratio": fmt_ratio(sub["edge_ratio_embedding_over_hamming"].mean()),
                "Mean Delta h": fmt_float(sub["within_lineage_h_delta_embedding_minus_hamming"].mean(), signed=True),
                "ESM-2 higher h": f"{higher}/{total}",
                "Typical structural read": summarize_graph_family(graph, sub),
            }
        )
    return pd.DataFrame(rows)


def summarize_graph_family(graph: str, sub: pd.DataFrame) -> str:
    mean_ratio = float(sub["edge_ratio_embedding_over_hamming"].mean())
    mean_delta = float(sub["within_lineage_h_delta_embedding_minus_hamming"].mean())
    if graph == "RNG":
        return "ESM-2 RNG is much sparser, usually with stronger same-lineage separation."
    if graph == "kNN-50":
        return "Dense local neighborhoods are slightly more same-lineage in Hamming."
    if graph == "kNN-5":
        return "Very local neighborhoods are similar, with cohort-specific direction."
    if graph == "MST":
        if mean_delta > 0:
            return "Tree backbone often shows stronger same-lineage structure in ESM-2."
        return "Tree backbone is mixed across cohorts."
    return f"Mean edge ratio {mean_ratio:.2f}, mean Delta h {mean_delta:+.3f}."


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]).replace("|", "\\|") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine ESM-2 and Hamming cohort graph summaries.")
    parser.add_argument(
        "--embedding-summary",
        type=Path,
        default=Path("analysis/cohort_validation/05_cross_cohort_comparison/esm2_main_n20000_embedding_graph_summary_with_rng.csv"),
    )
    parser.add_argument(
        "--hamming-summary",
        type=Path,
        default=Path("analysis/cohort_validation/05_cross_cohort_comparison/hamming_main_n20000_graph_summary.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis/cohort_validation/05_cross_cohort_comparison"),
    )
    args = parser.parse_args()

    embedding = add_derived_columns(pd.read_csv(args.embedding_summary), "embedding")
    hamming = add_derived_columns(pd.read_csv(args.hamming_summary), "hamming")
    combined = pd.concat([embedding, hamming], ignore_index=True, sort=False)
    combined = combined.sort_values(["cohort_dir", "graph_order", "metric_space"]).reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = args.out_dir / "main_n20000_embedding_hamming_combined_graph_metrics.csv"
    paired_path = args.out_dir / "main_n20000_embedding_vs_hamming_graph_comparison.csv"
    overview_csv_path = args.out_dir / "main_n20000_embedding_vs_hamming_graph_family_overview.csv"
    overview_md_path = args.out_dir / "main_n20000_embedding_vs_hamming_graph_family_overview.md"
    presentation_csv_path = args.out_dir / "main_n20000_embedding_vs_hamming_summary_table.csv"
    presentation_md_path = args.out_dir / "main_n20000_embedding_vs_hamming_summary_table.md"

    paired = build_pairwise_table(combined)
    overview = build_graph_family_overview(paired)
    presentation = build_presentation_table(paired)

    combined.to_csv(combined_path, index=False)
    paired.to_csv(paired_path, index=False)
    overview.to_csv(overview_csv_path, index=False)
    write_markdown_table(overview, overview_md_path)
    presentation.to_csv(presentation_csv_path, index=False)
    write_markdown_table(presentation, presentation_md_path)

    print(f"Wrote {combined_path}")
    print(f"Wrote {paired_path}")
    print(f"Wrote {overview_csv_path}")
    print(f"Wrote {overview_md_path}")
    print(f"Wrote {presentation_csv_path}")
    print(f"Wrote {presentation_md_path}")


if __name__ == "__main__":
    main()
