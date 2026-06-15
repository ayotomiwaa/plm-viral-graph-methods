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

GRAPH_ORDER = {"MST": 0, "kNN-5": 1, "kNN-50": 2, "RNG": 3}
MISSING_LABELS = {"", "nan", "none", "null", "nat"}


def normalize_graph(row: pd.Series) -> str:
    graph_type = str(row["graph_type"])
    k = row.get("k")
    if graph_type.endswith("_mst"):
        return "MST"
    if graph_type.endswith("_rng_exact"):
        return "RNG"
    if graph_type.endswith("_knn") or graph_type.endswith("knn"):
        return f"kNN-{int(float(k))}" if pd.notna(k) else "kNN"
    return graph_type


def endpoint_expected_same(nodes_path: Path, label_col: str) -> float:
    nodes = pd.read_csv(nodes_path, usecols=[label_col, "degree"], low_memory=False)
    labels = nodes[label_col].astype(str).str.strip()
    valid = ~labels.str.lower().isin(MISSING_LABELS)
    nodes = nodes[valid].copy()
    labels = labels[valid]
    total_degree = float(nodes["degree"].sum())
    if total_degree <= 0:
        return np.nan
    degree_by_label = nodes.groupby(labels)["degree"].sum()
    endpoint_fraction = degree_by_label / total_degree
    return float((endpoint_fraction**2).sum())


def add_metric_space(df: pd.DataFrame, metric_space: str) -> pd.DataFrame:
    out = df.copy()
    out["Metric_Space"] = metric_space
    out["Cohort"] = out["cohort_dir"].map(COHORT_LABELS).fillna(out["cohort_dir"])
    out["Graph"] = out.apply(normalize_graph, axis=1)
    out["graph_order"] = out["Graph"].map(GRAPH_ORDER).fillna(99).astype(int)
    return out


def build_table(combined: pd.DataFrame, label_col: str) -> pd.DataFrame:
    rows = []
    observed_col = f"{label_col}_observed_same_fraction"
    node_expected_col = f"{label_col}_nodepair_expected_same_fraction"
    node_ratio_col = f"{label_col}_nodepair_enrichment_ratio"

    for _, row in combined.sort_values(["cohort_dir", "Metric_Space", "graph_order"]).iterrows():
        graph_dir = Path(row["graph_dir"])
        nodes_path = graph_dir / "nodes.csv"
        expected_same = endpoint_expected_same(nodes_path, label_col)
        observed_same = float(row[observed_col])
        coleman_h = (
            (observed_same - expected_same) / (1 - expected_same)
            if pd.notna(expected_same) and expected_same != 1
            else np.nan
        )
        observed_node = observed_same
        expected_node = float(row[node_expected_col])
        assortativity = float(row[node_ratio_col])

        rows.append(
            {
                "Cohort": row["Cohort"],
                "Metric_Space": row["Metric_Space"],
                "Graph": row["Graph"],
                "Edges": int(row["n_edges"]),
                "Components": int(row["n_components"]),
                "Giant_Fraction": float(row["giant_component_frac"]),
                "Observed_same": observed_same,
                "Expected_same": expected_same,
                "Coleman h Index": coleman_h,
                "observed_node": observed_node,
                "expected_node": expected_node,
                "assortatirtivty_coefficient": assortativity,
            }
        )
    table = pd.DataFrame(rows)
    h_values = table.pivot_table(
        index=["Cohort", "Graph"],
        columns="Metric_Space",
        values="Coleman h Index",
        aggfunc="first",
    )
    if {"ESM-2", "Hamming"}.issubset(h_values.columns):
        h_values["Delta h = ESM2h - Hamming h"] = h_values["ESM-2"] - h_values["Hamming"]
        table = table.merge(
            h_values[["Delta h = ESM2h - Hamming h"]].reset_index(),
            on=["Cohort", "Graph"],
            how="left",
        )
    else:
        table["Delta h = ESM2h - Hamming h"] = np.nan
    return table


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Edges"] = out["Edges"].map(lambda x: f"{int(x):,}")
    for col in [
        "Giant_Fraction",
        "Observed_same",
        "Expected_same",
        "Coleman h Index",
        "Delta h = ESM2h - Hamming h",
        "observed_node",
        "expected_node",
        "assortatirtivty_coefficient",
    ]:
        out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
    return out


def write_markdown(df: pd.DataFrame, path: Path) -> None:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "\\|") for col in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create labeled assortativity summary table for cohort graph comparisons.")
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
    parser.add_argument("--label-col", default="within_lineage_label")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis/cohort_validation/05_cross_cohort_comparison"),
    )
    parser.add_argument("--output-prefix", default="main_n20000_assortativity_summary_table")
    args = parser.parse_args()

    embedding = add_metric_space(pd.read_csv(args.embedding_summary), "ESM-2")
    hamming = add_metric_space(pd.read_csv(args.hamming_summary), "Hamming")
    combined = pd.concat([embedding, hamming], ignore_index=True, sort=False)
    table = build_table(combined, args.label_col)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / f"{args.output_prefix}.csv"
    formatted_csv_path = args.out_dir / f"{args.output_prefix}_formatted.csv"
    md_path = args.out_dir / f"{args.output_prefix}.md"

    table.to_csv(raw_path, index=False)
    formatted = format_table(table)
    formatted.to_csv(formatted_csv_path, index=False)
    write_markdown(formatted, md_path)

    print(f"Wrote {raw_path}")
    print(f"Wrote {formatted_csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
