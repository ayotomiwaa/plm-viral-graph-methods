#!/usr/bin/env python3
"""Plot synthetic and biological RNG degree probability-mass functions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(
    "analysis/cohort_validation/29_seed42_20k_rng_degree_dimension_calibration/"
    "random_full_dataset_seed42/seed_42"
)
COLORS = {
    "synthetic_raw": "#2563EB",
    "synthetic_refined": "#D97706",
    "biological_raw": "#111827",
    "biological_refined": "#6B7280",
}


def positive_xy(frame: pd.DataFrame, y_column: str) -> tuple[np.ndarray, np.ndarray]:
    used = frame.loc[frame[y_column] > 0].sort_values("degree")
    return used["degree"].to_numpy(), used[y_column].to_numpy()


def add_series(
    ax: plt.Axes,
    frame: pd.DataFrame,
    y_column: str,
    label: str,
    color: str,
    linestyle: str,
    linewidth: float,
    marker: str | None = None,
) -> None:
    x, y = positive_xy(frame, y_column)
    ax.plot(
        x,
        y,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        marker=marker,
        markersize=3.2 if marker else 0,
        markerfacecolor="white" if marker else color,
        markeredgewidth=0.9,
        label=label,
    )


def plot_method_matched(
    biological: pd.DataFrame,
    synthetic: pd.DataFrame,
    comparison_n: int,
    out_dir: Path,
) -> None:
    dimensions = sorted(synthetic.loc[synthetic["n_points"] == comparison_n, "dimension"].unique())
    if not dimensions:
        raise ValueError(f"no synthetic degree distributions found for n_points={comparison_n}")
    raw_bio = biological.loc[
        biological["representation"] == "biological_unique_collapsed_original"
    ]
    refined_bio = biological.loc[
        biological["representation"] == "biological_unique_refined_direct"
    ]
    if raw_bio.empty or refined_bio.empty:
        raise ValueError("method-matched biological degree distributions are incomplete")

    ncols = 3
    nrows = int(np.ceil(len(dimensions) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.2 * nrows), sharex=True, sharey=True)
    axes_array = np.atleast_1d(axes).ravel()
    x_max = int(
        max(
            biological.loc[
                biological["representation"].isin(
                    ["biological_unique_collapsed_original", "biological_unique_refined_direct"]
                ),
                "degree",
            ].max(),
            synthetic.loc[synthetic["n_points"] == comparison_n, "degree"].max(),
        )
    )

    for ax, dimension in zip(axes_array, dimensions):
        panel = synthetic.loc[
            (synthetic["n_points"] == comparison_n)
            & (synthetic["dimension"] == dimension)
        ]
        for graph_state, color in [("raw", COLORS["synthetic_raw"]), ("refined", COLORS["synthetic_refined"])]:
            condition = panel.loc[panel["graph_state"] == graph_state].sort_values("degree")
            x = condition["degree"].to_numpy()
            ax.fill_between(
                x,
                condition["min_fraction_nodes"].to_numpy(),
                condition["max_fraction_nodes"].to_numpy(),
                color=color,
                alpha=0.10,
                linewidth=0,
            )
            add_series(
                ax,
                condition,
                "mean_fraction_nodes",
                f"Synthetic {graph_state}",
                color,
                "-",
                1.8,
            )
        add_series(
            ax,
            raw_bio,
            "fraction_nodes",
            "Biological unique raw",
            COLORS["biological_raw"],
            "--",
            1.8,
            "o",
        )
        add_series(
            ax,
            refined_bio,
            "fraction_nodes",
            "Biological unique refined",
            COLORS["biological_refined"],
            ":",
            2.0,
            "s",
        )
        ax.set_title(f"Known dimension d={int(dimension)}", fontsize=11, color="#111827")
        ax.set_yscale("log")
        ax.set_xlim(0, x_max + 1)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors="#374151", labelsize=9)

    for ax in axes_array[len(dimensions) :]:
        ax.set_visible(False)
    for ax in axes_array[-ncols:]:
        if ax.get_visible():
            ax.set_xlabel("Node degree", color="#374151")
    for row in range(nrows):
        axes_array[row * ncols].set_ylabel("Fraction of nodes", color="#374151")

    handles, labels = axes_array[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.935))
    fig.suptitle("RNG degree distributions by known dimension", fontsize=16, color="#111827", y=0.99)
    fig.text(
        0.5,
        0.952,
        f"L1 uniform points at N={comparison_n:,} (replicate range) versus biological unique-coordinate RNG; log probability scale",
        ha="center",
        va="top",
        fontsize=10,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0.03, 0.04, 0.99, 0.90))
    for suffix in ["png", "pdf"]:
        fig.savefig(out_dir / f"rng_degree_distribution_method_matched.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_biological_scopes(biological: pd.DataFrame, out_dir: Path) -> None:
    specs = [
        (
            "20,000 biological records",
            "biological_record_original",
            "biological_record_refined",
            "symlog",
        ),
        (
            "8,921 unique coordinates",
            "biological_unique_collapsed_original",
            "biological_unique_refined_direct",
            "linear",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, (title, raw_name, refined_name, xscale) in zip(axes, specs):
        raw = biological.loc[biological["representation"] == raw_name]
        refined = biological.loc[biological["representation"] == refined_name]
        add_series(ax, raw, "fraction_nodes", "Original", COLORS["synthetic_raw"], "-", 2.0)
        add_series(ax, refined, "fraction_nodes", "Refined", COLORS["synthetic_refined"], "--", 2.0)
        ax.set_title(title, fontsize=12, color="#111827")
        ax.set_yscale("log")
        if xscale == "symlog":
            ax.set_xscale("symlog", linthresh=20, linscale=1.0)
        ax.set_xlabel("Node degree", color="#374151")
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors="#374151", labelsize=9)
    axes[0].set_ylabel("Fraction of nodes", color="#374151")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.90))
    fig.suptitle("Biological RNG degree distributions", fontsize=16, color="#111827", y=0.99)
    fig.text(
        0.5,
        0.945,
        "Record-level ties inflate the degree scale; panels use log probability and separate degree axes",
        ha="center",
        va="top",
        fontsize=10,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0.03, 0.05, 0.99, 0.86))
    for suffix in ["png", "pdf"]:
        fig.savefig(out_dir / f"rng_degree_distribution_biological_scopes.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    biological_path = args.analysis_root / "biological_rng_degree_distribution.csv"
    synthetic_path = args.analysis_root / "synthetic_rng_degree_distribution_summary.csv"
    biological = pd.read_csv(biological_path)
    synthetic = pd.read_csv(synthetic_path)
    if args.comparison_n is None:
        manifest = json.loads((args.analysis_root / "calibration_manifest.json").read_text())
        comparison_n = int(manifest["n_unique_biological_coordinates"])
    else:
        comparison_n = args.comparison_n
    out_dir = args.analysis_root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_method_matched(biological, synthetic, comparison_n, out_dir)
    plot_biological_scopes(biological, out_dir)
    print(f"Wrote degree-distribution figures under {out_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--comparison-n", type=int, default=None)
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
