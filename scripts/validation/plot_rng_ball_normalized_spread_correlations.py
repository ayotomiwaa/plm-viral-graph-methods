#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_CITYBLOCK_SUMMARY = Path(
    "analysis/cohort_validation/15_seed42_20k/rng_ball_temporal_spread/"
    "cityblock_pow2_local_shuffle/all_rng_ball_temporal_spread_radius_summary.csv"
)
DEFAULT_HAMMING_SUMMARY = Path(
    "analysis/cohort_validation/15_seed42_20k/rng_ball_temporal_spread/"
    "hamming_pow2_local_shuffle/all_rng_ball_temporal_spread_radius_summary.csv"
)
DEFAULT_OUT_DIR = Path(
    "analysis/cohort_validation/15_seed42_20k/rng_ball_temporal_spread/"
    "normalized_spread_correlations"
)


GRAPH_LABELS = {
    "rng_embedding": "RNG embedding",
    "rng_hamming": "RNG Hamming",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def load_pyplot(out_path: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Cannot write {out_path}: missing Python package {exc.name!r}") from exc
    return plt


def parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def first_radius_at_threshold(curve: pd.DataFrame, threshold: float) -> float:
    hits = curve[curve["y_norm"] >= threshold].sort_values("radius")
    if hits.empty:
        return math.nan
    first = hits.iloc[0]
    return float(first["radius"])


def normalized_radius(radius: pd.Series, denominator: float) -> pd.Series:
    if np.isfinite(denominator) and denominator > 0:
        return (radius / denominator).clip(lower=0.0, upper=1.0)
    return pd.Series(np.zeros(radius.shape[0], dtype=float), index=radius.index)


def read_summary(
    path: Path,
    graph_key: str,
    date_assignment: str,
    date_shuffle_index: int | None,
) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if "radius" not in frame.columns:
        raise ValueError(f"{path} is missing required column: radius")
    if "date_assignment" in frame.columns:
        frame = frame[frame["date_assignment"].astype(str) == date_assignment].copy()
    if date_shuffle_index is not None and "date_shuffle_index" in frame.columns:
        frame = frame[pd.to_numeric(frame["date_shuffle_index"], errors="coerce") == date_shuffle_index].copy()
    if frame.empty:
        raise ValueError(
            f"No rows remain in {path} after filtering date_assignment={date_assignment!r} "
            f"and date_shuffle_index={date_shuffle_index!r}"
        )
    frame["graph_key"] = graph_key
    frame["source_summary_csv"] = str(path)
    frame["radius"] = pd.to_numeric(frame["radius"], errors="coerce")
    return frame


def normalize_curves(
    summaries: pd.DataFrame,
    spread_family: str,
    stats: list[str],
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_rows: list[dict[str, Any]] = []
    saturation_rows: list[dict[str, Any]] = []
    for graph_key, graph_frame in summaries.groupby("graph_key", sort=False):
        graph_frame = graph_frame.sort_values("radius").copy()
        sampled_max_radius = float(graph_frame["radius"].max())
        if not np.isfinite(sampled_max_radius) or sampled_max_radius <= 0:
            raise ValueError(f"{graph_key} has invalid sampled max radius: {sampled_max_radius}")

        for stat in stats:
            value_col = f"{spread_family}_{stat}"
            if value_col not in graph_frame.columns:
                raise ValueError(f"Input summary is missing required column: {value_col}")
            values = pd.to_numeric(graph_frame[value_col], errors="coerce")
            finite = graph_frame["radius"].notna() & values.notna()
            curve = graph_frame.loc[finite, ["radius"]].copy()
            curve["value"] = values[finite].astype(float)
            curve = curve.sort_values("radius").reset_index(drop=True)
            if curve.empty:
                raise ValueError(f"{graph_key} {value_col} has no finite values")

            terminal_rows = curve[curve["radius"] == curve["radius"].max()]
            saturation = float(terminal_rows["value"].iloc[-1])
            if not np.isfinite(saturation) or saturation == 0:
                raise ValueError(f"{graph_key} {value_col} has invalid saturation value: {saturation}")

            curve["graph_key"] = graph_key
            curve["graph_label"] = GRAPH_LABELS.get(graph_key, graph_key)
            curve["spread_family"] = spread_family
            curve["statistic"] = stat
            curve["value_col"] = value_col
            curve["saturation_value"] = saturation
            curve["value_capped_at_saturation"] = curve["value"].clip(upper=saturation)
            curve["y_norm_uncapped"] = curve["value"] / saturation
            curve["y_norm"] = curve["value_capped_at_saturation"] / saturation
            r95 = first_radius_at_threshold(curve, threshold=threshold)
            x_saturation_radius = r95 if np.isfinite(r95) and r95 > 0 else sampled_max_radius
            curve["sampled_max_radius"] = sampled_max_radius
            curve["x_saturation_stat"] = stat
            curve["x_saturation_rule"] = "per_stat_positive_r95"
            curve["x_saturation_radius"] = x_saturation_radius
            curve["max_radius"] = x_saturation_radius
            curve["x_norm"] = normalized_radius(curve["radius"], x_saturation_radius)
            if np.isfinite(x_saturation_radius) and x_saturation_radius > 0:
                curve["in_saturation_interval"] = curve["radius"] <= x_saturation_radius
            else:
                curve["in_saturation_interval"] = curve["radius"] == 0
            r95_norm = (
                float(normalized_radius(pd.Series([r95]), x_saturation_radius).iloc[0])
                if np.isfinite(r95)
                else math.nan
            )

            saturation_rows.append(
                {
                    "graph_key": graph_key,
                    "graph_label": GRAPH_LABELS.get(graph_key, graph_key),
                    "spread_family": spread_family,
                    "statistic": stat,
                    "value_col": value_col,
                    "sampled_max_radius": sampled_max_radius,
                    "x_saturation_stat": stat,
                    "x_saturation_rule": "per_stat_positive_r95",
                    "x_saturation_radius": x_saturation_radius,
                    "max_radius": x_saturation_radius,
                    "saturation_value": saturation,
                    "threshold": threshold,
                    "threshold_value": threshold * saturation,
                    "r95_radius": r95,
                    "r95_x_norm": r95_norm,
                    "n_radius_points": int(curve.shape[0]),
                    "n_values_above_terminal_saturation": int((curve["value"] > saturation).sum()),
                }
            )
            curve_rows.extend(curve.to_dict("records"))

    return pd.DataFrame(curve_rows), pd.DataFrame(saturation_rows)


def assign_normalized_bins(curves: pd.DataFrame, bins: int) -> pd.DataFrame:
    if bins <= 0:
        raise ValueError("--bins must be positive")
    frame = curves.copy()
    if "in_saturation_interval" in frame.columns:
        frame = frame[frame["in_saturation_interval"].astype(bool)].copy()
    bin_index = np.floor(frame["x_norm"].to_numpy(dtype=float) * bins).astype(int)
    bin_index = np.clip(bin_index, 0, bins - 1)
    frame["x_bin"] = bin_index
    frame["x_bin_left"] = frame["x_bin"] / bins
    frame["x_bin_right"] = (frame["x_bin"] + 1) / bins
    frame["x_bin_mid"] = (frame["x_bin_left"] + frame["x_bin_right"]) / 2
    grouped = (
        frame.groupby(
            [
                "graph_key",
                "graph_label",
                "spread_family",
                "statistic",
                "value_col",
                "sampled_max_radius",
                "x_saturation_radius",
                "max_radius",
                "saturation_value",
                "x_bin",
                "x_bin_left",
                "x_bin_right",
                "x_bin_mid",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            y_norm_mean=("y_norm", "mean"),
            y_norm_min=("y_norm", "min"),
            y_norm_max=("y_norm", "max"),
            raw_value_mean=("value", "mean"),
            radius_min=("radius", "min"),
            radius_max=("radius", "max"),
            n_radius_points=("radius", "size"),
        )
        .sort_values(["spread_family", "statistic", "graph_key", "x_bin"])
        .reset_index(drop=True)
    )
    return grouped


def correlation_row(x: pd.Series, y: pd.Series) -> dict[str, float | int]:
    valid = x.notna() & y.notna()
    x = x[valid]
    y = y[valid]
    out: dict[str, float | int] = {
        "n_bins": int(valid.sum()),
        "n_unique_x": int(x.nunique()),
        "n_unique_y": int(y.nunique()),
    }
    if out["n_bins"] >= 2 and out["n_unique_x"] >= 2 and out["n_unique_y"] >= 2:
        out["pearson_r"] = float(x.corr(y, method="pearson"))
        out["spearman_r"] = float(x.corr(y, method="spearman"))
    else:
        out["pearson_r"] = math.nan
        out["spearman_r"] = math.nan
    return out


def compute_correlations(binned: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in binned.groupby(["graph_key", "graph_label", "spread_family", "statistic"], sort=False):
        graph_key, graph_label, spread_family, statistic = keys
        corr = correlation_row(group["x_bin_mid"], group["y_norm_mean"])
        rows.append(
            {
                "graph_key": graph_key,
                "graph_label": graph_label,
                "spread_family": spread_family,
                "statistic": statistic,
                **corr,
            }
        )
    return pd.DataFrame(rows).sort_values(["spread_family", "statistic", "graph_key"]).reset_index(drop=True)


def compute_graph_pair_correlations(binned: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in binned.groupby(["spread_family", "statistic"], sort=False):
        spread_family, statistic = keys
        wide = group.pivot_table(index="x_bin", columns="graph_key", values="y_norm_mean", aggfunc="first")
        for graph_key in ["rng_hamming", "rng_embedding"]:
            if graph_key not in wide.columns:
                wide[graph_key] = np.nan
        valid = wide["rng_hamming"].notna() & wide["rng_embedding"].notna()
        paired = wide.loc[valid, ["rng_hamming", "rng_embedding"]]
        row: dict[str, Any] = {
            "spread_family": spread_family,
            "statistic": statistic,
            "comparison": "rng_hamming_vs_rng_embedding",
            "n_matched_bins": int(paired.shape[0]),
            "n_unique_hamming_y": int(paired["rng_hamming"].nunique()),
            "n_unique_embedding_y": int(paired["rng_embedding"].nunique()),
        }
        if (
            row["n_matched_bins"] >= 2
            and row["n_unique_hamming_y"] >= 2
            and row["n_unique_embedding_y"] >= 2
        ):
            row["pearson_r"] = float(paired["rng_hamming"].corr(paired["rng_embedding"], method="pearson"))
            row["spearman_r"] = float(paired["rng_hamming"].corr(paired["rng_embedding"], method="spearman"))
        else:
            row["pearson_r"] = math.nan
            row["spearman_r"] = math.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["spread_family", "statistic"]).reset_index(drop=True)


def write_wide_tables(correlations: pd.DataFrame, out_dir: Path, stats: list[str]) -> list[Path]:
    written: list[Path] = []
    for spread_family, family in correlations.groupby("spread_family", sort=False):
        for method in ["pearson_r", "spearman_r"]:
            table = family.pivot(index="statistic", columns="graph_key", values=method)
            table = table.reindex(stats)
            table = table.rename_axis(index="correlation_statistic", columns=None).reset_index()
            for graph_key in ["rng_hamming", "rng_embedding"]:
                if graph_key not in table.columns:
                    table[graph_key] = np.nan
            table = table[["correlation_statistic", "rng_hamming", "rng_embedding"]]
            out_path = out_dir / f"{spread_family}_normalized_radius_spread_{method}_table.csv"
            table.to_csv(out_path, index=False)
            written.append(out_path)
    return written


def write_graph_pair_wide_tables(pair_correlations: pd.DataFrame, out_dir: Path, stats: list[str]) -> list[Path]:
    written: list[Path] = []
    for spread_family, family in pair_correlations.groupby("spread_family", sort=False):
        for method in ["pearson_r", "spearman_r"]:
            table = family.pivot(index="statistic", columns="comparison", values=method)
            table = table.reindex(stats)
            table = table.rename_axis(index="correlation_statistic", columns=None).reset_index()
            if "rng_hamming_vs_rng_embedding" not in table.columns:
                table["rng_hamming_vs_rng_embedding"] = np.nan
            table = table[["correlation_statistic", "rng_hamming_vs_rng_embedding"]]
            out_path = out_dir / f"{spread_family}_normalized_hamming_embedding_curve_{method}_table.csv"
            table.to_csv(out_path, index=False)
            written.append(out_path)
    return written


def plot_family_bars(ax, frame: pd.DataFrame, method: str, spread_family: str) -> None:
    stat_order = ["min", "q1", "median", "q3", "max"]
    stats = [stat for stat in stat_order if stat in set(frame["statistic"])]
    x = np.arange(len(stats), dtype=float)
    width = 0.34
    colors = {"rng_hamming": "#3366aa", "rng_embedding": "#cc7a29"}
    offsets = {"rng_hamming": -width / 2, "rng_embedding": width / 2}

    for graph_key in ["rng_hamming", "rng_embedding"]:
        graph_frame = frame[frame["graph_key"] == graph_key].set_index("statistic")
        values = [graph_frame.loc[stat, method] if stat in graph_frame.index else np.nan for stat in stats]
        ax.bar(
            x + offsets[graph_key],
            values,
            width=width,
            label=GRAPH_LABELS.get(graph_key, graph_key),
            color=colors[graph_key],
            edgecolor="#333333",
            linewidth=0.7,
        )
    ax.axhline(0.0, color="#5f646b", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(stats)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Boxplot statistic")
    ax.set_title(spread_family.replace("_", " "), fontsize=12)
    ax.grid(axis="y", alpha=0.22)


def plot_pair_bars(ax, frame: pd.DataFrame, method: str, spread_family: str) -> None:
    stat_order = ["min", "q1", "median", "q3", "max"]
    stats = [stat for stat in stat_order if stat in set(frame["statistic"])]
    values = []
    by_stat = frame.set_index("statistic")
    for stat in stats:
        values.append(by_stat.loc[stat, method] if stat in by_stat.index else np.nan)
    x = np.arange(len(stats), dtype=float)
    ax.bar(
        x,
        values,
        width=0.58,
        color="#4a6f9f",
        edgecolor="#333333",
        linewidth=0.7,
    )
    ax.axhline(0.0, color="#5f646b", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(stats)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Boxplot statistic")
    ax.set_title(spread_family.replace("_", " "), fontsize=12)
    ax.grid(axis="y", alpha=0.22)


def write_correlation_plot(correlations: pd.DataFrame, spread_family: str, out_path: Path, method: str) -> None:
    plt = load_pyplot(out_path)
    frame = correlations[correlations["spread_family"] == spread_family].copy()
    frame[method] = pd.to_numeric(frame[method], errors="coerce")
    frame = frame[frame[method].notna()]
    if frame.empty:
        log(f"Skipping {out_path}: no finite {method} values")
        return

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    plot_family_bars(ax, frame=frame, method=method, spread_family=spread_family)
    ax.set_ylabel(f"{method.replace('_', ' ')} of normalized radius vs normalized spread")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_graph_pair_correlation_plot(pair_correlations: pd.DataFrame, out_path: Path, method: str) -> None:
    plt = load_pyplot(out_path)
    frame = pair_correlations.copy()
    frame[method] = pd.to_numeric(frame[method], errors="coerce")
    frame = frame[frame[method].notna()]
    spread_families = [name for name in ["max_delta_days", "mean_delta_days"] if name in set(frame["spread_family"])]
    if not spread_families:
        log(f"Skipping {out_path}: no finite {method} values")
        return

    fig, axes = plt.subplots(1, len(spread_families), figsize=(6.8 * len(spread_families), 4.8), sharey=True)
    if len(spread_families) == 1:
        axes = [axes]
    for ax, spread_family in zip(axes, spread_families):
        family = frame[frame["spread_family"] == spread_family]
        plot_pair_bars(ax, frame=family, method=method, spread_family=spread_family)
    axes[0].set_ylabel(f"{method.replace('_', ' ')} between normalized curves")
    fig.suptitle("Correlation between normalized RNG ball-spread curves", fontsize=14, y=1.04)
    fig.tight_layout(rect=[0.03, 0.0, 1.0, 0.96])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_normalized_curve_plot(curves: pd.DataFrame, spread_family: str, out_path: Path) -> None:
    plt = load_pyplot(out_path)
    frame = curves[curves["spread_family"] == spread_family].copy()
    if frame.empty:
        log(f"Skipping {out_path}: no rows for {spread_family}")
        return

    stat_order = ["min", "q1", "median", "q3", "max"]
    stats = [stat for stat in stat_order if stat in set(frame["statistic"])]
    n_cols = min(3, len(stats))
    n_rows = int(math.ceil(len(stats) / n_cols))
    finite_y = pd.to_numeric(frame["y_norm"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    y_upper = 1.08
    if not finite_y.empty:
        y_upper = max(1.08, float(finite_y.max()) * 1.08)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 3.8 * n_rows), sharex=True, sharey=True)
    axes_flat = np.array(axes, dtype=object).reshape(-1)
    colors = {"rng_hamming": "#3366aa", "rng_embedding": "#cc7a29"}
    markers = {"rng_hamming": "o", "rng_embedding": "s"}

    for ax, stat in zip(axes_flat, stats):
        stat_frame = frame[frame["statistic"] == stat]
        for graph_key in ["rng_hamming", "rng_embedding"]:
            graph_frame = stat_frame[stat_frame["graph_key"] == graph_key].sort_values("x_norm")
            if graph_frame.empty:
                continue
            ax.plot(
                graph_frame["x_norm"].to_numpy(dtype=float),
                graph_frame["y_norm"].to_numpy(dtype=float),
                marker=markers[graph_key],
                linewidth=1.8,
                markersize=4.5,
                color=colors[graph_key],
                label=GRAPH_LABELS.get(graph_key, graph_key),
            )
        ax.axhline(0.95, color="#666666", linewidth=1.0, linestyle="--", alpha=0.6)
        ax.set_title(stat)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.05, y_upper)
        ax.grid(axis="y", alpha=0.22)
    for ax in axes_flat[len(stats) :]:
        ax.set_axis_off()
    for ax in axes_flat[::n_cols]:
        ax.set_ylabel("Normalized temporal spread")
    for ax in axes_flat[-n_cols:]:
        ax.set_xlabel("Normalized RNG graph-distance radius")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"{spread_family.replace('_', ' ')} normalized curves", fontsize=14, y=1.06)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_combined_correlation_plot(correlations: pd.DataFrame, out_path: Path, method: str) -> None:
    plt = load_pyplot(out_path)
    frame = correlations.copy()
    frame[method] = pd.to_numeric(frame[method], errors="coerce")
    frame = frame[frame[method].notna()]
    spread_families = [name for name in ["max_delta_days", "mean_delta_days"] if name in set(frame["spread_family"])]
    if not spread_families:
        log(f"Skipping {out_path}: no finite {method} values")
        return

    fig, axes = plt.subplots(1, len(spread_families), figsize=(6.6 * len(spread_families), 4.8), sharey=True)
    if len(spread_families) == 1:
        axes = [axes]
    for ax, spread_family in zip(axes, spread_families):
        family = frame[frame["spread_family"] == spread_family]
        plot_family_bars(ax, frame=family, method=method, spread_family=spread_family)
    axes[0].set_ylabel(f"{method.replace('_', ' ')} of normalized radius vs normalized spread")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Normalized radius-spread correlations", fontsize=14, y=1.08)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Normalize RNG ball radius-summary curves by each graph's own terminal radius and "
            "terminal temporal-spread value, then correlate the normalized Hamming and embedding "
            "curves on shared normalized-radius bins."
        )
    )
    ap.add_argument("--embedding-summary-csv", type=Path, default=DEFAULT_CITYBLOCK_SUMMARY)
    ap.add_argument("--hamming-summary-csv", type=Path, default=DEFAULT_HAMMING_SUMMARY)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--date-assignment", default="observed")
    ap.add_argument(
        "--date-shuffle-index",
        type=int,
        default=None,
        help="Optional date_shuffle_index filter, useful when --date-assignment is shuffled_dates.",
    )
    ap.add_argument("--spread-families", default="max_delta_days,mean_delta_days")
    ap.add_argument("--stats", default="min,q1,median,q3,max")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--plot-correlation", choices=["pearson_r", "spearman_r"], default="pearson_r")
    args = ap.parse_args()

    spread_families = parse_csv_list(args.spread_families)
    stats = parse_csv_list(args.stats)
    if not spread_families:
        raise ValueError("--spread-families must contain at least one value")
    if not stats:
        raise ValueError("--stats must contain at least one value")

    summaries = pd.concat(
        [
            read_summary(args.embedding_summary_csv, "rng_embedding", args.date_assignment, args.date_shuffle_index),
            read_summary(args.hamming_summary_csv, "rng_hamming", args.date_assignment, args.date_shuffle_index),
        ],
        ignore_index=True,
    )

    curve_frames: list[pd.DataFrame] = []
    saturation_frames: list[pd.DataFrame] = []
    for spread_family in spread_families:
        curves, saturation = normalize_curves(
            summaries,
            spread_family=spread_family,
            stats=stats,
            threshold=args.threshold,
        )
        curve_frames.append(curves)
        saturation_frames.append(saturation)

    curves = pd.concat(curve_frames, ignore_index=True)
    saturation = pd.concat(saturation_frames, ignore_index=True)
    binned = assign_normalized_bins(curves, bins=args.bins)
    pair_correlations = compute_graph_pair_correlations(binned)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    curve_path = args.out_dir / "normalized_radius_spread_curve_points.csv"
    saturation_path = args.out_dir / "normalized_radius_spread_saturation_points.csv"
    binned_path = args.out_dir / "normalized_radius_spread_binned_points.csv"
    pair_corr_path = args.out_dir / "normalized_hamming_embedding_curve_correlations_long.csv"

    curves.to_csv(curve_path, index=False)
    saturation.to_csv(saturation_path, index=False)
    binned.to_csv(binned_path, index=False)
    pair_correlations.to_csv(pair_corr_path, index=False)
    pair_wide_paths = write_graph_pair_wide_tables(pair_correlations, args.out_dir, stats=stats)

    plot_paths: list[Path] = []
    for spread_family in spread_families:
        curve_plot_path = args.out_dir / f"{spread_family}_normalized_hamming_embedding_curves.png"
        write_normalized_curve_plot(curves, spread_family=spread_family, out_path=curve_plot_path)
        plot_paths.append(curve_plot_path)
    pair_plot_path = args.out_dir / f"normalized_hamming_embedding_curve_{args.plot_correlation}_plot.png"
    write_graph_pair_correlation_plot(pair_correlations, out_path=pair_plot_path, method=args.plot_correlation)
    plot_paths.append(pair_plot_path)

    for path in [
        curve_path,
        saturation_path,
        binned_path,
        pair_corr_path,
        *pair_wide_paths,
        *plot_paths,
    ]:
        log(f"Wrote {path}")


if __name__ == "__main__":
    main()
