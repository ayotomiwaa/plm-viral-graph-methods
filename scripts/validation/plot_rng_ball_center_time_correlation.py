#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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


def compute_correlations(ball_frame: pd.DataFrame, value_col: str, date_col: str) -> pd.DataFrame:
    required = {"radius", value_col, date_col}
    missing = required.difference(ball_frame.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    frame = ball_frame.copy()
    frame["radius"] = pd.to_numeric(frame["radius"], errors="coerce")
    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
    dates = pd.to_datetime(frame[date_col], errors="coerce")
    frame["center_collection_date_day"] = np.nan
    valid_dates = dates.notna()
    if valid_dates.any():
        frame.loc[valid_dates, "center_collection_date_day"] = (
            dates[valid_dates].to_numpy(dtype="datetime64[D]").astype(np.int64).astype(float)
        )
    frame = frame[frame["radius"].notna() & frame[value_col].notna() & frame["center_collection_date_day"].notna()]
    if frame.empty:
        raise ValueError("No rows have finite radius, finite value, and valid center collection date")

    rows: list[dict[str, Any]] = []
    for radius, group in frame.groupby("radius", dropna=False):
        x = pd.to_numeric(group["center_collection_date_day"], errors="coerce")
        y = pd.to_numeric(group[value_col], errors="coerce")
        valid = x.notna() & y.notna()
        x = x[valid]
        y = y[valid]
        row: dict[str, Any] = {
            "radius": float(radius),
            "radius_label": f"{float(radius):.4g}",
            "value_col": value_col,
            "date_col": date_col,
            "n_balls": int(valid.sum()),
            "n_unique_center_dates": int(x.nunique()),
            "n_unique_values": int(y.nunique()),
        }
        if row["n_balls"] >= 2 and row["n_unique_center_dates"] >= 2 and row["n_unique_values"] >= 2:
            row["pearson_r"] = float(x.corr(y, method="pearson"))
            row["spearman_r"] = float(x.corr(y, method="spearman"))
        else:
            row["pearson_r"] = math.nan
            row["spearman_r"] = math.nan
        rows.append(row)

    return pd.DataFrame(rows).sort_values("radius").reset_index(drop=True)


def write_plot(corr_frame: pd.DataFrame, corr_col: str, out_path: Path) -> None:
    plt = load_pyplot(out_path)
    frame = corr_frame.copy()
    frame[corr_col] = pd.to_numeric(frame[corr_col], errors="coerce")
    frame = frame[frame[corr_col].notna()]
    if frame.empty:
        log(f"Skipping {out_path}: no finite {corr_col} values")
        return

    radii = frame["radius"].to_numpy(dtype=float)
    labels = frame["radius_label"].astype(str).tolist()
    x = np.arange(len(frame), dtype=float)

    fig_width = max(8.0, min(16.0, 0.55 * len(frame) + 4.0))
    fig, ax = plt.subplots(figsize=(fig_width, 5.0), constrained_layout=True)
    ax.plot(x, frame[corr_col].to_numpy(dtype=float), marker="o", linewidth=1.8, color="#3366aa")
    ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.65)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("RNG graph-distance radius")
    ax.set_ylabel(f"{corr_col.replace('_', ' ')}: center collection date vs ball temporal spread")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Compute one correlation per RNG ball radius: x=center collection date, "
            "y=ball temporal spread, then plot correlation versus radius."
        )
    )
    ap.add_argument("--ball-csv", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-png", type=Path, required=True)
    ap.add_argument("--value-col", default="mean_pairwise_delta_days")
    ap.add_argument("--date-col", default="center_collection_date")
    ap.add_argument("--correlation", choices=["pearson_r", "spearman_r"], default="pearson_r")
    args = ap.parse_args()

    ball_frame = pd.read_csv(args.ball_csv, low_memory=False)
    corr_frame = compute_correlations(ball_frame, value_col=args.value_col, date_col=args.date_col)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    corr_frame.to_csv(args.out_csv, index=False)
    write_plot(corr_frame, corr_col=args.correlation, out_path=args.out_png)
    log(f"Wrote {args.out_csv}")
    log(f"Wrote {args.out_png}")


if __name__ == "__main__":
    main()
