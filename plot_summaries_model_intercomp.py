#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import rasterio
from PIL import Image, ImageDraw

INPUT_DIR = Path("fintercomp")
OUTPUT_DIR = INPUT_DIR / "viz"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_intercomparison_csvs(input_dir: Path) -> pd.DataFrame:
    csvs = sorted(input_dir.glob("hourly_*_intercomparison.csv"))
    frames = []
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        df["source_csv"] = str(csv_path)

        if "field" not in df.columns or "member" not in df.columns:
            stem = csv_path.stem
            m = re.match(r"hourly_(.+)_([^_]+)_intercomparison", stem)
            if m:
                df["field"] = m.group(1)
                df["member"] = m.group(2)

        if "hour" in df.columns:
            df["hour"] = pd.to_numeric(df["hour"], errors="coerce")

        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No intercomparison CSVs found in {input_dir}")

    combo = pd.concat(frames, ignore_index=True)
    combo.to_csv(OUTPUT_DIR / "combined_intercomparison_metrics.csv", index=False)
    return combo


def save_fig(fig: go.Figure, out_png: Path, caption: str, description: str = ""):
    fig.write_image(str(out_png))
    with open(str(out_png) + ".meta.json", "w") as f:
        json.dump({"caption": caption, "description": description}, f)


def plot_hourly_metric(df: pd.DataFrame, metric: str, field: str):
    sdf = df[df["field"] == field].copy()
    if sdf.empty or metric not in sdf.columns:
        return

    sdf = sdf.sort_values(["member", "hour"])
    fig = px.line(sdf, x="hour", y=metric, color="member", markers=True)
    fig.update_layout(
        title={
            "text": f"{metric} by hour ({field})<br><span style='font-size: 18px; font-weight: normal;'>Each line is one ensemble member</span>"
        },
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
    )
    fig.update_xaxes(title_text="Hour")
    fig.update_yaxes(title_text=metric[:15])
    save_fig(fig, OUTPUT_DIR / f"line_{field}_{metric}.png", f"{metric} by hour for {field}")


def plot_member_hour_heatmap(df: pd.DataFrame, metric: str, field: str):
    sdf = df[df["field"] == field].copy()
    if sdf.empty or metric not in sdf.columns:
        return

    piv = sdf.pivot_table(index="member", columns="hour", values=metric, aggfunc="mean")
    fig = px.imshow(piv, aspect="auto", labels={"x": "Hour", "y": "Member", "color": metric[:15]})
    fig.update_layout(
        title={
            "text": f"{metric} heatmap ({field})<br><span style='font-size: 18px; font-weight: normal;'>Rows are members and columns are hours</span>"
        }
    )
    fig.update_xaxes(title_text="Hour")
    fig.update_yaxes(title_text="Member")
    save_fig(fig, OUTPUT_DIR / f"heatmap_{field}_{metric}.png", f"{metric} heatmap for {field}")


def plot_metric_scatter(df: pd.DataFrame, xmetric: str, ymetric: str, field: str):
    sdf = df[df["field"] == field].copy()
    if sdf.empty or xmetric not in sdf.columns or ymetric not in sdf.columns:
        return

    fig = px.scatter(sdf, x=xmetric, y=ymetric, color="hour", symbol="member", hover_data=["member", "hour"])
    fig.update_traces(cliponaxis=False)
    fig.update_layout(
        title={
            "text": f"{ymetric} vs {xmetric} ({field})<br><span style='font-size: 18px; font-weight: normal;'>Points are hour-member pairs</span>"
        }
    )
    fig.update_xaxes(title_text=xmetric[:15])
    fig.update_yaxes(title_text=ymetric[:15])
    save_fig(fig, OUTPUT_DIR / f"scatter_{field}_{xmetric}_vs_{ymetric}.png", f"{ymetric} vs {xmetric} for {field}")


def plot_metric_correlation(df: pd.DataFrame, field: str, metrics: list[str]):
    sdf = df[df["field"] == field].copy()
    keep = [m for m in metrics if m in sdf.columns]
    if len(keep) < 2:
        return

    corr = sdf[keep].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", zmin=-1, zmax=1)
    fig.update_layout(
        title={
            "text": f"Metric correlation ({field})<br><span style='font-size: 18px; font-weight: normal;'>Correlation across all hour-member rows</span>"
        }
    )
    fig.update_xaxes(title_text="Metric")
    fig.update_yaxes(title_text="Metric")
    save_fig(fig, OUTPUT_DIR / f"corr_{field}.png", f"Metric correlation for {field}")


def _normalize_continuous(arr: np.ndarray, global_vmax: float | None = None) -> np.ndarray:
    arr = arr.astype(np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros((*arr.shape, 3), dtype=np.uint8)

    vmax = global_vmax if global_vmax is not None else np.nanpercentile(np.abs(arr[finite]), 98)
    vmax = float(max(vmax, 1e-6))

    clipped = np.clip(arr, -vmax, vmax)
    norm = (clipped + vmax) / (2.0 * vmax)

    rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    rgb[..., 0] = (norm * 255).astype(np.uint8)
    rgb[..., 2] = ((1.0 - norm) * 255).astype(np.uint8)
    rgb[..., 1] = (100 * (1.0 - np.abs(norm - 0.5) * 2.0)).astype(np.uint8)
    rgb[~finite] = np.array([0, 0, 0], dtype=np.uint8)
    return rgb


def _render_label(frame: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, img.width, 24), fill=(255, 255, 255))
    draw.text((8, 6), text, fill=(0, 0, 0))
    return np.array(img)


def read_diff_for_gif(paths: list[Path], mode: str = "continuous"):
    frames = []
    global_vmax = None

    if mode == "continuous" and paths:
        vals = []
        for p in paths:
            with rasterio.open(p) as src:
                arr = src.read(1).astype(np.float32)
                finite = np.isfinite(arr)
                if np.any(finite):
                    vals.append(np.nanpercentile(np.abs(arr[finite]), 98))
        global_vmax = max(vals) if vals else 1.0

    for p in sorted(paths):
        with rasterio.open(p) as src:
            if mode == "continuous":
                arr = src.read(1).astype(np.float32)
                rgb = _normalize_continuous(arr, global_vmax=global_vmax)
            else:
                diff = src.read(3).astype(np.int16)
                rgb = np.zeros((diff.shape[0], diff.shape[1], 3), dtype=np.uint8)
                rgb[:] = np.array([240, 240, 240], dtype=np.uint8)
                rgb[diff == 1] = np.array([230, 57, 70], dtype=np.uint8)
                rgb[diff == -1] = np.array([29, 78, 216], dtype=np.uint8)

            rgb = _render_label(rgb, p.stem)
            frames.append(rgb)

    return frames


def make_gif(paths: list[Path], out_path: Path, mode: str = "continuous", duration: float = 0.7):
    if not paths:
        return
    frames = read_diff_for_gif(paths, mode=mode)
    imageio.mimsave(out_path, frames, duration=duration, loop=0)


def build_mean_diff_map(paths: list[Path], out_tif: Path):
    if not paths:
        return

    stack = []
    profile = None
    for p in paths:
        with rasterio.open(p) as src:
            arr = src.read(1).astype(np.float32)
            stack.append(arr)
            if profile is None:
                profile = src.profile.copy()

    mean_arr = np.nanmean(np.stack(stack, axis=0), axis=0).astype(np.float32)
    profile.update(dtype="float32", count=1, compress="deflate", nodata=np.nan)

    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(mean_arr, 1)


def build_png_preview_from_tif(src_tif: Path, out_png: Path, mode: str = "continuous"):
    with rasterio.open(src_tif) as src:
        if mode == "continuous":
            arr = src.read(1).astype(np.float32)
            rgb = _normalize_continuous(arr)
        else:
            diff = src.read(3).astype(np.int16)
            rgb = np.zeros((diff.shape[0], diff.shape[1], 3), dtype=np.uint8)
            rgb[:] = np.array([240, 240, 240], dtype=np.uint8)
            rgb[diff == 1] = np.array([230, 57, 70], dtype=np.uint8)
            rgb[diff == -1] = np.array([29, 78, 216], dtype=np.uint8)

    Image.fromarray(rgb).save(out_png)


def main():
    df = load_intercomparison_csvs(INPUT_DIR)
    fields = sorted(df["field"].dropna().unique())

    for field in fields:
        for metric in ["rmse", "bias", "pearson_r", "perim_mIoU", "perim_hausdorff", "perim_ch_IoU"]:
            plot_hourly_metric(df, metric, field)

        for metric in ["rmse", "perim_mIoU", "perim_hausdorff"]:
            plot_member_hour_heatmap(df, metric, field)

        plot_metric_scatter(df, "rmse", "perim_mIoU", field)
        plot_metric_scatter(df, "bias", "perim_mIoU", field)
        plot_metric_correlation(
            df,
            field,
            ["rmse", "bias", "pearson_r", "perim_mIoU", "perim_hausdorff", "perim_SSIM", "perim_ch_IoU", "perim_ch_SSIM"],
        )

        diff_paths = sorted((INPUT_DIR / "diff_rasters").glob(f"{field}_*_diff_*.tif"))
        perim_paths = sorted((INPUT_DIR / "perimeter_diff_rasters").glob(f"{field}_*_perim_diff_*.tif"))

        make_gif(diff_paths, OUTPUT_DIR / f"{field}_continuous_diff.gif", mode="continuous", duration=0.6)
        make_gif(perim_paths, OUTPUT_DIR / f"{field}_perimeter_diff.gif", mode="perimeter", duration=0.6)

        mean_tif = OUTPUT_DIR / f"{field}_mean_diff.tif"
        build_mean_diff_map(diff_paths, mean_tif)
        build_png_preview_from_tif(mean_tif, OUTPUT_DIR / f"{field}_mean_diff_preview.png", mode="continuous")

        if perim_paths:
            build_png_preview_from_tif(perim_paths[0], OUTPUT_DIR / f"{field}_perimeter_diff_preview.png", mode="perimeter")


if __name__ == "__main__":
    main()
