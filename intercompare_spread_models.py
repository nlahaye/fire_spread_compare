#!/usr/bin/env python3
"""
Intercompare hourly flame length products from two fire spread models.

Assumptions
-----------
- Each model outputs one raster per hour (e.g., GeoTIFF).
- File names contain an hour token that can be parsed consistently.
- Flame length units are the same between models.
- Rasters may differ in CRS, resolution, extent, or nodata handling.

"""

from __future__ import annotations

import argparse
import json
import math
import re
import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject



def collect_hourly_files(folder: Path, pattern: str, hour_regex: str) -> dict[str, Path]:
    rx = re.compile(hour_regex)
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    matched = {}
 
    for fp in files:
        print(fp, hour_regex)
        m = rx.search(fp)
        if not m:
            continue
        hour_key = m.group(1)
        matched[hour_key] = fp

    return matched


def read_masked(src_path: Path, band: int = 1) -> tuple[np.ndarray, dict]:
    with rasterio.open(src_path) as src:
        arr = src.read(band).astype("float32")
        profile = src.profile.copy()
        nodata = src.nodata

        if nodata is not None:
            arr[arr == nodata] = 0.0

        return arr, {
            "profile": profile,
            "transform": src.transform,
            "crs": src.crs,
            "width": src.width,
            "height": src.height,
            "nodata": src.nodata,
            "bounds": src.bounds,
        }


def reproject_to_match(src_path: Path, match_path: Path, resampling: Resampling) -> np.ndarray:
    with rasterio.open(src_path) as src, rasterio.open(match_path) as match:
        dst = np.full((match.height, match.width), 0.0, dtype="float32")

        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=match.transform,
            dst_crs=match.crs,
            dst_nodata=0.0,
            resampling=resampling,
        )
        return dst


def apply_valid_range(arr: np.ndarray, valid_min: float, valid_max: float | None) -> np.ndarray:
    out = arr.copy()
    if valid_min is not None:
        out[out < valid_min] = 0.0
    if valid_max is not None:
        out[out > valid_max] = 0.0
    return out


def summarize_pair(a: np.ndarray, b: np.ndarray) -> dict:
    valid = np.isfinite(a) & np.isfinite(b)
    n = int(valid.sum())

    if n == 0:
        return {
            "n_valid": 0,
            "mean_a": 0.0,
            "mean_b": 0.0,
            "mean_diff": 0.0,
            "mean_abs_diff": 0.0,
            "rmse": 0.0,
            "bias": 0.0,
            "pearson_r": 0.0,
            "p50_abs_diff": 0.0,
            "p90_abs_diff": 0.0,
            "max_abs_diff": 0.0,
            "agreement_frac_within_0p5": 0.0,
            "agreement_frac_within_1p0": 0.0,
        }

    av = a[valid]
    bv = b[valid]
    diff = av - bv
    abs_diff = np.abs(diff)

    rmse = float(np.sqrt(np.mean(diff**2)))
    bias = float(np.mean(diff))

    if n > 1 and np.nanstd(av) > 0 and np.nanstd(bv) > 0:
        pearson_r = float(np.corrcoef(av, bv)[0, 1])
    else:
        pearson_r = 0.0

    return {
        "n_valid": n,
        "mean_a": float(np.mean(av)),
        "mean_b": float(np.mean(bv)),
        "mean_diff": float(np.mean(diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "rmse": rmse,
        "bias": bias,
        "pearson_r": pearson_r,
        "p50_abs_diff": float(np.percentile(abs_diff, 50)),
        "p90_abs_diff": float(np.percentile(abs_diff, 90)),
        "max_abs_diff": float(np.max(abs_diff)),
        "agreement_frac_within_0p5": float(np.mean(abs_diff <= 0.5)),
        "agreement_frac_within_1p0": float(np.mean(abs_diff <= 1.0)),
    }


def write_diff_raster(reference_path: Path, diff_arr: np.ndarray, out_path: Path):
    with rasterio.open(reference_path) as ref:
        profile = ref.profile.copy()
        profile.update(
            dtype="float32",
            count=1,
            compress="deflate",
            nodata=0.0,
        )

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(diff_arr.astype("float32"), 1)


def main():
    #args = parse_args()
    #model_a_dir = Path(args.model_a_dir)
    #model_b_dir = Path(args.model_b_dir)

    model_a_dir = "fires/creek_fire/pyretechnics-deck/"
    model_b_dir = "fires/creek_fire/elmfire-deck/"
    output_dir = "fires/creek_fire/model_compare/"

    
    os.makedirs(output_dir, exist_ok=True)

    fields = ["crown-fire", "flame-length", "spread-rate",  "hours-since-burned"] #"time-of-arrival", "hours-since-burned"]
    ensemble_members = 10

    ensemble_str = ["10", "20", "30", "40", "50", "60", "70", "80", "90", "99"]

    hour_regex = "(\d{3})"

    valid_min = None
    valid_max = None

    for field in fields:

        for i in range(ensemble_members):
            a_dir = os.path.join(model_a_dir, field)
            b_dir = os.path.join(model_b_dir, field)

            if "time-of-arrival" == field:
                glob_pattern = field + "*.tif"
            else:
                glob_pattern = field + "_" + ensemble_str[i] + "*.tif"
            
            files_a = collect_hourly_files(a_dir, glob_pattern, hour_regex)
            files_b = collect_hourly_files(b_dir, glob_pattern, hour_regex)

            common_hours = sorted(set(files_a) & set(files_b))
            only_a = sorted(set(files_a) - set(files_b))
            only_b = sorted(set(files_b) - set(files_a))

            if not common_hours:
                raise RuntimeError("No overlapping hourly files found between Model A and Model B. " + field)

            resampling = Resampling.bilinear  #get_resampling(args.resampling)
            rows = []

            for hour in common_hours:
                a_path = files_a[hour]
                b_path = files_b[hour]

                #if args.reference == "A":
                ref_path = a_path
                a = reproject_to_match(a_path, ref_path, resampling=Resampling.nearest)
                b = reproject_to_match(b_path, ref_path, resampling=resampling)
                #else:
                #    ref_path = b_path
                #    a = reproject_to_match(a_path, ref_path, resampling=resampling)
                #    b = reproject_to_match(b_path, ref_path, resampling=Resampling.nearest)

                a = apply_valid_range(a, valid_min, valid_max)
                b = apply_valid_range(b, valid_min, valid_max)

                stats = summarize_pair(a, b)
                stats["hour"] = hour
                stats["file_a"] = str(a_path)
                stats["file_b"] = str(b_path)

                with np.errstate(invalid="ignore"):
                    diff = a - b

                out_diff = os.path.join(output_dir,"diff_rasters",f"{field}_diff_{hour}.tif")
                write_diff_raster(ref_path, diff, out_diff)
                stats["diff_raster"] = str(out_diff)

                rows.append(stats)

            df = pd.DataFrame(rows).sort_values("hour")

            summary = {
                "n_common_hours": len(common_hours),
                "hours_only_in_a": only_a,
                "hours_only_in_b": only_b,
                "reference_grid": "pyretechnics",
                "resampling": resampling,
                "valid_min": valid_min,
                "valid_max": valid_max,
                "overall": {
                    "mean_rmse": float(df["rmse"].mean()),
                    "median_rmse": float(df["rmse"].median()),
                    "mean_bias": float(df["bias"].mean()),
                    "mean_pearson_r": float(df["pearson_r"].mean()),
                    "total_valid_pixels_across_hours": int(df["n_valid"].sum()),
                },
            }

            df.to_csv(os.path.join(output_dir, "hourly" + field + "_intercomparison.csv"), index=False)

            with open(os.path.join(output_dir, "hourly" + field + "_intercomparison_summary.json"), "w") as f:
                json.dump(summary, f, indent=2)

            print(f"Wrote CSV: {os.path.join(output_dir, 'hourly_' + field + '_intercomparison.csv')}")
            print(f"Wrote JSON: {os.path.join(output_dir, 'hourly_' + field + '_intercomparison_summary.json')}")
            print(f"Wrote diff rasters to: {os.path.join(output_dir, 'diff_rasters')}")


if __name__ == "__main__":
    main()
