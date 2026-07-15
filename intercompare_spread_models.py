#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling

from fire_compare_utils import (
    apply_valid_range,
    collect_hourly_files,
    compare_binary_fire_perimeters,
    get_resampling,
    reproject_to_match,
    summarize_continuous_pair,
    write_csv,
    write_json,
    write_single_band_raster,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Intercompare hourly spread products from two model directories.")
    parser.add_argument("--model-a-dir", type=Path, required=True)
    parser.add_argument("--model-b-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fields", nargs="+", default=["crown-fire", "flame-length", "spread-rate", "hours-since-burned"])
    parser.add_argument("--ensemble-str", nargs="+", default=["10", "20", "30", "40", "50", "60", "70", "80", "90", "99"])
    parser.add_argument("--hour-regex", default=r"(\d{3})")
    parser.add_argument("--valid-min", type=float, default=None)
    parser.add_argument("--valid-max", type=float, default=None)
    parser.add_argument("--reference", choices=["A", "B"], default="A")
    parser.add_argument("--binary-threshold", type=float, default=0.0)
    parser.add_argument("--resampling", choices=["nearest", "bilinear", "cubic"], default="bilinear")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    resampling = get_resampling(args.resampling)

    for field in args.fields:
        for member_tag in args.ensemble_str:

            a_dir = args.model_a_dir / field
            b_dir = args.model_b_dir / field

            if field == "time-of-arrival":
                glob_pattern = f"{field}*.tif"
                member_name = "toa"
            else:
                glob_pattern = f"{field}_{member_tag}*.tif"
                member_name = member_tag

            files_a = collect_hourly_files(a_dir, glob_pattern, args.hour_regex)
            files_b = collect_hourly_files(b_dir, glob_pattern, args.hour_regex)
            
            common_hours = sorted(set(files_a) & set(files_b))
            only_a = sorted(set(files_a) - set(files_b))
            only_b = sorted(set(files_b) - set(files_a))

            if not common_hours:
                continue

            rows = []
            for hour in common_hours:
                a_path = files_a[hour]
                b_path = files_b[hour]

                if args.reference == "A":
                    ref_path = a_path
                    a = reproject_to_match(a_path, ref_path, Resampling.nearest)
                    b = reproject_to_match(b_path, ref_path, resampling)
                else:
                    ref_path = b_path
                    a = reproject_to_match(a_path, ref_path, resampling)
                    b = reproject_to_match(b_path, ref_path, Resampling.nearest)

                a = apply_valid_range(a, args.valid_min, args.valid_max)
                b = apply_valid_range(b, args.valid_min, args.valid_max)

                stats = summarize_continuous_pair(a, b)
                stats.update(
                    {
                        "field": field,
                        "member": member_name,
                        "hour": str(hour),
                        "file_a": str(a_path),
                        "file_b": str(b_path),
                    }
                )

                with np.errstate(invalid="ignore"):
                    diff = a - b

                diff_path = output_dir / "diff_rasters" / f"{field}_{member_name}_diff_{hour}.tif"
                write_single_band_raster(ref_path, diff, diff_path, dtype="float32", nodata=0.0)
                stats["diff_raster"] = str(diff_path)

                with rasterio.open(ref_path) as ref_src:
                    ref_meta = {
                        "transform": ref_src.transform,
                        "crs": ref_src.crs,
                        "width": ref_src.width,
                        "height": ref_src.height,
                    }

                perim_diff = output_dir / "perimeter_diff_rasters" / f"{field}_{member_name}_perim_diff_{hour}.tif"
                perim_ch_diff = output_dir / "perimeter_diff_rasters" / f"{field}_{member_name}_perim_ch_diff_{hour}.tif"

                perim_stats = compare_binary_fire_perimeters(
                    a_arr=a,
                    b_arr=b,
                    ref_meta=ref_meta,
                    fire_threshold=args.binary_threshold,
                    diff_output_path=perim_diff,
                    ch_diff_output_path=perim_ch_diff,
                )

                stats.update(perim_stats)
                stats["perim_diff_raster"] = str(perim_diff)
                stats["perim_ch_diff_raster"] = str(perim_ch_diff)
                rows.append(stats)

            df = pd.DataFrame(rows).sort_values("hour")
            csv_path = output_dir / f"hourly_{field}_{member_name}_intercomparison.csv"
            json_path = output_dir / f"hourly_{field}_{member_name}_intercomparison_summary.json"

            summary = {
                "field": field,
                "member": member_name,
                "n_common_hours": len(common_hours),
                "hours_only_in_a": only_a,
                "hours_only_in_b": only_b,
                "reference_grid": args.reference,
                "resampling": args.resampling,
                "valid_min": args.valid_min,
                "valid_max": args.valid_max,
                "binary_threshold": args.binary_threshold,
                "overall": {
                    "mean_rmse": float(df["rmse"].mean()),
                    "median_rmse": float(df["rmse"].median()),
                    "mean_bias": float(df["bias"].mean()),
                    "mean_pearson_r": float(df["pearson_r"].mean()),
                    "total_valid_pixels_across_hours": int(df["n_valid"].sum()),
                    "mean_perim_miou": float(df["perim_mIoU"].mean()) if "perim_mIoU" in df.columns else float("nan"),
                    "mean_perim_hausdorff": float(df["perim_hausdorff"].mean()) if "perim_hausdorff" in df.columns else float("nan"),
                    "mean_perim_ssim": float(df["perim_SSIM"].mean()) if "perim_SSIM" in df.columns else float("nan"),
                    "mean_perim_ch_iou": float(df["perim_ch_IoU"].mean()) if "perim_ch_IoU" in df.columns else float("nan"),
                    "mean_perim_ch_ssim": float(df["perim_ch_SSIM"].mean()) if "perim_ch_SSIM" in df.columns else float("nan"),
                },
            }

            write_csv(df, csv_path)
            write_json(summary, json_path)
            print(f"Wrote CSV: {csv_path}")
            print(f"Wrote JSON: {json_path}")


if __name__ == "__main__":
    main()
