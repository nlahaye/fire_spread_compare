#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from utils import read_yaml
from fire_compare_utils import (
    compare_binary_fire_perimeters,
    compute_raster_grid,
    dissolve_to_polygon,
    rasterize_geom,
    write_single_band_raster,
)


def build_observed_masks(yaml_conf: dict):
    x = gpd.read_parquet(yaml_conf["fname"])
    x = x[x["mergeid"] == yaml_conf["mergeid"]].copy()

    x["fline"] = x["fline"].set_crs(epsg=9311, allow_override=True)
    x["fline"] = x["fline"].to_crs("EPSG:4326")
    x["hull"] = x["hull"].to_crs("EPSG:4326")

    pt = x.sort_values(by="t").iloc[yaml_conf["fcompare"]]
    pt_init = x.sort_values(by="t").iloc[yaml_conf["fstart"]]

    resolution = yaml_conf["resolution_deg"]
    init_geom = dissolve_to_polygon(pt_init["fline"])
    observed_geom = dissolve_to_polygon(pt["fline"])

    transform_init, width_init, height_init = compute_raster_grid(init_geom, resolution)
    transform, width, height = compute_raster_grid(observed_geom, resolution)

    mask_init = rasterize_geom(init_geom, transform_init, width_init, height_init)
    mask_observed = rasterize_geom(observed_geom, transform, width, height)
    crs = x["hull"].crs

    return {
        "gdf": x,
        "transform": transform,
        "width": width,
        "height": height,
        "crs": crs,
        "mask_init": mask_init.astype(np.int16),
        "mask_observed": mask_observed.astype(np.int16),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", required=True, help="YAML file with config params")
    return parser.parse_args()


def main(yaml_conf: dict):
    output_dir = Path(yaml_conf["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    observed = build_observed_masks(yaml_conf)
    output_field = yaml_conf["output_field"]
    model_dir = Path(yaml_conf["model_dir"]) / output_field
    is_cawfe = yaml_conf.get("is_cawfe", False)
    percentiles = yaml_conf["percentiles"] if not is_cawfe else ["00"]

    ref_path = output_dir / f"{yaml_conf['fire_name']}_final.tif"
    profile = {
        "driver": "GTiff",
        "dtype": "int16",
        "width": observed["mask_observed"].shape[1],
        "height": observed["mask_observed"].shape[0],
        "count": 1,
        "crs": observed["crs"].to_wkt(),
        "transform": observed["transform"],
        "compress": "lzw",
        "predictor": 2,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "nodata": 0,
    }
    with rasterio.open(ref_path, "w", **profile) as dst:
        dst.write(observed["mask_observed"].astype(np.int16), 1)

    init_ref_path = output_dir / f"{yaml_conf['fire_name']}_init.tif"
    with rasterio.open(init_ref_path, "w", **profile) as dst:
        dst.write(observed["mask_init"].astype(np.int16), 1)

    hist_vals = {"IoU": [], "ch_SSIM": [], "hausdorff": [], "ch_IoU": []}

    for pct in percentiles:

        if is_cawfe:
            fdir_full = yaml_conf["cawfe_csv"]
            model_path = os.path.splitext(fdir_full)[0] + ".tif"
            mask_modeled = cawfe_csv_to_raster(fdir_full, model_path, yaml_conf)
        else:
            model_path = model_dir / f"{output_field}_{pct}_{yaml_conf['compare_tint']}.tif"

        
        with rasterio.open(model_path) as src:
            resampled = np.zeros(observed["mask_observed"].shape, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=resampled,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=observed["transform"],
                dst_crs=observed["crs"],
                resampling=Resampling.nearest,
            )

        resampled[np.where(resampled > 0)] = 1
        resampled[np.where(resampled < 1)] = 0
        resampled = resampled.astype(np.int16)

        model_mask_path = output_dir / f"{yaml_conf['fire_name']}_{output_field}_{pct}_{yaml_conf['compare_tint']}.tif"
        write_single_band_raster(ref_path, resampled, model_mask_path, dtype="int16", nodata=0)

        ref_meta = {
            "transform": observed["transform"],
            "crs": observed["crs"],
            "width": observed["width"],
            "height": observed["height"],
        }

        diff_path = output_dir / f"Diff_{yaml_conf['fire_name']}_{output_field}_{pct}_{yaml_conf['compare_tint']}.tif"
        ch_diff_path = output_dir / f"Diff_CH_{yaml_conf['fire_name']}_{output_field}_{pct}_{yaml_conf['compare_tint']}.tif"

        perim_stats = compare_binary_fire_perimeters(
            a_arr=resampled,
            b_arr=observed["mask_observed"],
            ref_meta=ref_meta,
            fire_threshold=0.0,
            diff_output_path=diff_path,
            ch_diff_output_path=ch_diff_path,
        )

        hist_vals["IoU"].append(perim_stats["perim_mIoU"])
        hist_vals["hausdorff"].append(perim_stats["perim_hausdorff"])
        hist_vals["ch_SSIM"].append(perim_stats["perim_ch_SSIM"])
        hist_vals["ch_IoU"].append(perim_stats["perim_ch_IoU"])
        print(perim_stats)

    stair_bins = [0] + list(percentiles)

    plt.stairs(hist_vals["IoU"], stair_bins, fill=True)
    plt.savefig(output_dir / f"{yaml_conf['fire_name']}_{output_field}_IoU_hist.png", dpi=400)
    plt.clf()

    plt.stairs(hist_vals["hausdorff"], stair_bins, fill=True)
    plt.savefig(output_dir / f"{yaml_conf['fire_name']}_{output_field}_hausdorff_hist.png", dpi=400)
    plt.clf()

    plt.stairs(hist_vals["ch_SSIM"], stair_bins, fill=True)
    plt.savefig(output_dir / f"{yaml_conf['fire_name']}_{output_field}_chSSIM_hist.png", dpi=400)
    plt.clf()

    plt.stairs(hist_vals["ch_IoU"], stair_bins, fill=True)
    plt.savefig(output_dir / f"{yaml_conf['fire_name']}_{output_field}_chIoU_hist.png", dpi=400)


if __name__ == "__main__":
    args = parse_args()
    yaml_conf = read_yaml(args.yaml)
    main(yaml_conf)
