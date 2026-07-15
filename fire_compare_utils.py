from __future__ import annotations

import copy
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from scipy.spatial.distance import directed_hausdorff
from shapely import set_precision
from shapely.geometry import MultiPoint, mapping
from shapely.ops import unary_union
from skimage.metrics import structural_similarity


def collect_hourly_files(folder: Path, pattern: str, hour_regex: str) -> dict[str, Path]:
    rx = re.compile(hour_regex)
    files = sorted(glob.glob(os.path.join(str(folder), pattern)))
    matched: dict[str, Path] = {}
    for fp in files:
        m = rx.search(os.path.basename(fp))
        if not m:
            continue
        matched[m.group(1)] = Path(fp)
    return matched


def get_resampling(name: str) -> Resampling:
    lut = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
    }
    return lut[name]


def cawfe_csv_to_raster(csv_fname, yaml_conf):
    df = pd.read_csv(csv_fname)
    x = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.LON, df.LAT), crs="EPSG:4326"
    )   

    x = x.to_crs("EPSG:4326")
    resolution = yaml_conf["resolution_deg"]


    geom = dissolve_to_polygon(x["geometry"])

    transform, width, height = compute_raster_grid(geom, resolution)
    mask = rasterize_geom(geom, transform, width, height)

    profile = {
            "driver" : "GTiff",
            "dtype" : "int16",
            "width": mask.shape[1],
            "height" : mask.shape[0],
            "count" : 1,
            "crs" : x.crs.to_wkt(),
            "transform" : transform,
            "compress" : "lzw",
            "predictor" : 2,
            "tiled" : True,
            "blockxsize" : 256,
            "blockysize" : 256
    }

    output_path = os.path.join(yaml_conf["output_dir"], yaml_conf["fire_name"] + "_CAWFE.tif")

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mask.astype(np.int16), 1)

    return mask, output_path


def read_raster(src_path: Path, band: int = 1) -> tuple[np.ndarray, dict]:
    with rasterio.open(src_path) as src:
        arr = src.read(band).astype("float32")
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = 0.0
        meta = {
            "profile": src.profile.copy(),
            "transform": src.transform,
            "crs": src.crs,
            "width": src.width,
            "height": src.height,
            "nodata": src.nodata,
            "bounds": src.bounds,
        }
    return arr, meta


def reproject_to_match(
    src_path: Path,
    match_path: Path,
    resampling: Resampling,
    band: int = 1,
    dst_nodata: float = 0.0,
) -> np.ndarray:
    with rasterio.open(src_path) as src, rasterio.open(match_path) as match:
        dst = np.full((match.height, match.width), dst_nodata, dtype="float32")
        reproject(
            source=rasterio.band(src, band),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=match.transform,
            dst_crs=match.crs,
            dst_nodata=dst_nodata,
            resampling=resampling,
        )
    return dst


def apply_valid_range(arr: np.ndarray, valid_min: float | None, valid_max: float | None) -> np.ndarray:
    out = arr.copy()
    if valid_min is not None:
        out[out < valid_min] = 0.0
    if valid_max is not None:
        out[out > valid_max] = 0.0
    return out


def summarize_continuous_pair(a: np.ndarray, b: np.ndarray) -> dict:
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


def binarize_fire_mask(arr: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    out = np.zeros(arr.shape, dtype=np.int16)
    out[np.isfinite(arr) & (arr > threshold)] = 1
    return out


def rasterize_geom(geom, transform, width: int, height: int, all_touched: bool = False) -> np.ndarray:
    arr = rasterize(
        [(mapping(geom), 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.int16,
        all_touched=all_touched,
    )
    return arr.astype(np.int16)


def dissolve_to_polygon(geoms, precision: float = 0.0027):
    geom = set_precision(geoms, precision)
    return unary_union(geom)


def compute_raster_grid(geom, resolution: float, pad_frac: float = 0.05):
    minx, miny, maxx, maxy = geom.bounds
    pad_x = (maxx - minx) * pad_frac
    pad_y = (maxy - miny) * pad_frac
    minx -= pad_x
    maxx += pad_x
    miny -= pad_y
    maxy += pad_y
    width = max(1, int(np.ceil((maxx - minx) / resolution)))
    height = max(1, int(np.ceil((maxy - miny) / resolution)))
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    return transform, width, height


def compute_miou(modeled: np.ndarray, observed: np.ndarray, results: dict | None = None) -> dict:
    if results is None:
        results = {}

    pred = modeled == 1
    label = observed == 1
    inter = np.logical_and(pred, label).sum()
    union = np.logical_or(pred, label).sum()
    iou = float(inter) / float(union) if union > 0 else float("nan")

    results["fire"] = {"intersection": int(inter), "union": int(union), "iou": iou}
    valid_ious = [
        v["iou"] for v in results.values()
        if isinstance(v, dict) and "iou" in v and not np.isnan(v["iou"])
    ]
    results["mIoU"] = float(np.mean(valid_ious)) if valid_ious else float("nan")
    return results


def compute_hausdorff(modeled: np.ndarray, observed: np.ndarray, results: dict | None = None) -> dict:
    if results is None:
        results = {}

    rows_m, cols_m = np.where(modeled > 0)
    rows_o, cols_o = np.where(observed > 0)

    if len(rows_m) == 0 or len(rows_o) == 0:
        results["hausdorff"] = float("nan")
        return results

    pts_m = np.column_stack([rows_m, cols_m])
    pts_o = np.column_stack([rows_o, cols_o])

    d_mo = directed_hausdorff(pts_m, pts_o)[0]
    d_om = directed_hausdorff(pts_o, pts_m)[0]
    results["hausdorff"] = float(max(d_mo, d_om))
    return results


def compute_ssim(modeled: np.ndarray, observed: np.ndarray, results: dict | None = None) -> dict:
    if results is None:
        results = {}

    if modeled.shape != observed.shape:
        raise ValueError("compute_ssim requires arrays with identical shape")

    win_size = min(modeled.shape[0], modeled.shape[1], 7)
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        results["SSIM"] = float("nan")
        return results

    ssim = structural_similarity(
        modeled.astype(np.float32),
        observed.astype(np.float32),
        data_range=1,
        gaussian_weights=False,
        win_size=win_size,
    )
    results["SSIM"] = float(ssim)
    return results


def create_convex_hull_mask(binary_mask: np.ndarray, transform) -> np.ndarray:
    rows, cols = np.where(binary_mask == 1)
    if len(rows) < 3:
        return np.zeros(binary_mask.shape, dtype=np.int16)

    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    points = np.column_stack([xs, ys])
    hull = MultiPoint(points).convex_hull

    hull_mask = rasterize(
        [(mapping(hull), 1)],
        out_shape=binary_mask.shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="int16",
    )
    return hull_mask.astype(np.int16)


def convex_hull_metrics(modeled: np.ndarray, observed: np.ndarray, transform, results: dict | None = None):
    if results is None:
        results = {}

    ch_modeled = create_convex_hull_mask(modeled, transform)
    ch_observed = create_convex_hull_mask(observed, transform)

    tmp = compute_miou(ch_modeled, ch_observed)
    results["ch_IoU"] = tmp["mIoU"]

    tmp = compute_ssim(ch_modeled, ch_observed)
    results["ch_SSIM"] = tmp["SSIM"]
    return results, ch_modeled, ch_observed


def summarize_binary_pair(a_bin: np.ndarray, b_bin: np.ndarray, transform) -> tuple[dict, np.ndarray, np.ndarray]:
    results = {}
    results = compute_miou(a_bin, b_bin, results)
    results = compute_hausdorff(a_bin, b_bin, results)
    results = compute_ssim(a_bin, b_bin, results)
    results, ch_a, ch_b = convex_hull_metrics(a_bin, b_bin, transform, results)
    results["n_fire_a"] = int((a_bin > 0).sum())
    results["n_fire_b"] = int((b_bin > 0).sum())
    results["fire_area_ratio_a_to_b"] = (
        float(results["n_fire_a"]) / float(results["n_fire_b"])
        if results["n_fire_b"] > 0 else float("nan")
    )
    return results, ch_a, ch_b


def flatten_metrics(metrics: dict, prefix: str | None = None) -> dict:
    flat = {}
    for k, v in metrics.items():
        key = f"{prefix}_{k}" if prefix else k
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"{key}_{kk}"] = vv
        else:
            flat[key] = v
    return flat


def write_single_band_raster(
    reference_path: Path,
    arr: np.ndarray,
    out_path: Path,
    dtype: str = "float32",
    nodata: float = 0.0,
):
    with rasterio.open(reference_path) as ref:
        profile = ref.profile.copy()
        profile.update(dtype=dtype, count=1, compress="deflate", nodata=nodata)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr.astype(dtype), 1)


def write_difference_tif(
    modeled: np.ndarray,
    observed: np.ndarray,
    transform,
    crs_wkt,
    output_path: Path,
    metrics: dict,
):
    modeled_bin = copy.deepcopy(modeled).astype(np.int16)
    modeled_bin[modeled_bin > 0] = 1

    observed_bin = copy.deepcopy(observed).astype(np.int16)
    observed_bin[observed_bin > 0] = 1

    diff = modeled_bin - observed_bin
    profile = {
        "driver": "GTiff",
        "dtype": "int16",
        "width": diff.shape[1],
        "height": diff.shape[0],
        "count": 3,
        "crs": crs_wkt,
        "transform": transform,
        "compress": "lzw",
        "predictor": 2,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(observed_bin, 1)
        dst.write(modeled_bin, 2)
        dst.write(diff.astype(np.int16), 3)

        dst.update_tags(
            BAND1_DESCRIPTION="Reference mask — (0=background, 1=fire)",
            BAND2_DESCRIPTION="Prediction mask — (0=background, 1=fire)",
            BAND3_DESCRIPTION="Difference: 0=Agree, 1=FP, -1=FN",
            MIOU=str(metrics.get("mIoU", np.nan)),
            IOU_FIRE=str(metrics.get("fire", {}).get("iou", np.nan)),
            HAUSDORFF=str(metrics.get("hausdorff", np.nan)),
            SSIM=str(metrics.get("SSIM", np.nan)),
            CH_IOU=str(metrics.get("ch_IoU", np.nan)),
            CH_SSIM=str(metrics.get("ch_SSIM", np.nan)),
        )

        dst.set_band_description(1, "Reference")
        dst.set_band_description(2, "Prediction")
        dst.set_band_description(3, "Difference map")


def compare_binary_fire_perimeters(
    a_arr: np.ndarray,
    b_arr: np.ndarray,
    ref_meta: dict,
    fire_threshold: float = 0.0,
    diff_output_path: Path | None = None,
    ch_diff_output_path: Path | None = None,
) -> dict:
    a_bin = binarize_fire_mask(a_arr, threshold=fire_threshold)
    b_bin = binarize_fire_mask(b_arr, threshold=fire_threshold)

    metrics, ch_a, ch_b = summarize_binary_pair(a_bin, b_bin, ref_meta["transform"])
    metrics["binary_threshold"] = fire_threshold

    crs_wkt = ref_meta["crs"].to_wkt() if hasattr(ref_meta["crs"], "to_wkt") else str(ref_meta["crs"])

    if diff_output_path is not None:
        write_difference_tif(a_bin, b_bin, ref_meta["transform"], crs_wkt, diff_output_path, metrics)
    if ch_diff_output_path is not None:
        write_difference_tif(ch_a, ch_b, ref_meta["transform"], crs_wkt, ch_diff_output_path, metrics)

    return flatten_metrics(metrics, prefix="perim")


def write_json(data: dict, output_path: Path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def write_csv(df: pd.DataFrame, output_path: Path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)




