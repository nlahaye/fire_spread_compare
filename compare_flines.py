from shapely.geometry import LineString, MultiPoint, Point, mapping
from shapely import to_geojson, get_precision, set_precision
from shapely.ops import unary_union
from shapely.validation import make_valid

import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling


import numpy as np
import random
import geopandas as gpd
import argparse
import copy
import json 
import os

import cv2

from utils import read_yaml

from scipy.spatial.distance import directed_hausdorff
from skimage.metrics import structural_similarity

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rasterize_geom(geom, transform, width, height):
    shapes = [mapping(geom), 1]
    arr = rasterize(shapes,  out_shape = (height, width), transform=transform,\
            fill=0, dtype=np.int16,
            all_touched=False)
    return arr.astype(np.int16)

def compute_miou(modeled, observed, results = None):

    if results is None:
        results = {}

    for cls, name in [(1,"fire")]:
        pred = (modeled == cls)
        label = (observed == cls)
        inter = np.logical_and(pred, label).sum()
        union = np.logical_or(pred, label).sum()
        iou = float(inter) / float(union) if union > 0 else float("nan")
        results[name] = {"intersection": int(inter), "union": int(union), "iou":iou}

    valid_ious = [v["iou"] for v in results.values() if not np.isnan(v["iou"])]
    results["mIoU"] = float(np.mean(valid_ious)) if valid_ious else float("nan")
    return results


def compute_hausdorff(modeled, observed, results = None):
 
    if results is None:
        results = {}

    rows_m, cols_m = np.where(modeled > 0)
    pts_m = np.column_stack([rows_m, cols_m])

    rows_o, cols_o = np.where(observed > 0)
    pts_o = np.column_stack([rows_o, cols_o])
 
    d_mo, i_m, j_o = directed_hausdorff(pts_m, pts_o)
    d_om, i_o, j_m = directed_hausdorff(pts_o, pts_m)
     
    d = max(d_mo, d_om)

    results["hausdorff"] = d

    return results
    

def compute_ssim(modeled, observed, results = None):

    if results is None:
        results = {}
 
    win_size = min(modeled.shape[0], modeled.shape[1])
    if win_size % 2 == 0:
        win_size = win_size -1
    ssim = structural_similarity(modeled.astype(np.float32), observed.astype(np.float32), data_range=1, gaussian_weights=False, win_size=win_size)

    results["SSIM"] = ssim

    return results

def convex_hull_metrics(modeled, observed, transform, results = None):

    if results is None:
        results = {}

    ch_modeled = create_convex_hull_2(modeled.copy(), transform)
    ch_observed = create_convex_hull_2(observed.copy(), transform)

    results_tmp = compute_miou(ch_modeled, ch_observed)
    results["ch_IoU"] = results_tmp["mIoU"]

    results_tmp = compute_ssim(ch_modeled, ch_observed)
    results["ch_SSIM"] = results_tmp["SSIM"]

    return results, ch_modeled, ch_observed


def create_convex_hull_2(fline_raster, transform):

    rows, cols = np.where(fline_raster == 1)

    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    points = np.column_stack([xs, ys])

    hull = MultiPoint(points).convex_hull
    hull_mask = rasterize(
            [(mapping(hull), 1)],
            out_shape=fline_raster.shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype="int16")

    return hull_mask


def create_convex_hull(fline_raster):

    fline_raster[np.where(fline_raster < 0)] = 0
    fline_raster[np.where(fline_raster > 0)] = 1
    fline_raster = fline_raster.astype(np.uint8) * 255
    fline_raster2 = fline_raster.copy()
    fline_raster3 = fline_raster.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = fline_raster.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(fline_raster3, mask, (0,0), 255)
    cv2.floodFill(fline_raster3, mask, (w-1,0), 255)
    cv2.floodFill(fline_raster3, mask, (w-1, h-1), 255)
    cv2.floodFill(fline_raster3, mask, (0, h-1), 255)
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(fline_raster3)
 

    # Combine the two images to get the foreground.
    im_out = fline_raster | im_floodfill_inv
    im_out = im_out.astype(np.uint8) * 255

    # Find Canny edges
    edged = cv2.Canny(im_out, 30, 200)

    # Finding Contours
    # Use a copy of the image e.g. edged.copy()
    # since findContours alters the image
    contours, hierarchy = cv2.findContours(im_out, \
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print("Number of Contours found = " + str(len(contours)))

    # Draw all contours
    ret = np.zeros(edged.shape)
    cv2.drawContours(ret, contours, -1, 1, thickness=cv2.FILLED)

    return ret



#TODO - edit and integrate as a third option/middle ground for polygon comparisons

def close_fireline_and_fill(
    raster_path,
    fire_value=1,
    band=1,
    out_path=None,
    all_touched=True,
    sort_points=True,
):
    """
    Take a raster fire line, close it into a polygon, and fill that polygon.

    Parameters
    ----------
    raster_path : str
        Input raster path.
    fire_value : int or float, default=1
        Raster value representing the fire line.
    band : int, default=1
        Band to read.
    out_path : str or None, default=None
        Optional output GeoTIFF path for the filled polygon raster.
    all_touched : bool, default=True
        Passed to rasterize().
    sort_points : bool, default=True
        If True, sort fire-line pixels by angle around centroid before closing.

    Returns
    -------
    dict
        Contains polygon geometry, raster mask, bounds, area, CRS, and transform.
    """
    with rasterio.open(raster_path) as src:
        arr = src.read(band)
        transform = src.transform
        crs = src.crs
        profile = src.profile.copy()

        rows, cols = np.where(arr == fire_value)

        if len(rows) < 3:
            raise ValueError("Need at least 3 fire-line pixels to form a polygon.")

        xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
        coords = np.column_stack([xs, ys])

        if sort_points:
            centroid = coords.mean(axis=0)
            angles = np.arctan2(coords[:, 1] - centroid[1], coords[:, 0] - centroid[0])
            coords = coords[np.argsort(angles)]

        # Close the line if needed
        if not np.allclose(coords[0], coords[-1]):
            coords = np.vstack([coords, coords[0]])

        line = LineString(coords)

        if not line.is_closed:
            raise ValueError("Failed to create a closed fire line.")

        polygon = Polygon(line)

        if not polygon.is_valid:
            polygon = make_valid(polygon)

        if polygon.is_empty:
            raise ValueError("Polygon is empty after closing/validation.")

        filled = rasterize(
            [(polygon, 1)],
            out_shape=arr.shape,
            transform=transform,
            fill=0,
            all_touched=all_touched,
            dtype="uint8",
        )

        if out_path is not None:
            profile.update(dtype="uint8", count=1, nodata=0, compress="lzw")
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(filled, 1)

        return {
            "polygon": polygon,
            "filled_raster": filled if out_path is None else None,
            "bounds": polygon.bounds,
            "area": polygon.area,
            "crs": crs,
            "transform": transform,
            "out_path": out_path,
        }



def write_difference_tif(modeled, observed, transform, crs_wkt, output_path, metrics):

    diff = np.zeros_like(modeled, np.int16)
    #diff[modeled & observed] = 0
    #diff[~modeled & ~observed] = 0
    #diff[~modeled & observed] = -1
    #diff[modeled & ~observed] = 1
    modeled_bin = copy.deepcopy(modeled)
    modeled_bin[np.where(modeled > 0)] = 1
    diff = modeled_bin - observed

    profile = {
            "driver" : "GTiff",
            "dtype" : "int16",
            "width": diff.shape[1],
            "height" : diff.shape[0],
            "count" : 3,
            "crs" : crs_wkt,
            "transform" : transform,
            "compress" : "lzw",
            "predictor" : 2,
            "tiled" : True,
            "blockxsize" : 256,
            "blockysize" : 256
    }
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(modeled.astype(np.int16), 2)
        dst.write(observed.astype(np.int16), 1)
        dst.write(diff, 3)

        dst.update_tags(
            BAND1_DESCRIPTION="Reference mask — (0=background, 1=fire)",
            BAND2_DESCRIPTION="Prediction mask — (0=background, 1=fire)",
            BAND3_DESCRIPTION="Difference: 0=Agree, 1=FP, -1=FN",
            MIOU=f"{metrics['mIoU']:.6f}",
            IOU_FIRE=f"{metrics['fire']['iou']:.6f}",
        )
        dst.set_band_description(1, "Reference (GeoParquet)")
        dst.set_band_description(2, "Prediction (GeoJSON)")
        dst.set_band_description(3, "Difference map")

    print(f"[INFO] GeoTIFF written --> {output_path}")


def dissolve_to_polygon(gdf):

    geom = set_precision(gdf, 0.0027)
    geom = unary_union(geom)
    geom_type = geom.geom_type.lower()

    return geom


def compute_raster_grid(geom_observed, resolution):
 
    minx = geom_observed.bounds[0]
    miny = geom_observed.bounds[1]
    maxx = geom_observed.bounds[2]
    maxy = geom_observed.bounds[3]

    pad_x = (maxx - minx) * 0.05
    pad_y = (maxy - miny) * 0.05

    minx -= pad_x
    maxx += pad_x
    miny -= pad_y
    maxy += pad_y

    width = max(1, int(np.ceil((maxx - minx) / resolution)))
    height = max(1, int(np.ceil((maxy - miny) / resolution)))

    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    return transform, width, height





 
def main(yaml_conf):
 
    if "parquet" in yaml_conf["fname"]:
        x = gpd.read_parquet(yaml_conf["fname"])
        x = x[x['mergeid'] == yaml_conf["mergeid"]]

    x['fline'] = x['fline'].set_crs(epsg=9311, inplace=True)
    x['fline'] = x['fline'].to_crs("EPSG:4326")
    x['hull'] = x['hull'].to_crs("EPSG:4326")
    pt = x.sort_values(by='t').iloc[yaml_conf["fcompare"]]
    pt_init = x.sort_values(by='t').iloc[yaml_conf["fstart"]]

    resolution = yaml_conf["resolution_deg"]
 
    init_geom = dissolve_to_polygon(pt_init["fline"])
    geom_observed = dissolve_to_polygon(pt["fline"])

    transform_init, width_init, height_init = compute_raster_grid(init_geom, resolution)
    transform, width, height = compute_raster_grid(geom_observed, resolution)

    mask_init = rasterize_geom(init_geom, transform_init, width_init, height_init)
    mask_observed = rasterize_geom(geom_observed, transform, width, height)

    output_field = yaml_conf["output_field"]

    fdir = os.path.join(yaml_conf["model_dir"], output_field) #"pyretechnics-deck/flame-length/"
    percentiles = yaml_conf["percentiles"] #[10,20,30,40,50,60,70,80,90,99]

 
    profile = {
            "driver" : "GTiff",
            "dtype" : "int16",
            "width": mask_observed.shape[1],
            "height" : mask_observed.shape[0],
            "count" : 1,
            "crs" : x["hull"].crs.to_wkt(),
            "transform" : transform,
            "compress" : "lzw",
            "predictor" : 2,
            "tiled" : True,
            "blockxsize" : 256,
            "blockysize" : 256
    } 

    output_path = os.path.join(yaml_conf["output_dir"], yaml_conf["fire_name"] + "_final.tif")

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mask_observed.astype(np.int16), 1)


    profile_init = {
            "driver" : "GTiff",
            "dtype" : "int16",
            "width": mask_init.shape[1],
            "height" : mask_init.shape[0],
            "count" : 1,
            "crs" : x["hull"].crs.to_wkt(),
            "transform" : transform,
            "compress" : "lzw",
            "predictor" : 2,
            "tiled" : True,
            "blockxsize" : 256,
            "blockysize" : 256
    }

    output_path = os.path.join(yaml_conf["output_dir"], yaml_conf["fire_name"] + "_init.tif")

    with rasterio.open(output_path, "w", **profile_init) as dst:
        dst.write(mask_init.astype(np.int16), 1)


 
    resampled = None
    hist_vals = { "IoU" : [], "ch_SSIM" : [], "hausdorff" : [], "ch_IoU" : [] }

    for i in range(len(percentiles)):
        fdir_full = os.path.join(fdir, output_field + "_" + str(percentiles[i]) + "_" + yaml_conf["compare_tint"] + ".tif")
        src = rasterio.open(fdir_full)
        mask_modeled = src.read(1)
        mask_modeled[np.where(mask_modeled > 0)] =  1
        mask_modeled[np.where(mask_modeled < 1)] = 0
        mask_modeled = mask_modeled.astype(np.int16)
 
        resampled = np.empty(mask_observed.shape)
        #add_axis for band
        reproject(
                source = rasterio.band(src, 1),
                destination = resampled,
                src_transform=src.transform,
                src_crs = src.crs,
                dst_transform=transform,
                dst_crs= x["hull"].crs,
                resampling=Resampling.nearest)

        with rasterio.open(os.path.join(yaml_conf["output_dir"], yaml_conf["fire_name"] + "_" + output_field + "_" + str(percentiles[i]) + "_" + yaml_conf["compare_tint"] + ".tif"), "w", **profile) as dst:
            dst.write(np.squeeze(resampled.astype(np.int16)), 1)

        resampled = resampled.astype(np.int16)


        modeled_bin = copy.deepcopy(resampled)
        modeled_bin[np.where(resampled > 0)] = 1

        metrics = None
        metrics = compute_miou(modeled_bin, mask_observed, metrics)
        metrics = compute_hausdorff(modeled_bin, mask_observed, metrics)
        metrics, ch_modeled, ch_observed = convex_hull_metrics(modeled_bin, mask_observed, transform, metrics)
 
        hist_vals["IoU"].append(metrics["mIoU"])
        hist_vals["hausdorff"].append(metrics["hausdorff"])
        hist_vals["ch_SSIM"].append(metrics["ch_SSIM"])
        hist_vals["ch_IoU"].append(metrics["ch_IoU"])

        print(metrics)

        uid = yaml_conf["fire_name"] + "_" + output_field + "_" + str(percentiles[i]) + "_" + yaml_conf["compare_tint"]
        target_crs =src.crs
        write_difference_tif(resampled, mask_observed, transform, x["hull"].crs.to_wkt(), os.path.join(yaml_conf["output_dir"],"Diff_" + uid + ".tif"), metrics)

        write_difference_tif(ch_modeled, ch_observed, transform, x["hull"].crs.to_wkt(), os.path.join(yaml_conf["output_dir"],"Diff_CH_" + uid + ".tif"), metrics)


    percentiles = [0] + percentiles

    plt.stairs(hist_vals["IoU"], percentiles, fill=True)
    plt.savefig(os.path.join(yaml_conf["output_dir"], yaml_conf["fire_name"] + "_" + output_field + "_IoU_hist.png"), dpi=400)

    plt.clf()
    plt.stairs(hist_vals["hausdorff"], percentiles, fill=True)
    plt.savefig(os.path.join(yaml_conf["output_dir"], yaml_conf["fire_name"] + "_" + output_field + "_hausdorff_hist.png"), dpi=400)

    plt.clf()
    plt.stairs(hist_vals["ch_SSIM"], percentiles, fill=True)
    plt.savefig(os.path.join(yaml_conf["output_dir"], yaml_conf["fire_name"] + "_" + output_field + "_chSSIM_hist.png"), dpi=400)

    plt.clf()
    plt.stairs(hist_vals["ch_IoU"], percentiles, fill=True)
    plt.savefig(os.path.join(yaml_conf["output_dir"], yaml_conf["fire_name"] + "_" + output_field + "_chIoU_hist.png"), dpi=400)

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file with config params")
    args = parser.parse_args()

    yaml_conf = read_yaml(args.yaml)
    main(yaml_conf)



