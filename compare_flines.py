from shapely.geometry import LineString, MultiPoint, Point, mapping
from shapely import to_geojson, get_precision, set_precision
from shapely.ops import unary_union
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

from utils import read_yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rasterize_geom(geom, transform, width, height):
    shapes = [mapping(geom), 1]
    arr = rasterize(shapes,  out_shape = (height, width), transform=transform,\
            fill=0, dtype=np.int16,
            all_touched=False)
    return arr.astype(np.int16)

def compute_miou(modeled, observed):
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
    hist_vals = []

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

        metrics = compute_miou(modeled_bin, mask_observed)

        hist_vals.append(metrics["mIoU"])

        print(metrics)

        uid = yaml_conf["fire_name"] + "_" + output_field + "_" + str(percentiles[i]) + "_" + yaml_conf["compare_tint"]
        target_crs =src.crs
        write_difference_tif(resampled, mask_observed, transform, x["hull"].crs.to_wkt(), os.path.join(yaml_conf["output_dir"],"Diff_" + uid + ".tif"), metrics)


    percentiles = [0] + percentiles
    print(hist_vals)
    plt.stairs(hist_vals, percentiles, fill=True)
    plt.savefig(os.path.join(yaml_conf["output_dir"], yaml_conf["fire_name"] + "_" + output_field + "_hist.png"), dpi=400)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file with config params")
    args = parser.parse_args()

    yaml_conf = read_yaml(args.yaml)
    main(yaml_conf)



