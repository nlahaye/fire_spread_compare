
from shapely.geometry import MultiLineString, LineString, MultiPoint, Point, MultiPolygon, Polygon
from shapely import to_geojson
import shapely
import rioxarray
import rasterio
import random
import geopandas as gpd
import argparse
import copy
import json 
import os

import numpy as np
import datetime

from utils import read_yaml

init_int_pt_dict = {
  "type": "FeatureCollection",
  "crs": {
    "type": "name",
    "properties": {
      "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
    }
  },
  "features": [
    {
      "type": "Feature",
      "properties": {
          "t": "2020-09-27T12:00:00", #"2020-09-05T00:00:00",
        "FireID": 613,
        "duration": 0.0,
        "FRAPid": 243,
        "FRAPfnm": "GLASS" #"CREEK"
      },
      "geometry": {
        "type": "MultiPoint",
        "coordinates": [
          #[
          #  -119.271812,
          #  37.201797
          #]
        ]
      }
    }
  ]
}


init_fline_dict = {
  "type": "FeatureCollection",
  "crs": {
    "type": "name",
    "properties": {
      "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
    }
  },
  "features": [
    {
      "type": "Feature",
      "properties": {
          "t": "2020-09-27T12:00:00", #"2020-09-05T00:00:00",
        "FireID": 613,
        "duration": 0.0,
        "FRAPid": 243,
        "FRAPfnm": "GLASS" #"CREEK"
      },
      "geometry": {
        "type": "LineString",
        "coordinates": [
          [
            #-119.27352072991958,
            #37.17634614659672
          ],
          ...
        ]
      }
    }
  ]
}



def sample_linestring_points(line: LineString, percent: float, seed: int | None = None) -> MultiPoint:
    """
    Randomly sample a percentage of vertices from a LineString and return a MultiPoint.

    Parameters
    ----------
    line : shapely.geometry.LineString
        Input LineString.
    percent : float
        Percentage of vertices to sample, between 0 and 100.
    seed : int | None
        Optional random seed for reproducibility.

    Returns
    -------
    shapely.geometry.MultiPoint
    """
    if not (0 <= percent <= 100):
        raise ValueError("percent must be between 0 and 100")

    coords = []
    if isinstance(line, MultiLineString) or isinstance(line, MultiPoint):
        coords_tmp = [list(sline.coords) for sline in line.geoms]
        for i in range(len(coords_tmp)):
            coords.extend(coords_tmp[i])
    elif isinstance(line, MultiPolygon) or isinstance(line, Polygon):
        coords = shapely.get_coordinates(line)
    else:
        coords = list(line.coords)

    n = len(coords)

    if n == 0 or percent == 0:
        return MultiPoint([])

    k = max(1, round(n * percent / 100))
    k = min(k, n)

    rng = random.Random(seed)
    indices = rng.sample(range(n), k)
    points = [coords[i] for i in indices]

    return MultiPoint(points)


def multipoint_to_single_multilinestring(multipoint):
    coords = [(pt.x, pt.y) for pt in multipoint.geoms]

    if len(coords) < 2:
        return multipoint

    lines = []
    group_size = 3
    for i in range(0, len(coords), group_size):
        chunk = coords[i:i + group-size]

        if len(chunk) < 2:
            continue

        if len(chunk) > 2:
            chunk = chunk + [chunk[0]]

        lines.append(LineString(chunk))


    return MultiLineString([lines])


def main(yaml_conf):
  
    is_point = False
    if "parquet" in yaml_conf["fname"]:
        x = gpd.read_parquet(yaml_conf["fname"])
        x = x[x['mergeid'] == yaml_conf["mergeid"]] #64500]

        #TODO - handle FIRMS-based CSVs
 
        x['fline'] = x['fline'].set_crs(epsg=9311, inplace=True)
        x['fline'] = x['fline'].to_crs("EPSG:4326")
        pt = x.sort_values(by='t').iloc[yaml_conf["fstart"]]

        tmp =  gpd.GeoSeries(pt['fline'])
        tmp = tmp.set_crs("EPSG:4326")

        fline = pt['fline']
        time = pt['t_st'].strftime('%Y-%m-%dT%H:%M:%S')
        fid = pt['mergeid']
    elif "tif" in yaml_conf["fname"]:
        raster = rioxarray.open_rasterio(yaml_conf["fname"])
        band = raster.sel(band=1)
        mask = band > 0

        results = (
            {'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v) in enumerate(
                rasterio.features.shapes(
                band.values,
                mask=mask.values, # Only vectorize where mask is True
                transform=band.rio.transform()
               )
             )
        )

        geoms = list(results)

        x = gpd.GeoDataFrame.from_features(geoms, crs=band.rio.crs)
        x["geometry"]  = x["geometry"].to_crs("EPSG:4326")
        tmp = gpd.GeoSeries(x.iloc[0]["geometry"])
        tmp = tmp.set_crs("EPSG:4326")
        
        dt = datetime.datetime.strptime(yaml_conf["time"], "%Y-%m-%dT%H:%M:%S")
        time = dt.strftime('%Y-%m-%dT%H:%M:%S')
        fid = yaml_conf["fire_id"]


    elif "fire_nrt" in yaml_conf["fname"] or "fire_archive" in yaml_conf["fname"]:
        is_point = True
        x = gpd.read_file(yaml_conf["fname"])
        dt = datetime.datetime.strptime(yaml_conf["date"], "%Y-%m-%d")
        min_time = yaml_conf["min_time"]
        max_time = yaml_conf["max_time"]
        x['ACQ_TIME'] = x['ACQ_TIME'].astype(np.int64)
        x = x[(x['ACQ_DATE'] == dt) & (x['ACQ_TIME'] >= min_time) & (x['ACQ_TIME'] <= max_time)]


        tmp = gpd.GeoSeries(MultiPoint(x.geometry.values))
        tmp = multipoint_to_single_multilinestring(tmp)

        time = dt.strftime('%Y-%m-%dT%H:%M:%S')
        fid = yaml_conf["fire_id"]
    else:
        x = gpd.read_file(yaml_conf["fname"])
        x["geometry"]  = x["geometry"].to_crs("EPSG:4326")
        tmp = gpd.GeoSeries(x.iloc[0]["geometry"])
        tmp = tmp.set_crs("EPSG:4326")

        dt = datetime.datetime.strptime(yaml_conf["time"], "%Y-%m-%dT%H:%M:%S")
        time = dt.strftime('%Y-%m-%dT%H:%M:%S')
        fid = yaml_conf["fire_id"]
 

    duration = 0.0
 
    pt_dict = copy.deepcopy(init_int_pt_dict)
    pt_dict["features"][-1]["properties"]["t"] = time
    pt_dict["features"][-1]["properties"]["duration"] = duration
    pt_dict["features"][-1]["properties"]["FireID"] = fid
    pt_dict["features"][-1]["properties"]["FRAPfnm"] = yaml_conf["FRAPfnm"]

    fline_dict = copy.deepcopy(init_fline_dict)
    fline_dict["features"][-1]["properties"]["t"] = time
    fline_dict["features"][-1]["properties"]["duration"] = duration
    fline_dict["features"][-1]["properties"]["FireID"] = fid
    fline_dict["features"][-1]["properties"]["FRAPfnm"] = yaml_conf["FRAPfnm"]

    point = tmp.geometry.centroid #iloc[-1].coords[int(tmp.geometry.iloc[-1].length / 2.0)]

 
    if not is_point:
        points = sample_linestring_points(tmp.geometry.iloc[-1], 100) 
    else:
        points = tmp.geometry.iloc[-1]

    print(point)

    pt_dict["features"][-1]["geometry"] = points.__geo_interface__
    fline_dict["features"][-1]["geometry"] = tmp.__geo_interface__["features"][-1]["geometry"]

 
    with open(os.path.join(yaml_conf["config_dir"], str(fid) + "_int_pts.json"), "w") as fp:
        json.dump(pt_dict , fp) 


    with open(os.path.join(yaml_conf["config_dir"], str(fid) + "_fline.json"), "w") as fp:
        json.dump(fline_dict , fp)




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file with config params")
    args = parser.parse_args()
    
    yaml_conf = read_yaml(args.yaml)
    main(yaml_conf)

