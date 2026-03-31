
from shapely.geometry import LineString, MultiPoint, Point
from shapely import to_geojson
import random
import geopandas as gpd
import copy
import json 


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
        "t": "2020-09-05T00:00:00",
        "FireID": 613,
        "duration": 0.0,
        "FRAPid": 243,
        "FRAPfnm": "CREEK"
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
        "t": "2020-09-05T00:00:00",
        "FireID": 613,
        "duration": 0.0,
        "FRAPid": 243,
        "FRAPfnm": "CREEK"
      },
      "geometry": {
        "type": "LineString",
        "coordinates": [
          [
            -119.27352072991958,
            37.17634614659672
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
    if not isinstance(line, LineString):
        raise TypeError("line must be a shapely.geometry.LineString")
    if not (0 <= percent <= 100):
        raise ValueError("percent must be between 0 and 100")

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


x = gpd.read_parquet("/home/nlahaye/Fire_Fronts/feds_western_us_2020_af_postprocessed.parquet")
x = x[x['mergeid'] == 64500]
x['fline'] = x['fline'].set_crs(epsg=9311, inplace=True)
x['fline'] = x['fline'].to_crs("EPSG:4326")
pt = x.sort_values(by='t').iloc[0]

tmp =  gpd.GeoSeries(pt['fline'])
tmp = tmp.set_crs("EPSG:4326")

fline = pt['fline']
time = pt['t_st'].strftime('%Y-%m-%dT%H:%M:%S')
fid = pt['mergeid']
duration = 0.0
 
pt_dict = copy.deepcopy(init_int_pt_dict)
pt_dict["features"][-1]["properties"]["t"] = time
pt_dict["features"][-1]["properties"]["duration"] = duration
pt_dict["features"][-1]["properties"]["FireID"] = fid

fline_dict = copy.deepcopy(init_fline_dict)

print(tmp.geometry.iloc[-1].coords[int(tmp.geometry.iloc[-1].length / 2.0)])

point = tmp.geometry.iloc[-1].coords[int(tmp.geometry.iloc[-1].length / 2.0)]


points = sample_linestring_points(tmp.geometry.iloc[-1], 100) 
print(points.__geo_interface__)
pt_dict["features"][-1]["geometry"] = points.__geo_interface__

print(tmp.geometry.iloc[-1].intersects(points))
print(points.intersects(Point(point)))

fline_dict["features"][-1]["properties"]["t"] = time
fline_dict["features"][-1]["properties"]["duration"] = duration
fline_dict["features"][-1]["properties"]["FireID"] = fid

fline_dict["features"][-1]["geometry"] = tmp.__geo_interface__

with open(str(fid) + "_int_pts.json", "w") as fp:
  json.dump(pt_dict , fp) 


with open(str(fid) + "_fline.json", "w") as fp:
  json.dump(fline_dict , fp)





