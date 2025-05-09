#!/usr/bin/env python3
import sys
from pathlib import Path

# ensure project root is on sys.path so `src` package is found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import zipfile
import tempfile
import numpy as np
import pandas as pd
import geopandas as gpd
import toml
from io import StringIO
from unidecode import unidecode
from datetime import timedelta
from scipy.interpolate import PchipInterpolator
from shapely.ops import substring, linemerge
from shapely.geometry import Point
from osmnx.features import features_from_bbox

from src.load_data import *
from src.feature_building_utils import *
from src.geometric_utils import *

# --------------------------------------------------
# Core spatial helpers (<20 lines)
# --------------------------------------------------


def extract_kml(kmz_path: Path) -> gpd.GeoDataFrame:
    """Extract and read KML from KMZ archive."""
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(kmz_path, "r") as z:
            z.extractall(tmp)
        kml_file = Path(tmp) / "doc.kml"
        return gpd.read_file(kml_file, driver="KML")


def build_segments(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Split a closed LineString into ordered segments with fixed location codes."""
    line = gdf.geometry.iloc[0]
    pts = gdf.geometry.iloc[1:].drop_duplicates().tolist()
    dists = sorted(line.project(pt) for pt in pts)
    wrap = linemerge(
        [substring(line, dists[-1], line.length), substring(line, 0.0, dists[0])]
    )
    segs = [wrap] + [substring(line, a, b) for a, b in zip(dists[:-1], dists[1:])]
    # fixed segment codes matching data['location']
    locs = ["FG", "GH", "AB", "BC", "CD", "DE", "EF"]
    return gpd.GeoDataFrame({"location": locs, "geometry": segs}, crs=gdf.crs)


def interpolate_dynamic(data: pd.DataFrame, segments: gpd.GeoDataFrame) -> pd.DataFrame:
    """Interpolate along segments for 'dynamic' regime points, preserving original index."""
    # select dynamic rows and preserve original index
    S_dyn = data[data["regime"] == "dynamic"]
    orig_idx = S_dyn.index
    # merge to bring in geometry
    sd = S_dyn.merge(segments, on="location", how="left")
    # restore original time index
    sd.index = orig_idx
    # drop unmatched locations
    sd = sd.dropna(subset=["geometry"])
    # interpolate along each segment
    sd["sample_pt"] = pd.NA
    for loc, grp in sd.groupby("location"):
        seg = grp.geometry.iloc[0]
        distances = np.linspace(0, seg.length, len(grp))
        sd.loc[grp.index, "sample_pt"] = [seg.interpolate(d) for d in distances]
    # extract x/y and drop helper columns
    result = sd.assign(
        x=lambda df: df.sample_pt.map(lambda p: p.x),
        y=lambda df: df.sample_pt.map(lambda p: p.y),
    ).drop(columns=["sample_pt", "geometry"])
    return result


def extract_static(data: pd.DataFrame, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Attach coordinates for 'static' regime points from KML points."""
    pts = (
        gdf.drop(0)
        .assign(
            location=lambda df: df.Name.str[0],
            x=lambda df: df.geometry.x,
            y=lambda df: df.geometry.y,
        )
        .drop(columns=["Name", "Description", "geometry"])
        .drop_duplicates("location")
        .set_index("location")
    )
    return data.query("regime=='static'").join(pts, on="location")


# --------------------------------------------------
# OSM proximity utilities (<20 lines)
# --------------------------------------------------


def add_osm_proximity(
    df: pd.DataFrame,
    gdf: gpd.GeoDataFrame,
    prefix: str,
    thresholds: List[float],
    type_col: Optional[Union[str, List[str]]] = None,
    types: Optional[Union[str, List[Union[str, List[str]]]]] = None,
) -> pd.DataFrame:
    """
    Add binary flags for proximity to OSM features, calling is_close_to() internally.
    Supports passing a single column or a list of columns, with matching lists of value‐lists.
    """
    for thr in thresholds:
        col = f"close2{prefix}_{thr}"
        df[col] = df.apply(
            lambda r: is_close_to(
                features=gdf,
                point=Point(r.x, r.y),
                threshold=thr,
                type_column=type_col,
                types=types,
            ),
            axis=1,
        ).astype(int)
    return df


def add_osm_count(
    df: pd.DataFrame,
    gdf: gpd.GeoDataFrame,
    prefix: str,
    thresholds: List[float],
    type_col: Optional[Union[str, List[str]]] = None,
    types: Optional[Union[str, List[Union[str, List[str]]]]] = None,
) -> pd.DataFrame:
    """
    Add counts of nearby OSM features, calling count_nearby() internally.
    Supports multiple filter columns/types via the updated signature.
    """
    for thr in thresholds:
        col = f"num_{prefix}_{thr}"
        df[col] = df.apply(
            lambda r: count_nearby(
                features=gdf,
                point=Point(r.x, r.y),
                threshold=thr,
                type_column=type_col,
                types=types,
            ),
            axis=1,
        ).astype(int)
    return df


def add_osm_proportion(
    df: pd.DataFrame,
    gdf: gpd.GeoDataFrame,
    prefix: str,
    thresholds: List[float],
    type_col: Optional[Union[str, List[str]]] = None,
    types: Optional[Union[str, List[Union[str, List[str]]]]] = None,
) -> pd.DataFrame:
    """
    Add land-cover proportions within thresholds, calling land_cover_proportion() internally.
    Handles multi-column / multi-value filters automatically.
    """
    for thr in thresholds:
        col = f"proportion_{prefix}_{thr}"
        df[col] = df.apply(
            lambda r: land_cover_proportion(
                features=gdf,
                point=Point(r.x, r.y),
                threshold=thr,
                type_column=type_col,
                types=types,
            ),
            axis=1,
        ).astype(float)
    return df


def add_osm_mean(df, gdf, prefix, thresholds, value_col):
    """
    Add the mean of a numeric column within given radii.
    """
    for thr in thresholds:
        col = f"average_{prefix}_{thr}"
        df[col] = df.apply(
            lambda r: get_nearest_rows(
                gdf,
                Point(r.x, r.y),
                radius_meters=thr,
            )[value_col].mean(),
            axis=1,
        ).astype(float)
    return df


# --------------------------------------------------
# New building helpers (<20 lines each)
# --------------------------------------------------


def load_buildings(bbox: dict) -> gpd.GeoDataFrame:
    """
    Fetch building footprints and normalize the 'building:levels' tag.
    """
    tags = {"building": True}
    gdf = features_from_bbox(bbox, tags)
    gdf = gdf.replace({"building:levels": {"piano terra": 0}})
    gdf.loc[gdf["level"] == "-1", "level"] = np.nan
    gdf["level"] = gdf["level"].astype(float)
    gdf["building:levels"] = gdf["building:levels"].astype(float).fillna(gdf["level"])
    return gdf.drop(columns=["level"])


def impute_building_levels(
    gdf: gpd.GeoDataFrame, max_rank: int = 50
) -> gpd.GeoDataFrame:
    """
    Run low-rank SVD imputation on building heights.
    """
    exclude = [
        "geometry",
        "name",
        "name:de",
        "name:es",
        "name:fr",
        "roof:levels",
        "wikidata",
        "wikimedia_commons",
        "wikipedia",
        "contact:street",
        "ref",
        "alt_name",
        "description",
        "operator",
        "website",
        "name:en",
        "name:it",
        "phone",
        "addr:housenumber",
        "loc_name",
        "short_name",
        "ref:vatin",
        "source",
        "start_date",
        "addr:unit",
        "end_date",
        "note",
        "internet_access",
        "internet_access:ssid",
        "internet_access:fee",
        "opening_hours",
        "ref:isil",
        "check_date",
        "check_date:opening_hours",
        "opening_hours:signed",
        "contact:email",
        "contact:website",
        "old_name",
        "contact:phone",
        "fax",
        "operator:wikipedia",
        "operator:wikidata",
    ]
    return impute_gdf(gdf, exclude=exclude, max_rank=max_rank)


# --------------------------------------------------
# Weather utilities (<20 lines)
# --------------------------------------------------


def parse_station_metadata(folder: Path) -> dict:
    """Parse station metadata (ID→name) from TXT files."""
    meta = {}
    for f in os.listdir(folder):
        if f.endswith(".txt"):
            lines = open(folder / f).readlines()[11:18]
            dfm = pd.read_csv(StringIO("".join(lines)))
            meta.update(dfm.set_index("Id_Sensore")["Nome_Sensore"].to_dict())
    return meta


def read_raw_weather(folder: Path, meta: dict) -> pd.DataFrame:
    """Read and concatenate raw CSV weather files, renaming columns."""
    dfs = []
    for f in os.listdir(folder):
        if f.endswith(".csv"):
            dft = pd.read_csv(folder / f, index_col="Data-Ora", parse_dates=True)
            orig = [c for c in dft.columns if c != "Id Sensore"][0]
            sid_val = dft["Id Sensore"].iat[0]
            cname = (
                unidecode(f"{meta[sid_val].strip()} {orig.strip()}")
                .replace(" ", "_")
                .lower()
            )
            print(f"Renaming {orig} to {cname}")
            dft.replace([777, 7777], 0, inplace=True)
            dft.replace([888, 8888, -999], pd.NA, inplace=True)
            dfs.append(dft.rename(columns={orig: cname}).drop(columns=["Id Sensore"]))
    return pd.concat(dfs, axis=1)


def interpolate_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Upsample weather to 1s using PCHIP interpolation."""
    df = df.sort_index()
    t0 = df.index.astype(np.int64) / 1e9
    t1 = pd.date_range(df.index[0], df.index[-1], freq="1S")
    t1s = t1.astype(np.int64) / 1e9
    out = {}
    for c in df.columns:
        mask = df[c].notna()
        p = PchipInterpolator(t0[mask], df[c][mask])
        out[c] = p(t1s)
    return pd.DataFrame(out, index=t1)


# --------------------------------------------------
# Main pipeline (<20 lines)
# --------------------------------------------------


def process_subject(subject_id: int):
    raw = Path("data/raw_data")
    out = Path("data/processed_data")

    data = load_subject(subject_id)
    kml = extract_kml(raw / "route.kmz")
    segs = build_segments(kml)
    S = pd.concat(
        [extract_static(data, kml), interpolate_dynamic(data, segs)]
    ).sort_index()
    time_delta = pd.to_timedelta(S.index.astype(str))
    base = pd.Timestamp("2022-11-11") + pd.Timedelta(days=subject_id - 1)
    S.index = base + time_delta

    feats = [
        # WATER-RELATED FEATURES ----------------------------------------------
        (
            "water",
            {"natural": ["water", "wetland"], "waterway": True, "water": True},
            [25, 100],
            None,
            None,
            "proximity",
        ),
        # SHOPPING-RELATED FEATURES -------------------------------------------
        (
            "smoking_shop",
            {"shop": True},
            [75],
            "shop",
            ["kiosk", "tobacco", "cannabis", "alcohol", "coffee"],
            "proximity",
        ),
        # MAN-MADE FEATURES ---------------------------------------------------
        ("chimney", {"man_made": True}, [150], "man_made", ["chimney"], "proximity"),
        ("water_tap", {"man_made": True}, [75], "man_made", ["water_tap"], "proximity"),
        # PUBLIC-TRANSPORT FEATURES -------------------------------------------
        ("public_transport", {"public_transport": True}, [15], None, None, "proximity"),
        (
            "public_transport",
            {"public_transport": True},
            [50, 100],
            None,
            None,
            "count",
        ),
        # LANDUSE FEATURES ----------------------------------------------------
        ("industry", {"landuse": True}, [400], "landuse", ["industrial"], "proximity"),
        (
            "residential",
            {"landuse": True},
            [10],
            "landuse",
            ["residential"],
            "proximity",
        ),
        (
            "residential",
            {"landuse": True},
            [400],
            "landuse",
            ["residential"],
            "proportion",
        ),
        ("railway", {"landuse": True}, [100], "landuse", ["railway"], "proximity"),
        ("grass", {"landuse": True}, [15, 100], "landuse", ["grass"], "proximity"),
        (
            "construction",
            {"landuse": True},
            [50],
            "landuse",
            ["construction"],
            "proximity",
        ),
        # NATURAL FEATURES ----------------------------------------------------
        (
            "trees",
            {"natural": True},
            [50, 500],
            "natural",
            ["tree", "tree_stump"],
            "count",
        ),
        (
            "woods",
            {"natural": True},
            [50, 200],
            "natural",
            ["wood", "shrub", "scrub"],
            "proximity",
        ),
        ("grassland", {"natural": True}, [200], "natural", ["grassland"], "proximity"),
        # PARK-RELATED FEATURES -----------------------------------------------
        (
            "park",
            {"leisure": ["park", "garden"], "amenity": ["town_square"]},
            [5, 25],
            "leisure",
            ["park"],
            "proximity",
        ),
        (
            "garden",
            {"leisure": ["park", "garden"], "amenity": ["town_square"]},
            [25],
            "leisure",
            ["garden"],
            "proximity",
        ),
        (
            "green",
            {"leisure": ["park", "garden"], "amenity": ["town_square"]},
            [200, 500, 1000],
            "leisure",
            ["garden", "park"],
            "count",
        ),
        (
            "green",
            {"leisure": ["park", "garden"], "amenity": ["town_square"]},
            [50, 100, 500],
            "leisure",
            ["garden", "park"],
            "proportion",
        ),
        # AMENITY-BASED FEATURES
        # 1) smoking-allowed amenity within 30m
        (
            "smoking_amenity",
            {"amenity": True},
            [30],
            ["outdoor_seating", "smoking"],
            [["yes"], ["yes", "outside"]],
            "proximity",
        ),
        # 2) cigarette dispenser within 50m
        (
            "cigarette_disposal",
            {"amenity": True},
            [50],
            "waste",
            ["cigarettes;mixed", "cigarettes"],
            "proximity",
        ),
        # 3) waste disposal bin within 25m
        (
            "waste_disposal",
            {"amenity": True},
            [25],
            "amenity",
            ["waste_basket", "recycling", "waste_disposal"],
            "proximity",
        ),
        # 4) parking space within 30m
        (
            "parking_space",
            {"amenity": True},
            [30],
            ["amenity", "parking"],
            [
                ["parking", "parking_space", "parking_entrance", "motorcycle_parking"],
                [],
            ],
            "proximity",
        ),
        # 5) fuel station within 40m
        (
            "fuel_station",
            {"amenity": True},
            [40],
            "amenity",
            ["fuel"],
            "proximity",
        ),
        # 6) drinking water source within 20m
        (
            "drinking_water",
            {"amenity": True},
            [30],
            "amenity",
            ["drinking_water", "fountain"],
            "proximity",
        ),
        # BUILDING-RELATED FEATURES -------------------------------------------
        (
            "industrial_building",
            {"building": True},
            [15, 100],
            "building",
            ["industrial"],
            "proximity",
        ),
        (
            "residential_buildings",
            {"building": True},
            [200],
            "building",
            ["residential", "apartments", "house", "dormitory", "semidetached_house"],
            "count",
        ),
        (
            "residential_buildings",
            {"building": True},
            [500],
            "building",
            ["residential", "apartments", "house", "dormitory", "semidetached_house"],
            "proportion",
        ),
        (
            "commercial_building",
            {"building": True},
            [25],
            "building",
            ["commercial"],
            "proximity",
        ),
        (
            "commercial_buildings",
            {"building": True},
            [200],
            "building",
            ["commercial", "service", "office", "kiosk", "retail"],
            "count",
        ),
        # Building height mean (uses imputed footprints)
        (
            "building_height",
            None,
            [100],
            None,
            None,
            "mean",
            "building:levels",
            "imp",
        ),
    ]

    # ----------------------------------------------------------------------
    # Pre-load OSM layers for non-mean modes
    # ----------------------------------------------------------------------
    gdf_cache = {}

    def freeze_tags(tags: dict) -> frozenset:
        return frozenset(
            (k, tuple(v) if isinstance(v, list) else v) for k, v in tags.items()
        )

    for prefix, tags, thresholds, tc, ts, mode, *params in feats:
        if mode != "mean":
            key = freeze_tags(tags)
            if key not in gdf_cache:
                gdf_cache[key] = features_from_bbox(BBOX, tags)

    # ----------------------------------------------------------------------
    # Load and prepare specific GeoDataFrames for "mean" modes
    # ----------------------------------------------------------------------
    gdf_sources = {}
    # example: building footprints
    gdf_sources["bld"] = load_buildings(BBOX)
    gdf_sources["imp"] = impute_building_levels(gdf_sources["bld"])
    # future gdfs can be added here, e.g. gdf_sources['custom'] = load_custom(...)

    # ----------------------------------------------------------------------
    # Apply all features in one loop
    # ----------------------------------------------------------------------
    for prefix, tags, thresholds, tc, ts, mode, *params in feats:
        if mode == "mean":
            value_col, source_key = params
            source_gdf = gdf_sources[source_key]
            S = add_osm_mean(S, source_gdf, prefix, thresholds, value_col)
        else:
            source_gdf = gdf_cache[freeze_tags(tags)]
            if mode == "proximity":
                S = add_osm_proximity(S, source_gdf, prefix, thresholds, tc, ts)
            elif mode == "count":
                S = add_osm_count(S, source_gdf, prefix, thresholds, tc, ts)
            elif mode == "proportion":
                S = add_osm_proportion(S, source_gdf, prefix, thresholds, tc, ts)
            else:
                raise ValueError(f"Unknown mode: {mode}")

    wd_dirs = list(raw.glob("RW_*"))
    if wd_dirs:
        wdir = wd_dirs[0]
        meta = parse_station_metadata(wdir)
        wr = read_raw_weather(wdir, meta)
        w1s = interpolate_weather(wr)
        S = S.join(w1s, how="left")

    out.mkdir(parents=True, exist_ok=True)
    S.to_parquet(out / f"S{subject_id}-coords.parquet")
    return S


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process subject pipeline")
    parser.add_argument("--subject", type=int, help="Subject ID to process")
    args = parser.parse_args()
    if args.subject:
        process_subject(args.subject)
    else:
        for sid in range(1, 21):
            process_subject(sid)
