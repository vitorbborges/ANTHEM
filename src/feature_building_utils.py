from typing import List, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from fancyimpute import SoftImpute
from shapely.geometry import Point

# GLOBAL VARIABLES
WEST, SOUTH, EAST, NORTH = 9.2257, 45.47162, 9.23768, 45.48537
BBOX = (WEST, SOUTH, EAST, NORTH)  # Bounding box for the area of interest
TOML_PATH = "documentation/feature_docs.toml"


def is_inside(
    features: gpd.GeoDataFrame,
    point: Point,
    type_column: Optional[str] = None,
    types: Optional[Union[str, List[str]]] = None,
) -> bool:
    """
    Return True if `point` lies within any geometry in `features`.

    Optionally filters `features` by a column and value(s) before testing.

    Parameters:
        features (GeoDataFrame): geometries in EPSG:4326
        point (Point): location in EPSG:4326
        type_column (str, optional): column name to filter on
        types (str or list, optional): value(s) in `type_column` to include

    Returns:
        bool: True if containment in at least one feature
    """
    # filter by type if requested
    if type_column and types is not None:
        vals = [types] if isinstance(types, str) else list(types)
        features = features[features[type_column].isin(vals)]
        if features.empty:
            return False

    # project for accurate geometry operations
    features_proj = features.to_crs("EPSG:3857")
    point_proj = gpd.GeoSeries([point], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]

    return features_proj.geometry.contains(point_proj).any()


def is_close_to(
    features: gpd.GeoDataFrame,
    point: Point,
    threshold: float,
    type_column: Optional[Union[str, List[str]]] = None,
    types: Optional[Union[str, List[Union[str, List[str]]]]] = None,
) -> bool:
    """
    Return True if `point` is within `threshold` meters of any geometry in `features`.
    Can filter by one or multiple columns/types before testing.
    """
    # filter by type(s) if requested
    # ─── Filter by type(s) exactly as in the standalone version ─────────────────
    if type_column and types is not None:
        # normalize column(s)
        cols = [type_column] if isinstance(type_column, str) else list(type_column)
        # take types as-is
        vals = types
        # if it's a single non-list/tuple, or a flat list rather than a list-of-lists,
        # repeat it for each column
        first = vals[0] if isinstance(vals, list) else vals
        if not isinstance(first, (list, tuple)):
            vals = [vals] * len(cols)
        # ensure each entry is a list
        vals_list = [v if isinstance(v, (list, tuple)) else [v] for v in vals]
        # build mask exactly like `mask = False; mask = mask | …`
        mask = False
        for col, allowed in zip(cols, vals_list):
            mask = mask | features[col].isin(allowed)
        features = features[mask]
        if features.empty:
            return False

    # ─── Projection and distance test ───────────────────────────────────────────
    feats_proj = features.to_crs("EPSG:3857")
    p_proj = gpd.GeoSeries([point], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
    distances = feats_proj.geometry.distance(p_proj)
    return (distances <= threshold).any()


def count_nearby(
    features: gpd.GeoDataFrame,
    point: Point,
    threshold: float,
    type_column: Optional[Union[str, List[str]]] = None,
    types: Optional[Union[str, List[Union[str, List[str]]]]] = None,
) -> int:
    """
    Count how many geometries in `features` are within `threshold` meters of `point`.
    Supports filtering by one or multiple columns/types.
    """
    # filter by type(s) if requested
    if type_column and types is not None:
        cols = [type_column] if isinstance(type_column, str) else list(type_column)
        vals = types
        if not isinstance(vals[0] if isinstance(vals, list) else vals, (list, tuple)):
            vals = [vals] * len(cols)
        vals_list = [v if isinstance(v, (list, tuple)) else [v] for v in vals]
        mask = False
        for col, allowed in zip(cols, vals_list):
            mask = mask | features[col].isin(allowed)
        features = features[mask]
        if features.empty:
            return 0

    features_proj = features.to_crs("EPSG:3857")
    point_proj = gpd.GeoSeries([point], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]

    distances = features_proj.geometry.distance(point_proj)
    return int((distances <= threshold).sum())


def land_cover_proportion(
    features: gpd.GeoDataFrame,
    point: Point,
    threshold: float = 100.0,
    type_column: Optional[Union[str, List[str]]] = None,
    types: Optional[Union[str, List[Union[str, List[str]]]]] = None,
) -> float:
    """
    Proportion of area within `threshold` meters of `point` covered by `features`.
    Supports filtering by one or multiple columns/types.
    """
    # filter by type(s) if requested
    if type_column and types is not None:
        cols = [type_column] if isinstance(type_column, str) else list(type_column)
        vals = types
        if not isinstance(vals[0] if isinstance(vals, list) else vals, (list, tuple)):
            vals = [vals] * len(cols)
        vals_list = [v if isinstance(v, (list, tuple)) else [v] for v in vals]
        mask = False
        for col, allowed in zip(cols, vals_list):
            mask = mask | features[col].isin(allowed)
        features = features[mask]
        if features.empty:
            return 0.0

    features_proj = features.to_crs("EPSG:3857")
    point_proj = gpd.GeoSeries([point], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
    buffer_geom = point_proj.buffer(threshold)

    intersection_area = features_proj.geometry.intersection(buffer_geom).area.sum()
    buffer_area = buffer_geom.area

    return intersection_area / buffer_area if buffer_area > 0 else 0.0


def impute_gdf(
    gdf: gpd.GeoDataFrame, exclude: list[str], max_rank: int = 20, max_iters: int = 100
) -> gpd.GeoDataFrame:
    """
    Impute missing values in a GeoDataFrame using a low-rank SVD approach.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame with some missing values.
    exclude : list[str]
        Column names to skip (e.g., IDs, names, geometry).
    max_rank : int, optional
        Maximum rank for the SoftImpute SVD approximation (default=20).
    max_iters : int, optional
        Maximum number of SoftImpute iterations (default=100).

    Returns
    -------
    GeoDataFrame
        A copy of `gdf` with numerical and categorical columns imputed.
    """
    # 1) Select columns to impute
    use_cols = [c for c in gdf.columns if c not in exclude]

    # 2) Collapse list-valued columns: take the first element
    X = gdf[use_cols].copy()
    list_cols = [c for c in use_cols if X[c].apply(lambda v: isinstance(v, list)).any()]
    for col in list_cols:
        X[col] = X[col].apply(lambda v: v[0] if isinstance(v, list) and v else np.nan)

    # 3) Coerce “numeric-looking” object columns -> floats
    for col in X.columns:
        if X[col].dtype == object:
            coerced = pd.to_numeric(X[col], errors="coerce")
            if coerced.notna().sum() > len(coerced) / 2:
                X[col] = coerced

    # 4) Identify numeric vs. categorical
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # 5) One-hot encode categoricals
    dummies = pd.get_dummies(X[cat_cols], dummy_na=False)
    M = pd.concat([X[num_cols], dummies], axis=1).astype(float)

    # 6) Apply SoftImpute
    filled = SoftImpute(
        max_rank=max_rank, max_iters=max_iters, verbose=False
    ).fit_transform(M.values)
    M_filled = pd.DataFrame(filled, columns=M.columns, index=M.index)

    # 7) Build output GeoDataFrame
    imputed = gdf.copy()

    # 7a) Numeric columns
    for col in num_cols:
        imputed[col] = M_filled[col]

    # 7b) Categorical columns: pick dummy with highest score
    for col in cat_cols:
        pref = f"{col}_"
        dcols = [c for c in M_filled.columns if c.startswith(pref)]
        if not dcols:
            continue
        best = M_filled[dcols].idxmax(axis=1).str[len(pref) :]
        imputed[col] = best.astype(gdf[col].dtype)

    # 7c) Restore geometry (in case it was dropped)
    if "geometry" in gdf.columns:
        imputed.geometry = gdf.geometry

    return imputed
