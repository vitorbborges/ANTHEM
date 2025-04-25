import geopandas as gpd
from shapely.geometry import Point
from typing import Optional, Union, List

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
    type_column: Optional[str] = None,
    types: Optional[Union[str, List[str]]] = None,
) -> bool:
    """
    Return True if `point` is within `threshold` meters of any geometry in `features`.

    Optionally filters `features` by a column and value(s) before testing.

    Parameters:
        features (GeoDataFrame): geometries in EPSG:4326
        point (Point): location in EPSG:4326
        threshold (float): distance in meters
        type_column (str, optional): column name to filter on
        types (str or list, optional): value(s) in `type_column` to include

    Returns:
        bool: True if at least one feature is within `threshold` meters
    """
    # filter by type if requested
    if type_column and types is not None:
        vals = [types] if isinstance(types, str) else list(types)
        features = features[features[type_column].isin(vals)]
        if features.empty:
            return False

    features_proj = features.to_crs("EPSG:3857")
    point_proj = gpd.GeoSeries([point], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]

    distances = features_proj.geometry.distance(point_proj)
    return (distances <= threshold).any()


def count_nearby(
    features: gpd.GeoDataFrame,
    point: Point,
    threshold: float,
    type_column: Optional[str] = None,
    types: Optional[Union[str, List[str]]] = None,
) -> int:
    """
    Count how many geometries in `features` lie within `threshold` meters of `point`.

    Optionally filters `features` by a column and value(s) before counting.

    Parameters:
        features (GeoDataFrame): geometries in EPSG:4326
        point (Point): location in EPSG:4326
        threshold (float): distance in meters
        type_column (str, optional): column name to filter on
        types (str or list, optional): value(s) in `type_column` to include

    Returns:
        int: number of features within `threshold` meters
    """
    # filter by type if requested
    if type_column and types is not None:
        vals = [types] if isinstance(types, str) else list(types)
        features = features[features[type_column].isin(vals)]
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
    type_column: Optional[str] = None,
    types: Optional[Union[str, List[str]]] = None,
) -> float:
    """
    Calculate the proportion of area within `threshold` meters of `point`
    that is covered by the geometries in `features`, optionally filtering
    by feature type.

    Parameters:
        features (GeoDataFrame): geometries in EPSG:4326.
        point (Point): center point in EPSG:4326.
        threshold (float): buffer radius in meters (default: 100.0).
        type_column (str, optional): column name to filter on.
        types (str or list, optional): value(s) in `type_column` to include.

    Returns:
        float: fraction (0.0–1.0) of the buffer’s area occupied by the features.
    """
    # Filter by type if requested
    if type_column and types is not None:
        vals = [types] if isinstance(types, str) else list(types)
        features = features[features[type_column].isin(vals)]
        if features.empty:
            return 0.0

    # Project to metric CRS for accurate buffering & area calcs
    features_proj = features.to_crs("EPSG:3857")
    point_proj = gpd.GeoSeries([point], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]

    # Build the buffer
    buffer_geom = point_proj.buffer(threshold)

    # Compute intersection area
    intersection_area = features_proj.geometry.intersection(buffer_geom).area.sum()

    # Compute buffer area
    buffer_area = buffer_geom.area

    return intersection_area / buffer_area
