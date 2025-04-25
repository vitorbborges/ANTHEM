from shapely.geometry import Point, LineString
import geopandas as gpd


def count_nearest_parks(
    parks: gpd.GeoDataFrame, point: Point, threshold: float = 25
) -> bool:
    """
    Determine whether a given point lies within a specified distance of any park.

    Parameters:
        parks (GeoDataFrame): A GeoDataFrame of park polygons or points, in EPSG:4326.
        point (shapely.geometry.Point): The location to test, in EPSG:4326.
        threshold (float): Distance threshold in meters. Defaults to 25m.

    Returns:
        bool:
            True if the point is at most `threshold` meters away from at least one park;
            False otherwise.
    """
    # Project to a metric CRS for accurate distance calculations
    target_crs = "EPSG:3857"
    parks_proj = parks.to_crs(target_crs)

    # Project the point
    point_proj = gpd.GeoSeries([point], crs="EPSG:4326").to_crs(target_crs).iloc[0]

    # Check if any park is within the threshold distance
    distances = parks_proj.geometry.distance(point_proj)
    return (distances <= threshold).any()


def is_close_to_park(
    parks: gpd.GeoDataFrame, point: Point, threshold: float = 25
) -> bool:
    """
    Returns True if `point` lies within `threshold` meters of any feature
    in `parks` classified as a 'park'.

    Assumes `parks` has a column (e.g. 'type') with the value 'park'
    for park geometries. Adjust `feature_column` and `park_value` if yours differs.

    Parameters:
        parks (GeoDataFrame): in EPSG:4326, with a column 'type' == 'park'
        point (Point): in EPSG:4326
        threshold (float): distance in meters

    Returns:
        bool
    """
    # filter only park geometries
    parks_only = parks[parks["type"] == "park"]
    if parks_only.empty:
        return False

    # project to metric CRS
    parks_proj = parks_only.to_crs("EPSG:3857")
    point_proj = gpd.GeoSeries([point], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]

    # check distances
    return (parks_proj.geometry.distance(point_proj) <= threshold).any()


def is_close_to_garden(
    parks: gpd.GeoDataFrame, point: Point, threshold: float = 25
) -> bool:
    """
    Returns True if `point` lies within `threshold` meters of any feature
    in `parks` classified as a 'garden'.

    Assumes `parks` has a column (e.g. 'type') with the value 'garden'
    for garden geometries. Adjust `feature_column` and `garden_value` if yours differs.

    Parameters:
        parks (GeoDataFrame): in EPSG:4326, with a column 'type' == 'garden'
        point (Point): in EPSG:4326
        threshold (float): distance in meters

    Returns:
        bool
    """
    # filter only garden geometries
    gardens_only = parks[parks["type"] == "garden"]
    if gardens_only.empty:
        return False

    # project to metric CRS
    gardens_proj = gardens_only.to_crs("EPSG:3857")
    point_proj = gpd.GeoSeries([point], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]

    # check distances
    return (gardens_proj.geometry.distance(point_proj) <= threshold).any()


def is_indoor(buildings: gpd.GeoDataFrame, point: Point) -> bool:
    """
    Checks if a given point is inside any building polygon.

    Parameters:
        buildings (GeoDataFrame): GeoDataFrame of building geometries in EPSG:4326
        point (Point): The shapely Point to check, in EPSG:4326

    Returns:
        bool: True if the point is inside any building, False otherwise
    """
    # Ensure buildings are in lat/lon
    if buildings.crs != "EPSG:4326":
        buildings = buildings.to_crs(epsg=4326)

    # Vectorized containment check
    return buildings.geometry.contains(point).any()
