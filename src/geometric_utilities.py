from shapely.geometry import Point, LineString
import geopandas as gpd

def get_closest_edge(gdf: gpd.GeoDataFrame, point: Point) -> gpd.GeoSeries:
    """
    Returns the row of the GeoDataFrame whose LineString geometry 
    is closest to the given point.
    
    Parameters:
        gdf (GeoDataFrame): GeoDataFrame of LineStrings
        point (Point): The Shapely Point to compare against

    Returns:
        GeoSeries: The row (as a Series) corresponding to the closest LineString
    """
    distances = gdf.geometry.distance(point)
    closest_idx = distances.idxmin()
    return gdf.loc[closest_idx]


def get_nearest_point_on_line(line: LineString, point: Point) -> Point:
    """
    Returns the nearest point *on the line* to the given point.

    Parameters:
        line (LineString): The Shapely LineString to project onto
        point (Point): The point to find the closest location on the line

    Returns:
        Point: The point on the LineString closest to the input point
    """
    return line.interpolate(line.project(point))

def is_close_to_park(parks: gpd.GeoDataFrame, point: Point, threshold=25) -> bool:
    """
    Checks if a given point is within a certain distance (in meters) from any park.

    Parameters:
        parks (GeoDataFrame): GeoDataFrame containing park geometries in EPSG:4326
        point (Point): The shapely Point to check, in EPSG:4326
        threshold_meters (float): The distance threshold in meters

    Returns:
        bool: True if the point is close to any park, False otherwise
    """
    # Project parks and point to a CRS in meters (Web Mercator here)
    target_crs = "EPSG:3857"
    
    # Project parks
    parks_proj = parks.to_crs(target_crs)

    # Create GeoSeries from point and project it
    point_gdf = gpd.GeoSeries([point], crs="EPSG:4326").to_crs(target_crs)
    point_proj = point_gdf.iloc[0]

    # Check distance to each park
    for _, park in parks_proj.iterrows():
        if point_proj.distance(park.geometry) <= threshold:
            return True
    return False
