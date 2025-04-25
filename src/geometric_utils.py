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


def get_closest_node(nodes: gpd.GeoDataFrame, point: Point) -> gpd.GeoSeries:
    """
    Returns the row of the GeoDataFrame whose Point geometry
    is closest to the given point.

    Parameters:
        nodes (GeoDataFrame): GeoDataFrame of Point geometries
        point (Point): The Shapely Point to compare against

    Returns:
        GeoSeries: The row (as a Series) corresponding to the closest node
    """
    # compute distances from each node to the target point
    distances = nodes.geometry.distance(point)
    # find the index of the minimum distance
    closest_idx = distances.idxmin()
    # return the full row (attributes + geometry)
    return nodes.loc[closest_idx]


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


def get_nearest_nodes(
    nodes: gpd.GeoDataFrame, point: Point, radius_meters: float = 25
) -> gpd.GeoDataFrame:
    """
    Returns all nodes whose Point geometries lie within a given radius (in meters)
    of the specified point.

    Parameters:
        nodes (GeoDataFrame): GeoDataFrame of Point geometries in EPSG:4326
        point (Point): The Shapely Point to search around, in EPSG:4326
        radius_meters (float): Search radius in meters (default: 25)

    Returns:
        GeoDataFrame: Subset of `nodes` within `radius_meters` of `point`
    """
    # 1. Reproject to a metric CRS (Web Mercator)
    target_crs = "EPSG:3857"
    nodes_proj = nodes.to_crs(target_crs)

    # 2. Project the single point
    point_gs = gpd.GeoSeries([point], crs="EPSG:4326").to_crs(target_crs)
    point_proj = point_gs.iloc[0]

    # 3. Compute distances and filter
    dists = nodes_proj.geometry.distance(point_proj)
    within_mask = dists <= radius_meters

    # 4. Return the original rows (in original CRS) that satisfy the mask
    return nodes.loc[within_mask.values]


def get_nearest_edges(
    edges: gpd.GeoDataFrame, point: Point, radius_meters: float = 25
) -> gpd.GeoDataFrame:
    """
    Returns all edges whose LineString geometries lie within a given radius (in meters)
    of the specified point.

    Parameters:
        edges (GeoDataFrame): GeoDataFrame of LineString geometries in EPSG:4326
        point (Point): The Shapely Point to search around, in EPSG:4326
        radius_meters (float): Search radius in meters (default: 25)

    Returns:
        GeoDataFrame: Subset of `edges` within `radius_meters` of `point`
    """
    # 1. Reproject to a metric CRS (Web Mercator)
    target_crs = "EPSG:3857"
    edges_proj = edges.to_crs(target_crs)

    # 2. Project the single point
    point_gs = gpd.GeoSeries([point], crs="EPSG:4326").to_crs(target_crs)
    point_proj = point_gs.iloc[0]

    # 3. Compute distances and filter
    dists = edges_proj.geometry.distance(point_proj)
    within_mask = dists <= radius_meters

    # 4. Return the original rows (in original CRS) that satisfy the mask
    return edges.loc[within_mask.values]
