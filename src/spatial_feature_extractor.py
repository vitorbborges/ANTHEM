import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from osmnx.features import features_from_bbox
from shapely.geometry import LineString, Point
from shapely.ops import linemerge, substring


class SpatialFeatureExtractor:
    """
    Encapsulates methods for deriving spatial features from various geospatial data sources.

    Provides utilities for:
      - Parsing KML/KMZ geometries (static points and line segments).
      - Caching and querying OpenStreetMap (OSM) layers.
      - Computing proximity flags, counts, land-cover proportions, and mean values.

    Attributes
    ----------
    bbox : tuple
        Bounding box (north, south, east, west) for OSM queries.
    osm_layer_cache : Dict[frozenset, gpd.GeoDataFrame]
        Cached OSM layer data keyed by tag sets.
    geo_sources : Dict[str, gpd.GeoDataFrame]
        Registered GeoDataFrame sources for mean-aggregation methods.
    """

    def __init__(self, bbox: tuple) -> None:
        self.bbox: tuple = bbox
        self.osm_layer_cache: Dict[frozenset, gpd.GeoDataFrame] = {}
        self.geo_sources: Dict[str, gpd.GeoDataFrame] = {}
        # Graph/network attributes
        self._graph = None
        self._graph_nodes: Optional[gpd.GeoDataFrame] = None
        self._graph_edges: Optional[gpd.GeoDataFrame] = None
        self._graph_imputed_edges: Optional[gpd.GeoDataFrame] = None
        self._graph_cars: Optional[gpd.GeoDataFrame] = None
        # Register graph edges and car-only edges for mean calculations
        self.load_mean_source("edges", lambda: self._get_graph_edges())
        self.load_mean_source("cars", lambda: self._get_graph_cars())

    def _load_graph(self) -> None:
        """
        Load and cache the OSM street network graph, extract nodes and edges,
        process and impute edge attributes, and derive the car-only subset.
        """
        if self._graph is None:
            # Fetch full street network graph
            north, south, east, west = self.bbox
            self._graph = ox.graph_from_bbox(
                north, south, east, west, network_type="all", simplify=False
            )
            # Convert to GeoDataFrames
            nodes, edges = ox.graph_to_gdfs(self._graph)
            # Process original edges for imputation
            edges_orig = edges.copy()
            edges_orig["lanes"] = edges_orig["lanes"].fillna(0)
            edges_orig["oneway"] = edges_orig["oneway"].astype(int)
            edges_orig["reversed"] = edges_orig["reversed"].astype(int)
            edges_orig["maxspeed"] = edges_orig["maxspeed"].astype(float)
            # Enforce single lane on one-way streets
            edges_orig.loc[edges_orig["oneway"] == 1, "lanes"] = 1
            # Set zero lanes to NaN for imputation
            edges_orig.loc[edges_orig["lanes"] == 0, "lanes"] = np.nan
            # Cache nodes and raw edges
            self._graph_nodes = nodes
            self._graph_edges = edges_orig
            # Impute edge attributes using MICE
            imputed = FeatureImputer.impute_edges(edges_orig)
            self._graph_imputed_edges = imputed
            # Derive car-only edges (exclude footway-like types)
            foot_types = [
                "footway",
                "pedestrian",
                "unclassified",
                "steps",
                "corridor",
                "path",
            ]
            mask_foot = imputed["highway"].isin(foot_types)
            self._graph_cars = imputed.loc[~mask_foot].copy()

    def _get_graph_nodes(self) -> gpd.GeoDataFrame:
        if self._graph_nodes is None:
            self._load_graph()
        return self._graph_nodes

    def _get_graph_edges(self) -> gpd.GeoDataFrame:
        if self._graph_edges is None:
            self._load_graph()
        return self._graph_edges

    def _get_graph_imputed_edges(self) -> gpd.GeoDataFrame:
        if self._graph_imputed_edges is None:
            self._load_graph()
        return self._graph_imputed_edges

    def _get_graph_cars(self) -> gpd.GeoDataFrame:
        if self._graph_cars is None:
            self._load_graph()
        return self._graph_cars

    def extract_kml(self, kmz_path: Path) -> gpd.GeoDataFrame:
        """
        Extract and read KML geometries from a KMZ archive.

        Parameters
        ----------
        kmz_path : Path
            Path to the KMZ file containing a 'doc.kml'.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame of all geometries in the KML.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(kmz_path, "r") as archive:
                archive.extractall(tmpdir)
            return gpd.read_file(Path(tmpdir) / "doc.kml", driver="KML")

    def build_segments(self, kml_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Split a closed LineString into ordered segments based on KML points.

        Parameters
        ----------
        kml_gdf : gpd.GeoDataFrame
            GeoDataFrame containing one LineString and multiple Point breaks.

        Returns
        -------
        gpd.GeoDataFrame
            Segmented line pieces with location labels.
        """
        line: LineString = kml_gdf.geometry.iloc[0]
        pts = kml_gdf.geometry.iloc[1:].drop_duplicates().tolist()
        dists = sorted(line.project(pt) for pt in pts)
        wrap = linemerge(
            [substring(line, dists[-1], line.length), substring(line, 0.0, dists[0])]
        )
        segments = [wrap] + [
            substring(line, a, b) for a, b in zip(dists[:-1], dists[1:])
        ]
        locs = ["FG", "GH", "AB", "BC", "CD", "DE", "EF"]
        return gpd.GeoDataFrame(
            {"location": locs, "geometry": segments}, crs=kml_gdf.crs
        )

    def extract_static(
        self, df: pd.DataFrame, kml_gdf: gpd.GeoDataFrame
    ) -> pd.DataFrame:
        """
        Attach static regime points from KML to a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with a 'regime' column containing 'static'.
        kml_gdf : gpd.GeoDataFrame
            KML GeoDataFrame with named points.

        Returns
        -------
        pd.DataFrame
            Subset of df with x, y coordinates from the KML points.
        """
        pts = (
            kml_gdf.drop(0)
            .assign(
                location=lambda d: d.Name.str[0],
                x=lambda d: d.geometry.x,
                y=lambda d: d.geometry.y,
            )
            .drop(columns=["Name", "Description", "geometry"])
            .drop_duplicates("location")
            .set_index("location")
        )
        return df.query("regime=='static'").join(pts, on="location")

    def extract_dynamic(
        self, df: pd.DataFrame, segments: gpd.GeoDataFrame
    ) -> pd.DataFrame:
        """
        Interpolate dynamic regime points evenly along line segments.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with a 'regime' column containing 'dynamic'.
        segments : gpd.GeoDataFrame
            GeoDataFrame of line segments labeled by 'location'.

        Returns
        -------
        pd.DataFrame
            Original rows augmented with computed x, y coordinates.
        """
        dyn = df[df["regime"] == "dynamic"]
        orig_idx = dyn.index
        merged = dyn.merge(segments, on="location", how="left")
        merged.index = orig_idx
        merged = merged.dropna(subset=["geometry"]).copy()

        merged["sample_pt"] = None
        for loc, grp in merged.groupby("location"):
            seg: LineString = grp.geometry.iloc[0]
            distances = np.linspace(0, seg.length, len(grp))
            merged.loc[grp.index, "sample_pt"] = [seg.interpolate(d) for d in distances]

        return merged.assign(
            x=lambda d: d.sample_pt.map(lambda p: p.x),
            y=lambda d: d.sample_pt.map(lambda p: p.y),
        ).drop(columns=["sample_pt", "geometry"])

    def load_buildings(self) -> gpd.GeoDataFrame:
        """
        Fetch building footprints from OSM and normalize building levels.

        Returns
        -------
        gpd.GeoDataFrame
            Building footprints with numeric 'building:levels'.
        """
        tags = {"building": True}
        gdf = features_from_bbox(self.bbox, tags)
        gdf = gdf.replace({"building:levels": {"piano terra": 0}})
        if "level" in gdf.columns:
            gdf.loc[gdf["level"] == "-1", "level"] = np.nan
            gdf["level"] = gdf["level"].astype(float)
        gdf["building:levels"] = (
            gdf["building:levels"].astype(float).fillna(gdf.get("level", np.nan))
        )
        return gdf.drop(columns=["level"], errors="ignore")

    def get_closest_row(self, gdf: gpd.GeoDataFrame, point: Point) -> gpd.GeoSeries:
        """
        Return the GeoDataFrame row whose geometry is closest to the specified point.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame containing geometries in any CRS.
        point : Point
            Shapely Point in the same CRS as `gdf.geometry`.

        Returns
        -------
        gpd.GeoSeries
            The row from `gdf` with the minimum distance to `point`.
        """
        distances = gdf.geometry.distance(point)
        return gdf.loc[distances.idxmin()]

    def get_nearest_point_on_line(self, line: LineString, point: Point) -> Point:
        """
        Project a point onto a line and return the closest point on that line.

        Parameters
        ----------
        line : LineString
            LineString to which `point` will be projected.
        point : Point
            Shapely Point to project.

        Returns
        -------
        Point
            The point on `line` nearest to `point`.
        """
        return line.interpolate(line.project(point))

    def get_nearest_rows(
        self, gdf: gpd.GeoDataFrame, point: Point, radius_meters: float = 25.0
    ) -> gpd.GeoDataFrame:
        """
        Retrieve all rows whose geometries lie within a given radius of a point.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame with geometries in EPSG:4326.
        point : Point
            Query point in EPSG:4326.
        radius_meters : float, default=25.0
            Search radius in meters (projected to EPSG:3857).

        Returns
        -------
        gpd.GeoDataFrame
            Subset of `gdf` where geometries are within `radius_meters` of `point`.
        """
        proj = gdf.to_crs("EPSG:3857")
        p_proj = gpd.GeoSeries([point], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
        mask = proj.geometry.distance(p_proj) <= radius_meters
        return gdf.loc[mask.values]

    def _get_osm_layer(self, tags: Dict[str, Any]) -> gpd.GeoDataFrame:
        """
        Fetch and cache an OpenStreetMap layer matching specified tags.

        Parameters
        ----------
        tags : dict[str, Any]
            OSM feature tags to query (e.g., {'amenity': 'school'}).

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame of OSM features matching `tags`.
        """
        key = frozenset(
            (k, tuple(v) if isinstance(v, (list, tuple)) else v)
            for k, v in tags.items()
        )
        if key not in self.osm_layer_cache:
            self.osm_layer_cache[key] = features_from_bbox(self.bbox, tags)
        return self.osm_layer_cache[key]

    def is_close_to(
        self,
        features: gpd.GeoDataFrame,
        point: Point,
        threshold: float,
        type_column: Optional[Union[str, Sequence[str]]] = None,
        types: Optional[Union[str, Sequence[Union[str, Sequence[str]]]]] = None,
    ) -> bool:
        """
        Determine whether any feature lies within a specified distance from a given point.

        Parameters
        ----------
        features : gpd.GeoDataFrame
            GeoDataFrame of candidate geometries in EPSG:4326.
        point : Point
            Query point in EPSG:4326.
        threshold : float
            Distance threshold in meters (metric CRS EPSG:3857).
        type_column : str or sequence of str, optional
            Column(s) in `features` to filter by.
        types : str or sequence of str, optional
            Allowed value(s) corresponding to `type_column` filter.

        Returns
        -------
        bool
            True if at least one geometry in `features` is within `threshold` meters of `point`.
        """
        if type_column and types is not None:
            cols = [type_column] if isinstance(type_column, str) else list(type_column)
            vals = types if isinstance(types, (list, tuple)) else [types]
            mask = pd.Series(False, index=features.index)
            for col, allowed in zip(cols, vals):
                allowed_list = (
                    allowed if isinstance(allowed, (list, tuple)) else [allowed]
                )
                mask |= features[col].isin(allowed_list)
            features = features.loc[mask]
            if features.empty:
                return False

        feats_proj = features.to_crs("EPSG:3857")
        p_proj = gpd.GeoSeries([point], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
        return (feats_proj.geometry.distance(p_proj) <= threshold).any()

    def count_nearby(
        self,
        features: gpd.GeoDataFrame,
        point: Point,
        threshold: float,
        type_column: Optional[Union[str, Sequence[str]]] = None,
        types: Optional[Union[str, Sequence[Union[str, Sequence[str]]]]] = None,
    ) -> int:
        """
        Count how many geometries lie within a specified distance of a point.

        Parameters
        ----------
        features : gpd.GeoDataFrame
            GeoDataFrame of candidate geometries in EPSG:4326.
        point : Point
            Query point in EPSG:4326.
        threshold : float
            Distance threshold in meters.
        type_column : str or sequence of str, optional
            Column(s) in `features` to filter by category.
        types : str or sequence of str, optional
            Allowed values for the filter columns.

        Returns
        -------
        int
            The count of geometries within `threshold` meters of `point`.
        """
        sub = features.copy()
        if type_column and types is not None:
            cols = [type_column] if isinstance(type_column, str) else list(type_column)
            vals = types if isinstance(types, (list, tuple)) else [types]
            mask = pd.Series(False, index=sub.index)
            for col, allowed in zip(cols, vals):
                allowed_list = (
                    allowed if isinstance(allowed, (list, tuple)) else [allowed]
                )
                mask |= sub[col].isin(allowed_list)
            sub = sub.loc[mask]
            if sub.empty:
                return 0

        proj = sub.to_crs("EPSG:3857")
        p_proj = gpd.GeoSeries([point], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
        return int((proj.geometry.distance(p_proj) <= threshold).sum())

    def add_proximity(
        self,
        df: pd.DataFrame,
        prefix: str,
        tags: Dict[str, Any],
        radii: Sequence[float],
        type_column: Optional[Union[str, Sequence[str]]] = None,
        types: Optional[Union[str, Sequence[Union[str, Sequence[str]]]]] = None,
    ) -> pd.DataFrame:
        """
        Add binary proximity flags for specified OSM features.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing 'x' and 'y' columns in EPSG:4326.
        prefix : str
            Prefix for the new proximity columns.
        tags : dict
            OSM feature tags for the proximity query.
        radii : sequence of float
            Distance thresholds in meters.
        type_column : str or sequence, optional
            Feature attribute to filter by before proximity check.
        types : str or sequence, optional
            Allowed attribute values for filtering.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with boolean columns 'close2{prefix}_{r}'.
        """
        layer = self._get_osm_layer(tags)
        result = df.copy()
        for r in radii:
            col = f"close2{prefix}_{int(r)}"
            result[col] = result.apply(
                lambda row: int(
                    self.is_close_to(layer, Point(row.x, row.y), r, type_column, types)
                ),
                axis=1,
            )
        return result

    def add_count(
        self,
        df: pd.DataFrame,
        prefix: str,
        tags: Dict[str, Any],
        radii: Sequence[float],
        type_column: Optional[Union[str, Sequence[str]]] = None,
        types: Optional[Union[str, Sequence[Union[str, Sequence[str]]]]] = None,
    ) -> pd.DataFrame:
        """
        Add integer counts of OSM features near each point in the DataFrame.

        For each radius in `radii`, computes how many geometries tagged by `tags`
        lie within that distance of the (x, y) coordinate in each row.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with columns 'x' and 'y' in EPSG:4326.
        prefix : str
            Column name prefix (e.g., 'schools').
        tags : Dict[str, Any]
            OSM feature tags to query (e.g., {'amenity': 'school'}).
        radii : Sequence[float]
            Distances in meters for count buffers.
        type_column : str or sequence of str, optional
            Column(s) in OSM layer to filter by before counting.
        types : str or sequence of str, optional
            Allowed value(s) corresponding to `type_column` filter.

        Returns
        -------
        pd.DataFrame
            A copy of `df` with new integer columns named
            'num_{prefix}_{radius}' for each radius.
        """
        layer = self._get_osm_layer(tags)
        result = df.copy()
        for r in radii:
            col = f"num_{prefix}_{int(r)}"
            result[col] = result.apply(
                lambda row: self.count_nearby(
                    layer, Point(row.x, row.y), r, type_column, types
                ),
                axis=1,
            ).astype(int)
        return result

    def add_proportion(
        self,
        df: pd.DataFrame,
        prefix: str,
        tags: Dict[str, Any],
        radii: Sequence[float],
        type_column: Optional[Union[str, Sequence[str]]] = None,
        types: Optional[Union[str, Sequence[Union[str, Sequence[str]]]]] = None,
    ) -> pd.DataFrame:
        """
        Add fractional land-cover proportion of OSM features within each radius.

        For each radius, calculates the ratio of area covered by features tagged
        by `tags` to the circular buffer area around each (x, y) point.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with 'x' and 'y' in EPSG:4326.
        prefix : str
            Prefix for new proportion columns.
        tags : Dict[str, Any]
            OSM feature tags to query for land-cover.
        radii : Sequence[float]
            Buffer distances in meters.
        type_column : str or sequence of str, optional
            Feature attribute to filter by before proportion calculation.
        types : str or sequence of str, optional
            Allowed values for the filter column(s).

        Returns
        -------
        pd.DataFrame
            Copy of `df` with new float columns 'proportion_{prefix}_{radius}'.
        """
        layer = self._get_osm_layer(tags)
        result = df.copy()
        for r in radii:
            col = f"proportion_{prefix}_{int(r)}"
            result[col] = result.apply(
                lambda row: self.land_cover_proportion(
                    layer, Point(row.x, row.y), r, type_column, types
                ),
                axis=1,
            )
        return result

    def land_cover_proportion(
        self,
        features: gpd.GeoDataFrame,
        point: Point,
        threshold: float = 100.0,
        type_column: Optional[Union[str, Sequence[str]]] = None,
        types: Optional[Union[str, Sequence[Union[str, Sequence[str]]]]] = None,
    ) -> float:
        """
        Compute the fraction of buffer area around a point covered by features.

        Buffers a point by `threshold` meters, intersects with `features`, and
        returns the ratio of intersection area to buffer area.

        Parameters
        ----------
        features : gpd.GeoDataFrame
            GeoDataFrame of geometries in EPSG:4326.
        point : Point
            Center of buffer in EPSG:4326.
        threshold : float, default=100.0
            Buffer radius in meters.
        type_column : str or sequence of str, optional
            Column(s) to filter `features` by.
        types : str or sequence of str, optional
            Allowed category value(s) for filter.

        Returns
        -------
        float
            Proportion of buffer area covered by filtered features (0.0–1.0).
        """
        if type_column and types is not None:
            cols = [type_column] if isinstance(type_column, str) else list(type_column)
            vals = types if isinstance(types, (list, tuple)) else [types]
            mask = pd.Series(False, index=features.index)
            for col, allowed in zip(cols, vals):
                allowed_list = (
                    allowed if isinstance(allowed, (list, tuple)) else [allowed]
                )
                mask |= features[col].isin(allowed_list)
            features = features.loc[mask]
            if features.empty:
                return 0.0

        feats_proj = features.to_crs("EPSG:3857")
        p_proj = gpd.GeoSeries([point], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
        buffer_geom = p_proj.buffer(threshold)
        inter_area = feats_proj.geometry.intersection(buffer_geom).area.sum()
        buf_area = buffer_geom.area
        return inter_area / buf_area if buf_area > 0 else 0.0

    def add_mean(
        self,
        df: pd.DataFrame,
        prefix: str,
        source_key: str,
        radii: Sequence[float],
        value_column: str,
    ) -> pd.DataFrame:
        """
        Add spatially-averaged values of a registered geo-source within buffers.

        For each radius, computes the mean of `value_column` from the layer
        registered under `source_key` within `r` meters of each point.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with 'x' and 'y' in EPSG:4326.
        prefix : str
            Prefix for new average columns.
        source_key : str
            Key under which a GeoDataFrame is registered via `load_mean_source`.
        radii : Sequence[float]
            Buffer distances in meters for mean calculation.
        value_column : str
            Numeric column in the source GeoDataFrame to average.

        Returns
        -------
        pd.DataFrame
            Copy of `df` with new float columns 'average_{prefix}_{radius}'.
        """
        layer = self.geo_sources.get(source_key)
        result = df.copy()
        for r in radii:
            col = f"average_{prefix}_{int(r)}"
            result[col] = result.apply(
                lambda row: layer.loc[
                    layer.geometry.distance(Point(row.x, row.y)) <= r, value_column
                ].mean(),
                axis=1,
            )
        return result

    def load_mean_source(
        self, key: str, loader: Callable[[], gpd.GeoDataFrame]
    ) -> None:
        """
        Register a GeoDataFrame for subsequent spatial averaging operations.

        Parameters
        ----------
        key : str
            Identifier for the geo-source.
        loader : callable
            Function returning a GeoDataFrame when called.

        Returns
        -------
        None
            Side-effect: stores the loaded GeoDataFrame under `self.geo_sources[key]`.
        """
        self.geo_sources[key] = loader()
