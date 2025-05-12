import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Union

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from osmnx.features import features_from_bbox
from shapely.geometry import LineString, Point
from shapely.ops import linemerge, substring

from src.feature_imputer import FeatureImputer


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
        """
        Initialize without doing any heavy I/O up front.
        All sources—whether named strings or arbitrary OSM‐tag dicts—
        will be loaded lazily via _get_source().
        """
        # bounding box for any OSM queries
        self.bbox: tuple = bbox

        # unified cache: keys are either:
        #  - simple strings for named sources ("edges", "cars", "buildings", etc.)
        #  - frozenset versions of tag‐dicts for arbitrary OSM lookups
        self._source_cache: Dict[Any, gpd.GeoDataFrame] = {}

        # registry of your three special, named loaders
        self._named_loaders: Dict[str, Callable[[], gpd.GeoDataFrame]] = {
            "nodes": self._get_graph_nodes,
            "edges": self._get_graph_edges,
            "imputed_edges": self._get_graph_imputed_edges,
            "cars": self._get_graph_cars,
            "crossing_nodes": self._get_crossing_nodes,
            "traffic_nodes": self._get_traffic_nodes,
            "residential_cars": self._get_residential_cars,
            "service_cars": self._get_service_cars,
            "pedestrian_edges": self._get_pedestrian_edges,
            "imputed_buildings": self._get_imputed_buildings,
        }

        # placeholders for lazy‐loading the graph & buildings
        self._graph: Optional[ox.utils_graph.Graph] = None
        self._graph_nodes: Optional[gpd.GeoDataFrame] = None
        self._graph_edges: Optional[gpd.GeoDataFrame] = None
        self._graph_imputed_edges: Optional[gpd.GeoDataFrame] = None
        self._graph_cars: Optional[gpd.GeoDataFrame] = None
        self._graph_crossing_nodes: Optional[gpd.GeoDataFrame] = None
        self._graph_traffic_nodes: Optional[gpd.GeoDataFrame] = None
        self._graph_residential_cars: Optional[gpd.GeoDataFrame] = None
        self._graph_service_cars: Optional[gpd.GeoDataFrame] = None
        self._graph_pedestrian_edges: Optional[gpd.GeoDataFrame] = None
        self._imputed_buildings: Optional[gpd.GeoDataFrame] = None

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

    def _get_crossing_nodes(self) -> gpd.GeoDataFrame:
        if self._graph_crossing_nodes is None:
            self._load_graph()
        return self._graph_crossing_nodes

    def _get_traffic_nodes(self) -> gpd.GeoDataFrame:
        if self._graph_traffic_nodes is None:
            self._load_graph()
        return self._graph_traffic_nodes

    def _get_residential_cars(self) -> gpd.GeoDataFrame:
        if self._graph_residential_cars is None:
            self._load_graph()
        return self._graph_residential_cars

    def _get_service_cars(self) -> gpd.GeoDataFrame:
        if self._graph_service_cars is None:
            self._load_graph()
        return self._graph_service_cars

    def _get_pedestrian_edges(self) -> gpd.GeoDataFrame:
        if self._graph_pedestrian_edges is None:
            self._load_graph()
        return self._graph_pedestrian_edges

    def _get_imputed_buildings(self) -> gpd.GeoDataFrame:
        if self._imputed_buildings is None:
            self._load_buildings()
        return self._imputed_buildings

    def _get_osm_layer(self, tags: Dict[str, Any]) -> gpd.GeoDataFrame:
        """
        Fetch and cache an OpenStreetMap layer matching specified tags,
        using the unified _source_cache.

        Parameters
        ----------
        tags : dict[str, Any]
            OSM feature tags to query (e.g., {'amenity': 'school'}).

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame of OSM features matching `tags`.
        """
        # Build a hashable key from the tags dict
        tagkey = frozenset(
            (k, tuple(v) if isinstance(v, (list, tuple)) else v)
            for k, v in tags.items()
        )

        # If not already loaded, fetch and cache under the unified cache
        if tagkey not in self._source_cache:
            self._source_cache[tagkey] = features_from_bbox(self.bbox, tags)

        return self._source_cache[tagkey]

    def _get_source(self, key: Union[str, Dict[str, Any]]) -> gpd.GeoDataFrame:
        """
        Retrieve a GeoDataFrame from the unified cache, loading it on first access.

        Parameters
        ----------
        key : str or dict
            - If str: must be one of the named sources in self._named_loaders
            (e.g. "edges", "cars", "buildings", etc.).
            - If dict: an OSM‐tag lookup, e.g. {"amenity": "school"}.

        Returns
        -------
        gpd.GeoDataFrame
            The requested GeoDataFrame, either from cache or freshly loaded.
        """
        # 1) Named‐source case
        if isinstance(key, str):
            if key not in self._named_loaders:
                raise KeyError(f"Unknown named source: {key!r}")
            if key not in self._source_cache:
                # call the loader, cache its result
                df = self._named_loaders[key]()
                self._source_cache[key] = df
            return self._source_cache[key]

        # 2) OSM‐tag lookup case
        if isinstance(key, dict):
            # build the same frozenset key as _get_osm_layer
            tagkey = frozenset(
                (k, tuple(v) if isinstance(v, (list, tuple)) else v)
                for k, v in key.items()
            )
            if tagkey not in self._source_cache:
                # delegate to your existing _get_osm_layer
                df = self._get_osm_layer(key)
                self._source_cache[tagkey] = df
            return self._source_cache[tagkey]

        # 3) Anything else is invalid
        raise TypeError("key must be either a named‐source string or an OSM‐tag dict")

    def _load_buildings(self) -> gpd.GeoDataFrame:
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
        # Impute missing levels
        self._imputed_buildings = FeatureImputer.impute_gdf(
            gdf,
            exclude=["geometry", "name", "roof:levels", "wikidata", "operator"],
        )
        return gdf.drop(columns=["level"], errors="ignore")

    def _load_graph(self) -> None:
        """
        Load and cache the OSM street network graph, extract nodes and edges,
        process and impute edge attributes, and derive the car-only subset.
        """
        if self._graph is None:
            # Fetch full street network graph
            self._graph = ox.graph_from_bbox(
                self.bbox, network_type="all", simplify=False
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
            # car‐only edges
            mask_foot = imputed["highway"].isin(foot_types)
            self._graph_cars = imputed.loc[~mask_foot].copy()

            # pedestrian‐only edges (footway types)
            self._graph_pedestrian_edges = imputed.loc[mask_foot].copy()

            # crossing‐only nodes
            self._graph_crossing_nodes = self._graph_nodes[
                self._graph_nodes["highway"] == "crossing"
            ]

            # traffic_signals‐only nodes
            self._graph_traffic_nodes = self._graph_nodes[
                self._graph_nodes["highway"] == "traffic_signals"
            ]

            # residential car‐streets
            self._graph_residential_cars = self._graph_cars[
                self._graph_cars["highway"] == "residential"
            ]

            # service car‐streets
            self._graph_service_cars = self._graph_cars[
                self._graph_cars["highway"] == "service"
            ]

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
        # ─── Filter by type(s) exactly as in the standalone version ──────────

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

        # ─── Projection and distance test ────────────────────────────────────
        feats_proj = features.to_crs("EPSG:3857")
        p_proj = gpd.GeoSeries([point], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
        distances = feats_proj.geometry.distance(p_proj)
        return (distances <= threshold).any()

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
        sub = features
        # ——— Filter by type(s) exactly like the standalone version —————————————
        if type_column and types:
            # normalize column(s)
            cols = [type_column] if isinstance(type_column, str) else list(type_column)
            # take types as given
            vals = types
            # if it's a single non-list/tuple or a flat list, repeat it for each col
            first = vals[0] if isinstance(vals, list) else vals
            if not isinstance(first, (list, tuple)):
                vals = [vals] * len(cols)
            # ensure each entry is a list
            vals_list = [v if isinstance(v, (list, tuple)) else [v] for v in vals]

            mask = False
            for col, allowed in zip(cols, vals_list):
                mask = mask | sub[col].isin(allowed)
            sub = sub.loc[mask]
            if sub.empty:
                return 0

        # ——— Project and count distances ————————————————————————————————
        sub_proj = sub.to_crs("EPSG:3857")
        p_proj = gpd.GeoSeries([point], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
        dists = sub_proj.geometry.distance(p_proj)
        return int((dists <= threshold).sum())

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
        sub = features
        # ——— Filter by type(s) exactly like the standalone version —————————————
        if type_column and types is not None:
            cols = [type_column] if isinstance(type_column, str) else list(type_column)
            vals = types
            first = vals[0] if isinstance(vals, list) else vals
            if not isinstance(first, (list, tuple)):
                vals = [vals] * len(cols)
            vals_list = [v if isinstance(v, (list, tuple)) else [v] for v in vals]

            mask = False
            for col, allowed in zip(cols, vals_list):
                mask = mask | sub[col].isin(allowed)
            sub = sub.loc[mask]
            if sub.empty:
                return 0.0

        # ——— Project to metric CRS, build buffer, and compute proportion ————————
        sub_proj = sub.to_crs("EPSG:3857")
        p_proj = gpd.GeoSeries([point], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
        buffer_geom = p_proj.buffer(threshold)
        inter_area = sub_proj.geometry.intersection(buffer_geom).area.sum()
        buffer_area = buffer_geom.area

        return inter_area / buffer_area if buffer_area > 0 else 0.0

    def add_proximity(
        self,
        df: pd.DataFrame,
        prefix: str,
        source: Union[str, Dict[str, Any]],
        radii: Sequence[float],
        column: Optional[Union[str, Sequence[str]]] = None,
        values: Optional[Union[str, Sequence[Union[str, Sequence[str]]]]] = None,
    ) -> pd.DataFrame:
        layer = self._get_source(source)
        result = df.copy()
        for r in radii:
            col = f"close2{prefix}_{int(r)}"
            result[col] = result.apply(
                lambda row: int(
                    self.is_close_to(layer, Point(row.x, row.y), r, column, values)
                ),
                axis=1,
            )
        return result

    def add_count(
        self,
        df: pd.DataFrame,
        prefix: str,
        source: Union[str, Dict[str, Any]],
        radii: Sequence[float],
        column: Optional[Union[str, Sequence[str]]] = None,
        values: Optional[Union[str, Sequence[Union[str, Sequence[str]]]]] = None,
    ) -> pd.DataFrame:
        layer = self._get_source(source)
        result = df.copy()
        for r in radii:
            col = f"num_{prefix}_{int(r)}"
            result[col] = result.apply(
                lambda row: self.count_nearby(
                    layer, Point(row.x, row.y), r, column, values
                ),
                axis=1,
            ).astype(int)
        return result

    def add_proportion(
        self,
        df: pd.DataFrame,
        prefix: str,
        source: Union[str, Dict[str, Any]],
        radii: Sequence[float],
        column: Optional[Union[str, Sequence[str]]] = None,
        values: Optional[Union[str, Sequence[Union[str, Sequence[str]]]]] = None,
    ) -> pd.DataFrame:
        layer = self._get_source(source)
        result = df.copy()
        for r in radii:
            col = f"proportion_{prefix}_{int(r)}"
            result[col] = result.apply(
                lambda row: self.land_cover_proportion(
                    layer, Point(row.x, row.y), r, column, values
                ),
                axis=1,
            )
        return result

    def add_sum(
        self,
        df: pd.DataFrame,
        prefix: str,
        source: str,  # now ALWAYS a string key into _named_loaders
        radii: Sequence[float],
        column: str,
        _values=None,  # unused
    ) -> pd.DataFrame:
        layer = self._get_source(source).to_crs(epsg=3857)
        pts = gpd.GeoSeries(
            [Point(x, y) for x, y in zip(df.x, df.y)], index=df.index, crs="EPSG:4326"
        ).to_crs(epsg=3857)
        out = df.copy()
        for r in radii:
            col = f"sum_{prefix}_{int(r)}"
            out[col] = [
                layer.loc[layer.geometry.distance(pt) <= r, column].sum() for pt in pts
            ]
        return out

    def add_mean(
        self,
        df: pd.DataFrame,
        prefix: str,
        source: str,  # string key only
        radii: Sequence[float],
        column: str,
        _values=None,  # unused
    ) -> pd.DataFrame:
        layer = self._get_source(source).to_crs(epsg=3857)
        pts = gpd.GeoSeries(
            [Point(x, y) for x, y in zip(df.x, df.y)], index=df.index, crs="EPSG:4326"
        ).to_crs(epsg=3857)
        out = df.copy()
        for r in radii:
            col = f"average_{prefix}_{int(r)}"
            out[col] = [
                layer.loc[layer.geometry.distance(pt) <= r, column].mean() for pt in pts
            ]
        return out
