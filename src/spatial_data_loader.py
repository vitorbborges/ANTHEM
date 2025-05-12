import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import geopandas as gpd
import numpy as np
import osmnx as ox
from osmnx.features import features_from_bbox
from shapely.geometry import LineString
from shapely.ops import linemerge, substring

from src.feature_imputer import FeatureImputer


class SpatialDataLoader:
    """
    Loads and caches spatial data sources, including OSM layers, street networks,
    and building footprints for a defined bounding box.

    Attributes
    ----------
    bbox : tuple
        Bounding box specified as (north, south, east, west) or similar.
    _source_cache : Dict[Any, gpd.GeoDataFrame]
        Cache for previously loaded GeoDataFrames, keyed by source name or tagset.
    _named_loaders : Dict[str, Callable[[], gpd.GeoDataFrame]]
        Mapping of source names to loader functions for those data types.
    _graph : Optional[ox.utils_graph.Graph]
        Cached OSM street network graph.
    ...
    """

    def __init__(self, bbox: tuple) -> None:
        # Initialize bounding box and caches
        self.bbox: tuple = bbox
        self._source_cache: Dict[Any, gpd.GeoDataFrame] = {}
        self._named_loaders: Dict[str, Callable[[], gpd.GeoDataFrame]] = {
            # Register named data loader methods
            "nodes": self._get_graph_nodes,
            "edges": self._get_graph_edges,
            "imputed_edges": self._get_graph_imputed_edges,
            "cars": self._get_graph_cars,
            "pedestrian_edges": self._get_pedestrian_edges,
            "crossing_nodes": self._get_crossing_nodes,
            "traffic_nodes": self._get_traffic_nodes,
            "residential_cars": self._get_residential_cars,
            "service_cars": self._get_service_cars,
            "imputed_buildings": self._get_imputed_buildings,
        }
        # Placeholders for data once loaded
        self._graph: Optional[ox.utils_graph.Graph] = None
        self._graph_nodes: Optional[gpd.GeoDataFrame] = None
        self._graph_edges: Optional[gpd.GeoDataFrame] = None
        self._graph_imputed_edges: Optional[gpd.GeoDataFrame] = None
        self._graph_cars: Optional[gpd.GeoDataFrame] = None
        self._graph_pedestrian_edges: Optional[gpd.GeoDataFrame] = None
        self._graph_crossing_nodes: Optional[gpd.GeoDataFrame] = None
        self._graph_traffic_nodes: Optional[gpd.GeoDataFrame] = None
        self._graph_residential_cars: Optional[gpd.GeoDataFrame] = None
        self._graph_service_cars: Optional[gpd.GeoDataFrame] = None
        self._imputed_buildings: Optional[gpd.GeoDataFrame] = None

    def get_source(self, key: Union[str, Dict[str, Any]]) -> gpd.GeoDataFrame:
        """
        Retrieve a GeoDataFrame by named source or OSM tag dictionary.

        Parameters
        ----------
        key : Union[str, dict]
            If str, must be a registered loader name. If dict, interpreted as
            OSM tags to filter when querying bounding box.

        Returns
        -------
        gpd.GeoDataFrame
            The requested spatial data frame, loaded and cached on first call.

        Raises
        ------
        KeyError
            If a string key is unknown in named loaders.
        TypeError
            If key is neither str nor dict.
        """
        if isinstance(key, str):
            # Lookup by named loader
            if key not in self._named_loaders:
                raise KeyError(f"Unknown named source: {key}")
            if key not in self._source_cache:
                # Call loader to populate cache
                self._source_cache[key] = self._named_loaders[key]()
            return self._source_cache[key]

        if isinstance(key, dict):
            # Convert tag dict into an immutable key
            tagkey = frozenset(
                (k, tuple(v) if isinstance(v, (list, tuple)) else v)
                for k, v in key.items()
            )
            if tagkey not in self._source_cache:
                # Query OSM features for the bounding box
                self._source_cache[tagkey] = features_from_bbox(self.bbox, key)
            return self._source_cache[tagkey]

        # Invalid key type
        raise TypeError("key must be a string or dict of OSM tags")

    def _load_graph(self) -> None:
        """
        Internal: Load and preprocess the full OSM street network.

        - Fetch graph from OSM.
        - Convert to GeoDataFrames for nodes and edges.
        - Clean and cast edge attributes.
        - Impute missing edge fields.
        - Partition into subsets for different transportation modes.
        """
        if self._graph is not None:
            return  # Already loaded
        # Fetch raw graph without simplification
        self._graph = ox.graph_from_bbox(
            self.bbox,
            network_type="all",
            simplify=False,
        )
        # Extract nodes and edges into GeoDataFrames
        nodes, edges = ox.graph_to_gdfs(self._graph)
        # Prepare edge attributes for imputation
        edges_orig = edges.copy()
        edges_orig["lanes"] = edges_orig["lanes"].fillna(0)
        edges_orig["oneway"] = edges_orig["oneway"].astype(int)
        edges_orig["reversed"] = edges_orig["reversed"].astype(int)
        edges_orig["maxspeed"] = edges_orig["maxspeed"].astype(float)
        # Ensure correct lane counts for oneway roads
        edges_orig.loc[edges_orig["oneway"] == 1, "lanes"] = 1
        # Replace zero lanes with NaN for imputation
        edges_orig.loc[edges_orig["lanes"] == 0, "lanes"] = np.nan
        self._graph_nodes = nodes
        self._graph_edges = edges_orig
        # Impute missing values on edges
        imputed = FeatureImputer.impute_edges(edges_orig)
        self._graph_imputed_edges = imputed
        # Identify pedestrian vs. vehicle edges
        foot_types = [
            "footway",
            "pedestrian",
            "unclassified",
            "steps",
            "corridor",
            "path",
        ]
        # Vehicle edges exclude foot types
        self._graph_cars = imputed[~imputed["highway"].isin(foot_types)].copy()
        # Pedestrian-only edges
        self._graph_pedestrian_edges = imputed[
            imputed["highway"].isin(foot_types)
        ].copy()
        # Nodes at crossings and traffic signals
        self._graph_crossing_nodes = nodes[nodes["highway"] == "crossing"]
        self._graph_traffic_nodes = nodes[nodes["highway"] == "traffic_signals"]
        # Subsets of car edges by residential/service classification
        self._graph_residential_cars = self._graph_cars[
            self._graph_cars["highway"] == "residential"
        ]
        self._graph_service_cars = self._graph_cars[
            self._graph_cars["highway"] == "service"
        ]

    # Named loader methods for each cached graph component
    def _get_graph_nodes(self) -> gpd.GeoDataFrame:
        """Return nodes GeoDataFrame from loaded graph."""
        self._load_graph()
        return self._graph_nodes  # type: ignore

    def _get_graph_edges(self) -> gpd.GeoDataFrame:
        """Return original edges GeoDataFrame with raw attributes."""
        self._load_graph()
        return self._graph_edges  # type: ignore

    def _get_graph_imputed_edges(self) -> gpd.GeoDataFrame:
        """Return edges GeoDataFrame after attribute imputation."""
        self._load_graph()
        return self._graph_imputed_edges  # type: ignore

    def _get_graph_cars(self) -> gpd.GeoDataFrame:
        """Return drivable edges excluding pedestrian ways."""
        self._load_graph()
        return self._graph_cars  # type: ignore

    def _get_pedestrian_edges(self) -> gpd.GeoDataFrame:
        """Return pedestrian-only edges GeoDataFrame."""
        self._load_graph()
        return self._graph_pedestrian_edges  # type: ignore

    def _get_crossing_nodes(self) -> gpd.GeoDataFrame:
        """Return node locations identified as crossings."""
        self._load_graph()
        return self._graph_crossing_nodes  # type: ignore

    def _get_traffic_nodes(self) -> gpd.GeoDataFrame:
        """Return node locations identified as traffic signals."""
        self._load_graph()
        return self._graph_traffic_nodes  # type: ignore

    def _get_residential_cars(self) -> gpd.GeoDataFrame:
        """Return residential vehicle edges."""
        self._load_graph()
        return self._graph_residential_cars  # type: ignore

    def _get_service_cars(self) -> gpd.GeoDataFrame:
        """Return service vehicle edges."""
        self._load_graph()
        return self._graph_service_cars  # type: ignore

    def _load_buildings(self) -> None:
        """
        Internal: Fetch building footprints from OSM and impute missing levels.

        - Query 'building' features within bbox.
        - Normalize 'building:levels' and 'level' fields.
        - Impute missing levels via FeatureImputer.
        """
        tags = {"building": True}
        gdf = features_from_bbox(self.bbox, tags)
        # Standardize text labels for ground floor
        gdf = gdf.replace({"building:levels": {"piano terra": 0}})
        if "level" in gdf.columns:
            gdf.loc[gdf["level"] == "-1", "level"] = np.nan
            gdf["level"] = gdf["level"].astype(float)
        # Combine explicit and inferred building levels
        gdf["building:levels"] = (
            gdf["building:levels"].astype(float).fillna(gdf.get("level", np.nan))
        )
        # Store imputed footprints
        self._imputed_buildings = FeatureImputer.impute_gdf(
            gdf,
            exclude=["geometry", "name", "roof:levels", "wikidata", "operator"],
        )

    def _get_imputed_buildings(self) -> gpd.GeoDataFrame:
        """Return building footprints with levels imputed."""
        if self._imputed_buildings is None:
            self._load_buildings()
        return self._imputed_buildings  # type: ignore

    def extract_kml(self, kmz_path: Path) -> gpd.GeoDataFrame:
        """
        Extract features from a KMZ file containing KML data.

        Parameters
        ----------
        kmz_path : Path
            Path to the KMZ archive.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame read from the contained KML document.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Unzip KMZ archive to temporary directory
            with zipfile.ZipFile(kmz_path, "r") as archive:
                archive.extractall(tmpdir)
            # Read extracted KML
            return gpd.read_file(Path(tmpdir) / "doc.kml", driver="KML")

    def build_segments(self, kml_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Split a closed LineString into ordered segments based on KML point breaks.

        Parameters
        ----------
        kml_gdf : gpd.GeoDataFrame
            GeoDataFrame with the first row as a closed LineString and subsequent
            rows as Point geometries indicating segment break locations.

        Returns
        -------
        gpd.GeoDataFrame
            Segmented line pieces with labels for each location segment.
        """
        line: LineString = kml_gdf.geometry.iloc[0]
        # Identify distinct break points along the line
        pts = kml_gdf.geometry.iloc[1:].drop_duplicates().tolist()
        # Compute distances along line for each break point
        dists = sorted(line.project(pt) for pt in pts)
        # Handle wrapping segment across the line end
        wrap = linemerge(
            [substring(line, dists[-1], line.length), substring(line, 0.0, dists[0])]
        )
        # Create all intermediate segments between break distances
        segments = [wrap] + [
            substring(line, a, b) for a, b in zip(dists[:-1], dists[1:])
        ]
        # Define location labels for segments
        locs = ["FG", "GH", "AB", "BC", "CD", "DE", "EF"]
        # Return as GeoDataFrame with original CRS
        return gpd.GeoDataFrame(
            {"location": locs, "geometry": segments}, crs=kml_gdf.crs
        )
