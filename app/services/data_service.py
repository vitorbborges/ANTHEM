# app/services/data_service.py - Data access layer
from pathlib import Path
from typing import Any, Dict

import geopandas as gpd
import streamlit as st

from app.core.config import AppConfig
from app.features.path_calculator import PathCalculator

# Import your existing pipeline
from src.data_processing.process_subject_pipeline import ProcessSubjectPipeline


class DataService:
    """Handles data loading and caching."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._pipeline = None
        self._loader = None

    @property
    def pipeline(self):
        """Lazy-loaded pipeline."""
        if self._pipeline is None:
            self._pipeline = ProcessSubjectPipeline(self.config.bbox.bbox_tuple)
        return self._pipeline

    @property
    def loader(self):
        """Lazy-loaded data loader."""
        if self._loader is None:
            self._loader = self.pipeline.extractor.loader
        return self._loader

    @st.cache_resource
    def get_route_data(_self) -> Dict[str, gpd.GeoDataFrame]:
        """Load and cache route data."""
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        route_file = PROJECT_ROOT / _self.config.data_path / "route.kmz"

        gdf = _self.loader.extract_kml(route_file)
        gdf = gdf.to_crs(epsg=4326)

        linestrings = gdf[gdf.geometry.geom_type == "LineString"].copy()
        points = gdf[gdf.geometry.geom_type == "Point"].copy()

        return {"linestrings": linestrings, "points": points}

    @st.cache_resource
    def get_path_calculator(_self) -> PathCalculator:
        """Load and cache path calculator."""
        nodes = _self.loader.get_source("nodes")
        edges = _self.loader.get_source("imputed_edges")
        return PathCalculator(nodes, edges)

    def get_layer_data(self, tags: Dict[str, Any]) -> gpd.GeoDataFrame:
        """Get OSM layer data."""
        return self.loader.get_source(tags).to_crs(epsg=4326)
