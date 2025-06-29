# app/services/exposure_service.py - CO2 exposure calculation service
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.load_data import load_subject
from src.data_processing.spatial_data_loader import SpatialDataLoader


class ExposureService:
    """Service for calculating CO₂ exposure and finding optimal paths."""

    def __init__(self, bbox, project_root: Path):
        self.bbox = bbox
        self.project_root = project_root
        self.loader = SpatialDataLoader(bbox)
        self._fvc_data = None
        self._walking_speeds = None
        self._avg_walking_speed = None
        self._street_predictions = None
        self._exposure_data = None

    @st.cache_data
    def load_fvc_data(_self):
        """Load FVC data with caching."""
        if _self._fvc_data is not None:
            return _self._fvc_data

        fvc_path = _self.project_root / "data" / "raw_data" / "fvc_data.csv"

        if not fvc_path.exists():
            st.error(f"FVC data file not found at {fvc_path}")
            return None

        try:
            _self._fvc_data = pd.read_csv(fvc_path)
            return _self._fvc_data
        except Exception as e:
            st.error(f"Error loading FVC data: {e}")
            return None

    def get_fvc_for_subject(self, age: int, sex: str, height: float) -> Optional[float]:
        """Get FVC value for subject based on age, sex, and height."""
        fvc_df = self.load_fvc_data()
        if fvc_df is None:
            return None

        # Filter by age and sex
        sex_filter = fvc_df["sex"] == sex
        age_filter = fvc_df["age"] == age

        filtered_df = fvc_df[sex_filter & age_filter]

        if filtered_df.empty:
            # Use closest age match
            closest_age = fvc_df[sex_filter]["age"].iloc[
                (fvc_df[sex_filter]["age"] - age).abs().argsort()[:1]
            ]
            if not closest_age.empty:
                filtered_df = fvc_df[
                    sex_filter & (fvc_df["age"] == closest_age.iloc[0])
                ]

        if filtered_df.empty:
            st.error(f"No FVC data available for sex={sex}")
            return None

        # Find closest height match
        height_diff = (filtered_df["height"] - height).abs()
        best_match_idx = height_diff.idxmin()

        fvc_value = filtered_df.loc[best_match_idx, "fvc_predicted"]
        return fvc_value

    def calculate_minute_ventilation(
        self, age: int, sex: str, fvc: float, fb: float = 20, hr: float = 100
    ) -> float:
        """Calculate minute ventilation using the research formula."""
        sex_code = 1 if sex.upper() == "M" else 2

        vm = (
            np.exp(-8.75)
            * (hr**1.72)
            * (fb**0.611)
            * (age**0.298)
            * (sex_code ** (-0.206))
            * (fvc**0.614)
        )

        # Convert from L/min to m³/min
        vm_m3 = vm / 1000
        return vm_m3

    @st.cache_data
    def calculate_walking_speeds(_self):
        """Calculate average walking speeds from subject data."""
        if _self._walking_speeds is not None and _self._avg_walking_speed is not None:
            return _self._walking_speeds, _self._avg_walking_speed

        try:
            # Load route data
            raw_dir = _self.project_root / "data" / "raw_data"
            kml = _self.loader.extract_kml(raw_dir / "route.kmz")

            # Calculate total route distance
            route_line = kml.geometry.iloc[0]
            lat_center = (_self.bbox[1] + _self.bbox[3]) / 2
            meters_per_degree = 111320 * np.cos(np.radians(lat_center))
            route_distance_meters = route_line.length * meters_per_degree

            # Calculate walking speeds for each subject
            walking_speeds = {}

            for subject_id in range(1, 21):
                try:
                    data = load_subject(subject_id)
                    if data.empty:
                        continue

                    start_time = data.index.min()
                    end_time = data.index.max()
                    duration_seconds = (end_time - start_time).total_seconds()

                    if duration_seconds <= 0:
                        continue

                    walking_speed = route_distance_meters / duration_seconds  # m/s
                    walking_speeds[subject_id] = walking_speed

                except Exception:
                    continue

            avg_speed = (
                np.mean(list(walking_speeds.values())) if walking_speeds else 1.4
            )  # Default walking speed

            _self._walking_speeds = walking_speeds
            _self._avg_walking_speed = avg_speed

            return walking_speeds, avg_speed

        except Exception as e:
            st.error(f"Error calculating walking speeds: {e}")
            return {}, 1.4  # Default walking speed

    @st.cache_data
    def load_street_predictions(_self):
        """Load street network CO₂ predictions."""
        if _self._street_predictions is not None:
            return _self._street_predictions

        # Try multiple possible file locations
        possible_files = [
            _self.project_root
            / "output"
            / "street_predictions"
            / "street_network_predictions_5m.csv",
            _self.project_root
            / "output"
            / "street_predictions"
            / "street_edges_predictions_5m.csv",
            _self.project_root
            / "output"
            / "grid_cache"
            / "proper_grid_predictions_100m.csv",
            _self.project_root
            / "output"
            / "grid_cache"
            / "proper_grid_predictions_50m.csv",
        ]

        for predictions_file in possible_files:
            if predictions_file.exists():
                try:
                    df = pd.read_csv(predictions_file)

                    # Check if this is street network data or grid data
                    if "point_type" in df.columns:
                        # Street network data
                        edge_df = df[df["point_type"] == "edge"].copy()
                        if "edge_id" not in edge_df.columns:
                            # Create edge_id if missing
                            if "point_id" in edge_df.columns:
                                edge_df["edge_id"] = (
                                    edge_df["point_id"].str.split("_").str[0]
                                )
                            else:
                                edge_df["edge_id"] = edge_df.index.astype(str)

                        if len(edge_df) > 0:
                            _self._street_predictions = edge_df
                            return edge_df

                    elif (
                        "predicted_co2" in df.columns
                        and "x" in df.columns
                        and "y" in df.columns
                    ):
                        # Grid data - convert to edge-like format
                        df["edge_id"] = df.index.astype(str)
                        df["point_type"] = "grid"
                        df["distance_along_edge"] = 0
                        _self._street_predictions = df
                        return df

                except Exception as e:
                    st.warning(f"Could not load {predictions_file.name}: {e}")
                    continue

        st.error(
            "No suitable prediction data found. Please run one of these scripts first:"
        )
        st.markdown(
            """
        - Street network predictions: `python src/visualization/street_graph_prediction.py`
        - Grid predictions: `python src/visualization/proper_grid_prediction.py`
        """
        )
        return None

    def calculate_edge_exposures(
        self, age: int, sex: str, height: float, fb: float = 20, hr: float = 100
    ) -> Optional[pd.DataFrame]:
        """Calculate exposure for each edge in the street network."""

        # Get FVC and calculate minute ventilation
        fvc = self.get_fvc_for_subject(age, sex, height)
        if fvc is None:
            st.error("Could not calculate FVC - exposure analysis unavailable")
            return None

        vm = self.calculate_minute_ventilation(age, sex, fvc, fb, hr)

        # Get walking speed and predictions
        _, avg_speed = self.calculate_walking_speeds()
        edge_predictions = self.load_street_predictions()

        if edge_predictions is None or edge_predictions.empty:
            st.error("No prediction data available for exposure calculation")
            return None

        # Check if we have the required columns
        required_cols = ["predicted_co2", "x", "y"]
        missing_cols = [
            col for col in required_cols if col not in edge_predictions.columns
        ]
        if missing_cols:
            st.error(f"Missing required columns in prediction data: {missing_cols}")
            return None

        # Handle different data formats
        if "edge_id" not in edge_predictions.columns:
            st.warning("Creating synthetic edge IDs from data indices")
            edge_predictions = edge_predictions.copy()
            edge_predictions["edge_id"] = edge_predictions.index.astype(str)

        # Group predictions by edge_id and calculate exposures
        try:
            edge_groups = edge_predictions.groupby("edge_id")
            edge_exposures = []

            for edge_id, edge_data in edge_groups:
                # Sort by distance if available
                if "distance_along_edge" in edge_data.columns:
                    edge_data = edge_data.sort_values("distance_along_edge")

                concentrations = edge_data["predicted_co2"].values

                # Remove NaN values
                valid_mask = ~np.isnan(concentrations)
                concentrations = concentrations[valid_mask]

                if len(concentrations) < 1:
                    continue

                # For single points, use point exposure
                if len(concentrations) == 1:
                    # Assume 5m segment exposure
                    time_exposure = 5 / avg_speed / 60  # minutes
                    exposure = concentrations[0] * vm * time_exposure
                else:
                    # Calculate time intervals (5m intervals by default)
                    time_per_interval = 5 / avg_speed  # seconds
                    time_intervals = [time_per_interval] * (len(concentrations) - 1)

                    # Trapezoidal integration
                    sz_p = 0.0
                    for i in range(len(concentrations) - 1):
                        avg_concentration = (
                            concentrations[i] + concentrations[i + 1]
                        ) / 2
                        sz_p += avg_concentration * time_intervals[i]

                    # Convert to minutes and calculate exposure
                    sz_p_minutes = sz_p / 60
                    exposure = sz_p_minutes * vm

                # Get edge metadata
                edge_info = edge_data.iloc[0]

                edge_exposures.append(
                    {
                        "edge_id": edge_id,
                        "exposure": exposure,
                        "mean_co2": np.mean(concentrations),
                        "edge_length": edge_info.get("edge_length", 5.0),  # Default 5m
                        "highway_type": edge_info.get("highway", "unknown"),
                        "start_x": edge_data["x"].iloc[0],
                        "start_y": edge_data["y"].iloc[0],
                        "end_x": (
                            edge_data["x"].iloc[-1]
                            if len(edge_data) > 1
                            else edge_data["x"].iloc[0]
                        ),
                        "end_y": (
                            edge_data["y"].iloc[-1]
                            if len(edge_data) > 1
                            else edge_data["y"].iloc[0]
                        ),
                    }
                )

            if not edge_exposures:
                st.error(
                    "No valid exposures could be calculated from the prediction data"
                )
                return None

            result_df = pd.DataFrame(edge_exposures)
            return result_df

        except Exception as e:
            st.error(f"Error calculating edge exposures: {str(e)}")
            return None

    def create_exposure_graph(self, exposure_df: pd.DataFrame) -> nx.Graph:
        """Create a NetworkX graph with exposure weights."""
        # Get the street network graph
        nodes_gdf = self.loader.get_source("nodes")
        edges_gdf = self.loader.get_source("imputed_edges")

        # Create the base graph
        G = nx.Graph()

        # Add nodes
        for idx, node in nodes_gdf.iterrows():
            G.add_node(idx, pos=(node.geometry.x, node.geometry.y))

        # Add edges with exposure weights
        exposure_dict = dict(zip(exposure_df["edge_id"], exposure_df["exposure"]))

        for idx, edge in edges_gdf.iterrows():
            # Extract edge ID in same format as exposure calculation
            if isinstance(idx, tuple):
                edge_id = f"{idx[0]}_{idx[1]}_{idx[2]}"
            else:
                edge_id = str(idx)

            # Get exposure weight, default to high value if not found
            exposure_weight = exposure_dict.get(edge_id, 1000.0)

            # Add edge (assuming idx is (u, v, key) format)
            if isinstance(idx, tuple) and len(idx) >= 2:
                u, v = idx[0], idx[1]
                G.add_edge(
                    u,
                    v,
                    weight=exposure_weight,
                    edge_id=edge_id,
                    length=edge.get("length", 100),
                )

        return G

    def find_least_exposure_path(
        self,
        point1: Dict[str, float],
        point2: Dict[str, float],
        age: int,
        sex: str,
        height: float,
    ) -> Optional[List[List[float]]]:
        """Find the path with least CO₂ exposure between two points."""

        try:
            # Calculate edge exposures
            exposure_df = self.calculate_edge_exposures(age, sex, height)
            if exposure_df is None or exposure_df.empty:
                st.error("Cannot calculate exposure path - no exposure data available")
                return None

            # For now, fall back to shortest path if exposure graph creation fails
            # This provides basic functionality while the street network data is being prepared
            try:
                # Create exposure-weighted graph
                G = self.create_exposure_graph(exposure_df)

                # Get street network for node finding
                import osmnx as ox

                street_graph = ox.graph_from_gdfs(
                    self.loader.get_source("nodes"),
                    self.loader.get_source("imputed_edges"),
                )

                # Find nearest nodes
                nearest_node1 = ox.distance.nearest_nodes(
                    street_graph, point1["lng"], point1["lat"]
                )
                nearest_node2 = ox.distance.nearest_nodes(
                    street_graph, point2["lng"], point2["lat"]
                )

                # Find shortest path by exposure
                if G.has_node(nearest_node1) and G.has_node(nearest_node2):
                    route = nx.shortest_path(
                        G, nearest_node1, nearest_node2, weight="weight"
                    )

                    # Extract coordinates
                    nodes_gdf = self.loader.get_source("nodes")
                    path_coords = []
                    for node_id in route:
                        if node_id in nodes_gdf.index:
                            node_data = nodes_gdf.loc[node_id]
                            path_coords.append(
                                [node_data.geometry.y, node_data.geometry.x]
                            )

                    return path_coords
                else:
                    st.warning(
                        "Could not find exposure-optimized path between selected points"
                    )
                    return None

            except Exception as graph_error:
                st.warning(
                    f"Exposure graph creation failed: {str(graph_error)[:100]}..."
                )
                st.info("Falling back to shortest distance path")

                # Fall back to regular shortest path
                from app.features.path_calculator import PathCalculator

                nodes = self.loader.get_source("nodes")
                edges = self.loader.get_source("imputed_edges")
                path_calc = PathCalculator(nodes, edges)
                return path_calc.calculate_shortest_path(point1, point2)

        except Exception as e:
            st.error(f"Error finding least exposure path: {str(e)[:100]}...")
            return None

    def get_path_exposure_comparison(
        self,
        point1: Dict[str, float],
        point2: Dict[str, float],
        age: int,
        sex: str,
        height: float,
    ) -> Dict[str, any]:
        """Compare exposure between shortest distance path and least exposure path."""

        # Get both paths
        from app.features.path_calculator import PathCalculator

        nodes = self.loader.get_source("nodes")
        edges = self.loader.get_source("imputed_edges")
        path_calc = PathCalculator(nodes, edges)

        shortest_path = path_calc.calculate_shortest_path(point1, point2)
        least_exposure_path = self.find_least_exposure_path(
            point1, point2, age, sex, height
        )

        # Calculate metrics for both paths
        comparison = {
            "shortest_path_coords": shortest_path,
            "least_exposure_path_coords": least_exposure_path,
            "shortest_path_exists": shortest_path is not None,
            "least_exposure_path_exists": least_exposure_path is not None,
        }

        if shortest_path:
            comparison["shortest_distance_km"] = self._calculate_path_distance(
                shortest_path
            )

        if least_exposure_path:
            comparison["least_exposure_distance_km"] = self._calculate_path_distance(
                least_exposure_path
            )

        return comparison

    def _calculate_path_distance(self, path_coords: List[List[float]]) -> float:
        """Calculate total distance of a path in kilometers."""
        if not path_coords or len(path_coords) < 2:
            return 0.0

        from app.features.calculations import GeoCalculations

        total_distance = 0.0
        for i in range(len(path_coords) - 1):
            lat1, lng1 = path_coords[i]
            lat2, lng2 = path_coords[i + 1]
            total_distance += GeoCalculations.calculate_distance(lat1, lng1, lat2, lng2)

        return total_distance
