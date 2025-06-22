from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import networkx as nx
import osmnx as ox

from app.features.calculations import GeoCalculations


class PathCalculator:
    """Handles shortest path calculations using OSMnx graph data."""

    def __init__(self, nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame):
        """
        Initialize PathCalculator with graph data.

        Parameters
        ----------
        nodes : gpd.GeoDataFrame
            Graph nodes GeoDataFrame
        edges : gpd.GeoDataFrame
            Graph edges GeoDataFrame
        """
        self.nodes = nodes
        self.edges = edges
        self._graph = None

    def _get_graph(self):
        """Lazy load the networkx graph from GeoDataFrames."""
        if self._graph is None:
            self._graph = ox.graph_from_gdfs(self.nodes, self.edges)
        return self._graph

    def calculate_shortest_path(
        self, point1: Dict[str, float], point2: Dict[str, float]
    ) -> Optional[List[List[float]]]:
        """
        Calculate shortest path between two points using the road network.

        Parameters
        ----------
        point1 : Dict[str, float]
            First point with 'lat' and 'lng' keys
        point2 : Dict[str, float]
            Second point with 'lat' and 'lng' keys

        Returns
        -------
        Optional[List[List[float]]]
            List of [lat, lng] coordinates for the path, or None if no path found
        """
        try:
            # Find nearest nodes to the clicked points
            nearest_node1 = ox.distance.nearest_nodes(
                self._get_graph(), point1["lng"], point1["lat"]
            )
            nearest_node2 = ox.distance.nearest_nodes(
                self._get_graph(), point2["lng"], point2["lat"]
            )

            # Calculate shortest path
            try:
                route = nx.shortest_path(
                    self._get_graph(), nearest_node1, nearest_node2, weight="length"
                )

                # Extract coordinates for the path
                path_coords = []
                for node_id in route:
                    node_data = self.nodes.loc[node_id]
                    path_coords.append([node_data.geometry.y, node_data.geometry.x])

                return path_coords

            except nx.NetworkXNoPath:
                return None

        except Exception as e:
            print(f"Error calculating shortest path: {e}")
            return None

    def calculate_path_distance(self, path_coords: List[List[float]]) -> float:
        """
        Calculate the total distance of a path.

        Parameters
        ----------
        path_coords : List[List[float]]
            List of [lat, lng] coordinates

        Returns
        -------
        float
            Total distance in kilometers
        """
        if not path_coords or len(path_coords) < 2:
            return 0.0

        total_distance = 0.0
        for i in range(len(path_coords) - 1):
            lat1, lng1 = path_coords[i]
            lat2, lng2 = path_coords[i + 1]
            total_distance += GeoCalculations.calculate_distance(lat1, lng1, lat2, lng2)

        return total_distance

    def get_path_metrics(
        self, point1: Dict[str, float], point2: Dict[str, float]
    ) -> Dict[str, any]:
        """
        Calculate comprehensive metrics for the shortest path between two points.

        Parameters
        ----------
        point1 : Dict[str, float]
            First point with 'lat' and 'lng' keys
        point2 : Dict[str, float]
            Second point with 'lat' and 'lng' keys

        Returns
        -------
        Dict[str, any]
            Dictionary containing path metrics
        """
        path_coords = self.calculate_shortest_path(point1, point2)

        metrics = {
            "path_exists": path_coords is not None,
            "path_coords": path_coords,
            "path_distance_km": 0.0,
            "path_distance_miles": 0.0,
            "direct_distance_km": 0.0,
            "direct_distance_miles": 0.0,
            "detour_factor": 0.0,
        }

        if path_coords:
            # Calculate path distance
            path_distance = self.calculate_path_distance(path_coords)
            metrics["path_distance_km"] = path_distance
            metrics["path_distance_miles"] = path_distance * 0.621371

            # Calculate direct distance for comparison
            direct_distance = GeoCalculations.calculate_distance(
                point1["lat"], point1["lng"], point2["lat"], point2["lng"]
            )
            metrics["direct_distance_km"] = direct_distance
            metrics["direct_distance_miles"] = direct_distance * 0.621371

            # Calculate detour factor (how much longer the path is vs direct)
            if direct_distance > 0:
                metrics["detour_factor"] = path_distance / direct_distance

        return metrics
