# app/services/map_service.py - Map creation service
from typing import Any, Dict, List, Optional

import folium
import geopandas as gpd

from app.core.config import AppConfig
from app.features.map_handler import MapHandler


class MapService:
    """Service for creating and managing maps."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.map_handler = MapHandler(config.bbox.bbox_tuple)

    def create_complete_map(
        self,
        route_gdf: Optional[gpd.GeoDataFrame] = None,
        kml_points: Optional[gpd.GeoDataFrame] = None,
        selected_points: Optional[List[Dict[str, float]]] = None,
        shortest_path_coords: Optional[List[List[float]]] = None,
        layer_data: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> folium.Map:
        """Create a complete map with all requested layers."""
        m = self.map_handler.create_base_map()

        # Add route data
        if route_gdf is not None and not route_gdf.empty:
            self.map_handler.add_route_data(m, route_gdf)

        # Add KML points
        if kml_points is not None and not kml_points.empty:
            self.map_handler.add_kml_points(m, kml_points)

        # Add selected points
        if selected_points:
            self.map_handler.add_point_markers(m, selected_points)

        # Add shortest path
        if shortest_path_coords:
            self.map_handler.add_shortest_path(m, shortest_path_coords)

        # Add OSM layers
        if layer_data:
            self._add_osm_layers(m, layer_data)

        # Add click handler
        self._add_click_handler(m)

        return m

    def _add_osm_layers(self, m: folium.Map, layer_data: Dict[str, Dict[str, Any]]):
        """Add OSM layers to the map."""
        for layer_name, data in layer_data.items():
            gdf = data["gdf"]
            color = data["color"]
            for _, row in gdf.iterrows():
                geom = row.geometry
                if geom.geom_type == "Point":
                    folium.CircleMarker(
                        location=[geom.y, geom.x],
                        radius=4,
                        color=color,
                        fill=True,
                        fill_opacity=0.7,
                        popup=layer_name,
                    ).add_to(m)
                elif geom.geom_type == "LineString":
                    folium.PolyLine(
                        locations=[[pt[1], pt[0]] for pt in geom.coords],
                        color=color,
                        weight=2,
                        popup=layer_name,
                    ).add_to(m)
                elif geom.geom_type == "Polygon":
                    folium.GeoJson(
                        geom,
                        name=layer_name,
                        style_function=lambda x, col=color: {
                            "color": col,
                            "weight": 2,
                            "fillOpacity": 0.3,
                        },
                    ).add_to(m)

    def _add_click_handler(self, m: folium.Map):
        """Add invisible rectangle for click handling."""
        folium.Rectangle(
            bounds=[
                [self.config.bbox.south, self.config.bbox.west],
                [self.config.bbox.north, self.config.bbox.east],
            ],
            color="transparent",
            fill=True,
            fillColor="transparent",
            fillOpacity=0.0,
            weight=0,
            interactive=True,
        ).add_to(m)
