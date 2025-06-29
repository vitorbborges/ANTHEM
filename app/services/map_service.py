# app/services/map_service.py - Updated map creation service with exposure path support
from typing import Any, Dict, List, Optional

import folium
import geopandas as gpd

from app.core.config import AppConfig
from app.features.map_handler import MapHandler


class MapService:
    """Service for creating and managing maps with exposure path visualization."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.map_handler = MapHandler(config.bbox.bbox_tuple)

    def create_complete_map(
        self,
        route_gdf: Optional[gpd.GeoDataFrame] = None,
        kml_points: Optional[gpd.GeoDataFrame] = None,
        selected_points: Optional[List[Dict[str, float]]] = None,
        shortest_path_coords: Optional[List[List[float]]] = None,
        exposure_path_coords: Optional[List[List[float]]] = None,
        layer_data: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> folium.Map:
        """Create a complete map with all requested layers including exposure paths."""
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

        # Add shortest path (regular blue path)
        if shortest_path_coords:
            self.map_handler.add_shortest_path(m, shortest_path_coords)

        # Add exposure-optimized path (green path)
        if exposure_path_coords:
            self.add_exposure_path(m, exposure_path_coords)

        # Add OSM layers
        if layer_data:
            self._add_osm_layers(m, layer_data)

        # Add click handler
        self._add_click_handler(m)

        return m

    def add_exposure_path(self, m: folium.Map, path_coords: List[List[float]]):
        """Add exposure-optimized path to the map with distinct styling."""
        if not path_coords:
            return

        path_group = folium.FeatureGroup(name="Least Exposure Path", show=True)

        # Create the path with distinct green styling
        folium.PolyLine(
            locations=path_coords,
            color="#28a745",  # Green color
            weight=8,  # Thicker than normal paths
            opacity=0.9,
            dash_array="10, 5",  # Dashed line to distinguish from shortest path
            popup=folium.Popup(
                """
                <div style='font-family: Arial, sans-serif; max-width: 200px;'>
                    <h4 style='color: #28a745; margin: 0 0 10px 0;'>🫁 Least Exposure Route</h4>
                    <p style='margin: 5px 0; font-size: 12px;'>
                        This route is optimized to minimize CO₂ exposure based on:
                    </p>
                    <ul style='font-size: 11px; margin: 5px 0; padding-left: 15px;'>
                        <li>Your personal physiological profile</li>
                        <li>Real-time CO₂ concentration predictions</li>
                        <li>Walking speed and exposure duration</li>
                    </ul>
                    <p style='font-size: 10px; color: #666; margin: 5px 0 0 0;'>
                        💡 May be longer in distance but healthier for air quality exposure
                    </p>
                </div>
                """,
                max_width=250,
            ),
            tooltip="🫁 Least CO₂ Exposure Path - Click for details",
        ).add_to(path_group)

        # Add start and end markers for the exposure path
        if len(path_coords) >= 2:
            # Start marker
            folium.CircleMarker(
                location=path_coords[0],
                radius=8,
                color="#28a745",
                fill=True,
                fillColor="#28a745",
                fillOpacity=0.8,
                popup="🫁 Exposure Route Start",
                tooltip="Start of least exposure route",
            ).add_to(path_group)

            # End marker
            folium.CircleMarker(
                location=path_coords[-1],
                radius=8,
                color="#28a745",
                fill=True,
                fillColor="#28a745",
                fillOpacity=0.8,
                popup="🫁 Exposure Route End",
                tooltip="End of least exposure route",
            ).add_to(path_group)

        path_group.add_to(m)

    def _add_osm_layers(self, m: folium.Map, layer_data: Dict[str, Dict[str, Any]]):
        """Add OSM layers to the map."""
        for layer_name, data in layer_data.items():
            gdf = data["gdf"]
            color = data["color"]

            # Create a feature group for this layer
            layer_group = folium.FeatureGroup(name=layer_name, show=True)

            for _, row in gdf.iterrows():
                geom = row.geometry

                if geom.geom_type == "Point":
                    folium.CircleMarker(
                        location=[geom.y, geom.x],
                        radius=4,
                        color=color,
                        fill=True,
                        fillOpacity=0.7,
                        popup=f"{layer_name}: {row.get('name', 'Unknown')}",
                        tooltip=layer_name,
                    ).add_to(layer_group)

                elif geom.geom_type == "LineString":
                    folium.PolyLine(
                        locations=[[pt[1], pt[0]] for pt in geom.coords],
                        color=color,
                        weight=2,
                        popup=f"{layer_name}: {row.get('name', 'Unknown')}",
                        tooltip=layer_name,
                    ).add_to(layer_group)

                elif geom.geom_type == "Polygon":
                    folium.GeoJson(
                        geom,
                        popup=f"{layer_name}: {row.get('name', 'Unknown')}",
                        tooltip=layer_name,
                        style_function=lambda x, col=color: {
                            "color": col,
                            "weight": 2,
                            "fillOpacity": 0.3,
                        },
                    ).add_to(layer_group)

            layer_group.add_to(m)

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

    def add_exposure_legend(self, m: folium.Map):
        """Add a legend explaining the different path types."""
        legend_html = """
        <div style="
            position: fixed;
            top: 120px; left: 10px; width: 200px;
            background-color: rgba(255, 255, 255, 0.95);
            border: 2px solid #333;
            z-index: 9999;
            font-size: 12px;
            color: black;
            padding: 12px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        ">
        <div style="text-align: center; margin-bottom: 10px; font-size: 14px; font-weight: bold; color: #333;">
            🗺️ Route Legend
        </div>

        <div style="margin-bottom: 8px; display: flex; align-items: center;">
            <div style="width: 20px; height: 4px; background-color: #518dda; margin-right: 8px; border-radius: 2px;"></div>
            <span style="font-size: 11px;">🛣️ Shortest Distance</span>
        </div>

        <div style="margin-bottom: 8px; display: flex; align-items: center;">
            <div style="width: 20px; height: 4px; background: repeating-linear-gradient(to right, #28a745 0, #28a745 10px, transparent 10px, transparent 15px); margin-right: 8px;"></div>
            <span style="font-size: 11px;">🫁 Least Exposure</span>
        </div>

        <div style="margin-top: 10px; font-size: 10px; color: #666; text-align: center;">
            Green route optimized for<br>minimal CO₂ exposure
        </div>
        </div>
        """

        m.get_root().html.add_child(folium.Element(legend_html))
