# app/features/map_handler.py - Keep your existing map handler (updated imports)
from typing import Any, Dict, List, Tuple

import folium
import geopandas as gpd


class MapHandler:
    """Handles map creation and styling for the interactive route viewer."""

    def __init__(self, bbox: Tuple[float, float, float, float]):
        """
        Initialize MapHandler with bounding box coordinates.

        Parameters
        ----------
        bbox : Tuple[float, float, float, float]
            Bounding box as (west, south, east, north)
        """
        self.west, self.south, self.east, self.north = bbox
        self.center_lat = (self.north + self.south) / 2
        self.center_lon = (self.east + self.west) / 2

    def create_base_map(self) -> folium.Map:
        """
        Create the base map with dark theme and controlled interactions.

        Returns
        -------
        folium.Map
            Configured base map
        """
        import math

        lat_span = self.north - self.south
        lng_span = self.east - self.west

        map_width_px = 700
        map_height_px = int(700 * lat_span / lng_span)

        zoom_lng = math.log2(360 * map_width_px / (256 * lng_span)) + 0.5
        zoom_lat = math.log2(180 * map_height_px / (256 * lat_span)) + 0.5
        zoom = min(zoom_lng, zoom_lat)

        m = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=zoom,
            tiles=None,
            zoom_control=False,
            scrollWheelZoom=False,
            doubleClickZoom=False,
            dragging=False,
            touchZoom=False,
            boxZoom=False,
            keyboard=False,
            attributionControl=False,
            prefer_canvas=True,
            crs="EPSG3857",
        )

        folium.TileLayer(
            tiles="https://cartodb-basemaps-{s}.global.ssl.fastly.net/dark_all/{z}/{x}/{y}.png",
            attr=" ",
            name="CartoDB Dark",
            overlay=False,
            control=False,
            max_zoom=20,
            min_zoom=10,
            no_wrap=True,
        ).add_to(m)

        m.fit_bounds(
            [[self.south, self.west], [self.north, self.east]],
            padding=(0, 0),
            max_zoom=None,
        )

        m.options["maxBounds"] = [[self.south, self.west], [self.north, self.east]]
        m.options["maxBoundsViscosity"] = 1.0
        m.options["bounceAtZoomLimits"] = False
        m.options["zoomSnap"] = 0.1
        m.options["zoomDelta"] = 0.1

        return m

    def add_route_data(
        self, m: folium.Map, gdf: gpd.GeoDataFrame, show_bbox: bool = False
    ) -> None:
        """
        Add route LineStrings to the map.

        Parameters
        ----------
        m : folium.Map
            Map to add data to
        gdf : gpd.GeoDataFrame
            GeoDataFrame containing route data
        show_bbox : bool
            Whether to show the bounding box outline
        """
        if show_bbox:
            folium.Rectangle(
                bounds=[[self.south, self.west], [self.north, self.east]],
                color="#00ff00",
                fill=False,
                weight=2,
                opacity=0.5,
                interactive=False,
            ).add_to(m)

        # Only process LineString geometries
        gdf_lines = gdf[gdf.geometry.geom_type == "LineString"].copy()

        if gdf_lines.empty:
            return

        route_group = folium.FeatureGroup(name="Routes", show=True)

        for _, row in gdf_lines.iterrows():
            coords = []
            for coord in row.geometry.coords:
                if len(coord) >= 2:
                    lon, lat = coord[0], coord[1]
                    coords.append([lat, lon])

            folium.PolyLine(
                locations=coords,
                color="#F18F01",
                weight=4,
                opacity=0.9,
                interactive=False,
            ).add_to(route_group)

        route_group.add_to(m)

    def add_kml_points(self, m: folium.Map, points_gdf: gpd.GeoDataFrame) -> None:
        """
        Add KML point markers with hover-triggered text names to the map.

        Parameters
        ----------
        m : folium.Map
            Map to add markers to
        points_gdf : gpd.GeoDataFrame
            GeoDataFrame containing point data with names
        """
        if points_gdf.empty:
            return

        points_group = folium.FeatureGroup(name="KML Points", show=True)

        for idx, row in points_gdf.iterrows():
            point_name = row.get("Name", f"Point {idx}")
            if point_name is None or str(point_name).strip() == "":
                point_name = f"Point {idx}"

            lat, lng = row.geometry.y, row.geometry.x

            # Use a standard Marker with custom icon for guaranteed compatibility
            folium.Marker(
                location=[lat, lng],
                popup=folium.Popup(
                    f"<b>{point_name}</b><br>Lat: {lat:.6f}<br>Lng: {lng:.6f}",
                    max_width=200,
                ),
                tooltip=point_name,
                icon=folium.Icon(color="red", icon="info-sign", prefix="glyphicon"),
            ).add_to(points_group)

        points_group.add_to(m)

    def add_shortest_path(self, m: folium.Map, path_coords: List[List[float]]) -> None:
        """
        Add shortest path polyline to the map.

        Parameters
        ----------
        m : folium.Map
            Map to add path to
        path_coords : List[List[float]]
            List of [lat, lng] coordinates for the path
        """
        if not path_coords:
            return

        path_group = folium.FeatureGroup(name="Shortest Path", show=True)

        folium.PolyLine(
            locations=path_coords,
            color="#00FF00",
            weight=6,
            opacity=0.8,
            popup="Shortest Path via Road Network",
        ).add_to(path_group)

        path_group.add_to(m)

    def add_point_markers(self, m: folium.Map, points: List[Dict[str, float]]) -> None:
        """
        Add point markers to the map.

        Parameters
        ----------
        m : folium.Map
            Map to add markers to
        points : List[Dict[str, float]]
            List of point dictionaries with 'lat' and 'lng' keys
        """
        if not points:
            return

        colors = ["#FF6B6B", "#4ECDC4"]
        labels = ["Point 1", "Point 2"]
        marker_group = folium.FeatureGroup(name="Markers", show=True)

        for i, point in enumerate(points):
            icon_html = f"""
            <div style="
                background-color: {colors[i]};
                width: 24px;
                height: 24px;
                border: 2px solid white;
                border-radius: 50%;
                box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            "></div>
            """

            icon = folium.DivIcon(
                html=icon_html,
                icon_size=(24, 24),
                icon_anchor=(12, 12),
            )

            folium.Marker(
                location=[point["lat"], point["lng"]],
                icon=icon,
                popup=folium.Popup(
                    f"""
                    <div style='
                        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                        font-size: 12px;
                        background: #2F2F2F;
                        color: white;
                        padding: 12px;
                        border-radius: 8px;
                        min-width: 150px;
                        border: 1px solid {colors[i]};
                    '>
                        <div style='
                            color: {colors[i]};
                            font-weight: bold;
                            margin-bottom: 8px;
                            font-size: 14px;
                        '>{labels[i]}</div>
                        <div style='margin-bottom: 4px;'>
                            <span style='color: #999;'>Lat:</span>
                            <span style='font-family: monospace;'>{point['lat']:.6f}</span>
                        </div>
                        <div>
                            <span style='color: #999;'>Lng:</span>
                            <span style='font-family: monospace;'>{point['lng']:.6f}</span>
                        </div>
                    </div>
                    """,
                    max_width=200,
                ),
            ).add_to(marker_group)

        marker_group.add_to(m)
