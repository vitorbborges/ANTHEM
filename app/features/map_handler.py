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
            point_name = row.get("Name", "Unnamed Point")
            lat, lng = row.geometry.y, row.geometry.x

            # Use CircleMarker with tooltip for better hover behavior
            folium.CircleMarker(
                location=[lat, lng],
                radius=8,
                color="#FF6B6B",
                fill=True,
                fillColor="#FF6B6B",
                fillOpacity=0.8,
                weight=2,
                # Make it non-interactive for clicking to avoid conflicts
                popup=None,
                tooltip=folium.Tooltip(
                    point_name,
                    permanent=False,
                    sticky=True,
                    style="""
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                        font-size: 12px;
                        font-weight: bold;
                        color: white;
                        background-color: rgba(255, 107, 107, 0.9);
                        border: 1px solid white;
                        border-radius: 4px;
                        padding: 4px 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
                    """,
                ),
            ).add_to(points_group)

            # Add an invisible larger circle for better hover detection
            folium.CircleMarker(
                location=[lat, lng],
                radius=15,  # Larger radius for easier hovering
                color="transparent",
                fill=True,
                fillColor="transparent",
                fillOpacity=0.0,
                weight=0,
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
                        border: 1px solid #FF6B6B;
                    '>
                        <div style='
                            color: #FF6B6B;
                            font-weight: bold;
                            margin-bottom: 8px;
                            font-size: 14px;
                        '>{point_name}</div>
                        <div style='margin-bottom: 4px;'>
                            <span style='color: #999;'>Lat:</span>
                            <span style='font-family: monospace;'>{lat:.6f}</span>
                        </div>
                        <div>
                            <span style='color: #999;'>Lng:</span>
                            <span style='font-family: monospace;'>{lng:.6f}</span>
                        </div>
                    </div>
                    """,
                    max_width=200,
                ),
                tooltip=folium.Tooltip(
                    point_name,
                    permanent=False,
                    sticky=True,
                    style="""
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                        font-size: 12px;
                        font-weight: bold;
                        color: white;
                        background-color: rgba(255, 107, 107, 0.9);
                        border: 1px solid white;
                        border-radius: 4px;
                        padding: 4px 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
                    """,
                ),
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

    def create_complete_map(
        self,
        route_gdf: gpd.GeoDataFrame = None,
        kml_points: gpd.GeoDataFrame = None,
        selected_points: List[Dict[str, float]] = None,
        shortest_path_coords: List[List[float]] = None,
        show_route: bool = True,
        show_kml_points: bool = False,
        show_shortest_path: bool = False,
    ) -> folium.Map:
        """
        Create a complete map with optional route data, KML points, selected points, and shortest path.

        Parameters
        ----------
        route_gdf : gpd.GeoDataFrame, optional
            GeoDataFrame containing route linestring data
        kml_points : gpd.GeoDataFrame, optional
            GeoDataFrame containing KML point data
        selected_points : List[Dict[str, float]], optional
            List of selected points
        shortest_path_coords : List[List[float]], optional
            Coordinates for the shortest path
        show_route : bool, default True
            Whether to show the route linestring
        show_kml_points : bool, default False
            Whether to show KML points
        show_shortest_path : bool, default False
            Whether to show the shortest path

        Returns
        -------
        folium.Map
            Complete map ready for display
        """
        m = self.create_base_map()

        # Add route linestring if enabled and data provided
        if show_route and route_gdf is not None:
            self.add_route_data(m, route_gdf)

        # Add KML points if enabled and data provided
        if show_kml_points and kml_points is not None:
            self.add_kml_points(m, kml_points)

        # Add selected point markers if provided
        if selected_points:
            self.add_point_markers(m, selected_points)

        # Add shortest path if enabled and coordinates provided
        if show_shortest_path and shortest_path_coords:
            self.add_shortest_path(m, shortest_path_coords)

        # Fit bounds to the bounding box
        m.fit_bounds([[self.south, self.west], [self.north, self.east]], padding=(0, 0))
        return m
