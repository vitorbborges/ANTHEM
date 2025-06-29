# app/core/application.py - Main application controller
from pathlib import Path

import streamlit as st
from streamlit_folium import st_folium

from app.core.config import AppConfig
from app.core.state import AppState
from app.handlers.click_handler import ClickHandler
from app.services.data_service import DataService
from app.services.heatmap_service import CO2HeatmapService
from app.services.map_service import MapService
from app.ui.components import UIComponentFactory
from app.ui.layout import LayoutManager


class RouteMapApplication:
    """Main application controller."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.state = AppState()
        self.data_service = DataService(config)
        self.map_service = MapService(config)
        self.layout_manager = LayoutManager(config)
        self.ui_factory = UIComponentFactory()
        self.click_handler = ClickHandler(config.bbox)

        # Initialize CO2 heatmap service
        project_root = Path(__file__).resolve().parents[2]  # Go up to project root
        self.heatmap_service = CO2HeatmapService(project_root)

        self._configure_page()

    def _configure_page(self):
        """Configure Streamlit page settings."""
        st.set_page_config(page_title="Interactive Route Map", layout="wide")
        self._inject_custom_css()

    def _inject_custom_css(self):
        """Inject custom CSS for styling."""
        css = """
        <style>
            /* Base text color to black for better readability */
            .stApp, .stApp * {
                color: black !important;
            }

            /* Specific overrides for Streamlit components */
            .stMarkdown, .stMarkdown * {
                color: black !important;
            }

            .stSelectbox label, .stSlider label, .stCheckbox label {
                color: black !important;
            }

            /* Button styling - override black backgrounds */
            .stButton > button {
                color: white !important;
                background-color: #518dda !important;
                border: 1px solid #518dda !important;
                border-radius: 6px !important;
            }

            .stButton > button:hover {
                background-color: #4a7bc8 !important;
                border-color: #4a7bc8 !important;
            }

            .stButton > button[kind="secondary"] {
                background-color: #6c757d !important;
                border-color: #6c757d !important;
                color: white !important;
            }

            .stButton > button[kind="secondary"]:hover {
                background-color: #5a6268 !important;
                border-color: #545b62 !important;
            }

            /* Override code/monospace styling */
            code {
                background-color: rgba(135, 206, 250, 0.9) !important;
                color: black !important;
                padding: 2px 6px !important;
                border-radius: 3px !important;
                border: none !important;
            }

            /* Map styling */
            iframe { border: none !important; outline: none !important; box-shadow: none !important; }
            .stApp > div > div > div > div > div > section > div > div > div > div > div {
                border: none !important; outline: none !important;
            }
            .folium-map { border: none !important; margin: 0 auto !important; padding: 0 !important; display: block !important; }
            .main .block-container { padding-top: 0.2rem; }
            html { scroll-behavior: auto !important; }
            .stApp { overflow-anchor: auto !important; }
            .stHorizontalBlock {
                align-items: flex-start !important;
                justify-content: center !important;
            }
            div[data-testid="stHorizontalBlock"] > div:nth-child(2) {
                display: flex !important;
                justify-content: center !important;
                align-items: flex-start !important;
                overflow: hidden !important;
            }
            iframe[title="streamlit_folium.st_folium"] {
                display: block !important;
                margin: 0 auto !important;
                vertical-align: top !important;
            }
            @media (max-width: 768px) {
                .stHorizontalBlock { flex-direction: column !important; }
            }
        </style>

        <script>
            document.addEventListener('DOMContentLoaded', function() {
                let scrollPosition = sessionStorage.getItem('scrollPosition');
                if (scrollPosition) {
                    window.scrollTo(0, parseInt(scrollPosition));
                    sessionStorage.removeItem('scrollPosition');
                }
                window.addEventListener('beforeunload', function() {
                    sessionStorage.setItem('scrollPosition', window.scrollY);
                });
            });
        </script>
        """
        st.markdown(css, unsafe_allow_html=True)

    def run(self):
        """Run the main application."""
        # Load data
        route_data = self.data_service.get_route_data()
        path_calculator = self.data_service.get_path_calculator()

        # Create layout
        left_col, center_col, right_col = self.layout_manager.create_layout()

        image_url = (
            "https://i.pinimg.com/736x/0f/c6/52/0fc6528ee6eedc52bc14a3750eadd500.jpg"
        )
        self.set_background(image_url)

        # Render left sidebar
        with left_col:
            self._render_left_sidebar()

        # Render main map
        with center_col:
            map_data = self._render_map(route_data, path_calculator)
            self._handle_map_interactions(map_data, route_data["points"])

        # Render right sidebar
        with right_col:
            self._render_right_sidebar(path_calculator)

    def _render_left_sidebar(self):
        """Render the left sidebar with controls."""
        # Layer controls only (buttons moved to right sidebar)
        self._render_layer_controls()

    def _render_layer_controls(self):
        """Render layer control checkboxes."""
        st.markdown(
            """
            <div style="color: black;">
                <h3>🗂️ Show Layers</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Route toggle
        self.state.show_route = st.checkbox(
            "Show Route LineString", value=self.state.show_route
        )

        # KML points toggle
        self.state.show_kml_points = st.checkbox(
            "Show Static Points", value=self.state.show_kml_points
        )

        # CO2 Heatmap section
        st.markdown(
            """
            <div style="color: black;">
                <h4>🔥 CO₂ Concentration Heatmap</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # CO2 heatmap toggle
        self.state.show_co2_heatmap = st.checkbox(
            "Show CO₂ Heatmap",
            value=getattr(self.state, "show_co2_heatmap", True),  # Default to True
        )

        # Show CO2 data summary if heatmap is enabled
        if self.state.show_co2_heatmap:
            self.heatmap_service.render_data_summary()

            # Fixed heatmap settings (no user controls)
            self.state.heatmap_settings = {
                "resolution": 150,
                "colormap": "RdYlBu_r",
                "alpha": 0.7,
                "interpolation_method": "linear",
                "show_legend": True,
            }

        # OSM layer toggles
        st.markdown(
            """
            <div style="color: black;">
                <strong>OSM Layers:</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
        enabled_layers = set()

        for label, layer_config in self.config.osm_layers.items():
            col1, col2 = st.columns([0.7, 0.3])

            with col1:
                if st.checkbox(label, key=f"layer_{label}"):
                    enabled_layers.add(label)

            with col2:
                # Color indicator
                st.markdown(
                    f"""<div style="width: 20px; height: 20px; background-color: {layer_config.color};
                        border-radius: 3px; margin-top: 4px; border: 1px solid #ddd;"></div>""",
                    unsafe_allow_html=True,
                )

        self.state.enabled_layers = enabled_layers

    def _render_map(self, route_data, path_calculator):
        """Render the main map."""
        # Calculate shortest path if needed
        shortest_path_coords = None
        if self.state.show_path and self.state.can_show_path():
            shortest_path_coords = path_calculator.calculate_shortest_path(
                self.state.selected_points[0], self.state.selected_points[1]
            )

        # Get enabled layer data
        layer_data = {}
        for layer_name in self.state.enabled_layers:
            if layer_name in self.config.osm_layers:
                layer_config = self.config.osm_layers[layer_name]
                layer_data[layer_name] = {
                    "gdf": self.data_service.get_layer_data(layer_config.tags),
                    "color": layer_config.color,
                }

        # Create map
        m = self.map_service.create_complete_map(
            route_gdf=route_data["linestrings"] if self.state.show_route else None,
            kml_points=route_data["points"] if self.state.show_kml_points else None,
            selected_points=self.state.selected_points,
            shortest_path_coords=shortest_path_coords,
            layer_data=layer_data,
        )

        # Add CO2 heatmap if enabled
        if self.state.show_co2_heatmap:
            heatmap_settings = getattr(self.state, "heatmap_settings", {})
            # Pass the bounding box to the heatmap service
            success = self.heatmap_service.add_heatmap_with_legend(
                m, self.config.bbox.bbox_tuple, **heatmap_settings
            )

            if not success:
                st.markdown(
                    """
                    <div style="color: #ff6b6b; background-color: rgba(255,255,255,0.8); padding: 10px; border-radius: 5px; margin: 10px 0;">
                        <strong>⚠️ Could not load CO₂ heatmap</strong><br>
                        Check if the data file exists: output/grid_cache/proper_grid_predictions_100m.csv
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Display map
        map_width = self.config.map_config.width
        map_height = int(map_width / self.config.bbox.aspect_ratio)

        return st_folium(
            m,
            width=map_width,
            height=map_height,
            key="interactive_map",
            returned_objects=["last_object_clicked", "bounds"],
        )

    def _handle_map_interactions(self, map_data, kml_points):
        """Handle map click interactions."""
        if self.click_handler.should_handle_click(map_data, self.state):
            point = self.click_handler.extract_click_point(map_data)
            if point and not self.click_handler.is_too_close_to_kml_points(
                point, kml_points
            ):
                self.state.add_point(point)
                # Add scroll position preservation
                st.markdown(
                    """
                    <script>
                        sessionStorage.setItem('scrollPosition', window.scrollY);
                    </script>
                """,
                    unsafe_allow_html=True,
                )
                st.rerun()

    def _render_right_sidebar(self, path_calculator):
        """Render the right sidebar with selected points and calculations."""
        # Selected points section (moved from left sidebar)
        self.ui_factory.create_points_panel(self.state.selected_points)

        # Control buttons (moved from left sidebar)
        col1, col2 = st.columns(2)
        with col1:
            if self.ui_factory.create_clear_button("right"):
                self.state.clear_points()
                st.rerun()

        with col2:
            if self.ui_factory.create_path_button(self.state.can_show_path(), "right"):
                if self.state.can_show_path():
                    self.state.show_path = True
                    st.rerun()

        # Show path status
        if self.state.show_path and self.state.can_show_path():
            st.markdown(
                """
                <div style="color: #28a745; background-color: rgba(173, 216, 230, 0.8); padding: 10px; border-radius: 5px; margin: 5px 0;">
                    <strong>🛣️ Shortest path is displayed</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif self.state.show_path:
            self.state.show_path = False  # Reset if points were cleared elsewhere

        # Add some spacing
        st.markdown("<br>", unsafe_allow_html=True)

        # Calculations panel
        self.ui_factory.create_calculations_panel(
            self.state.selected_points,
            path_calculator if self.state.show_path else None,
        )

    def set_background(self, image_path: str):
        """
        Set background image for the entire Streamlit app using CSS.
        image_path should be a relative path or a URL to the image.
        """
        # If you use a local image, you need to convert it to base64 or serve it publicly.
        # For simplicity, let's assume image_path is a URL or base64 string.
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("{image_path}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
