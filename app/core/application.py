# app/core/application.py - Main application controller
import streamlit as st
from streamlit_folium import st_folium

from app.core.config import AppConfig
from app.core.state import AppState
from app.handlers.click_handler import ClickHandler
from app.services.data_service import DataService
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

        self._configure_page()

    def _configure_page(self):
        """Configure Streamlit page settings."""
        st.set_page_config(page_title="Interactive Route Map", layout="wide")
        self._inject_custom_css()

    def _inject_custom_css(self):
        """Inject custom CSS for styling."""
        css = """
        <style>
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

        image_url = "https://i.pinimg.com/736x/0f/c6/52/0fc6528ee6eedc52bc14a3750eadd500.jpg"
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
        # Selected points section
        self.ui_factory.create_points_panel(self.state.selected_points)

        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if self.ui_factory.create_clear_button():
                self.state.clear_points()
                st.rerun()

        with col2:
            if self.ui_factory.create_path_button(self.state.can_show_path()):
                if self.state.can_show_path():
                    self.state.show_path = True
                    st.rerun()

        # Show path status
        if self.state.show_path and self.state.can_show_path():
            st.success("🛣️ Shortest path is displayed")
        elif self.state.show_path:
            self.state.show_path = False  # Reset if points were cleared elsewhere

        # Layer controls
        self._render_layer_controls()

    def _render_layer_controls(self):
        """Render layer control checkboxes."""
        st.markdown("### 🗂️ Show Layers")

        # Route toggle
        self.state.show_route = st.checkbox(
            "Show Route LineString", value=self.state.show_route
        )

        # In _render_layer_controls()
        self.state.show_kml_points = st.checkbox(
            "Show Static Points", value=self.state.show_kml_points
        )

        # OSM layer toggles
        st.markdown("**OSM Layers:**")
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
        """Render the right sidebar with calculations."""
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
            unsafe_allow_html=True
    )