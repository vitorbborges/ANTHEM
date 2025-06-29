# app/core/application.py - Updated main application controller with exposure features
from pathlib import Path

import streamlit as st
from streamlit_folium import st_folium

from app.core.config import AppConfig
from app.core.state import AppState
from app.handlers.click_handler import ClickHandler
from app.services.data_service import DataService
from app.services.exposure_service import ExposureService
from app.services.heatmap_service import CO2HeatmapService
from app.services.map_service import MapService
from app.ui.components import UIComponentFactory
from app.ui.layout import LayoutManager


class RouteMapApplication:
    """Main application controller with CO₂ exposure analysis."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.state = AppState()
        self.data_service = DataService(config)
        self.map_service = MapService(config)
        self.layout_manager = LayoutManager(config)
        self.ui_factory = UIComponentFactory()
        self.click_handler = ClickHandler(config.bbox)

        # Initialize services
        project_root = Path(__file__).resolve().parents[2]
        self.heatmap_service = CO2HeatmapService(project_root)
        self.exposure_service = ExposureService(config.bbox.bbox_tuple, project_root)

        self._configure_page()

    def _configure_page(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Interactive Route Map with CO₂ Exposure Analysis",
            layout="wide",
            initial_sidebar_state="expanded",
        )
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

            .stSelectbox label, .stSlider label, .stCheckbox label, .stNumberInput label {
                color: black !important;
            }

            /* Button styling - enhanced for exposure buttons */
            .stButton > button {
                color: white !important;
                background-color: #518dda !important;
                border: 1px solid #518dda !important;
                border-radius: 6px !important;
                font-weight: 500 !important;
            }

            .stButton > button:hover {
                background-color: #4a7bc8 !important;
                border-color: #4a7bc8 !important;
            }

            /* Special styling for exposure path button */
            .exposure-button > button {
                background-color: #28a745 !important;
                border-color: #28a745 !important;
                color: white !important;
                font-weight: bold !important;
            }

            .exposure-button > button:hover {
                background-color: #218838 !important;
                border-color: #1e7e34 !important;
            }

            /* User input section styling */
            .user-input-section {
                background-color: rgba(255, 255, 255, 0.9);
                padding: 15px;
                border-radius: 10px;
                margin: 10px 0;
                border: 2px solid #518dda;
            }

            /* Results display styling */
            .exposure-results {
                background-color: rgba(40, 167, 69, 0.1);
                padding: 10px;
                border-radius: 5px;
                border-left: 4px solid #28a745;
                margin: 10px 0;
            }

            /* Warning/info boxes */
            .info-box {
                background-color: rgba(135, 206, 250, 0.2);
                padding: 10px;
                border-radius: 5px;
                border-left: 4px solid #518dda;
                margin: 10px 0;
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

    def _render_user_input_section(self):
        """Render user input section for physiological parameters."""
        st.markdown(
            """
            <div class="user-input-section">
                <h3>🧑‍⚕️ User Profile for Exposure Analysis</h3>
                <p>Enter your details to calculate personalized CO₂ exposure for route optimization.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input(
                "Age (years)",
                min_value=18,
                max_value=100,
                value=st.session_state.get("user_age", 30),
                step=1,
                help="Your age affects breathing patterns and CO₂ exposure calculations",
            )
            st.session_state.user_age = age

        with col2:
            sex = st.selectbox(
                "Sex",
                options=["M", "F"],
                index=0 if st.session_state.get("user_sex", "M") == "M" else 1,
                help="Biological sex affects lung capacity calculations",
            )
            st.session_state.user_sex = sex

        with col3:
            height = st.number_input(
                "Height (cm)",
                min_value=140,
                max_value=220,
                value=st.session_state.get("user_height", 170),
                step=1,
                help="Height is used to determine lung capacity (FVC)",
            )
            st.session_state.user_height = height

        # Advanced parameters (optional)
        with st.expander("🔬 Advanced Physiological Parameters", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                fb = st.slider(
                    "Breathing Frequency (breaths/min)",
                    min_value=15,
                    max_value=30,
                    value=st.session_state.get("user_fb", 20),
                    step=1,
                    help="Typical walking: 20 breaths/min",
                )
                st.session_state.user_fb = fb

            with col2:
                hr = st.slider(
                    "Heart Rate (bpm)",
                    min_value=80,
                    max_value=150,
                    value=st.session_state.get("user_hr", 100),
                    step=5,
                    help="Typical walking: 100 bpm",
                )
                st.session_state.user_hr = hr

        return age, sex, height, fb, hr

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
        """Render the left sidebar with user input and controls."""
        # User input section
        age, sex, height, fb, hr = self._render_user_input_section()

        # Show current physiological parameters
        try:
            fvc = self.exposure_service.get_fvc_for_subject(age, sex, height)
            if fvc:
                vm = self.exposure_service.calculate_minute_ventilation(
                    age, sex, fvc, fb, hr
                )
                st.markdown(
                    f"""
                    <div class="info-box">
                        <strong>📊 Calculated Parameters:</strong><br>
                        FVC: {fvc:.2f} L<br>
                        Minute Ventilation: {vm*1000:.1f} L/min
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div style="color: #dc3545; background-color: rgba(255,255,255,0.8); padding: 10px; border-radius: 5px;">
                        <strong>⚠️ FVC data not available</strong><br>
                        Exposure analysis requires FVC data file
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        except Exception as e:
            st.markdown(
                f"""
                <div style="color: #dc3545; background-color: rgba(255,255,255,0.8); padding: 10px; border-radius: 5px;">
                    <strong>⚠️ Error calculating parameters:</strong><br>
                    {str(e)[:50]}...
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Layer controls
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
            value=getattr(self.state, "show_co2_heatmap", True),
        )

        # Show CO2 data summary if heatmap is enabled
        if self.state.show_co2_heatmap:
            self.heatmap_service.render_data_summary()

            # Fixed heatmap settings
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
        # Calculate paths if needed
        shortest_path_coords = None
        least_exposure_path_coords = None

        if self.state.can_show_path():
            if self.state.show_path:
                shortest_path_coords = path_calculator.calculate_shortest_path(
                    self.state.selected_points[0], self.state.selected_points[1]
                )

            if getattr(self.state, "show_exposure_path", False):
                age = st.session_state.get("user_age", 30)
                sex = st.session_state.get("user_sex", "M")
                height = st.session_state.get("user_height", 170)

                least_exposure_path_coords = (
                    self.exposure_service.find_least_exposure_path(
                        self.state.selected_points[0],
                        self.state.selected_points[1],
                        age,
                        sex,
                        height,
                    )
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

        # Add least exposure path if available
        if least_exposure_path_coords:
            self._add_exposure_path_to_map(m, least_exposure_path_coords)

        # Add CO2 heatmap if enabled
        if self.state.show_co2_heatmap:
            heatmap_settings = getattr(self.state, "heatmap_settings", {})
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

    def _add_exposure_path_to_map(self, m, path_coords):
        """Add least exposure path to map with distinct styling."""
        import folium

        if not path_coords:
            return

        path_group = folium.FeatureGroup(name="Least Exposure Path", show=True)

        folium.PolyLine(
            locations=path_coords,
            color="#28a745",  # Green color for low exposure
            weight=8,
            opacity=0.8,
            popup="🫁 Least CO₂ Exposure Path",
            tooltip="Optimized route for minimal CO₂ exposure",
        ).add_to(path_group)

        path_group.add_to(m)

    def _handle_map_interactions(self, map_data, kml_points):
        """Handle map click interactions."""
        if self.click_handler.should_handle_click(map_data, self.state):
            point = self.click_handler.extract_click_point(map_data)
            if point and not self.click_handler.is_too_close_to_kml_points(
                point, kml_points
            ):
                self.state.add_point(point)
                # Reset path displays when new point is added
                self.state.show_path = False
                if hasattr(self.state, "show_exposure_path"):
                    self.state.show_exposure_path = False
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
        # Selected points section
        self.ui_factory.create_points_panel(self.state.selected_points)

        # Control buttons section
        st.markdown(
            """
            <div style="color: black;">
                <h4>🎯 Route Options</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Button layout
        col1, col2 = st.columns(2)

        with col1:
            if self.ui_factory.create_clear_button("right"):
                self.state.clear_points()
                if hasattr(self.state, "show_exposure_path"):
                    self.state.show_exposure_path = False
                st.rerun()

        with col2:
            if self.ui_factory.create_path_button(self.state.can_show_path(), "right"):
                if self.state.can_show_path():
                    self.state.show_path = True
                    if hasattr(self.state, "show_exposure_path"):
                        self.state.show_exposure_path = False
                    st.rerun()

        # Exposure path button (full width)
        if self.state.can_show_path():
            # Check if exposure analysis is available
            exposure_available = True
            try:
                # Quick check if we can load prediction data
                test_data = self.exposure_service.load_street_predictions()
                if test_data is None or test_data.empty:
                    exposure_available = False
            except:
                exposure_available = False

            if exposure_available:
                st.markdown('<div class="exposure-button">', unsafe_allow_html=True)
                if st.button(
                    "🫁 Find Least Exposure Path",
                    use_container_width=True,
                    help="Calculate the route with minimal CO₂ exposure based on your profile",
                ):
                    self.state.show_exposure_path = True
                    self.state.show_path = False  # Hide regular shortest path
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown(
                    """
                    <div style="color: #6c757d; background-color: rgba(255,255,255,0.8); padding: 10px; border-radius: 5px; margin: 5px 0;">
                        <strong>🫁 Exposure Analysis Unavailable</strong><br>
                        <small>Run street network prediction script first:<br>
                        <code>python src/visualization/street_graph_prediction.py</code></small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Show path status
        if (
            getattr(self.state, "show_exposure_path", False)
            and self.state.can_show_path()
        ):
            st.markdown(
                """
                <div class="exposure-results">
                    <strong>🫁 Least exposure path is displayed</strong><br>
                    <small>Route optimized for minimal CO₂ exposure based on your profile</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif self.state.show_path and self.state.can_show_path():
            st.markdown(
                """
                <div style="color: #518dda; background-color: rgba(135, 206, 250, 0.2); padding: 10px; border-radius: 5px; margin: 5px 0;">
                    <strong>🛣️ Shortest distance path is displayed</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Path comparison section
        if self.state.can_show_path():
            self._render_path_comparison(path_calculator)

        # Add some spacing
        st.markdown("<br>", unsafe_allow_html=True)

        # Calculations panel
        self.ui_factory.create_calculations_panel(
            self.state.selected_points,
            path_calculator if self.state.show_path else None,
        )

    def _render_path_comparison(self, path_calculator):
        """Render path comparison between shortest and least exposure routes."""
        if not self.state.can_show_path():
            return

        st.markdown(
            """
            <div style="color: black;">
                <h4>📊 Path Analysis</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Get user parameters
        age = st.session_state.get("user_age", 30)
        sex = st.session_state.get("user_sex", "M")
        height = st.session_state.get("user_height", 170)

        try:
            # Calculate both paths
            comparison = self.exposure_service.get_path_exposure_comparison(
                self.state.selected_points[0],
                self.state.selected_points[1],
                age,
                sex,
                height,
            )

            if (
                comparison["shortest_path_exists"]
                or comparison["least_exposure_path_exists"]
            ):

                # Display comparison table
                comparison_data = []

                if comparison["shortest_path_exists"]:
                    comparison_data.append(
                        {
                            "Route Type": "🛣️ Shortest Distance",
                            "Distance (km)": f"{comparison.get('shortest_distance_km', 0):.2f}",
                            "Optimization": "Minimal distance",
                        }
                    )

                if comparison["least_exposure_path_exists"]:
                    comparison_data.append(
                        {
                            "Route Type": "🫁 Least Exposure",
                            "Distance (km)": f"{comparison.get('least_exposure_distance_km', 0):.2f}",
                            "Optimization": "Minimal CO₂ exposure",
                        }
                    )

                if comparison_data:
                    import pandas as pd

                    df_comparison = pd.DataFrame(comparison_data)
                    st.dataframe(
                        df_comparison, use_container_width=True, hide_index=True
                    )

                    # Show recommendation
                    if (
                        comparison["shortest_path_exists"]
                        and comparison["least_exposure_path_exists"]
                    ):

                        dist_diff = comparison.get(
                            "least_exposure_distance_km", 0
                        ) - comparison.get("shortest_distance_km", 0)

                        if dist_diff > 0:
                            st.markdown(
                                f"""
                                <div class="info-box">
                                    <strong>💡 Route Recommendation:</strong><br>
                                    The least exposure route is {dist_diff:.2f} km longer but may significantly
                                    reduce your CO₂ exposure during the journey.
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                """
                                <div class="exposure-results">
                                    <strong>🎉 Great news!</strong><br>
                                    The least exposure route is also the shortest distance route.
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

        except Exception as e:
            st.markdown(
                f"""
                <div style="color: #dc3545; background-color: rgba(255,255,255,0.8); padding: 10px; border-radius: 5px;">
                    <strong>⚠️ Path Analysis Error:</strong><br>
                    {str(e)[:100]}...
                </div>
                """,
                unsafe_allow_html=True,
            )

    def set_background(self, image_path: str):
        """Set background image for the entire Streamlit app using CSS."""
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
