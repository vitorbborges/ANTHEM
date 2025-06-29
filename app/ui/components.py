from typing import Dict, List, Optional

import streamlit as st

from app.features.calculations import GeoCalculations
from app.features.path_calculator import PathCalculator


class UIComponentFactory:
    """Factory for creating UI components."""

    def create_points_panel(self, selected_points: List[Dict[str, float]]):
        """Create the selected points panel."""
        st.markdown(
            """
            <div style="color: black;">
                <h3>📍 Selected Points</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not selected_points:
            st.markdown(
                """
                <div style="color: black; background-color: rgba(173, 216, 230, 0.8); padding: 10px; border-radius: 5px; margin: 5px 0;">
                    <p style="margin: 0;">No points selected yet</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        labels = ["Point 1", "Point 2"]
        for i, point in enumerate(selected_points):
            st.markdown(
                f"""
                <div style="color: black; background-color: rgba(173, 216, 230, 0.8); padding: 10px; border-radius: 5px; margin: 5px 0;">
                    <strong>{labels[i]}:</strong><br>
                    📍 Lat: <span style="background-color: rgba(135, 206, 250, 0.9); padding: 2px 6px; border-radius: 3px; font-family: monospace; color: black;">{point['lat']:.6f}</span><br>
                    📍 Lng: <span style="background-color: rgba(135, 206, 250, 0.9); padding: 2px 6px; border-radius: 3px; font-family: monospace; color: black;">{point['lng']:.6f}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    def create_clear_button(self, key_suffix: str = "") -> bool:
        """Create clear points button."""
        return st.button(
            "🗑️ Clear Points", type="secondary", key=f"clear_points_{key_suffix}"
        )

    def create_path_button(self, can_show_path: bool, key_suffix: str = "") -> bool:
        """Create show path button."""
        if can_show_path:
            return st.button(
                "🛣️ Show Path", type="primary", key=f"show_path_{key_suffix}"
            )
        else:
            st.button(
                "🛣️ Show Path",
                type="secondary",
                disabled=True,
                help="Select 2 points first",
                key=f"show_path_disabled_{key_suffix}",
            )
            return False

    def create_calculations_panel(
        self,
        selected_points: List[Dict[str, float]],
        path_calculator: Optional[PathCalculator] = None,
    ):
        """Create the calculations panel."""
        st.markdown(
            """
            <div style="color: black;">
                <h3>📊 Calculations</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if len(selected_points) != 2:
            st.markdown(
                """
                <div style="color: black; background-color: rgba(173, 216, 230, 0.8); padding: 10px; border-radius: 5px; margin: 5px 0;">
                    <p style="margin: 0;">🎯 Select 2 points to see calculations</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        point1, point2 = selected_points

        # Direct distance calculations
        self._render_basic_metrics(point1, point2)

        # Path calculations if available
        if path_calculator:
            self._render_path_metrics(point1, point2, path_calculator)

        # Detailed calculations
        self._render_detailed_calculations(point1, point2, path_calculator)

    def _render_basic_metrics(self, point1: Dict[str, float], point2: Dict[str, float]):
        """Render basic distance and bearing metrics."""
        distance_km = GeoCalculations.calculate_distance(
            point1["lat"], point1["lng"], point2["lat"], point2["lng"]
        )
        bearing = GeoCalculations.calculate_bearing(
            point1["lat"], point1["lng"], point2["lat"], point2["lng"]
        )
        distance_miles = distance_km * 0.621371
        cardinal_direction = GeoCalculations.get_cardinal_direction(bearing)

        # Create custom styled metrics with blue theme
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
                <div style="color: black; background-color: rgba(173, 216, 230, 0.8); padding: 10px; border-radius: 5px; margin: 5px 0;">
                    <strong>🏃 Direct Distance</strong><br>
                    <span style="font-size: 1.2em;">{distance_km:.3f} km</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div style="color: black; background-color: rgba(173, 216, 230, 0.8); padding: 10px; border-radius: 5px; margin: 5px 0;">
                    <strong>🏃 Direct Distance</strong><br>
                    <span style="font-size: 1.2em;">{distance_miles:.3f} mi</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div style="color: black; background-color: rgba(173, 216, 230, 0.8); padding: 10px; border-radius: 5px; margin: 5px 0;">
                    <strong>📐 Bearing</strong><br>
                    <span style="font-size: 1.2em;">{bearing:.1f}°</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div style="color: black; background-color: rgba(173, 216, 230, 0.8); padding: 10px; border-radius: 5px; margin: 5px 0;">
                    <strong>🧭 Direction</strong><br>
                    <span style="font-size: 1.2em;">{cardinal_direction}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    def _render_path_metrics(
        self,
        point1: Dict[str, float],
        point2: Dict[str, float],
        path_calculator: PathCalculator,
    ):
        """Render path-specific metrics."""
        path_metrics = path_calculator.get_path_metrics(point1, point2)

        if path_metrics["path_exists"]:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"""
                    <div style="color: black; background-color: rgba(135, 206, 250, 0.8); padding: 10px; border-radius: 5px; margin: 5px 0;">
                        <strong>🛣️ Path Distance</strong><br>
                        <span style="font-size: 1.2em;">{path_metrics['path_distance_km']:.3f} km</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f"""
                    <div style="color: black; background-color: rgba(135, 206, 250, 0.8); padding: 10px; border-radius: 5px; margin: 5px 0;">
                        <strong>🛣️ Path Distance</strong><br>
                        <span style="font-size: 1.2em;">{path_metrics['path_distance_miles']:.3f} mi</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"""
                <div style="color: black; background-color: rgba(135, 206, 250, 0.8); padding: 10px; border-radius: 5px; margin: 5px 0;">
                    <strong>📈 Detour Factor</strong><br>
                    <span style="font-size: 1.2em;">{path_metrics['detour_factor']:.2f}x</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="color: #ff6b6b; background-color: rgba(173, 216, 230, 0.8); padding: 10px; border-radius: 5px; margin: 5px 0;">
                    <strong>⚠️ No path found between points</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

    def _render_detailed_calculations(
        self,
        point1: Dict[str, float],
        point2: Dict[str, float],
        path_calculator: Optional[PathCalculator],
    ):
        """Render detailed calculations in an expander."""
        with st.expander("📋 Details"):
            delta_lat = abs(point2["lat"] - point1["lat"])
            delta_lng = abs(point2["lng"] - point1["lng"])
            mid_lat = (point1["lat"] + point2["lat"]) / 2
            mid_lng = (point1["lng"] + point2["lng"]) / 2

            st.markdown(
                f"""
                <div style="color: black; background-color: rgba(173, 216, 230, 0.6); padding: 10px; border-radius: 5px; margin: 5px 0;">
                    <strong>Coordinate Differences:</strong><br>
                    📏 Δ Lat: {delta_lat:.6f}°<br>
                    📏 Δ Lng: {delta_lng:.6f}°<br>
                    🎯 Midpoint: ({mid_lat:.6f}, {mid_lng:.6f})
                </div>
                """,
                unsafe_allow_html=True,
            )

            if path_calculator:
                path_metrics = path_calculator.get_path_metrics(point1, point2)
                if path_metrics["path_exists"]:
                    extra_distance = (
                        path_metrics["path_distance_km"]
                        - path_metrics["direct_distance_km"]
                    )
                    st.markdown(
                        f"""
                        <div style="color: black; background-color: rgba(135, 206, 250, 0.6); padding: 10px; border-radius: 5px; margin: 5px 0;">
                            <strong>Path Analysis:</strong><br>
                            🛣️ Path length: {path_metrics['path_distance_km']:.3f} km<br>
                            ✈️ Direct distance: {path_metrics['direct_distance_km']:.3f} km<br>
                            📈 Extra distance: {extra_distance:.3f} km
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
