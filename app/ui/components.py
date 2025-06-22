from typing import Dict, List, Optional

import streamlit as st

from app.features.calculations import GeoCalculations
from app.features.path_calculator import PathCalculator


class UIComponentFactory:
    """Factory for creating UI components."""

    def create_points_panel(self, selected_points: List[Dict[str, float]]):
        """Create the selected points panel."""
        st.markdown("### 📍 Selected Points")

        if not selected_points:
            st.info("No points selected yet")
            return

        labels = ["Point 1", "Point 2"]
        for i, point in enumerate(selected_points):
            st.markdown(
                f"""
                **{labels[i]}:**
                📍 Lat: `{point['lat']:.6f}`
                📍 Lng: `{point['lng']:.6f}`
            """
            )

    def create_clear_button(self) -> bool:
        """Create clear points button."""
        return st.button("🗑️ Clear Points", type="secondary")

    def create_path_button(self, can_show_path: bool) -> bool:
        """Create show path button."""
        if can_show_path:
            return st.button("🛣️ Show Path", type="primary", key="show_path_button")
        else:
            st.button(
                "🛣️ Show Path",
                type="secondary",
                disabled=True,
                help="Select 2 points first",
            )
            return False

    def create_calculations_panel(
        self,
        selected_points: List[Dict[str, float]],
        path_calculator: Optional[PathCalculator] = None,
    ):
        """Create the calculations panel."""
        st.markdown("### 📊 Calculations")

        if len(selected_points) != 2:
            st.info("🎯 Select 2 points to see calculations")
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

        st.metric("🏃 Direct Distance", f"{distance_km:.3f} km")
        st.metric("📐 Bearing", f"{bearing:.1f}°")
        st.metric("🏃 Direct Distance", f"{distance_miles:.3f} mi")
        st.metric("🧭 Direction", cardinal_direction)

    def _render_path_metrics(
        self,
        point1: Dict[str, float],
        point2: Dict[str, float],
        path_calculator: PathCalculator,
    ):
        """Render path-specific metrics."""
        path_metrics = path_calculator.get_path_metrics(point1, point2)

        if path_metrics["path_exists"]:
            st.metric("🛣️ Path Distance", f"{path_metrics['path_distance_km']:.3f} km")
            st.metric(
                "🛣️ Path Distance", f"{path_metrics['path_distance_miles']:.3f} mi"
            )
            st.metric("📈 Detour Factor", f"{path_metrics['detour_factor']:.2f}x")
        else:
            st.warning("⚠️ No path found between points")

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

            st.write("**Coordinate Differences:**")
            st.write(f"📏 Δ Lat: {delta_lat:.6f}°")
            st.write(f"📏 Δ Lng: {delta_lng:.6f}°")
            st.write(f"🎯 Midpoint: ({mid_lat:.6f}, {mid_lng:.6f})")

            if path_calculator:
                path_metrics = path_calculator.get_path_metrics(point1, point2)
                if path_metrics["path_exists"]:
                    st.write("**Path Analysis:**")
                    st.write(
                        f"🛣️ Path length: {path_metrics['path_distance_km']:.3f} km"
                    )
                    st.write(
                        f"✈️ Direct distance: {path_metrics['direct_distance_km']:.3f} km"
                    )
                    extra_distance = (
                        path_metrics["path_distance_km"]
                        - path_metrics["direct_distance_km"]
                    )
                    st.write(f"📈 Extra distance: {extra_distance:.3f} km")
