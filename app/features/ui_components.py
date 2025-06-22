from typing import Dict, List, Optional

import streamlit as st

from .calculations import GeoCalculations


class UIComponents:
    """UI components for the interactive map application."""

    @staticmethod
    def render_selected_points(selected_points: List[Dict[str, float]]) -> None:
        """
        Render the selected points sidebar.

        Parameters
        ----------
        selected_points : List[Dict[str, float]]
            List of selected points
        """
        st.markdown("### 📍 Selected Points")

        if len(selected_points) == 0:
            st.info("No points selected yet")
        else:
            labels = ["Point 1", "Point 2"]
            for i, point in enumerate(selected_points):
                st.markdown(
                    f"""
                **{labels[i]}:**
                📍 Lat: `{point['lat']:.6f}`
                📍 Lng: `{point['lng']:.6f}`
                """
                )

        if st.button("🗑️ Clear All Points", type="secondary"):
            st.session_state.selected_points = []
            st.rerun()

    @staticmethod
    def render_calculations(selected_points: List[Dict[str, float]]) -> None:
        """
        Render the calculations sidebar.

        Parameters
        ----------
        selected_points : List[Dict[str, float]]
            List of selected points
        """
        st.markdown("### 📊 Calculations")

        if len(selected_points) == 2:
            metrics = GeoCalculations.calculate_all_metrics(
                selected_points[0], selected_points[1]
            )

            # Create metrics with better styling
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("🏃 Distance", f"{metrics['distance_km']:.3f} km")
                st.metric("📐 Bearing", f"{metrics['bearing']:.1f}°")

            with col2b:
                st.metric("🏃 Distance", f"{metrics['distance_miles']:.3f} mi")
                st.metric("🧭 Direction", metrics["cardinal_direction"])

            # Additional calculations in an expander
            with st.expander("📋 Detailed Calculations"):
                st.write("**Coordinate Differences:**")
                st.write(f"📏 Δ Latitude: {metrics['delta_lat']:.6f}°")
                st.write(f"📏 Δ Longitude: {metrics['delta_lng']:.6f}°")
                st.write(
                    f"🎯 Midpoint: ({metrics['midpoint_lat']:.6f}, {metrics['midpoint_lng']:.6f})"
                )

        else:
            st.info("🎯 Select 2 points on the map to see calculations")


class ClickHandler:
    """Handles map click events and point selection logic."""

    @staticmethod
    def handle_map_click(map_data: Dict, bbox: tuple) -> bool:
        """
        Handle map click events and update selected points.

        Parameters
        ----------
        map_data : Dict
            Map data from st_folium
        bbox : tuple
            Bounding box (west, south, east, north)

        Returns
        -------
        bool
            True if a rerun is needed, False otherwise
        """
        if map_data.get("last_object_clicked") is not None:
            clicked_point = map_data["last_object_clicked"]

            # Check if it's a coordinate click and within bounds
            if (
                clicked_point
                and "lat" in clicked_point
                and "lng" in clicked_point
                and clicked_point.get("object") != "Marker"
            ):

                lat, lng = clicked_point["lat"], clicked_point["lng"]
                west, south, east, north = bbox

                # Allow clicks anywhere within the map bounds (not just on route)
                if south <= lat <= north and west <= lng <= east:
                    new_point = {"lat": lat, "lng": lng}

                    # Add point if we have less than 2 points
                    if len(st.session_state.selected_points) < 2:
                        st.session_state.selected_points.append(new_point)
                        return True
                    else:
                        # Reset and add new point (cycle through)
                        st.session_state.selected_points = [new_point]
                        return True

        return False
