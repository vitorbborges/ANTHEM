from typing import Any, Dict, Optional

import geopandas as gpd

from app.core.config import BoundingBox
from app.core.state import AppState


class ClickHandler:
    """Handles map click events."""

    def __init__(self, bbox: BoundingBox):
        self.bbox = bbox

    def should_handle_click(self, map_data: Dict[str, Any], state: AppState) -> bool:
        """Determine if a click should be handled."""
        if not map_data or "last_object_clicked" not in map_data:
            return False

        clicked = map_data["last_object_clicked"]
        if not clicked or not isinstance(clicked, dict):
            return False

        if "lat" not in clicked or "lng" not in clicked:
            return False

        # Ignore clicks with popups (existing markers)
        if clicked.get("popup") is not None:
            return False

        # Check if this is a new click
        lat, lng = clicked["lat"], clicked["lng"]
        click_id = f"{lat:.6f},{lng:.6f}"

        import streamlit as st

        if click_id == st.session_state.get("last_click_id"):
            return False

        # Update last click ID
        st.session_state.last_click_id = click_id

        return True

    def extract_click_point(
        self, map_data: Dict[str, Any]
    ) -> Optional[Dict[str, float]]:
        """Extract point coordinates from click data."""
        clicked = map_data["last_object_clicked"]
        lat, lng = clicked["lat"], clicked["lng"]

        # Check if click is within bounds
        if not (
            self.bbox.south <= lat <= self.bbox.north
            and self.bbox.west <= lng <= self.bbox.east
        ):
            return None

        return {"lat": lat, "lng": lng}

    def is_too_close_to_kml_points(
        self,
        point: Dict[str, float],
        kml_points: gpd.GeoDataFrame,
        threshold_meters: float = 50,
    ) -> bool:
        """Check if click is too close to existing KML points."""
        if kml_points.empty:
            return False

        lat, lng = point["lat"], point["lng"]

        for _, kml_point in kml_points.iterrows():
            kml_lat, kml_lng = kml_point.geometry.y, kml_point.geometry.x

            # Calculate distance in meters (rough approximation)
            lat_diff = abs(lat - kml_lat) * 111000  # degrees to meters
            lng_diff = (
                abs(lng - kml_lng) * 111000 * abs(kml_lat / 90)
            )  # adjust for latitude
            distance = (lat_diff**2 + lng_diff**2) ** 0.5

            if distance < threshold_meters:
                return True

        return False
