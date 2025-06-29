# app/core/state.py - Application state management
from typing import Any, Dict, List, Optional

import streamlit as st


class AppState:
    """Manages application state using Streamlit session state."""

    def __init__(self):
        self._init_state()

    def _init_state(self):
        """Initialize session state variables."""
        defaults = {
            "selected_points": [],
            "last_click_id": None,
            "click_update": False,
            "show_path": False,
            "show_route": True,
            "show_kml_points": True,
            "show_co2_heatmap": True,  # Start with heatmap enabled
            "heatmap_settings": {
                "resolution": 150,
                "colormap": "RdYlBu_r",  # Fixed red-yellow-blue color scheme
                "alpha": 0.7,  # Fixed transparency
                "interpolation_method": "linear",
                "show_legend": True,
            },
            "enabled_layers": set(),
        }

        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    @property
    def selected_points(self) -> List[Dict[str, float]]:
        return st.session_state.selected_points

    @selected_points.setter
    def selected_points(self, value: List[Dict[str, float]]):
        st.session_state.selected_points = value

    @property
    def show_path(self) -> bool:
        return st.session_state.show_path

    @show_path.setter
    def show_path(self, value: bool):
        st.session_state.show_path = value

    @property
    def show_route(self) -> bool:
        return st.session_state.show_route

    @show_route.setter
    def show_route(self, value: bool):
        st.session_state.show_route = value

    @property
    def show_kml_points(self) -> bool:
        return st.session_state.show_kml_points

    @show_kml_points.setter
    def show_kml_points(self, value: bool):
        st.session_state.show_kml_points = value

    @property
    def show_co2_heatmap(self) -> bool:
        return st.session_state.show_co2_heatmap

    @show_co2_heatmap.setter
    def show_co2_heatmap(self, value: bool):
        st.session_state.show_co2_heatmap = value

    @property
    def heatmap_settings(self) -> Dict[str, Any]:
        return st.session_state.heatmap_settings

    @heatmap_settings.setter
    def heatmap_settings(self, value: Dict[str, Any]):
        st.session_state.heatmap_settings = value

    @property
    def enabled_layers(self) -> set:
        return st.session_state.enabled_layers

    @enabled_layers.setter
    def enabled_layers(self, value: set):
        st.session_state.enabled_layers = value

    def clear_points(self):
        """Clear all selected points and reset related state."""
        self.selected_points = []
        self.show_path = False
        st.session_state.last_click_id = None
        st.session_state.click_update = False

    def add_point(self, point: Dict[str, float]):
        """Add a point to selected points."""
        if len(self.selected_points) < 2:
            self.selected_points.append(point)
        else:
            self.selected_points = [point]
        st.session_state.click_update = True

    def can_show_path(self) -> bool:
        """Check if path can be shown."""
        return len(self.selected_points) == 2
