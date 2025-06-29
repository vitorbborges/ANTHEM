# app/core/state.py - Updated application state management with exposure features
from typing import Any, Dict, List, Optional

import streamlit as st


class AppState:
    """Manages application state using Streamlit session state with exposure analysis features."""

    def __init__(self):
        self._init_state()

    def _init_state(self):
        """Initialize session state variables."""
        defaults = {
            "selected_points": [],
            "last_click_id": None,
            "click_update": False,
            "show_path": False,
            "show_exposure_path": False,  # New: for least exposure path
            "show_route": True,
            "show_kml_points": True,
            "show_co2_heatmap": True,
            "heatmap_settings": {
                "resolution": 150,
                "colormap": "RdYlBu_r",
                "alpha": 0.7,
                "interpolation_method": "linear",
                "show_legend": True,
            },
            "enabled_layers": set(),
            # User profile for exposure calculations
            "user_age": 30,
            "user_sex": "M",
            "user_height": 170,
            "user_fb": 20,  # breathing frequency
            "user_hr": 100,  # heart rate
            # Exposure analysis results
            "exposure_comparison": None,
            "last_exposure_calculation": None,
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
    def show_exposure_path(self) -> bool:
        return st.session_state.show_exposure_path

    @show_exposure_path.setter
    def show_exposure_path(self, value: bool):
        st.session_state.show_exposure_path = value

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

    # User profile properties
    @property
    def user_age(self) -> int:
        return st.session_state.user_age

    @user_age.setter
    def user_age(self, value: int):
        st.session_state.user_age = value

    @property
    def user_sex(self) -> str:
        return st.session_state.user_sex

    @user_sex.setter
    def user_sex(self, value: str):
        st.session_state.user_sex = value

    @property
    def user_height(self) -> float:
        return st.session_state.user_height

    @user_height.setter
    def user_height(self, value: float):
        st.session_state.user_height = value

    @property
    def user_fb(self) -> float:
        return st.session_state.user_fb

    @user_fb.setter
    def user_fb(self, value: float):
        st.session_state.user_fb = value

    @property
    def user_hr(self) -> float:
        return st.session_state.user_hr

    @user_hr.setter
    def user_hr(self, value: float):
        st.session_state.user_hr = value

    @property
    def exposure_comparison(self) -> Optional[Dict[str, Any]]:
        return st.session_state.get("exposure_comparison")

    @exposure_comparison.setter
    def exposure_comparison(self, value: Optional[Dict[str, Any]]):
        st.session_state.exposure_comparison = value

    def clear_points(self):
        """Clear all selected points and reset related state."""
        self.selected_points = []
        self.show_path = False
        self.show_exposure_path = False
        st.session_state.last_click_id = None
        st.session_state.click_update = False
        # Clear exposure analysis when points are cleared
        self.exposure_comparison = None

    def add_point(self, point: Dict[str, float]):
        """Add a point to selected points."""
        if len(self.selected_points) < 2:
            self.selected_points.append(point)
        else:
            self.selected_points = [point]
            # Reset path displays when starting new selection
            self.show_path = False
            self.show_exposure_path = False
            self.exposure_comparison = None

        st.session_state.click_update = True

    def can_show_path(self) -> bool:
        """Check if path can be shown."""
        return len(self.selected_points) == 2

    def get_user_profile(self) -> Dict[str, Any]:
        """Get complete user profile for exposure calculations."""
        return {
            "age": self.user_age,
            "sex": self.user_sex,
            "height": self.user_height,
            "fb": self.user_fb,
            "hr": self.user_hr,
        }

    def update_user_profile(
        self,
        age: int = None,
        sex: str = None,
        height: float = None,
        fb: float = None,
        hr: float = None,
    ):
        """Update user profile parameters."""
        if age is not None:
            self.user_age = age
        if sex is not None:
            self.user_sex = sex
        if height is not None:
            self.user_height = height
        if fb is not None:
            self.user_fb = fb
        if hr is not None:
            self.user_hr = hr

        # Clear previous exposure calculations when profile changes
        self.exposure_comparison = None

    def has_valid_user_profile(self) -> bool:
        """Check if user profile is complete and valid."""
        profile = self.get_user_profile()
        return (
            18 <= profile["age"] <= 100
            and profile["sex"] in ["M", "F"]
            and 140 <= profile["height"] <= 220
            and 15 <= profile["fb"] <= 30
            and 80 <= profile["hr"] <= 150
        )
