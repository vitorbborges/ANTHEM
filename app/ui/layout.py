# app/ui/layout.py - Layout management
from typing import Tuple

import streamlit as st

from app.core.config import AppConfig


class LayoutManager:
    """Manages application layout and responsive design."""

    def __init__(self, config: AppConfig):
        self.config = config

    def create_layout(self) -> Tuple[st.delta_generator.DeltaGenerator, ...]:
        """Create responsive column layout based on aspect ratio."""
        aspect_ratio = self.config.bbox.aspect_ratio

        if aspect_ratio > 1.5:
            return st.columns([0.8, 2.5, 0.8])
        elif aspect_ratio < 0.7:
            return st.columns([1.2, 1.5, 1.2])
        else:
            return st.columns([1, 2, 1])
