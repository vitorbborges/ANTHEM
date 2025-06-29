# app/services/heatmap_service.py - CO2 heatmap data service with continuous heatmap
import base64
import io
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import folium
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata

warnings.filterwarnings("ignore")


class CO2HeatmapService:
    """Service for loading and displaying CO2 prediction heatmaps."""

    def __init__(self, project_root: Path):
        """
        Initialize the CO2 heatmap service.

        Args:
            project_root: Path to the project root directory
        """
        self.project_root = project_root
        self.grid_cache_path = (
            project_root / "output" / "grid_cache" / "proper_grid_predictions_100m.csv"
        )
        self._cached_data = None

    @st.cache_data
    def load_co2_data(_self) -> Optional[pd.DataFrame]:
        """Load CO2 prediction data with caching."""
        try:
            if not _self.grid_cache_path.exists():
                st.warning(f"CO2 data file not found at {_self.grid_cache_path}")
                return None

            # Load the CSV data
            df = pd.read_csv(_self.grid_cache_path)

            # Validate required columns
            required_cols = ["x", "y", "predicted_co2"]
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                st.error(f"Missing required columns in CO2 data: {missing_cols}")
                return None

            # Clean the data
            df = df.dropna(subset=["x", "y", "predicted_co2"])

            # Rename columns for consistency
            df = df.rename(columns={"x": "lng", "y": "lat"})

            return df

        except Exception as e:
            st.error(f"Error loading CO2 data: {e}")
            return None

    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the loaded CO2 data."""
        df = self.load_co2_data()

        if df is None:
            return {"exists": False}

        return {
            "exists": True,
            "total_points": len(df),
            "co2_min": df["predicted_co2"].min(),
            "co2_max": df["predicted_co2"].max(),
            "co2_mean": df["predicted_co2"].mean(),
            "co2_std": df["predicted_co2"].std(),
            "lat_range": (df["lat"].min(), df["lat"].max()),
            "lng_range": (df["lng"].min(), df["lng"].max()),
        }

    def create_continuous_heatmap_overlay(
        self,
        bbox: Tuple[float, float, float, float],
        resolution: int = 100,
        colormap: str = "RdYlBu_r",
        alpha: float = 0.6,
        interpolation_method: str = "linear",
    ) -> Optional[Tuple[str, Tuple[float, float, float, float]]]:
        """
        Create a continuous heatmap overlay as a base64-encoded PNG.

        Args:
            bbox: Bounding box as (west, south, east, north)
            resolution: Grid resolution for interpolation
            colormap: Matplotlib colormap name
            alpha: Transparency of the heatmap
            interpolation_method: Interpolation method ('linear', 'cubic', 'nearest')

        Returns:
            Tuple of (base64_image_string, bounds) or None if failed
        """
        df = self.load_co2_data()

        if df is None:
            return None

        try:
            west, south, east, north = bbox

            # Filter data to bounding box
            df_filtered = df[
                (df["lat"] >= south)
                & (df["lat"] <= north)
                & (df["lng"] >= west)
                & (df["lng"] <= east)
            ].copy()

            if len(df_filtered) < 3:
                st.warning(
                    "Not enough data points in the bounding box for interpolation"
                )
                return None

            # Create regular grid for interpolation
            grid_x = np.linspace(west, east, resolution)
            grid_y = np.linspace(south, north, resolution)
            grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

            # Interpolate CO2 values to regular grid
            points = df_filtered[["lng", "lat"]].values
            values = df_filtered["predicted_co2"].values

            # Use griddata for interpolation
            grid_z = griddata(
                points,
                values,
                (grid_xx, grid_yy),
                method=interpolation_method,
                fill_value=np.nan,
            )

            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(10, 10))
            fig.patch.set_alpha(0)  # Transparent figure background
            ax.set_xlim(west, east)
            ax.set_ylim(south, north)
            ax.axis("off")  # Remove axes

            # Create heatmap
            im = ax.imshow(
                grid_z,
                extent=[west, east, south, north],
                origin="lower",
                cmap=colormap,
                alpha=alpha,
                aspect="auto",
                interpolation="bilinear",
            )

            # Remove any padding
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)

            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(
                buffer,
                format="png",
                bbox_inches="tight",
                pad_inches=0,
                transparent=True,
                dpi=150,
            )
            buffer.seek(0)

            # Encode to base64
            img_base64 = base64.b64encode(buffer.getvalue()).decode()

            plt.close(fig)  # Clean up

            return f"data:image/png;base64,{img_base64}", (south, north, west, east)

        except Exception as e:
            st.error(f"Error creating continuous heatmap: {e}")
            return None

    def add_continuous_heatmap_to_map(
        self,
        folium_map: folium.Map,
        bbox: Tuple[float, float, float, float],
        resolution: int = 100,
        colormap: str = "RdYlBu_r",
        alpha: float = 0.6,
        interpolation_method: str = "linear",
    ) -> bool:
        """
        Add continuous CO2 heatmap overlay to an existing Folium map.

        Args:
            folium_map: Existing Folium map object
            bbox: Bounding box as (west, south, east, north)
            resolution: Grid resolution for interpolation
            colormap: Matplotlib colormap name
            alpha: Transparency of the heatmap
            interpolation_method: Interpolation method

        Returns:
            bool: True if heatmap was added successfully, False otherwise
        """
        heatmap_result = self.create_continuous_heatmap_overlay(
            bbox, resolution, colormap, alpha, interpolation_method
        )

        if heatmap_result is None:
            return False

        img_base64, bounds = heatmap_result
        south, north, west, east = bounds

        try:
            # Add image overlay to map
            folium.raster_layers.ImageOverlay(
                image=img_base64,
                bounds=[[south, west], [north, east]],
                opacity=alpha,
                interactive=False,
                cross_origin=False,
                name="CO₂ Concentration Heatmap",
            ).add_to(folium_map)

            return True

        except Exception as e:
            st.error(f"Error adding heatmap overlay to map: {e}")
            return False

    def create_colorbar_legend(self, colormap: str = "RdYlBu_r") -> str:
        """Create a vertical colorbar legend as HTML."""
        df = self.load_co2_data()

        if df is None:
            return ""

        min_co2 = df["predicted_co2"].min()
        max_co2 = df["predicted_co2"].max()
        mean_co2 = df["predicted_co2"].mean()

        # Create colormap
        cmap = plt.get_cmap(colormap)

        # Generate gradient colors (reversed for vertical display)
        n_colors = 10
        colors_list = []
        for i in range(n_colors):
            rgba = cmap(
                (n_colors - 1 - i) / (n_colors - 1)
            )  # Reverse for top-to-bottom
            hex_color = colors.rgb2hex(rgba[:3])
            colors_list.append(hex_color)

        gradient_css = ", ".join(
            [f"{color} {i/(n_colors-1)*100}%" for i, color in enumerate(colors_list)]
        )

        legend_html = f"""
        <div style="
            position: fixed;
            top: 10px; right: 10px; width: 120px; height: 180px;
            background-color: rgba(255, 255, 255, 0.95);
            border: 1px solid #333;
            z-index: 9999;
            font-size: 10px;
            color: black;
            padding: 8px;
            border-radius: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        ">
        <div style="text-align: center; margin-bottom: 6px; font-size: 11px; font-weight: bold;">
            🌡️ CO₂ (ppm)
        </div>
        <div style="display: flex; align-items: center; height: 120px;">
            <div style="
                background: linear-gradient(to bottom, {gradient_css});
                width: 20px;
                height: 100%;
                margin-right: 8px;
                border: 1px solid #666;
                border-radius: 2px;
            "></div>
            <div style="display: flex; flex-direction: column; justify-content: space-between; height: 100%; font-size: 9px;">
                <span style="font-weight: bold;">{max_co2:.1f}</span>
                <span style="font-weight: bold; color: #666;">{mean_co2:.1f}</span>
                <span style="font-weight: bold;">{min_co2:.1f}</span>
            </div>
        </div>
        <div style="text-align: center; margin-top: 6px; font-size: 8px; color: #888;">
            Interpolated surface
        </div>
        </div>
        """
        return legend_html

    def add_heatmap_with_legend(
        self,
        folium_map: folium.Map,
        bbox: Tuple[float, float, float, float],
        show_legend: bool = True,
        **heatmap_kwargs,
    ) -> bool:
        """
        Add continuous CO2 heatmap to map with optional legend.

        Args:
            folium_map: Existing Folium map object
            bbox: Bounding box as (west, south, east, north)
            show_legend: Whether to show the legend
            **heatmap_kwargs: Additional arguments for heatmap

        Returns:
            bool: True if successful, False otherwise
        """
        # Add heatmap
        success = self.add_continuous_heatmap_to_map(folium_map, bbox, **heatmap_kwargs)

        if success and show_legend:
            # Add legend
            colormap = heatmap_kwargs.get("colormap", "RdYlBu_r")
            legend_html = self.create_colorbar_legend(colormap)
            folium_map.get_root().html.add_child(folium.Element(legend_html))

        return success

    def get_heatmap_controls(self) -> Dict[str, Any]:
        """
        Create Streamlit controls for heatmap customization.

        Returns:
            Dict with heatmap parameters from user controls
        """
        st.markdown(
            """
            <div style="color: black;">
                <strong>🔥 Continuous Heatmap Settings</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            resolution = st.slider(
                "Grid Resolution",
                min_value=50,
                max_value=300,
                value=50,
                step=25,
                help="Higher resolution = smoother heatmap but slower rendering",
            )

            alpha = st.slider(
                "Transparency",
                min_value=0.1,
                max_value=1.0,
                value=0.6,
                step=0.1,
                help="Opacity of the heatmap overlay",
            )

        with col2:
            colormap_options = {
                "Red-Yellow-Blue (Reversed)": "RdYlBu_r",
                "Plasma": "plasma",
                "Viridis": "viridis",
                "Hot": "hot",
                "Cool": "cool",
                "Jet": "jet",
                "Red-Blue": "RdBu_r",
            }

            colormap_choice = st.selectbox(
                "Color Scheme",
                options=list(colormap_options.keys()),
                index=0,
                help="Color scheme for the heatmap",
            )

            interpolation_options = {
                "Linear": "linear",
                "Cubic": "cubic",
                "Nearest": "nearest",
            }

            interpolation_choice = st.selectbox(
                "Interpolation",
                options=list(interpolation_options.keys()),
                index=0,
                help="Method for interpolating between data points",
            )

        show_legend = st.checkbox(
            "Show Legend", value=True, help="Display CO₂ concentration legend"
        )

        return {
            "resolution": resolution,
            "colormap": colormap_options[colormap_choice],
            "alpha": alpha,
            "interpolation_method": interpolation_options[interpolation_choice],
            "show_legend": show_legend,
        }

    def render_data_summary(self):
        """Render a summary of the CO2 data in Streamlit."""
        info = self.get_data_info()

        if not info["exists"]:
            st.markdown(
                """
                <div style="color: #ff6b6b; background-color: rgba(255,255,255,0.8); padding: 10px; border-radius: 5px;">
                    <strong>⚠️ CO₂ data not available</strong><br>
                    File not found: output/grid_cache/proper_grid_predictions_100m.csv
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        # Data summary in columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"""
                <div style="color: black; background-color: rgba(255,255,255,0.8); padding: 10px; border-radius: 5px; margin: 5px 0;">
                    <strong>📊 Total Points</strong><br>
                    <span style="font-size: 1.2em;">{info['total_points']:,}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div style="color: black; background-color: rgba(255,255,255,0.8); padding: 10px; border-radius: 5px; margin: 5px 0;">
                    <strong>🌡️ Mean CO₂</strong><br>
                    <span style="font-size: 1.2em;">{info['co2_mean']:.1f} ppm</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
                <div style="color: black; background-color: rgba(255,255,255,0.8); padding: 10px; border-radius: 5px; margin: 5px 0;">
                    <strong>📈 Range</strong><br>
                    <span style="font-size: 1.0em;">{info['co2_min']:.1f} - {info['co2_max']:.1f} ppm</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
