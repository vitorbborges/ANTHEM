#!/usr/bin/env python3
"""
Complete CO2 Heatmap Generator with OSM Features

This script generates a high-resolution heatmap of predicted CO2 values over the
target bounding box. It uses the same data processing pipeline to extract OSM
features for each grid point, then applies the trained model to predict CO2
concentrations.

Features:
- Creates a grid of points over the target region
- Extracts OSM features using the existing pipeline
- Generates CO2 predictions with uncertainty
- Creates publication-quality heatmaps
- Interactive maps with Folium
- Saves results for further analysis
"""

import warnings

warnings.filterwarnings("ignore")

import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import toml
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
from tqdm.auto import tqdm

# Add src to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Import our modules
try:
    from src.data_processing.spatial_data_loader import SpatialDataLoader
    from src.data_processing.spatial_feature_extractor import SpatialFeatureExtractor
    from src.data_processing.weather_processor import WeatherProcessor
    from src.modeling.best_model_predictor import UniversalKrigingPredictor
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# Target bounding box
WEST, SOUTH, EAST, NORTH = 9.2257, 45.47162, 9.23768, 45.48537
BBOX = (WEST, SOUTH, EAST, NORTH)


class CO2HeatmapGenerator:
    """Generate CO2 prediction heatmaps using OSM features and trained models."""

    def __init__(self, bbox: Tuple[float, float, float, float]):
        """Initialize with target bounding box."""
        self.bbox = bbox
        self.west, self.south, self.east, self.north = bbox

        # Initialize spatial data components
        self.loader = SpatialDataLoader(bbox)
        self.extractor = SpatialFeatureExtractor(self.loader)

        # Load model
        self.predictor = None
        self.load_model()

        # Load feature specifications
        self.feature_specs = self.load_feature_specs()

    def load_model(self, model_path: str = None):
        """Load the best trained model."""
        try:
            if model_path:
                self.predictor = UniversalKrigingPredictor(model_path=model_path)
            else:
                self.predictor = UniversalKrigingPredictor.load_best_model()

            model_info = self.predictor.get_model_info()
            print(f"✅ Loaded model from trial {model_info['trial_number']}")
            print(f"📊 Ensemble size: {model_info['n_subject_models']} subject models")

        except Exception as e:
            raise ValueError(f"Failed to load model: {e}")

    def load_feature_specs(self) -> list:
        """Load feature specifications from TOML file."""
        specs_file = (
            PROJECT_ROOT / "src" / "data_processing" / "config" / "feature_specs.toml"
        )

        if specs_file.exists():
            try:
                config = toml.load(specs_file)
                features = config.get("features", [])
                print(f"📋 Loaded {len(features)} feature specifications")
                return features
            except Exception as e:
                print(f"⚠️ Failed to load feature specs: {e}")
                return []
        else:
            print(f"⚠️ Feature specs file not found at {specs_file}")
            return []

    def create_prediction_grid(self, resolution_meters: float = 25.0) -> pd.DataFrame:
        """Create a grid of points for prediction."""
        # Convert bbox to approximate meters for grid spacing
        # Rough conversion: 1 degree ≈ 111km at this latitude
        lat_center = (self.south + self.north) / 2

        # Meters per degree (approximate)
        meters_per_deg_lon = 111320 * np.cos(np.radians(lat_center))
        meters_per_deg_lat = 110540

        # Calculate grid spacing in degrees
        lon_spacing = resolution_meters / meters_per_deg_lon
        lat_spacing = resolution_meters / meters_per_deg_lat

        # Create coordinate arrays
        lons = np.arange(self.west, self.east + lon_spacing, lon_spacing)
        lats = np.arange(self.south, self.north + lat_spacing, lat_spacing)

        # Create meshgrid
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Flatten to create point list
        points_df = pd.DataFrame(
            {
                "x": lon_grid.flatten(),
                "y": lat_grid.flatten(),
                "location": "grid",  # Dummy location for compatibility
                "regime": "dynamic",  # Dummy regime for compatibility
                "sub": 1,  # Dummy subject for compatibility
            }
        )

        print(f"📍 Created prediction grid: {len(points_df)} points")
        print(f"   Resolution: ~{resolution_meters}m")
        print(f"   Grid dimensions: {len(lons)} x {len(lats)}")

        return points_df

    def extract_osm_features(self, grid_df: pd.DataFrame) -> pd.DataFrame:
        """Extract OSM features for grid points using the existing pipeline."""
        print("🗺️ Extracting OSM features for grid points...")

        if not self.feature_specs:
            print("⚠️ No feature specifications available, returning basic grid")
            return grid_df

        # Start with the grid
        result_df = grid_df.copy()

        # Process each feature specification
        with tqdm(self.feature_specs, desc="Processing features", unit="feat") as pbar:
            for feature in pbar:
                prefix = feature["prefix"]
                pbar.set_postfix_str(prefix)

                try:
                    # Get the feature extraction method
                    mode = feature.get("mode", "proximity")

                    if hasattr(self.extractor, f"add_{mode}"):
                        fn = getattr(self.extractor, f"add_{mode}")

                        # Extract parameters
                        source = feature["source"]
                        radii = feature.get("radii", [])
                        column = feature.get("column")
                        values = feature.get("values", [])

                        # Apply feature extraction
                        result_df = fn(result_df, prefix, source, radii, column, values)

                    else:
                        print(f"⚠️ Unknown feature mode '{mode}' for feature '{prefix}'")

                except Exception as e:
                    print(f"⚠️ Failed to compute feature {prefix}: {e}")
                    continue

        print(f"✅ Feature extraction complete: {result_df.shape[1]} total columns")
        return result_df

    def add_weather_features(
        self, grid_df: pd.DataFrame, weather_conditions: Optional[dict] = None
    ) -> pd.DataFrame:
        """Add weather features to grid points."""
        result_df = grid_df.copy()

        if weather_conditions:
            print("🌡️ Adding specified weather conditions...")
            for key, value in weather_conditions.items():
                result_df[key] = value
                print(f"   {key}: {value}")
        else:
            # Try to load and use average weather conditions from training data
            print("🌡️ Using default weather conditions...")
            try:
                # Load training data to get average weather
                train_data = pd.read_parquet(
                    "data/processed_data/combined_subjects.parquet"
                )
                weather_cols = [
                    "velocita_vento_massimo",
                    "velocita_vento_medio",
                    "direzzione_vento_massimo",
                    "direzzione_vento_medio",
                    "radiazione_globale_medio",
                    "precipitazione_valore_cumulato",
                    "P",
                    "T",
                    "RH",
                ]

                available_weather = [
                    col for col in weather_cols if col in train_data.columns
                ]

                for col in available_weather:
                    avg_value = train_data[col].mean()
                    result_df[col] = avg_value
                    print(f"   {col}: {avg_value:.2f}")

            except Exception as e:
                print(f"⚠️ Could not load average weather conditions: {e}")

        return result_df

    def generate_predictions(self, grid_with_features: pd.DataFrame) -> dict:
        """Generate CO2 predictions for the grid."""
        print("🎯 Generating CO2 predictions...")

        # Prepare environmental data (use Italian names as in training data)
        env_features = [
            "velocita_vento_massimo",
            "velocita_vento_medio",
            "direzzione_vento_massimo",
            "direzzione_vento_medio",
            "radiazione_globale_medio",
            "precipitazione_valore_cumulato",
            "P",
            "T",
            "RH",
        ]

        available_env = [f for f in env_features if f in grid_with_features.columns]

        if available_env:
            env_data = {col: grid_with_features[col].values for col in available_env}
            print(f"   Using environmental features: {len(available_env)} features")
        else:
            env_data = None
            print("   No environmental features available")

        # Make predictions with uncertainty
        try:
            results = self.predictor.predict(
                x=grid_with_features["x"].values,
                y=grid_with_features["y"].values,
                environmental_data=env_data,
                return_uncertainty=True,
                return_individual_predictions=False,
            )

            print(f"✅ Generated predictions for {len(grid_with_features)} points")

            return {
                "predictions": results["predictions"],
                "uncertainty": results.get("uncertainty", None),
                "x": grid_with_features["x"].values,
                "y": grid_with_features["y"].values,
            }

        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            raise

    def create_heatmap(
        self,
        prediction_results: dict,
        interpolation_method: str = "cubic",
        save_path: Optional[str] = None,
    ) -> tuple:
        """Create high-quality heatmap visualizations."""

        x = prediction_results["x"]
        y = prediction_results["y"]
        predictions = prediction_results["predictions"]
        uncertainty = prediction_results.get("uncertainty", None)

        # Create high-resolution grid for interpolation
        xi = np.linspace(x.min(), x.max(), 200)
        yi = np.linspace(y.min(), y.max(), 200)
        xi_grid, yi_grid = np.meshgrid(xi, yi)

        # Interpolate predictions to high-res grid
        zi_pred = griddata(
            (x, y),
            predictions,
            (xi_grid, yi_grid),
            method=interpolation_method,
            fill_value=np.nan,
        )

        # Interpolate uncertainty if available
        zi_uncertainty = None
        if uncertainty is not None:
            zi_uncertainty = griddata(
                (x, y),
                uncertainty,
                (xi_grid, yi_grid),
                method=interpolation_method,
                fill_value=np.nan,
            )

        # Create figure with subplots
        n_plots = 2 if uncertainty is not None else 1
        fig, axes = plt.subplots(1, n_plots, figsize=(12 * n_plots, 10))

        if n_plots == 1:
            axes = [axes]

        # Plot 1: CO2 Predictions
        ax1 = axes[0]

        # Create custom colormap
        colors = ["#2E8B57", "#32CD32", "#FFFF00", "#FFA500", "#FF4500", "#8B0000"]
        n_bins = 256
        cmap_co2 = LinearSegmentedColormap.from_list("co2", colors, N=n_bins)

        # Main heatmap
        im1 = ax1.contourf(
            xi_grid, yi_grid, zi_pred, levels=50, cmap=cmap_co2, alpha=0.8
        )

        # Add contour lines
        contours = ax1.contour(
            xi_grid,
            yi_grid,
            zi_pred,
            levels=10,
            colors="black",
            alpha=0.4,
            linewidths=0.5,
        )
        ax1.clabel(contours, inline=True, fontsize=8, fmt="%.0f")

        # Scatter plot of original prediction points
        scatter1 = ax1.scatter(
            x,
            y,
            c=predictions,
            cmap=cmap_co2,
            s=20,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.5,
        )

        # Formatting
        ax1.set_xlabel("Longitude", fontsize=12)
        ax1.set_ylabel("Latitude", fontsize=12)
        ax1.set_title(
            "CO₂ Concentration Predictions (ppm)", fontsize=14, fontweight="bold"
        )
        ax1.grid(True, alpha=0.3)

        # Colorbar
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label("CO₂ (ppm)", fontsize=12)

        # Add statistics
        stats_text = f"Min: {predictions.min():.1f} ppm\n"
        stats_text += f"Max: {predictions.max():.1f} ppm\n"
        stats_text += f"Mean: {predictions.mean():.1f} ppm\n"
        stats_text += f"Std: {predictions.std():.1f} ppm"

        ax1.text(
            0.02,
            0.98,
            stats_text,
            transform=ax1.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            verticalalignment="top",
            fontsize=10,
        )

        # Plot 2: Uncertainty (if available)
        if uncertainty is not None:
            ax2 = axes[1]

            # Uncertainty heatmap
            im2 = ax2.contourf(
                xi_grid, yi_grid, zi_uncertainty, levels=50, cmap="viridis", alpha=0.8
            )

            # Scatter plot of uncertainty points
            scatter2 = ax2.scatter(
                x,
                y,
                c=uncertainty,
                cmap="viridis",
                s=20,
                alpha=0.7,
                edgecolors="black",
                linewidths=0.5,
            )

            # Formatting
            ax2.set_xlabel("Longitude", fontsize=12)
            ax2.set_ylabel("Latitude", fontsize=12)
            ax2.set_title("Prediction Uncertainty", fontsize=14, fontweight="bold")
            ax2.grid(True, alpha=0.3)

            # Colorbar
            cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
            cbar2.set_label("Uncertainty", fontsize=12)

            # Add uncertainty statistics
            unc_stats = f"Min: {uncertainty.min():.2f}\n"
            unc_stats += f"Max: {uncertainty.max():.2f}\n"
            unc_stats += f"Mean: {uncertainty.mean():.2f}\n"
            unc_stats += f"Std: {uncertainty.std():.2f}"

            ax2.text(
                0.02,
                0.98,
                unc_stats,
                transform=ax2.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment="top",
                fontsize=10,
            )

        plt.tight_layout()

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"💾 Heatmap saved: {save_path}")

        return fig, (zi_pred, zi_uncertainty)

    def create_interactive_map(
        self, prediction_results: dict, save_path: Optional[str] = None
    ) -> str:
        """Create an interactive Folium map."""
        try:
            import folium
            from folium import plugins
        except ImportError:
            print("⚠️ Folium not available. Install with: pip install folium")
            return None

        x = prediction_results["x"]
        y = prediction_results["y"]
        predictions = prediction_results["predictions"]

        # Create base map centered on the bbox
        center_lat = (self.south + self.north) / 2
        center_lon = (self.west + self.east) / 2

        m = folium.Map(
            location=[center_lat, center_lon], zoom_start=15, tiles="OpenStreetMap"
        )

        # Add heatmap layer
        heat_data = [[lat, lon, pred] for lat, lon, pred in zip(y, x, predictions)]

        plugins.HeatMap(
            heat_data,
            min_opacity=0.4,
            max_zoom=18,
            radius=15,
            blur=10,
            gradient={0.0: "green", 0.3: "yellow", 0.6: "orange", 1.0: "red"},
        ).add_to(m)

        # Add individual markers for high CO2 areas
        high_co2_threshold = np.percentile(predictions, 90)
        for i, (lat, lon, pred) in enumerate(zip(y, x, predictions)):
            if pred > high_co2_threshold:
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=5,
                    popup=f"CO₂: {pred:.1f} ppm",
                    color="red",
                    fillColor="red",
                    fillOpacity=0.6,
                ).add_to(m)

        # Add colormap legend
        colormap = folium.LinearColormap(
            colors=["green", "yellow", "orange", "red"],
            vmin=predictions.min(),
            vmax=predictions.max(),
            caption="CO₂ Concentration (ppm)",
        )
        colormap.add_to(m)

        # Save if requested
        if save_path:
            m.save(save_path)
            print(f"🌐 Interactive map saved: {save_path}")

        return m

    def run_full_analysis(
        self,
        resolution_meters: float = 25.0,
        weather_conditions: Optional[dict] = None,
        save_results: bool = True,
    ) -> dict:
        """Run complete heatmap generation pipeline."""

        print("🚀 Starting CO₂ heatmap generation...")
        print(f"📏 Target resolution: {resolution_meters}m")
        print(f"🗺️ Bounding box: {self.bbox}")

        # Step 1: Create prediction grid
        grid_df = self.create_prediction_grid(resolution_meters)

        # Step 2: Extract OSM features
        grid_with_features = self.extract_osm_features(grid_df)

        # Step 3: Add weather features
        grid_with_features = self.add_weather_features(
            grid_with_features, weather_conditions
        )

        # Step 4: Generate predictions
        prediction_results = self.generate_predictions(grid_with_features)

        # Step 5: Create visualizations
        print("📊 Creating visualizations...")

        results_dir = Path("heatmap_results")
        results_dir.mkdir(exist_ok=True)

        # Static heatmap
        heatmap_path = (
            results_dir / f"co2_heatmap_{resolution_meters}m.png"
            if save_results
            else None
        )
        fig, interpolated_grids = self.create_heatmap(
            prediction_results, save_path=heatmap_path
        )

        # Interactive map
        interactive_path = (
            results_dir / f"co2_interactive_map_{resolution_meters}m.html"
            if save_results
            else None
        )
        interactive_map = self.create_interactive_map(
            prediction_results, save_path=interactive_path
        )

        # Save raw results
        if save_results:
            results_df = pd.DataFrame(
                {
                    "x": prediction_results["x"],
                    "y": prediction_results["y"],
                    "co2_prediction": prediction_results["predictions"],
                    "uncertainty": prediction_results.get("uncertainty", np.nan),
                }
            )

            results_path = results_dir / f"co2_predictions_{resolution_meters}m.csv"
            results_df.to_csv(results_path, index=False)
            print(f"💾 Raw results saved: {results_path}")

            # Save grid with features for analysis
            features_path = (
                results_dir / f"grid_with_features_{resolution_meters}m.parquet"
            )
            grid_with_features.to_parquet(features_path)
            print(f"💾 Grid with features saved: {features_path}")

        # Summary statistics
        predictions = prediction_results["predictions"]
        uncertainty = prediction_results.get("uncertainty", None)

        print("\n" + "=" * 60)
        print("HEATMAP GENERATION SUMMARY")
        print("=" * 60)
        print(f"Grid points: {len(predictions)}")
        print(f"Resolution: ~{resolution_meters}m")
        print(f"Features extracted: {grid_with_features.shape[1]} columns")
        print(f"\nCO₂ Predictions:")
        print(f"  Min:  {predictions.min():.1f} ppm")
        print(f"  Max:  {predictions.max():.1f} ppm")
        print(f"  Mean: {predictions.mean():.1f} ppm")
        print(f"  Std:  {predictions.std():.1f} ppm")

        if uncertainty is not None:
            print(f"\nUncertainty:")
            print(f"  Min:  {uncertainty.min():.2f}")
            print(f"  Max:  {uncertainty.max():.2f}")
            print(f"  Mean: {uncertainty.mean():.2f}")

        # Identify hotspots
        hotspot_threshold = np.percentile(predictions, 90)
        hotspots = predictions > hotspot_threshold
        print(f"\nHotspots (>90th percentile, {hotspot_threshold:.1f} ppm):")
        print(f"  Count: {hotspots.sum()} locations")
        print(f"  Percentage: {hotspots.mean()*100:.1f}%")

        print("=" * 60)

        plt.show()

        return {
            "prediction_results": prediction_results,
            "grid_with_features": grid_with_features,
            "interpolated_grids": interpolated_grids,
            "figure": fig,
            "interactive_map": interactive_map,
            "summary_stats": {
                "min": predictions.min(),
                "max": predictions.max(),
                "mean": predictions.mean(),
                "std": predictions.std(),
                "hotspot_threshold": hotspot_threshold,
                "hotspot_count": hotspots.sum(),
            },
        }


if __name__ == "__main__":
    # Example usage
    print("🗺️ CO₂ Heatmap Generator")
    print("=" * 40)

    # Initialize generator
    generator = CO2HeatmapGenerator(BBOX)

    # Define weather conditions (optional)
    # You can specify custom weather conditions or leave as None for defaults
    weather_conditions = {
        "T": 20.0,  # Temperature in Celsius
        "RH": 65.0,  # Relative humidity in %
        "P": 1013.25,  # Pressure in hPa
        "velocita_vento_medio": 3.0,  # Wind speed in m/s
        "direzzione_vento_medio": 180.0,  # Wind direction in degrees
        "radiazione_globale_medio": 500.0,  # Solar radiation
        "precipitazione_valore_cumulato": 0.0,  # Precipitation
    }

    # Generate heatmap
    results = generator.run_full_analysis(
        resolution_meters=20.0,  # 20m resolution
        weather_conditions=weather_conditions,  # Or None for defaults
        save_results=True,
    )

    print("\n✅ Heatmap generation complete!")
    print("📁 Check the 'heatmap_results' directory for outputs")
