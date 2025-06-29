import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import toml
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")


class StreamlitGridLoader:
    """
    Streamlit-integrated grid loader for CO2 prediction visualization.
    Handles grid creation, feature extraction, prediction, and interactive visualization.
    """

    def __init__(
        self,
        project_root: Union[str, Path],
        bbox: Tuple[float, float, float, float] = (9.2257, 45.47162, 9.23768, 45.48537),
    ):
        """
        Initialize the StreamlitGridLoader.

        Args:
            project_root: Path to the project root directory
            bbox: Bounding box as (west, south, east, north)
        """
        self.project_root = Path(project_root)
        self.bbox = bbox

        # Set up paths
        self.config_toml = (
            self.project_root
            / "src"
            / "data_processing"
            / "config"
            / "feature_specs.toml"
        )
        self.raw_data_dir = self.project_root / "data" / "raw_data"
        self.models_dir = self.project_root / "models"
        self.output_dir = self.project_root / "output"

        # Initialize components (lazy loading)
        self._loader = None
        self._extractor = None
        self._weather_processor = None
        self._model_data = None

        # Cache for grid data
        self._cached_grid = None
        self._cached_features = None
        self._cached_predictions = None

    @st.cache_data
    def _import_dependencies(_self):
        """Import heavy dependencies with caching."""
        try:
            import sys

            sys.path.insert(0, str(_self.project_root))

            from src.data_processing.spatial_data_loader import SpatialDataLoader
            from src.data_processing.spatial_feature_extractor import (
                SpatialFeatureExtractor,
            )
            from src.data_processing.weather_processor import WeatherProcessor

            return SpatialDataLoader, SpatialFeatureExtractor, WeatherProcessor
        except ImportError as e:
            st.error(f"Failed to import dependencies: {e}")
            return None, None, None

    def _initialize_components(self):
        """Initialize spatial processing components."""
        if self._loader is None:
            SpatialDataLoader, SpatialFeatureExtractor, WeatherProcessor = (
                self._import_dependencies()
            )

            if SpatialDataLoader is not None:
                self._loader = SpatialDataLoader(self.bbox)
                self._extractor = SpatialFeatureExtractor(self._loader)
                self._weather_processor = WeatherProcessor

    def render_configuration_panel(self) -> Dict:
        """Render Streamlit configuration panel and return settings."""
        st.sidebar.header("🗺️ Grid Configuration")

        # Grid resolution
        resolution_options = {
            "Very Fast (100m)": 100,
            "Fast (50m)": 50,
            "Medium (25m)": 25,
            "Detailed (10m)": 10,
            "Very Detailed (5m)": 5,
        }

        resolution_choice = st.sidebar.selectbox(
            "Grid Resolution",
            options=list(resolution_options.keys()),
            index=1,  # Default to 50m
            help="Higher resolution = more detail but longer processing time",
        )

        resolution = resolution_options[resolution_choice]

        # Estimate grid size
        west, south, east, north = self.bbox
        lat_center = (south + north) / 2
        lon_to_m = 111320 * np.cos(np.radians(lat_center))
        lat_to_m = 111320

        width_m = (east - west) * lon_to_m
        height_m = (north - south) * lat_to_m

        n_cols = int(width_m / resolution)
        n_rows = int(height_m / resolution)
        total_points = n_cols * n_rows

        st.sidebar.metric("Grid Points", f"{total_points:,}")
        st.sidebar.metric("Grid Size", f"{n_cols} × {n_rows}")

        # Time estimation
        time_estimates = {
            100: "~2 min",
            50: "~8 min",
            25: "~30 min",
            10: "~2 hours",
            5: "~8 hours",
        }
        st.sidebar.info(
            f"⏱️ Estimated time: {time_estimates.get(resolution, '~unknown')}"
        )

        # Advanced options
        with st.sidebar.expander("🔧 Advanced Options"):
            extract_osm = st.checkbox(
                "Extract OSM Features",
                value=True,
                help="Uncheck for faster testing with cached data",
            )
            extract_weather = st.checkbox("Extract Weather Features", value=True)
            use_cache = st.checkbox(
                "Use Cached Results",
                value=True,
                help="Reuse previous computations when possible",
            )

        return {
            "resolution": resolution,
            "total_points": total_points,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "extract_osm": extract_osm,
            "extract_weather": extract_weather,
            "use_cache": use_cache,
        }

    @st.cache_data
    def create_grid(_self, resolution: int) -> Tuple[pd.DataFrame, int, int]:
        """Create spatial grid with caching."""
        west, south, east, north = _self.bbox

        # Convert to approximate meters
        lat_center = (south + north) / 2
        lon_to_m = 111320 * np.cos(np.radians(lat_center))
        lat_to_m = 111320

        width_m = (east - west) * lon_to_m
        height_m = (north - south) * lat_to_m

        n_cols = int(width_m / resolution)
        n_rows = int(height_m / resolution)

        # Create coordinates
        lon_step = (east - west) / n_cols
        lat_step = (north - south) / n_rows

        lons = np.linspace(west + lon_step / 2, east - lon_step / 2, n_cols)
        lats = np.linspace(south + lat_step / 2, north - lat_step / 2, n_rows)

        lon_grid, lat_grid = np.meshgrid(lons, lats)

        grid_df = pd.DataFrame(
            {
                "x": lon_grid.flatten(),
                "y": lat_grid.flatten(),
                "sub": 1,  # Required subject column
                "grid_col": np.tile(np.arange(n_cols), n_rows),
                "grid_row": np.repeat(np.arange(n_rows), n_cols),
            }
        )

        return grid_df, n_rows, n_cols

    def extract_features_with_progress(
        self, grid_df: pd.DataFrame, extract_osm: bool, extract_weather: bool
    ) -> pd.DataFrame:
        """Extract features with Streamlit progress tracking."""
        df = grid_df.copy()

        if extract_osm:
            if not self.config_toml.exists():
                st.warning(f"Feature specs file not found at {self.config_toml}")
                return df

            entries = toml.load(self.config_toml).get("features", [])

            if not entries:
                st.warning("No feature specifications found")
                return df

            self._initialize_components()

            st.info(f"🏘️ Extracting {len(entries)} OSM feature types...")

            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, feature in enumerate(entries):
                prefix = feature["prefix"]
                status_text.text(f"Processing: {prefix}")

                mode = feature.get("mode", "proximity")
                if hasattr(self._extractor, f"add_{mode}"):
                    fn = getattr(self._extractor, f"add_{mode}")
                    try:
                        df = fn(
                            df,
                            prefix,
                            feature["source"],
                            feature.get("radii", []),
                            feature.get("column"),
                            feature.get("values", []),
                        )
                    except Exception as e:
                        st.warning(f"Failed to compute {prefix}: {e}")
                        continue
                else:
                    st.warning(f"Unknown feature mode '{mode}' for feature '{prefix}'")

                # Update progress
                progress_bar.progress((i + 1) / len(entries))

            status_text.text("OSM feature extraction completed!")

            # Handle NaN values
            self._process_nan_values(df)

        if extract_weather:
            df = self._add_weather_features(df)

        return df

    def _process_nan_values(self, df: pd.DataFrame):
        """Process NaN values intelligently."""
        coord_cols = ["x", "y", "sub", "grid_col", "grid_row"]
        feature_cols = [col for col in df.columns if col not in coord_cols]

        nan_info = []

        for col in feature_cols:
            if df[col].isna().any():
                nan_count = df[col].isna().sum()

                if any(keyword in col.lower() for keyword in ["distance", "close2"]):
                    df[col] = df[col].fillna(200.0)
                    nan_info.append(f"{col}: {nan_count} NaN → 200m default")
                elif any(keyword in col.lower() for keyword in ["num_", "count"]):
                    df[col] = df[col].fillna(0)
                    nan_info.append(f"{col}: {nan_count} NaN → 0")
                elif any(
                    keyword in col.lower() for keyword in ["proportion_", "average_"]
                ):
                    median_val = df[col].median()
                    fill_val = median_val if not pd.isna(median_val) else 0.3
                    df[col] = df[col].fillna(fill_val)
                    nan_info.append(f"{col}: {nan_count} NaN → {fill_val:.2f}")
                else:
                    df[col] = df[col].fillna(0.0)
                    nan_info.append(f"{col}: {nan_count} NaN → 0")

        if nan_info:
            with st.expander("🔧 NaN Value Processing"):
                for info in nan_info[:10]:  # Show first 10
                    st.text(info)
                if len(nan_info) > 10:
                    st.text(f"... and {len(nan_info) - 10} more")

    def _add_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add weather features with Streamlit feedback."""
        weather_dirs = list(self.raw_data_dir.glob("RW_*"))

        if not weather_dirs:
            st.info("ℹ️ No weather data found, skipping weather features")
            return df

        try:
            with st.spinner("🌤️ Processing weather data..."):
                meta = self._weather_processor.parse_metadata(weather_dirs[0])
                raw_weather = self._weather_processor.read_raw(weather_dirs[0], meta)
                weather_data = self._weather_processor.interpolate(raw_weather)

                # Use middle time point
                mid_idx = len(weather_data) // 2
                weather_values = weather_data.iloc[mid_idx]

                for col, value in weather_values.items():
                    df[col] = value

                st.success(f"✅ Added weather data from {weather_data.index[mid_idx]}")

                # Show weather summary
                with st.expander("🌤️ Weather Data Summary"):
                    weather_df = pd.DataFrame(
                        {
                            "Feature": weather_values.index,
                            "Value": weather_values.values,
                        }
                    )
                    st.dataframe(weather_df)

                return df

        except Exception as e:
            st.error(f"❌ Weather processing failed: {e}")
            return df

    @st.cache_data
    def load_model(_self) -> Dict:
        """Load the ensemble model with caching."""
        model_path = _self.models_dir / "best_ensemble_model.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        return model_data

    def predict_with_progress(self, grid_df: pd.DataFrame) -> np.ndarray:
        """Make predictions with progress tracking."""
        st.info("🤖 Loading ensemble model...")
        model_data = self.load_model()
        ensemble_model = model_data["ensemble_model"]
        similarity_params = model_data.get("similarity_params", {})

        # Show model info
        with st.expander("🔍 Model Information"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Training Subjects", len(ensemble_model.get_training_subjects())
                )
                st.metric("Trial Number", model_data.get("trial_number", "N/A"))
            with col2:
                st.metric("CV MSE", f"{model_data.get('trial_value', 0):.2f}")
                st.text(
                    f"Similarity: {similarity_params.get('similarity_method', 'unknown')}"
                )

        # Test compatibility
        st.info("🔍 Testing model compatibility...")
        test_row = grid_df.iloc[:1].copy()
        working_subjects = []
        failed_subjects = []

        progress_bar = st.progress(0)
        subject_models = list(ensemble_model.subject_models.items())

        for i, (subject_id, subject_model) in enumerate(subject_models):
            try:
                _ = subject_model.predict(test_row)
                working_subjects.append(subject_id)
            except Exception:
                failed_subjects.append(subject_id)

            progress_bar.progress((i + 1) / len(subject_models))

        if not working_subjects:
            st.error("❌ No subject models are compatible with the grid data!")
            return None

        # Show compatibility results
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"✅ Compatible: {len(working_subjects)} subjects")
        with col2:
            if failed_subjects:
                st.warning(f"⚠️ Failed: {len(failed_subjects)} subjects")

        # Make predictions
        st.info("🔮 Making predictions...")
        subject_predictions = {}

        progress_bar = st.progress(0)
        for i, subject_id in enumerate(working_subjects):
            try:
                subject_model = ensemble_model.subject_models[subject_id]
                predictions = subject_model.predict(grid_df)
                subject_predictions[subject_id] = predictions
            except Exception as e:
                st.warning(f"Prediction failed for subject {subject_id}: {e}")

            progress_bar.progress((i + 1) / len(working_subjects))

        if not subject_predictions:
            st.error("❌ No predictions could be made!")
            return None

        # Ensemble aggregation
        if similarity_params.get("similarity_method") == "simple_average":
            st.info("Using simple average ensemble...")
            final_predictions = self._simple_average_ensemble(subject_predictions)
        else:
            st.info("Using similarity-weighted ensemble...")
            final_predictions = self._similarity_weighted_ensemble(
                grid_df, subject_predictions, ensemble_model, similarity_params
            )

        return final_predictions

    def _simple_average_ensemble(self, subject_predictions: Dict) -> np.ndarray:
        """Simple average ensemble."""
        n_samples = len(next(iter(subject_predictions.values())))
        final_predictions = []

        for i in range(n_samples):
            sample_preds = []
            for predictions in subject_predictions.values():
                pred_val = predictions[i]
                if not (np.isnan(pred_val) or np.isinf(pred_val)):
                    sample_preds.append(pred_val)

            if sample_preds:
                final_predictions.append(np.mean(sample_preds))
            else:
                final_predictions.append(np.nan)

        return np.array(final_predictions)

    def _similarity_weighted_ensemble(
        self, grid_df, subject_predictions, ensemble_model, similarity_params
    ):
        """Similarity-weighted ensemble."""
        try:
            # Import similarity calculator
            import sys

            sys.path.insert(0, str(self.project_root))
            from src.ensemble.env_similarity import UnifiedEnvironmentalSimilarity

            train_features_dict = ensemble_model.get_subject_training_data()

            similarity_calc = UnifiedEnvironmentalSimilarity(
                method=similarity_params.get("similarity_method", "combined"),
                normalize_features=similarity_params.get("normalize_features", True),
            )

            # Use mean grid conditions for similarity
            grid_mean_conditions = (
                grid_df.select_dtypes(include=[np.number]).mean().to_frame().T
            )
            grid_mean_conditions["x"] = grid_df["x"].mean()
            grid_mean_conditions["y"] = grid_df["y"].mean()
            grid_mean_conditions["sub"] = 1

            weights = similarity_calc.calculate_similarity(
                grid_mean_conditions, train_features_dict
            )

            # Show weights
            with st.expander("⚖️ Similarity Weights"):
                weights_df = pd.DataFrame(
                    list(weights.items()), columns=["Subject", "Weight"]
                )
                st.dataframe(weights_df)

            # Apply weights
            n_samples = len(grid_df)
            final_predictions = []

            for i in range(n_samples):
                sample_preds = []
                sample_weights = []

                for subject_id, predictions in subject_predictions.items():
                    pred_val = predictions[i]
                    if not (np.isnan(pred_val) or np.isinf(pred_val)):
                        sample_preds.append(pred_val)
                        sample_weights.append(weights.get(subject_id, 1.0))

                if sample_preds:
                    sample_weights = np.array(sample_weights)
                    sample_weights = sample_weights / np.sum(sample_weights)
                    weighted_pred = np.average(sample_preds, weights=sample_weights)
                    final_predictions.append(weighted_pred)
                else:
                    final_predictions.append(np.nan)

            return np.array(final_predictions)

        except Exception as e:
            st.warning(f"Similarity weighting failed: {e}, using simple average")
            return self._simple_average_ensemble(subject_predictions)

    def create_interactive_map(
        self, grid_df: pd.DataFrame, predictions: np.ndarray, n_rows: int, n_cols: int
    ):
        """Create interactive Plotly map."""

        # Prepare data
        valid_mask = ~np.isnan(predictions)
        plot_df = grid_df[valid_mask].copy()
        plot_df["predicted_co2"] = predictions[valid_mask]

        # Create Plotly figure
        fig = px.scatter_mapbox(
            plot_df,
            lat="y",
            lon="x",
            color="predicted_co2",
            size_max=15,
            zoom=13,
            mapbox_style="open-street-map",
            color_continuous_scale="RdYlBu_r",
            title=f"CO₂ Concentration Predictions ({len(plot_df):,} points)",
            labels={"predicted_co2": "CO₂ (ppm)"},
        )

        fig.update_layout(height=600, margin=dict(l=0, r=0, t=50, b=0))

        return fig

    def create_heatmap(self, predictions: np.ndarray, n_rows: int, n_cols: int):
        """Create heatmap visualization."""

        # Reshape to grid
        pred_grid = predictions.reshape(n_rows, n_cols)

        west, south, east, north = self.bbox

        fig = go.Figure(
            data=go.Heatmap(
                z=pred_grid,
                x=np.linspace(west, east, n_cols),
                y=np.linspace(south, north, n_rows),
                colorscale="RdYlBu_r",
                colorbar=dict(title="CO₂ (ppm)"),
            )
        )

        fig.update_layout(
            title="CO₂ Concentration Heatmap",
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            height=500,
        )

        return fig

    def display_results(
        self, grid_df: pd.DataFrame, predictions: np.ndarray, n_rows: int, n_cols: int
    ):
        """Display results with interactive visualizations."""

        # Statistics
        valid_predictions = predictions[~np.isnan(predictions)]

        st.header("📊 Results")

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Points", f"{len(predictions):,}")
        with col2:
            st.metric("Valid Predictions", f"{len(valid_predictions):,}")
        with col3:
            st.metric("Mean CO₂", f"{np.mean(valid_predictions):.1f} ppm")
        with col4:
            st.metric("Std CO₂", f"{np.std(valid_predictions):.1f} ppm")

        # Visualization tabs
        tab1, tab2, tab3 = st.tabs(["🗺️ Interactive Map", "🔥 Heatmap", "📈 Statistics"])

        with tab1:
            st.plotly_chart(
                self.create_interactive_map(grid_df, predictions, n_rows, n_cols),
                use_container_width=True,
            )

        with tab2:
            st.plotly_chart(
                self.create_heatmap(predictions, n_rows, n_cols),
                use_container_width=True,
            )

        with tab3:
            # Distribution plot
            fig_hist = px.histogram(
                x=valid_predictions,
                nbins=50,
                title="CO₂ Concentration Distribution",
                labels={"x": "CO₂ (ppm)", "y": "Count"},
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            # Summary statistics
            st.subheader("Summary Statistics")
            stats_df = pd.DataFrame(
                {
                    "Statistic": [
                        "Count",
                        "Mean",
                        "Std",
                        "Min",
                        "25%",
                        "50%",
                        "75%",
                        "Max",
                    ],
                    "Value": [
                        len(valid_predictions),
                        np.mean(valid_predictions),
                        np.std(valid_predictions),
                        np.min(valid_predictions),
                        np.percentile(valid_predictions, 25),
                        np.percentile(valid_predictions, 50),
                        np.percentile(valid_predictions, 75),
                        np.max(valid_predictions),
                    ],
                }
            )
            st.dataframe(stats_df)

    def save_results(
        self, grid_df: pd.DataFrame, predictions: np.ndarray, config: Dict
    ):
        """Save results with download options."""

        # Prepare output data
        output_df = grid_df.copy()
        output_df["predicted_co2"] = predictions

        # Save to output directory
        self.output_dir.mkdir(exist_ok=True)

        resolution = config["resolution"]
        csv_path = self.output_dir / f"streamlit_grid_predictions_{resolution}m.csv"
        output_df.to_csv(csv_path, index=False)

        st.success(f"✅ Results saved to {csv_path}")

        # Download button
        csv_data = output_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results CSV",
            data=csv_data,
            file_name=f"co2_predictions_{resolution}m.csv",
            mime="text/csv",
        )

    def run_streamlit_app(self):
        """Main Streamlit app interface."""

        st.set_page_config(
            page_title="CO₂ Grid Prediction",
            page_icon="🗺️",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        st.title("🗺️ CO₂ Concentration Grid Prediction")
        st.markdown(
            "Interactive spatial prediction of CO₂ concentrations using ensemble models"
        )

        # Configuration panel
        config = self.render_configuration_panel()

        # Main process button
        if st.button("🚀 Generate Predictions", type="primary"):

            try:
                # Step 1: Create grid
                with st.spinner("📐 Creating spatial grid..."):
                    grid_df, n_rows, n_cols = self.create_grid(config["resolution"])
                    st.success(f"✅ Created {config['total_points']:,} grid points")

                # Step 2: Extract features
                if config["extract_osm"] or config["extract_weather"]:
                    grid_with_features = self.extract_features_with_progress(
                        grid_df, config["extract_osm"], config["extract_weather"]
                    )
                else:
                    grid_with_features = grid_df

                # Step 3: Make predictions
                predictions = self.predict_with_progress(grid_with_features)

                if predictions is not None:
                    # Step 4: Display results
                    self.display_results(
                        grid_with_features, predictions, n_rows, n_cols
                    )

                    # Step 5: Save results
                    self.save_results(grid_with_features, predictions, config)

            except Exception as e:
                st.error(f"❌ Pipeline failed: {e}")
                st.exception(e)


# Example usage
if __name__ == "__main__":
    # Initialize the grid loader
    project_root = Path(__file__).parent  # Adjust to your project root
    grid_loader = StreamlitGridLoader(project_root)

    # Run the Streamlit app
    grid_loader.run_streamlit_app()
