#!/usr/bin/env python3
"""
Proper grid prediction pipeline that:
1. Extracts real OSM features for the grid
2. Handles the similarity calculation correctly for grid vs training data
3. Makes proper ensemble predictions
"""

import pickle
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toml
from tqdm.auto import tqdm

# Suppress warnings
warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.spatial_data_loader import SpatialDataLoader
from src.data_processing.spatial_feature_extractor import SpatialFeatureExtractor
from src.data_processing.weather_processor import WeatherProcessor

# Configuration
BBOX = (9.2257, 45.47162, 9.23768, 45.48537)
CONFIG_TOML = PROJECT_ROOT / "src" / "data_processing" / "config" / "feature_specs.toml"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw_data"
BEST_MODEL_PATH = PROJECT_ROOT / "models" / "best_ensemble_model.pkl"


class ProperGridPrediction:
    """Grid prediction with proper OSM feature extraction and ensemble prediction."""

    def __init__(self, bbox, grid_resolution_meters=50):
        self.bbox = bbox
        self.grid_resolution = grid_resolution_meters
        self.loader = SpatialDataLoader(bbox)
        self.extractor = SpatialFeatureExtractor(self.loader)

        print(f"Initialized grid prediction pipeline")
        print(f"Resolution: {grid_resolution_meters}m")

    def create_grid(self):
        """Create spatial grid."""
        west, south, east, north = self.bbox

        # Convert to approximate meters
        lat_center = (south + north) / 2
        lon_to_m = 111320 * np.cos(np.radians(lat_center))
        lat_to_m = 111320

        width_m = (east - west) * lon_to_m
        height_m = (north - south) * lat_to_m

        n_cols = int(width_m / self.grid_resolution)
        n_rows = int(height_m / self.grid_resolution)

        print(f"Grid: {n_cols} × {n_rows} = {n_cols * n_rows:,} points")
        print(f"Area: {width_m:.0f}m × {height_m:.0f}m")

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

    def extract_osm_features(self, grid_df):
        """Extract OSM features using the actual TOML configuration."""

        if not CONFIG_TOML.exists():
            print(f"Warning: Feature specs file not found at {CONFIG_TOML}")
            return grid_df

        entries = toml.load(CONFIG_TOML).get("features", [])

        if not entries:
            print("Warning: No feature specifications found")
            return grid_df

        df = grid_df.copy()

        print(f"Extracting {len(entries)} OSM feature types...")

        # Process each feature with progress bar
        with tqdm(entries, desc="OSM Features", unit="feat") as feat_bar:
            for feature in feat_bar:
                prefix = feature["prefix"]
                feat_bar.set_postfix_str(prefix, refresh=False)

                mode = feature.get("mode", "proximity")
                if hasattr(self.extractor, f"add_{mode}"):
                    fn = getattr(self.extractor, f"add_{mode}")
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
                        print(f"Warning: Failed to compute {prefix}: {e}")
                        continue
                else:
                    print(
                        f"Warning: Unknown feature mode '{mode}' for feature '{prefix}'"
                    )

        # Handle NaN values intelligently
        print("Processing NaN values...")

        # Get coordinate columns to preserve
        coord_cols = ["x", "y", "sub", "grid_col", "grid_row"]
        feature_cols = [col for col in df.columns if col not in coord_cols]

        if feature_cols:
            # Strategy for different types of features
            for col in feature_cols:
                if df[col].isna().any():
                    if any(
                        keyword in col.lower() for keyword in ["distance", "close2"]
                    ):
                        # Distance features: use a reasonable default
                        df[col] = df[col].fillna(200.0)  # 200m default distance
                    elif any(keyword in col.lower() for keyword in ["num_", "count"]):
                        # Count features: use 0
                        df[col] = df[col].fillna(0)
                    elif any(
                        keyword in col.lower()
                        for keyword in ["proportion_", "average_"]
                    ):
                        # Proportion/average features: use median or reasonable default
                        median_val = df[col].median()
                        if pd.isna(median_val):
                            df[col] = df[col].fillna(0.3)  # Default proportion
                        else:
                            df[col] = df[col].fillna(median_val)
                    elif any(keyword in col.lower() for keyword in ["sum_", "len"]):
                        # Sum/length features: use 0
                        df[col] = df[col].fillna(0.0)
                    else:
                        # Other features: use median or 0
                        median_val = df[col].median()
                        if pd.isna(median_val):
                            df[col] = df[col].fillna(0.0)
                        else:
                            df[col] = df[col].fillna(median_val)

        # Final check
        remaining_nans = df.isna().sum().sum()
        if remaining_nans > 0:
            print(f"Warning: {remaining_nans} NaN values remain, filling with 0")
            df = df.fillna(0)

        print(f"OSM features extracted: {len(feature_cols)} features")

        return df

    def add_weather_features(self, grid_df):
        """Add weather features."""
        weather_dirs = list(RAW_DATA_DIR.glob("RW_*"))

        if not weather_dirs:
            print("No weather data found, skipping...")
            return grid_df

        try:
            print("Processing weather data...")
            meta = WeatherProcessor.parse_metadata(weather_dirs[0])
            raw_weather = WeatherProcessor.read_raw(weather_dirs[0], meta)
            weather_data = WeatherProcessor.interpolate(raw_weather)

            # Use middle time point
            mid_idx = len(weather_data) // 2
            weather_values = weather_data.iloc[mid_idx]

            df = grid_df.copy()
            for col, value in weather_values.items():
                df[col] = value

            print(f"Added weather data from {weather_data.index[mid_idx]}")
            return df

        except Exception as e:
            print(f"Weather processing failed: {e}")
            return grid_df

    def predict_with_ensemble(self, grid_df):
        """Make predictions using the ensemble model with proper similarity handling."""

        print("Loading ensemble model...")
        with open(BEST_MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)

        ensemble_model = model_data["ensemble_model"]
        similarity_params = model_data.get("similarity_params", {})

        print(f"Model info:")
        print(f"  Training subjects: {len(ensemble_model.get_training_subjects())}")
        print(
            f"  Similarity method: {similarity_params.get('similarity_method', 'unknown')}"
        )

        # Test model compatibility
        test_row = grid_df.iloc[:1].copy()
        working_subjects = []

        print("Testing model compatibility...")
        for subject_id, subject_model in ensemble_model.subject_models.items():
            try:
                _ = subject_model.predict(test_row)
                working_subjects.append(subject_id)
            except Exception as e:
                print(f"Subject {subject_id} incompatible: {str(e)[:50]}...")

        if not working_subjects:
            raise RuntimeError("No subject models are compatible with the grid data!")

        print(
            f"Compatible subjects: {len(working_subjects)}/{len(ensemble_model.subject_models)}"
        )

        # Make predictions with working subjects
        print("Making individual subject predictions...")
        subject_predictions = {}

        for subject_id in tqdm(working_subjects, desc="Subject predictions"):
            try:
                subject_model = ensemble_model.subject_models[subject_id]
                predictions = subject_model.predict(grid_df)
                subject_predictions[subject_id] = predictions
            except Exception as e:
                print(f"Prediction failed for subject {subject_id}: {e}")

        if not subject_predictions:
            raise RuntimeError("No predictions could be made!")

        # Handle ensemble aggregation based on similarity method
        if similarity_params.get("similarity_method") == "simple_average":
            print("Using simple average ensemble...")
            # Simple average of all working subjects
            final_predictions = self._simple_average_ensemble(subject_predictions)
        else:
            print("Using similarity-weighted ensemble...")
            # Use similarity weighting, but handle the grid properly
            final_predictions = self._similarity_weighted_ensemble(
                grid_df, subject_predictions, ensemble_model, similarity_params
            )

        return final_predictions

    def _simple_average_ensemble(self, subject_predictions):
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
        """Similarity-weighted ensemble that handles grid data properly."""

        # Get training features for similarity calculation
        train_features_dict = ensemble_model.get_subject_training_data()

        # Import similarity calculator
        from src.ensemble.env_similarity import UnifiedEnvironmentalSimilarity

        similarity_calc = UnifiedEnvironmentalSimilarity(
            method=similarity_params.get("similarity_method", "combined"),
            normalize_features=similarity_params.get("normalize_features", True),
        )

        # The similarity calculation expects to compare against training data
        # For a grid, we need to calculate similarity for the representative conditions

        # Option 1: Use mean conditions of the grid
        grid_mean_conditions = (
            grid_df.select_dtypes(include=[np.number]).mean().to_frame().T
        )
        grid_mean_conditions["x"] = grid_df["x"].mean()
        grid_mean_conditions["y"] = grid_df["y"].mean()
        grid_mean_conditions["sub"] = 1

        try:
            # Calculate weights based on mean grid conditions
            weights = similarity_calc.calculate_similarity(
                grid_mean_conditions, train_features_dict
            )

            # Apply these weights to all grid points
            print(f"Using similarity weights: {weights}")

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
            print(f"Similarity weighting failed: {e}, falling back to simple average")
            return self._simple_average_ensemble(subject_predictions)

    def save_grid_data(self, grid_df, predictions, n_rows, n_cols):
        """Save grid data in multiple formats optimized for fast Streamlit loading."""

        output_dir = PROJECT_ROOT / "output" / "grid_cache"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create output dataframe
        output_df = grid_df.copy()
        output_df["predicted_co2"] = predictions

        # Add some useful computed columns for Streamlit
        output_df["valid_prediction"] = ~np.isnan(predictions)

        # Save in multiple formats for different use cases

        # 1. Parquet - Fast loading, compressed, maintains dtypes
        parquet_path = output_dir / f"grid_predictions_{self.grid_resolution}m.parquet"
        output_df.to_parquet(parquet_path, index=False)
        print(f"✅ Parquet saved: {parquet_path}")

        # 2. CSV - Human readable, universal compatibility
        csv_path = output_dir / f"grid_predictions_{self.grid_resolution}m.csv"
        output_df.to_csv(csv_path, index=False)
        print(f"✅ CSV saved: {csv_path}")

        # 3. Numpy arrays - Ultra fast loading for just coordinates and predictions
        coords_and_preds = {
            "x": output_df["x"].values,
            "y": output_df["y"].values,
            "predicted_co2": predictions,
            "grid_rows": n_rows,
            "grid_cols": n_cols,
            "resolution_meters": self.grid_resolution,
            "bbox": self.bbox,
            "valid_mask": ~np.isnan(predictions),
        }

        numpy_path = output_dir / f"grid_fast_{self.grid_resolution}m.npz"
        np.savez_compressed(numpy_path, **coords_and_preds)
        print(f"✅ Numpy cache saved: {numpy_path}")

        # 4. Reshaped grid for direct plotting - fastest for heatmaps
        pred_grid = predictions.reshape(n_rows, n_cols)

        heatmap_data = {
            "prediction_grid": pred_grid,
            "extent": [
                self.bbox[0],
                self.bbox[2],
                self.bbox[1],
                self.bbox[3],
            ],  # [west, east, south, north]
            "resolution_meters": self.grid_resolution,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "stats": {
                "min": float(np.nanmin(predictions)),
                "max": float(np.nanmax(predictions)),
                "mean": float(np.nanmean(predictions)),
                "std": float(np.nanstd(predictions)),
                "valid_count": int(np.sum(~np.isnan(predictions))),
                "total_count": len(predictions),
            },
        }

        heatmap_path = output_dir / f"heatmap_data_{self.grid_resolution}m.npz"
        np.savez_compressed(heatmap_path, **heatmap_data)
        print(f"✅ Heatmap cache saved: {heatmap_path}")

        # 5. Metadata file for Streamlit app
        import json
        from datetime import datetime

        metadata = {
            "creation_time": datetime.now().isoformat(),
            "resolution_meters": self.grid_resolution,
            "grid_dimensions": {"rows": n_rows, "cols": n_cols},
            "total_points": len(predictions),
            "valid_predictions": int(np.sum(~np.isnan(predictions))),
            "bbox": {
                "west": self.bbox[0],
                "south": self.bbox[1],
                "east": self.bbox[2],
                "north": self.bbox[3],
            },
            "prediction_stats": {
                "min": float(np.nanmin(predictions)),
                "max": float(np.nanmax(predictions)),
                "mean": float(np.nanmean(predictions)),
                "std": float(np.nanstd(predictions)),
                "percentiles": {
                    "25th": float(np.nanpercentile(predictions, 25)),
                    "50th": float(np.nanpercentile(predictions, 50)),
                    "75th": float(np.nanpercentile(predictions, 75)),
                    "95th": float(np.nanpercentile(predictions, 95)),
                },
            },
            "files": {
                "parquet": f"grid_predictions_{self.grid_resolution}m.parquet",
                "csv": f"grid_predictions_{self.grid_resolution}m.csv",
                "numpy_cache": f"grid_fast_{self.grid_resolution}m.npz",
                "heatmap_cache": f"heatmap_data_{self.grid_resolution}m.npz",
            },
            "features_included": list(grid_df.columns),
        }

        metadata_path = output_dir / f"metadata_{self.grid_resolution}m.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Metadata saved: {metadata_path}")

        # Print file sizes for reference
        print(f"\n📁 File sizes:")
        for path in [parquet_path, csv_path, numpy_path, heatmap_path, metadata_path]:
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"  {path.name}: {size_mb:.2f} MB")

        print(f"\n📊 Cache Summary:")
        print(f"  Directory: {output_dir}")
        print(f"  Resolution: {self.grid_resolution}m")
        print(f"  Grid size: {n_cols} × {n_rows} = {len(predictions):,} points")
        print(f"  Valid predictions: {np.sum(~np.isnan(predictions)):,}")

        return output_dir
        """Create heatmap visualization."""

        # Handle NaN predictions
        valid_predictions = predictions[~np.isnan(predictions)]
        if len(valid_predictions) == 0:
            print("No valid predictions to plot!")
            return

        # Reshape to grid
        pred_grid = predictions.reshape(n_rows, n_cols)

        # Create plot
        plt.figure(figsize=(14, 10))

        west, south, east, north = self.bbox

        # Create heatmap
        im = plt.imshow(
            pred_grid,
            extent=[west, east, south, north],
            origin="lower",
            cmap="RdYlBu_r",
            interpolation="bilinear",
            aspect="auto",
        )

        # Colorbar
        cbar = plt.colorbar(im, shrink=0.8, pad=0.02)
        cbar.set_label(
            "Predicted CO₂ Concentration (ppm)", rotation=270, labelpad=20, fontsize=12
        )

        # Labels and title
        plt.xlabel("Longitude", fontsize=12)
        plt.ylabel("Latitude", fontsize=12)
        plt.title(
            f"CO₂ Concentration Prediction Map\n({self.grid_resolution}m Resolution, {len(valid_predictions):,} Valid Predictions)",
            fontsize=14,
            pad=20,
        )

        # Statistics
        stats_text = f"""Grid: {n_cols}×{n_rows} ({self.grid_resolution}m)
Valid: {len(valid_predictions):,}/{len(predictions):,}
Range: {np.nanmin(predictions):.1f} - {np.nanmax(predictions):.1f}
Mean: {np.nanmean(predictions):.1f} ± {np.nanstd(predictions):.1f}"""

        plt.text(
            0.02,
            0.98,
            stats_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.grid(True, alpha=0.3, linewidth=0.5)
        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Heatmap saved to {save_path}")

        plt.show()

    def run_pipeline(self):
        """Run the complete pipeline."""

        print("🗺️  Proper Grid Prediction Pipeline")
        print("=" * 60)

        # Step 1: Create grid
        print("\n1️⃣ Creating grid...")
        grid_df, n_rows, n_cols = self.create_grid()

        # Step 2: Extract OSM features (this will take time)
        print("\n2️⃣ Extracting OSM features...")
        print("⚠️  This step may take 10-20 minutes for large grids...")
        grid_with_osm = self.extract_osm_features(grid_df)

        # Step 3: Add weather features
        print("\n3️⃣ Adding weather features...")
        grid_with_features = self.add_weather_features(grid_with_osm)

        print(f"\nFinal dataset: {grid_with_features.shape}")
        print(f"Features: {len(grid_with_features.columns)}")

        # Step 4: Make predictions
        print("\n4️⃣ Making ensemble predictions...")
        predictions = self.predict_with_ensemble(grid_with_features)

        # Step 5: Create visualization
        print("\n5️⃣ Creating visualization...")
        output_dir = PROJECT_ROOT / "output"
        save_path = output_dir / f"proper_co2_heatmap_{self.grid_resolution}m.png"

        self.create_heatmap(grid_with_features, predictions, n_rows, n_cols, save_path)

        # Step 6: Save data in multiple formats for fast Streamlit loading
        print("\n6️⃣ Saving results...")
        self.save_grid_data(grid_with_features, predictions, n_rows, n_cols)

        print("\n🎉 Pipeline completed successfully!")

        # Print summary
        valid_preds = np.sum(~np.isnan(predictions))
        print(f"\n📊 Results Summary:")
        print(f"  Grid points: {len(predictions):,}")
        print(
            f"  Valid predictions: {valid_preds:,} ({100*valid_preds/len(predictions):.1f}%)"
        )
        print(
            f"  Prediction range: {np.nanmin(predictions):.1f} - {np.nanmax(predictions):.1f}"
        )
        print(
            f"  Mean ± std: {np.nanmean(predictions):.1f} ± {np.nanstd(predictions):.1f}"
        )


def main():
    """Main function."""

    # Configuration options
    resolutions = {
        1: 100,  # Fast test
        2: 50,  # Medium detail
        3: 25,  # High detail
        4: 10,  # Very high detail (will take a long time)
    }

    print("Grid Resolution Options:")
    for key, res in resolutions.items():
        est_points = (1500 // res) * (950 // res)
        time_est = "~5 min" if res >= 50 else "~15 min" if res >= 25 else "~30+ min"
        print(f"  {key}. {res}m resolution (~{est_points:,} points, {time_est})")

    try:
        choice = int(input("\nSelect resolution (1-4): "))
        resolution = resolutions.get(choice, 50)
    except (ValueError, KeyboardInterrupt):
        print("Using default 50m resolution")
        resolution = 50

    print(f"\nStarting pipeline with {resolution}m resolution...")
    print("Note: OSM feature extraction will take time proportional to grid size")

    # Run pipeline
    pipeline = ProperGridPrediction(BBOX, grid_resolution_meters=resolution)
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
