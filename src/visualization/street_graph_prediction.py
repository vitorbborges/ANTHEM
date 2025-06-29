#!/usr/bin/env python3
"""
Street network CO₂ prediction pipeline that:
1. Extracts street network nodes and edges from OSM
2. Creates prediction points at 5-meter intervals along edges
3. Extracts OSM features for all prediction points
4. Makes ensemble predictions for both nodes and edge points
"""

import pickle
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toml
from shapely.geometry import LineString, Point
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


class StreetNetworkPrediction:
    """CO₂ predictions for street network nodes and edges at 5m intervals."""

    def __init__(self, bbox, edge_interval_meters=5):
        self.bbox = bbox
        self.edge_interval = edge_interval_meters
        self.loader = SpatialDataLoader(bbox)
        self.extractor = SpatialFeatureExtractor(self.loader)

        print(f"Initialized street network prediction pipeline")
        print(f"Edge sampling interval: {edge_interval_meters}m")

    def extract_street_network(self):
        """Extract street network nodes and edges."""
        print("Loading street network...")

        # Get nodes and edges from the spatial loader
        nodes_gdf = self.loader.get_source("nodes")
        edges_gdf = self.loader.get_source("imputed_edges")

        print(f"Loaded {len(nodes_gdf):,} nodes and {len(edges_gdf):,} edges")

        return nodes_gdf, edges_gdf

    def create_edge_points(self, edges_gdf):
        """Create points at regular intervals along each edge."""
        print(f"Creating points every {self.edge_interval}m along edges...")

        edge_points = []
        edge_metadata = []

        for idx, edge in tqdm(
            edges_gdf.iterrows(), total=len(edges_gdf), desc="Processing edges"
        ):
            line = edge.geometry
            line_length = line.length

            # Convert length from degrees to approximate meters
            # This is a rough approximation - for precise work, use proper projections
            lat_center = (self.bbox[1] + self.bbox[3]) / 2
            meters_per_degree = 111320 * np.cos(np.radians(lat_center))
            length_meters = line_length * meters_per_degree

            if length_meters < self.edge_interval:
                # For very short edges, just use the midpoint
                mid_point = line.interpolate(0.5, normalized=True)
                edge_points.append(mid_point)
                edge_metadata.append(
                    {
                        "edge_id": (
                            f"{idx[0]}_{idx[1]}_{idx[2]}"
                            if isinstance(idx, tuple)
                            else str(idx)
                        ),
                        "point_id": 0,
                        "distance_along_edge": length_meters / 2,
                        "edge_length": length_meters,
                        "highway": edge.get("highway", "unknown"),
                        "maxspeed": edge.get("maxspeed", np.nan),
                        "lanes": edge.get("lanes", np.nan),
                        "oneway": edge.get("oneway", 0),
                    }
                )
            else:
                # Create points at regular intervals
                num_points = int(length_meters / self.edge_interval)
                for i in range(num_points):
                    # Calculate normalized distance along line (0 to 1)
                    distance_along = (i + 0.5) * self.edge_interval / length_meters
                    if distance_along <= 1.0:
                        point = line.interpolate(distance_along, normalized=True)
                        edge_points.append(point)
                        edge_metadata.append(
                            {
                                "edge_id": (
                                    f"{idx[0]}_{idx[1]}_{idx[2]}"
                                    if isinstance(idx, tuple)
                                    else str(idx)
                                ),
                                "point_id": i,
                                "distance_along_edge": (i + 0.5) * self.edge_interval,
                                "edge_length": length_meters,
                                "highway": edge.get("highway", "unknown"),
                                "maxspeed": edge.get("maxspeed", np.nan),
                                "lanes": edge.get("lanes", np.nan),
                                "oneway": edge.get("oneway", 0),
                            }
                        )

        # Create GeoDataFrame of edge points
        edge_points_gdf = gpd.GeoDataFrame(
            edge_metadata, geometry=edge_points, crs=edges_gdf.crs
        )

        print(f"Created {len(edge_points_gdf):,} points along edges")
        return edge_points_gdf

    def prepare_prediction_dataframe(self, nodes_gdf, edge_points_gdf):
        """Combine nodes and edge points into a single prediction dataframe."""
        print("Preparing unified prediction dataset...")

        # Prepare nodes data
        nodes_df = pd.DataFrame(
            {
                "x": nodes_gdf.geometry.x,
                "y": nodes_gdf.geometry.y,
                "sub": 1,  # Required subject column
                "point_type": "node",
                "point_id": nodes_gdf.index.astype(str),
                "highway": nodes_gdf.get("highway", "intersection"),
                "maxspeed": np.nan,
                "lanes": np.nan,
                "oneway": 0,
            }
        )

        # Prepare edge points data
        edge_points_df = pd.DataFrame(
            {
                "x": edge_points_gdf.geometry.x,
                "y": edge_points_gdf.geometry.y,
                "sub": 1,  # Required subject column
                "point_type": "edge",
                "point_id": edge_points_gdf["edge_id"]
                + "_"
                + edge_points_gdf["point_id"].astype(str),
                "highway": edge_points_gdf["highway"],
                "maxspeed": edge_points_gdf["maxspeed"],
                "lanes": edge_points_gdf["lanes"],
                "oneway": edge_points_gdf["oneway"],
            }
        )

        # Combine datasets
        combined_df = pd.concat([nodes_df, edge_points_df], ignore_index=True)

        print(
            f"Combined dataset: {len(nodes_df):,} nodes + {len(edge_points_df):,} edge points = {len(combined_df):,} total points"
        )

        return combined_df

    def extract_osm_features(self, prediction_df):
        """Extract OSM features for all prediction points."""

        if not CONFIG_TOML.exists():
            print(f"Warning: Feature specs file not found at {CONFIG_TOML}")
            return prediction_df

        entries = toml.load(CONFIG_TOML).get("features", [])

        if not entries:
            print("Warning: No feature specifications found")
            return prediction_df

        df = prediction_df.copy()

        print(f"Extracting {len(entries)} OSM feature types for {len(df):,} points...")

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

        # Get coordinate and metadata columns to preserve
        preserve_cols = [
            "x",
            "y",
            "sub",
            "point_type",
            "point_id",
            "highway",
            "maxspeed",
            "lanes",
            "oneway",
        ]
        feature_cols = [col for col in df.columns if col not in preserve_cols]

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

    def add_weather_features(self, prediction_df):
        """Add weather features."""
        weather_dirs = list(RAW_DATA_DIR.glob("RW_*"))

        if not weather_dirs:
            print("No weather data found, skipping...")
            return prediction_df

        try:
            print("Processing weather data...")
            meta = WeatherProcessor.parse_metadata(weather_dirs[0])
            raw_weather = WeatherProcessor.read_raw(weather_dirs[0], meta)
            weather_data = WeatherProcessor.interpolate(raw_weather)

            # Use middle time point
            mid_idx = len(weather_data) // 2
            weather_values = weather_data.iloc[mid_idx]

            df = prediction_df.copy()
            for col, value in weather_values.items():
                df[col] = value

            print(f"Added weather data from {weather_data.index[mid_idx]}")
            return df

        except Exception as e:
            print(f"Weather processing failed: {e}")
            return prediction_df

    def predict_with_ensemble(self, prediction_df):
        """Make predictions using the ensemble model."""

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
        test_row = prediction_df.iloc[:1].copy()
        working_subjects = []

        print("Testing model compatibility...")
        for subject_id, subject_model in ensemble_model.subject_models.items():
            try:
                _ = subject_model.predict(test_row)
                working_subjects.append(subject_id)
            except Exception as e:
                print(f"Subject {subject_id} incompatible: {str(e)[:50]}...")

        if not working_subjects:
            raise RuntimeError(
                "No subject models are compatible with the street network data!"
            )

        print(
            f"Compatible subjects: {len(working_subjects)}/{len(ensemble_model.subject_models)}"
        )

        # Make predictions with working subjects
        print("Making individual subject predictions...")
        subject_predictions = {}

        for subject_id in tqdm(working_subjects, desc="Subject predictions"):
            try:
                subject_model = ensemble_model.subject_models[subject_id]
                predictions = subject_model.predict(prediction_df)
                subject_predictions[subject_id] = predictions
            except Exception as e:
                print(f"Prediction failed for subject {subject_id}: {e}")

        if not subject_predictions:
            raise RuntimeError("No predictions could be made!")

        # Use simple average ensemble for street network
        print("Using simple average ensemble...")
        final_predictions = self._simple_average_ensemble(subject_predictions)

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

    def create_street_visualization(self, prediction_df, predictions, save_path=None):
        """Create comprehensive visualization of predictions on street network."""

        # Handle NaN predictions
        valid_mask = ~np.isnan(predictions)
        valid_predictions = predictions[valid_mask]

        if len(valid_predictions) == 0:
            print("No valid predictions to plot!")
            return

        # Separate nodes and edge points
        nodes_mask = prediction_df["point_type"] == "node"
        edges_mask = prediction_df["point_type"] == "edge"

        node_coords = prediction_df[nodes_mask & valid_mask][["x", "y"]].values
        edge_coords = prediction_df[edges_mask & valid_mask][["x", "y"]].values

        node_predictions = predictions[nodes_mask & valid_mask]
        edge_predictions = predictions[edges_mask & valid_mask]

        # Create comprehensive plot with 3 subplots
        fig = plt.figure(figsize=(24, 8))

        west, south, east, north = self.bbox

        # Plot 1: Street Network Scatter Plot
        ax1 = plt.subplot(1, 3, 1)

        if len(edge_predictions) > 0:
            scatter1 = ax1.scatter(
                edge_coords[:, 0],
                edge_coords[:, 1],
                c=edge_predictions,
                cmap="RdYlBu_r",
                s=3,
                alpha=0.6,
                label=f"Edge Points ({len(edge_predictions):,})",
                marker=".",
                vmin=np.nanmin(predictions),
                vmax=np.nanmax(predictions),
            )

        if len(node_predictions) > 0:
            scatter1 = ax1.scatter(
                node_coords[:, 0],
                node_coords[:, 1],
                c=node_predictions,
                cmap="RdYlBu_r",
                s=25,
                alpha=0.9,
                label=f"Nodes ({len(node_predictions):,})",
                marker="o",
                edgecolors="black",
                linewidth=0.3,
                vmin=np.nanmin(predictions),
                vmax=np.nanmax(predictions),
            )

        ax1.set_xlim(west, east)
        ax1.set_ylim(south, north)
        ax1.set_xlabel("Longitude", fontsize=10)
        ax1.set_ylabel("Latitude", fontsize=10)
        ax1.set_title(
            f"Street Network CO₂ Predictions\n({len(valid_predictions):,} points)",
            fontsize=11,
        )
        ax1.grid(True, alpha=0.3, linewidth=0.5)
        ax1.legend(fontsize=9)

        # Add colorbar
        cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.8, pad=0.02)
        cbar1.set_label(
            "CO₂ Concentration (ppm)", rotation=270, labelpad=15, fontsize=9
        )

        # Plot 2: Interpolated Heatmap
        ax2 = plt.subplot(1, 3, 2)

        # Create interpolated heatmap
        from scipy.interpolate import griddata

        # Get all valid coordinates and predictions
        valid_coords = prediction_df[valid_mask][["x", "y"]].values

        # Create grid for interpolation
        grid_resolution = 100  # Number of points in each dimension
        xi = np.linspace(west, east, grid_resolution)
        yi = np.linspace(south, north, grid_resolution)
        xi_grid, yi_grid = np.meshgrid(xi, yi)

        # Interpolate predictions onto regular grid
        try:
            zi = griddata(
                valid_coords,
                valid_predictions,
                (xi_grid, yi_grid),
                method="cubic",
                fill_value=np.nan,
            )

            # Create heatmap
            im = ax2.imshow(
                zi,
                extent=[west, east, south, north],
                origin="lower",
                cmap="RdYlBu_r",
                alpha=0.8,
                aspect="auto",
                vmin=np.nanmin(predictions),
                vmax=np.nanmax(predictions),
            )

            # Overlay actual points
            scatter2 = ax2.scatter(
                valid_coords[:, 0],
                valid_coords[:, 1],
                c=valid_predictions,
                cmap="RdYlBu_r",
                s=1,
                alpha=0.4,
                edgecolors="none",
                vmin=np.nanmin(predictions),
                vmax=np.nanmax(predictions),
            )

            ax2.set_xlim(west, east)
            ax2.set_ylim(south, north)
            ax2.set_xlabel("Longitude", fontsize=10)
            ax2.set_ylabel("Latitude", fontsize=10)
            ax2.set_title(
                "Interpolated CO₂ Heatmap\nwith Street Network Overlay", fontsize=11
            )
            ax2.grid(True, alpha=0.3, linewidth=0.5)

            # Add colorbar
            cbar2 = plt.colorbar(im, ax=ax2, shrink=0.8, pad=0.02)
            cbar2.set_label(
                "CO₂ Concentration (ppm)", rotation=270, labelpad=15, fontsize=9
            )

        except Exception as e:
            print(f"Warning: Could not create interpolated heatmap: {e}")
            ax2.text(
                0.5,
                0.5,
                "Interpolation\nFailed",
                transform=ax2.transAxes,
                ha="center",
                va="center",
                fontsize=14,
            )
            ax2.set_xlim(west, east)
            ax2.set_ylim(south, north)

        # Plot 3: Highway Type Analysis
        ax3 = plt.subplot(1, 3, 3)

        # Analyze predictions by highway type
        highway_stats = {}
        for highway_type in prediction_df["highway"].unique():
            if pd.isna(highway_type):
                continue
            mask = (prediction_df["highway"] == highway_type) & valid_mask
            if mask.sum() > 0:
                highway_preds = predictions[mask]
                highway_stats[highway_type] = {
                    "mean": np.nanmean(highway_preds),
                    "std": np.nanstd(highway_preds),
                    "count": len(highway_preds),
                }

        if highway_stats:
            # Sort by mean concentration
            sorted_highways = sorted(
                highway_stats.items(), key=lambda x: x[1]["mean"], reverse=True
            )

            highway_names = [item[0] for item in sorted_highways]
            highway_means = [item[1]["mean"] for item in sorted_highways]
            highway_stds = [item[1]["std"] for item in sorted_highways]
            highway_counts = [item[1]["count"] for item in sorted_highways]

            # Create color map based on concentration levels
            colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(highway_names)))

            bars = ax3.barh(
                range(len(highway_names)),
                highway_means,
                xerr=highway_stds,
                capsize=3,
                color=colors,
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
            )

            ax3.set_yticks(range(len(highway_names)))
            ax3.set_yticklabels(
                [
                    f"{name}\n(n={count})"
                    for name, count in zip(highway_names, highway_counts)
                ],
                fontsize=8,
            )
            ax3.set_xlabel("CO₂ Concentration (ppm)", fontsize=10)
            ax3.set_title("CO₂ by Highway Type\n(Mean ± Std)", fontsize=11)
            ax3.grid(True, alpha=0.3, axis="x")

            # Add value labels on bars
            for i, (mean, std) in enumerate(zip(highway_means, highway_stds)):
                ax3.text(
                    mean + std + 1,
                    i,
                    f"{mean:.1f}",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                )
        else:
            ax3.text(
                0.5,
                0.5,
                "No Highway\nData Available",
                transform=ax3.transAxes,
                ha="center",
                va="center",
                fontsize=14,
            )

        # Overall statistics text
        stats_text = f"""Street Network CO₂ Predictions Summary

Sampling: {self.edge_interval}m intervals on edges
Total Points: {len(predictions):,}
Valid Predictions: {len(valid_predictions):,} ({100*len(valid_predictions)/len(predictions):.1f}%)

Breakdown:
• Street Nodes: {len(node_predictions):,}
• Edge Points: {len(edge_predictions):,}

Statistics:
• Range: {np.nanmin(predictions):.1f} - {np.nanmax(predictions):.1f} ppm
• Mean ± Std: {np.nanmean(predictions):.1f} ± {np.nanstd(predictions):.1f} ppm
• Median: {np.nanmedian(predictions):.1f} ppm"""

        fig.text(
            0.02,
            0.98,
            stats_text,
            transform=fig.transFigure,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="gray"
            ),
        )

        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"Comprehensive visualization saved to {save_path}")

        plt.show()

    def create_simple_heatmap(self, prediction_df, predictions, save_path=None):
        """Create a simple heatmap focusing on the spatial distribution."""

        valid_mask = ~np.isnan(predictions)
        valid_predictions = predictions[valid_mask]

        if len(valid_predictions) == 0:
            print("No valid predictions for heatmap!")
            return

        plt.figure(figsize=(12, 10))

        west, south, east, north = self.bbox
        valid_coords = prediction_df[valid_mask][["x", "y"]].values

        # Create high-resolution scatter plot that looks like a heatmap
        scatter = plt.scatter(
            valid_coords[:, 0],
            valid_coords[:, 1],
            c=valid_predictions,
            cmap="RdYlBu_r",
            s=8,
            alpha=0.8,
            edgecolors="none",
        )

        plt.xlim(west, east)
        plt.ylim(south, north)
        plt.xlabel("Longitude", fontsize=12)
        plt.ylabel("Latitude", fontsize=12)
        plt.title(
            f"Street Network CO₂ Concentration Heatmap\n{len(valid_predictions):,} Points at {self.edge_interval}m Resolution",
            fontsize=14,
            pad=20,
        )

        # Enhanced colorbar
        cbar = plt.colorbar(scatter, shrink=0.8, pad=0.02)
        cbar.set_label(
            "CO₂ Concentration (ppm)", rotation=270, labelpad=20, fontsize=12
        )

        # Add grid and styling
        plt.grid(True, alpha=0.3, linewidth=0.5)

        # Statistics box
        stats_text = f"""Statistics:
Range: {np.nanmin(predictions):.1f} - {np.nanmax(predictions):.1f} ppm
Mean: {np.nanmean(predictions):.1f} ± {np.nanstd(predictions):.1f} ppm
Points: {len(valid_predictions):,}"""

        plt.text(
            0.02,
            0.98,
            stats_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
        )

        plt.tight_layout()

        if save_path:
            heatmap_path = save_path.parent / save_path.name.replace(
                ".png", "_heatmap.png"
            )
            plt.savefig(heatmap_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"Heatmap saved to {heatmap_path}")

        plt.show()

    def run_pipeline(self):
        """Run the complete street network prediction pipeline."""

        print("🛣️  Street Network CO₂ Prediction Pipeline")
        print("=" * 60)

        # Step 1: Extract street network
        print("\n1️⃣ Extracting street network...")
        nodes_gdf, edges_gdf = self.extract_street_network()

        # Step 2: Create edge points
        print("\n2️⃣ Creating edge sampling points...")
        edge_points_gdf = self.create_edge_points(edges_gdf)

        # Step 3: Prepare prediction dataframe
        print("\n3️⃣ Preparing prediction dataset...")
        prediction_df = self.prepare_prediction_dataframe(nodes_gdf, edge_points_gdf)

        # Step 4: Extract OSM features
        print("\n4️⃣ Extracting OSM features...")
        print("⚠️  This step may take 10-30 minutes for large street networks...")
        prediction_with_osm = self.extract_osm_features(prediction_df)

        # Step 5: Add weather features
        print("\n5️⃣ Adding weather features...")
        prediction_with_features = self.add_weather_features(prediction_with_osm)

        print(f"\nFinal dataset: {prediction_with_features.shape}")
        print(f"Features: {len(prediction_with_features.columns)}")

        # Step 6: Make predictions
        print("\n6️⃣ Making ensemble predictions...")
        predictions = self.predict_with_ensemble(prediction_with_features)

        # Step 7: Create visualizations
        print("\n7️⃣ Creating visualizations...")
        output_dir = PROJECT_ROOT / "output"
        save_path = (
            output_dir
            / "plots"
            / f"street_network_co2_predictions_{self.edge_interval}m.png"
        )

        # Create comprehensive visualization
        self.create_street_visualization(
            prediction_with_features, predictions, save_path
        )

        # Create simple heatmap
        print("Creating additional heatmap visualization...")
        self.create_simple_heatmap(prediction_with_features, predictions, save_path)

        # Step 8: Save results
        print("\n8️⃣ Saving results...")
        output_df = prediction_with_features.copy()
        output_df["predicted_co2"] = predictions

        # Save separate files for nodes and edge points
        nodes_df = output_df[output_df["point_type"] == "node"].copy()
        edges_df = output_df[output_df["point_type"] == "edge"].copy()

        csv_dir = output_dir / "street_predictions"
        csv_dir.mkdir(parents=True, exist_ok=True)

        nodes_csv = csv_dir / f"street_nodes_predictions_{self.edge_interval}m.csv"
        edges_csv = csv_dir / f"street_edges_predictions_{self.edge_interval}m.csv"
        combined_csv = csv_dir / f"street_network_predictions_{self.edge_interval}m.csv"

        nodes_df.to_csv(nodes_csv, index=False)
        edges_df.to_csv(edges_csv, index=False)
        output_df.to_csv(combined_csv, index=False)

        print(f"Node predictions saved to {nodes_csv}")
        print(f"Edge predictions saved to {edges_csv}")
        print(f"Combined predictions saved to {combined_csv}")

        print("\n🎉 Street network pipeline completed successfully!")

        # Print summary
        valid_preds = np.sum(~np.isnan(predictions))
        node_count = len(nodes_df)
        edge_count = len(edges_df)

        print(f"\n📊 Results Summary:")
        print(f"  Total prediction points: {len(predictions):,}")
        print(f"  Street nodes: {node_count:,}")
        print(f"  Edge points ({self.edge_interval}m intervals): {edge_count:,}")
        print(
            f"  Valid predictions: {valid_preds:,} ({100*valid_preds/len(predictions):.1f}%)"
        )
        print(
            f"  Prediction range: {np.nanmin(predictions):.1f} - {np.nanmax(predictions):.1f}"
        )
        print(
            f"  Mean ± std: {np.nanmean(predictions):.1f} ± {np.nanstd(predictions):.1f}"
        )

        return output_df


def main():
    """Main function."""

    # Configuration options
    intervals = {
        1: 10,  # 10m intervals - moderate detail
        2: 5,  # 5m intervals - high detail
        3: 3,  # 3m intervals - very high detail
        4: 1,  # 1m intervals - extremely high detail (very slow)
    }

    print("Edge Sampling Interval Options:")
    for key, interval in intervals.items():
        # Rough estimate of points (depends on actual street network)
        est_points = (
            "~10k-50k" if interval >= 5 else "~50k-200k" if interval >= 3 else "~200k+"
        )
        time_est = (
            "~10 min" if interval >= 5 else "~20 min" if interval >= 3 else "~30+ min"
        )
        print(f"  {key}. {interval}m intervals ({est_points} points, {time_est})")

    try:
        choice = int(input("\nSelect sampling interval (1-4): "))
        interval = intervals.get(choice, 5)
    except (ValueError, KeyboardInterrupt):
        print("Using default 5m interval")
        interval = 5

    print(f"\nStarting street network pipeline with {interval}m edge intervals...")
    print("Note: Processing time depends on street network complexity")

    # Run pipeline
    pipeline = StreetNetworkPrediction(BBOX, edge_interval_meters=interval)
    results_df = pipeline.run_pipeline()

    return results_df


if __name__ == "__main__":
    results = main()
