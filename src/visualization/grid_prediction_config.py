#!/usr/bin/env python3
"""
Configurable grid prediction script.
Allows you to adjust grid resolution and other parameters.
"""

import sys
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from grid_prediction_heatmap import GridPredictionPipeline

# Configuration parameters
CONFIG = {
    # Grid resolution in meters (1m = very high detail, 10m = medium, 50m = fast)
    "grid_resolution_meters": 10,  # Start with 10m for reasonable performance
    # Study area bounding box
    "bbox": (9.2257, 45.47162, 9.23768, 45.48537),  # (west, south, east, north)
    # Output settings
    "save_heatmap": True,
    "save_grid_data": True,
    "output_dir": PROJECT_ROOT / "output",
    # Display settings
    "show_plot": True,
    "figure_size": (12, 10),
    "dpi": 300,
}


def run_configurable_pipeline():
    """Run the pipeline with configurable parameters."""

    print("🗺️  Configurable Grid Prediction Pipeline")
    print("=" * 60)
    print(f"Grid resolution: {CONFIG['grid_resolution_meters']}m")
    print(f"Bounding box: {CONFIG['bbox']}")
    print(f"Output directory: {CONFIG['output_dir']}")

    # Estimate grid size
    west, south, east, north = CONFIG["bbox"]
    lat_center = (south + north) / 2
    lon_to_m = 111320 * np.cos(np.radians(lat_center))
    lat_to_m = 111320

    width_m = (east - west) * lon_to_m
    height_m = (north - south) * lat_to_m

    n_cols = int(width_m / CONFIG["grid_resolution_meters"])
    n_rows = int(height_m / CONFIG["grid_resolution_meters"])
    total_points = n_cols * n_rows

    print(f"Estimated grid: {n_cols} × {n_rows} = {total_points:,} points")

    if total_points > 100000:
        print("⚠️  WARNING: Large grid detected!")
        print("   This may take a long time and use significant memory.")
        print("   Consider increasing grid_resolution_meters for faster execution.")

        try:
            confirm = input("Continue anyway? (y/N): ").lower().strip()
            if confirm != "y":
                print(
                    "Cancelled. Modify CONFIG['grid_resolution_meters'] and try again."
                )
                return
        except KeyboardInterrupt:
            print("\nCancelled by user.")
            return

    # Initialize pipeline
    pipeline = GridPredictionPipeline(
        CONFIG["bbox"], grid_resolution_meters=CONFIG["grid_resolution_meters"]
    )

    try:
        # Step 1: Create grid
        print("\n📐 Creating spatial grid...")
        grid_df, n_rows, n_cols = pipeline.create_grid()

        # Step 2: Extract OSM features
        print("\n🏘️  Extracting OSM features...")
        grid_with_osm = pipeline.extract_osm_features(grid_df)

        # Step 3: Add weather features
        print("\n🌤️  Adding weather features...")
        grid_with_weather = pipeline.add_weather_features(grid_with_osm)

        # Step 3.5: Final data validation
        print("\n🔍 Validating final dataset...")

        # Check for any remaining issues
        total_rows = len(grid_with_weather)
        total_cols = len(grid_with_weather.columns)
        total_nans = grid_with_weather.isna().sum().sum()

        print(f"Final dataset: {total_rows:,} rows × {total_cols} columns")
        print(f"Total NaN values: {total_nans:,}")

        if total_nans > 0:
            print("Columns with NaN values:")
            nan_cols = grid_with_weather.isna().sum()
            for col, nan_count in nan_cols[nan_cols > 0].items():
                print(f"  {col}: {nan_count:,} ({100*nan_count/total_rows:.1f}%)")

        grid_with_features = grid_with_weather

        # Step 4: Load ensemble model
        print("\n🤖 Loading ensemble model...")
        model_data = pipeline.load_ensemble_model()

        # Step 5: Make predictions
        print("\n🔮 Making predictions...")
        predictions = pipeline.predict_grid(grid_with_features, model_data)

        # Step 6: Create outputs
        CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)

        if CONFIG["save_heatmap"]:
            print("\n📊 Creating heatmap...")
            save_path = (
                CONFIG["output_dir"]
                / f"co2_heatmap_{CONFIG['grid_resolution_meters']}m.png"
            )
            pipeline.create_heatmap(
                grid_with_features, predictions, n_rows, n_cols, save_path=save_path
            )

        if CONFIG["save_grid_data"]:
            print("\n💾 Saving grid data...")
            output_df = grid_with_features.copy()
            output_df["predicted_co2"] = predictions

            csv_path = (
                CONFIG["output_dir"]
                / f"grid_predictions_{CONFIG['grid_resolution_meters']}m.csv"
            )
            output_df.to_csv(csv_path, index=False)
            print(f"Grid data saved to {csv_path}")

            # Also save a summary
            summary = {
                "grid_resolution_meters": CONFIG["grid_resolution_meters"],
                "total_points": len(predictions),
                "valid_predictions": int(np.sum(~np.isnan(predictions))),
                "min_concentration": float(np.nanmin(predictions)),
                "max_concentration": float(np.nanmax(predictions)),
                "mean_concentration": float(np.nanmean(predictions)),
                "std_concentration": float(np.nanstd(predictions)),
                "median_concentration": float(np.nanmedian(predictions)),
            }

            import json

            summary_path = (
                CONFIG["output_dir"]
                / f"prediction_summary_{CONFIG['grid_resolution_meters']}m.json"
            )
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"Summary saved to {summary_path}")

        print("\n🎉 Pipeline completed successfully!")
        print(f"Check {CONFIG['output_dir']} for outputs.")

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    import numpy as np

    print("Available grid resolutions:")
    print("  - 1m:  Very high detail (~400K+ points, slow)")
    print("  - 5m:  High detail (~16K points, moderate)")
    print("  - 10m: Medium detail (~4K points, fast)")
    print("  - 25m: Low detail (~650 points, very fast)")
    print("  - 50m: Very low detail (~160 points, instant)")

    print(f"\nCurrent setting: {CONFIG['grid_resolution_meters']}m")
    print("Edit CONFIG['grid_resolution_meters'] in this script to change.")

    run_configurable_pipeline()
