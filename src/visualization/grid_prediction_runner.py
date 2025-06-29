#!/usr/bin/env python3
"""
Simple runner script for grid prediction pipeline.
Run this script to generate CO2 predictions for a 1m×1m grid and create a heatmap.
"""

import sys
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import and run the pipeline
try:
    from grid_prediction_heatmap import main as run_pipeline

    if __name__ == "__main__":
        print("🚀 Starting CO2 Grid Prediction Pipeline")
        print("This will:")
        print("  1. Create a 1m×1m grid covering the study area")
        print("  2. Extract OSM features for each grid point")
        print("  3. Add weather data to all points")
        print("  4. Load the best ensemble model")
        print("  5. Predict CO2 concentration for each point")
        print("  6. Generate and display a heatmap")
        print("\nNote: This may take several minutes due to the high resolution grid.")

        # Ask for confirmation due to computational intensity
        try:
            confirm = input("\nProceed with 1m resolution? (y/N): ").lower().strip()
            if confirm != "y":
                print(
                    "Cancelled. You can modify the grid_resolution_meters parameter for faster execution."
                )
                sys.exit(0)
        except KeyboardInterrupt:
            print("\nCancelled by user.")
            sys.exit(0)

        # Run the pipeline
        run_pipeline()

except ImportError as e:
    print(f"❌ Import error: {e}")
    print(
        "Make sure all required modules are installed and the project structure is correct."
    )
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
