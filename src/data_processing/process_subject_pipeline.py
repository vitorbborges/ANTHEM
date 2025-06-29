#!/usr/bin/env python3

# --- START: MODIFIED SECTION FOR WARNING SUPPRESSION ---
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module="shapely.measurement",
    message="invalid value encountered in distance",
)

import numpy as np

np.seterr(invalid="ignore", divide="ignore", over="ignore", under="ignore")
# --- END: MODIFIED SECTION FOR WARNING SUPPRESSION ---

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import hydra
import numpy as np
import osmnx as ox
import pandas as pd
import toml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm.auto import tqdm

# ensure project root is on sys.path so that 'src' modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.load_data import load_subject
from src.data_processing.spatial_data_loader import SpatialDataLoader
from src.data_processing.spatial_feature_extractor import SpatialFeatureExtractor
from src.data_processing.weather_processor import WeatherProcessor

# Global constants
CONFIG_TOML = Path(__file__).resolve().parent / "config/feature_specs.toml"
WEST, SOUTH, EAST, NORTH = 9.2257, 45.47162, 9.23768, 45.48537
BBOX = (WEST, SOUTH, EAST, NORTH)


class ProcessSubjectPipeline:
    """
    Orchestrator for processing a single subject's data through:
      1. Loading raw GPS and route (KML/KMZ) data,
      2. Extracting static and dynamic spatial features with resampling,
      3. Applying TOML-defined feature specifications,
      4. Optionally joining weather observations,
      5. Saving the processed DataFrame.
    """

    def __init__(self, bbox: tuple):
        self.bbox = bbox
        self.loader = SpatialDataLoader(bbox)
        self.extractor = SpatialFeatureExtractor(self.loader)
        self.weather_processor = WeatherProcessor
        self.specs_file = CONFIG_TOML

    def run(self, subject_id: int, job_id: int = 0) -> pd.DataFrame:
        # Use absolute paths based on PROJECT_ROOT so Hydra's working dir doesn't break us
        raw_dir = PROJECT_ROOT / "data" / "raw_data"
        out_dir = PROJECT_ROOT / "data" / "processed_data"

        # Load raw subject measurements and tag with subject_id
        data = load_subject(subject_id)
        data["subject_id"] = subject_id

        # Extract route and sensor segments using the loader
        kml = self.extractor.loader.extract_kml(raw_dir / "route.kmz")
        segments = self.extractor.loader.build_segments(kml)

        # Extract only dynamic points with resampling using the new method
        # Skip static data processing for now
        df = self.extractor.resample_dynamic(
            data, segments, subject_id, interpolation_meters=5
        )

        # If DataFrame is empty, save empty result and return
        if df.empty:
            out_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out_dir / f"S{subject_id}-coords.parquet")
            return df

        # Reset index for proper timestamp conversion
        df = df.reset_index(drop=True)

        # Convert integer index to timestamps based on a base date + subject offset
        base = pd.Timestamp("2022-11-11") + pd.Timedelta(days=subject_id - 1)
        # Create a proper timedelta from integer indices (assuming they represent seconds)
        df.index = base + pd.to_timedelta(df.index, unit="s")

        # Load feature specifications from TOML file if it exists
        if self.specs_file.exists():
            entries = toml.load(self.specs_file).get("features", [])

            # Compute each feature with an inner progress bar
            with tqdm(
                entries,
                desc=f"S{subject_id} features",
                unit="feat",
                dynamic_ncols=True,
                position=job_id + 1,
                leave=True,
                ncols=80,
            ) as feat_bar:
                for feature in feat_bar:
                    prefix = feature["prefix"]
                    feat_bar.set_postfix_str(prefix, refresh=False)

                    # Dynamically call add_proximity, add_sum, add_mean, etc.
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
                            print(
                                f"Warning: Failed to compute feature {prefix} with mode {mode}: {e}"
                            )
                            continue
                    else:
                        print(
                            f"Warning: Unknown feature mode '{mode}' for feature '{prefix}'"
                        )

        # If weather raw directories exist, parse and interpolate weather data
        wdirs = list(raw_dir.glob("RW_*"))
        if wdirs:
            try:
                meta = WeatherProcessor.parse_metadata(wdirs[0])
                raww = WeatherProcessor.read_raw(wdirs[0], meta)
                weather_data = WeatherProcessor.interpolate(raww)
                df = df.join(weather_data, how="left")
            except Exception as e:
                print(
                    f"Warning: Failed to process weather data for subject {subject_id}: {e}"
                )

        # Ensure output directory exists and write parquet file
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_dir / f"S{subject_id}-coords.parquet")
        return df


# Hydra entrypoint: multi-run config
@hydra.main(
    version_base="1.1", config_path="config", config_name="featurization_settings"
)
def hydra_main(cfg: DictConfig):
    # Re-apply warning filters within the Hydra job context
    warnings.filterwarnings("ignore")
    np.seterr(invalid="ignore", divide="ignore", over="ignore", under="ignore")

    job_id = HydraConfig.get().job.num
    pipeline = ProcessSubjectPipeline(BBOX)

    # Process the current subject (when running in multirun mode)
    subject_id = cfg.get("subject")
    if subject_id is not None:
        pipeline.run(int(subject_id), job_id=job_id)


# CLI entrypoint for argparse
def cli_main():
    parser = argparse.ArgumentParser(description="Process one or many subjects' data")
    parser.add_argument("--subject", type=int, help="Subject ID to process (1-20)")
    args = parser.parse_args()

    pipeline = ProcessSubjectPipeline(BBOX)
    subjects = [args.subject] if args.subject else list(range(1, 21))

    # Outer progress bar over subjects
    with tqdm(
        subjects,
        desc="All Subjects",
        unit="subj",
        dynamic_ncols=True,
        position=0,
        leave=True,
        ncols=80,
    ) as subj_bar:
        for subject_id in subj_bar:
            subj_bar.set_postfix_str(f"S{subject_id}", refresh=False)
            pipeline.run(subject_id)


if __name__ == "__main__":
    # If called with -m (Hydra multirun), use Hydra; otherwise use argparse CLI
    if "-m" in sys.argv:
        hydra_main()
    else:
        cli_main()

    import os

    # -----------------------------------------------------------------------------
    # Data concatenation and cleanup for combined_subjects.parquet
    # -----------------------------------------------------------------------------
    import pandas as pd

    df = pd.concat(
        [
            pd.read_parquet(os.path.join("data", "processed_data", file))
            for file in os.listdir(os.path.join("data", "processed_data"))
            if file != "combined_subjects.parquet" and file.endswith(".parquet")
        ]
    )
    # Drop unwanted column if exists
    df.drop(columns=["PM25"], inplace=True, errors="ignore")
    # Fill missing values for key features
    fill_zero_cols = [
        "average_nearby_num_lanes_50",
        "average_nearby_maxspeed_50",
        "average_nearby_streets_len_50",
        "average_building_height_100",
        "average_building_height_200",
    ]
    for col in fill_zero_cols:
        if col in df.columns:
            # Fill NaN values with 0 for specified columns
            df.loc[df[col].isna(), col] = 0
    # Drop any remaining missing rows
    df.dropna(inplace=True)
    # Write combined parquet
    output_path = os.path.join("data", "processed_data", "combined_subjects.parquet")
    df.to_parquet(output_path)
