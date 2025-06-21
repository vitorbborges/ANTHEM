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
    def __init__(self, bbox: tuple):
        self.bbox = bbox
        self.loader = SpatialDataLoader(bbox)
        self.extractor = SpatialFeatureExtractor(self.loader)
        self.specs_file = CONFIG_TOML

    def run(self, subject_id: int, job_id: int = 0) -> pd.DataFrame:
        # Use absolute paths based on PROJECT_ROOT so Hydra's working dir doesn't break us
        raw_dir = PROJECT_ROOT / "data" / "raw_data"
        out_dir = PROJECT_ROOT / "data" / "processed_data"

        data = load_subject(subject_id)
        data["subject_id"] = subject_id

        # Extract route and sensor segments
        kml = self.extractor.loader.extract_kml(raw_dir / "route.kmz")
        segments = self.extractor.loader.build_segments(kml)
        static_df = self.extractor.extract_static(data, kml)
        dynamic_df = self.extractor.extract_dynamic(data, segments)
        # TODO: remove the "segments" parameter from extract_dynamics()
        # TODO: delete the todo comments after finished to debloat the code

        df = pd.concat([static_df, dynamic_df]).sort_index()

        # Timestamp adjustment
        base = pd.Timestamp("2022-11-11") + pd.Timedelta(days=subject_id - 1)
        df.index = base + pd.to_timedelta(df.index.astype(str))

        # Feature computation
        entries = toml.load(self.specs_file).get("features", [])
        with tqdm(
            entries,
            desc=f"S{subject_id} features",
            unit="feat",
            dynamic_ncols=True,
            position=job_id + 1,
            leave=False,
        ) as feat_bar:
            for feature in feat_bar:
                prefix = feature["prefix"]
                feat_bar.set_postfix_str(prefix, refresh=False)
                fn = getattr(self.extractor, f"add_{feature['mode']}")
                df = fn(
                    df,
                    prefix,
                    feature["source"],
                    feature.get("radii", []),
                    feature.get("column"),
                    feature.get("values", []),
                )

        # Optional weather
        wdirs = list((raw_dir).glob("RW_*"))
        if wdirs:
            meta = WeatherProcessor.parse_metadata(wdirs[0])
            raww = WeatherProcessor.read_raw(wdirs[0], meta)
            df = df.join(WeatherProcessor.interpolate(raww), how="left")

        # Write output
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
    for subject_id in subjects:
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
    df.loc[df[col].isna(), col] = 0
# Drop any remaining missing rows
df.dropna(inplace=True)
# Write combined parquet
output_path = os.path.join("data", "processed_data", "combined_subjects.parquet")
df.to_parquet(output_path)
