import sys
from pathlib import Path

# Ensure project root is on sys.path so that `src` modules can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings

import osmnx as ox
import pandas as pd
import toml
from tqdm.auto import tqdm

# Custom modules for data loading, feature extraction, and weather processing
from src.feature_imputer import FeatureImputer
from src.load_data import load_subject
from src.spatial_data_loader import SpatialDataLoader
from src.spatial_feature_extractor import SpatialFeatureExtractor
from src.weather_processor import WeatherProcessor

# Suppress pandas FutureWarnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

# Global constants for configuration and study area
CONFIG_TOML = Path("config/feature_specs.toml")
WEST, SOUTH, EAST, NORTH = 9.2257, 45.47162, 9.23768, 45.48537
BBOX = (WEST, SOUTH, EAST, NORTH)  # Bounding box for the area of interest


class ProcessSubjectPipeline:
    """
    Orchestrator for processing a single subject's data through:
      1. Loading raw GPS and route (KML/KMZ) data,
      2. Extracting static and dynamic spatial features,
      3. Applying TOML-defined feature specifications,
      4. Optionally joining weather observations,
      5. Saving the processed DataFrame.
    """

    def __init__(self, bbox: tuple):
        # Store bounding box and initialize helper classes
        self.bbox = bbox
        self.loader = SpatialDataLoader(bbox)
        self.extractor = SpatialFeatureExtractor(self.loader)
        # Path to TOML file defining which features to compute
        self.specs_file = CONFIG_TOML

    def run(self, subject_id: int) -> pd.DataFrame:
        # Directories for raw inputs and outputs
        raw_dir = Path("data/raw_data")
        out_dir = Path("data/processed_data")

        # Load raw subject measurements and tag with subject_id
        data = load_subject(subject_id)
        data["subject_id"] = subject_id

        # Split into static (route shape) and dynamic (sensor) data
        kml = self.extractor.loader.extract_kml(raw_dir / "route.kmz")
        segments = self.extractor.loader.build_segments(kml)
        static_df = self.extractor.extract_static(data, kml)
        dynamic_df = self.extractor.extract_dynamic(data, segments)
        # Combine static + dynamic and sort by time index
        df = pd.concat([static_df, dynamic_df]).sort_index()

        # Convert integer index to timestamps based on a base date + subject offset
        base = pd.Timestamp("2022-11-11") + pd.Timedelta(days=subject_id - 1)
        df.index = base + pd.to_timedelta(df.index.astype(str))

        # Load feature specifications from TOML file
        entries = toml.load(self.specs_file).get("features", [])

        # Compute each feature with an inner progress bar
        with tqdm(
            entries,
            desc=f"S{subject_id} features",
            unit="feat",
            dynamic_ncols=True,
            position=1,
            leave=False,
        ) as feat_bar:
            for feature in feat_bar:
                prefix = feature["prefix"]
                feat_bar.set_postfix_str(prefix, refresh=False)
                # Dynamically call add_proximity, add_sum, add_mean, etc.
                fn = getattr(self.extractor, f"add_{feature['mode']}")
                df = fn(
                    df,
                    prefix,
                    feature["source"],
                    feature.get("radii", []),
                    feature.get("column"),
                    feature.get("values", []),
                )

        # If weather raw directories exist, parse and interpolate weather data
        wdirs = list(raw_dir.glob("RW_*"))
        if wdirs:
            meta = WeatherProcessor.parse_metadata(wdirs[0])
            raww = WeatherProcessor.read_raw(wdirs[0], meta)
            df = df.join(WeatherProcessor.interpolate(raww), how="left")

        # Ensure output directory exists and write parquet file
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_dir / f"S{subject_id}-coords.parquet")
        return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int)
    args = parser.parse_args()

    pipeline = ProcessSubjectPipeline(BBOX)
    # Process either a single subject or subjects 1–20 by default
    subjects = [args.subject] if args.subject else list(range(1, 21))

    # Outer progress bar over subjects
    with tqdm(
        subjects,
        desc="All Subjects",
        unit="subj",
        dynamic_ncols=True,
        position=0,
    ) as subj_bar:
        for sid in subj_bar:
            subj_bar.set_postfix_str(f"S{sid}", refresh=False)
            pipeline.run(sid)
