import sys
from pathlib import Path

# ensure project root is on sys.path so `src` package is found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import osmnx as ox
import pandas as pd
import toml
from tqdm import tqdm

from src.feature_imputer import FeatureImputer
from src.load_data import load_subject
from src.spatial_feature_extractor import SpatialFeatureExtractor
from src.weather_processor import WeatherProcessor

# GLOBAL CONSTANTS
TOML_PATH = Path("config/feature_specs.toml")
WEST, SOUTH, EAST, NORTH = 9.2257, 45.47162, 9.23768, 45.48537
BBOX = (WEST, SOUTH, EAST, NORTH)  # Bounding box for the area of interest


# --------------------------------------------------
# Orchestrator with tqdm
# --------------------------------------------------
class ProcessSubjectPipeline:
    def __init__(self, bbox: tuple):
        self.bbox = bbox
        self.extractor = SpatialFeatureExtractor(bbox)
        self.specs_file = TOML_PATH

    def run(self, subject_id: int) -> pd.DataFrame:
        raw_dir = Path("data/raw_data")
        out_dir = Path("data/processed_data")
        data = load_subject(subject_id)
        data["subject_id"] = subject_id
        kml = self.extractor.extract_kml(raw_dir / "route.kmz")
        segments = self.extractor.build_segments(kml)
        static_df = self.extractor.extract_static(data, kml)
        dynamic_df = self.extractor.extract_dynamic(data, segments)
        df = pd.concat([static_df, dynamic_df]).sort_index()
        base = pd.Timestamp("2022-11-11") + pd.Timedelta(days=subject_id - 1)
        df.index = base + pd.to_timedelta(df.index.astype(str))

        entries = toml.load(self.specs_file).get("features", [])
        feature_bar = tqdm(entries, desc=f"Subject {subject_id}", unit="feature")

        for feature in feature_bar:
            prefix = feature["prefix"]
            mode = feature["mode"]
            source = feature["source"]
            radii = feature.get("radii", [])
            column = feature.get("column")
            values = feature.get("values", [])

            feature_bar.set_description(f"Subject {subject_id} | {prefix}")

            method_name = f"add_{mode}"
            if not hasattr(self.extractor, method_name):
                raise ValueError(f"Unknown mode: {mode!r}")

            fn = getattr(self.extractor, method_name)
            df = fn(df, prefix, source, radii, column, values)

        wdirs = list(raw_dir.glob("RW_*"))
        if wdirs:
            meta = WeatherProcessor.parse_metadata(wdirs[0])
            raww = WeatherProcessor.read_raw(wdirs[0], meta)
            df = df.join(WeatherProcessor.interpolate(raww), how="left")

        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_dir / f"S{subject_id}-coords.parquet")
        return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int)
    args = parser.parse_args()
    pipeline = ProcessSubjectPipeline(BBOX)
    # pipeline.run(3)
    if args.subject:
        pipeline.run(args.subject)
    else:
        for sid in tqdm(range(1, 21), desc="All Subjects", unit="subject"):
            pipeline.run(sid)
