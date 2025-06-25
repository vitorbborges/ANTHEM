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
import geopandas as gpd
from scipy.io import loadmat
from scipy.interpolate import interp1d

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
        self.weather_processor = WeatherProcessor
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

        segment_points = segments['geometry'][1:]

        INTERPOLATION_METERS = 5

        dynamic_df = self.extractor.extract_dynamic(data, segments)
        df_resampled_measure = pd.DataFrame(columns = ['x', 'y', 'CO2' , 'location' , 'regime' , 'sub'])

        # Add lenght in the segments in order to knowing the ratio between true mesurements
        # and interpolated 
        gdf = gpd.GeoDataFrame(geometry=segments['geometry'].values)
        gdf.set_crs(epsg=4326, inplace=True)  # WGS84 (lat/lon)
        segments['length_m'] = gdf.to_crs(epsg=32632).geometry.length
        

        sd = dynamic_df.merge(
                    segments[['location','geometry']],
                    on='location',
                    how='left'
            )
        sd['sample_pt'] = None # column for the intepolated signal

        print(sd.head())

        ## start interpolation
        for loc, group in sd.groupby('location'):

            values = group["CO2"].to_numpy()
            n_measured_points = group.shape[0]
            length_of_segment = int(np.floor(segments[segments['location']==loc]['length_m'].values[0]))  
            n_interpolated_points   = int(np.floor(length_of_segment//INTERPOLATION_METERS ))

            print(f"subj = {subject_id} location ={loc} n.points = {n_measured_points} length={length_of_segment} N_points={n_interpolated_points}")
            
            
            seg = group.geometry.iloc[0]         # the LineString for this location
            L   = seg.length                     # its total length
            dists = np.linspace(0, L, n_interpolated_points) # linaspace of equally distanced points

            intepolated_pts = [seg.interpolate(d) for d in dists]
            df_interpolated_coords = pd.DataFrame([(p.x, p.y) for p in intepolated_pts], columns=['x', 'y'])
            interpolated_data = {}

            kind = 'cubic' if n_measured_points >= 4 else 'linear'

            old_idx = np.linspace(0, n_measured_points - 1, num=n_measured_points)
            new_idx = np.linspace(0, n_measured_points - 1, num=n_interpolated_points)

            f_interp = interp1d(old_idx, values, kind=kind, bounds_error=False, fill_value="extrapolate")
            f_interp= f_interp(new_idx)


            tmp = pd.DataFrame({
                'x': df_interpolated_coords['x'],
                'y': df_interpolated_coords['y'],
                'CO2': f_interp,
                'location': [loc] * len(f_interp),
                'regime': [group['regime'].iloc[0]] * len(f_interp),
                'sub': subject_id* len(f_interp),
            })

            df_resampled_measure = pd.concat([df_resampled_measure, tmp], ignore_index=True)
            print(df_resampled_measure.head())


        # TODO: it is just the resampling without the osmx and static data, in prder to works
        # with previous model I guess that should be concatened with segments and also with 
        # with static, i'M REALLY SRY but I have to spend more time in order to understand
        # how to navigate in this prohject structure so I'll go further on Colab and I'll merge
        # with the app in the end
        out_dir.mkdir(parents=True, exist_ok=True)
        df_resampled_measure.to_parquet(out_dir / f"S{subject_id}-coords.parquet")
        return df_resampled_measure




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
