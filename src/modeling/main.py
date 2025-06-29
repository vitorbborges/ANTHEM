# run_optimization.py

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.modeling.optimization import OptimizationRunner

if __name__ == "__main__":
    runner = OptimizationRunner(
        data_path="data/processed_data/combined_subjects.parquet",
        temporal_folds=4,
        spatial_folds=10,
    )

    study = runner.run(n_trials=5)
