# src/modeling/utils.py
import os
from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# Create directories for outputs
os.makedirs("metrics", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Global log file
log_file = f"metrics/optimization_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"


def log_message(message: str):
    """Log a message to both console and file."""
    print(message)
    with open(log_file, "a") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")


def stratify_continuous(
    y: Union[pd.Series, np.ndarray], n_bins: int = 10
) -> np.ndarray:
    """
    Create stratification bins for continuous target variable.
    This allows us to use stratified CV with regression problems.
    """
    if isinstance(y, pd.Series):
        bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
    else:
        bins = pd.qcut(pd.Series(y), q=n_bins, labels=False, duplicates="drop")
    return bins.values


def adjusted_r2_score(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
    """
    Calculate adjusted R² score.

    Adjusted R² = 1 - (1 - R²) * (n - 1) / (n - p - 1)
    where n is number of samples and p is number of features
    """
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)

    # Avoid division by zero or negative denominators
    if n <= n_features + 1:
        return float("-inf")  # Invalid case

    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return adj_r2


def load_data(
    file_path: str = "data/processed_data/S3-coords.parquet",
) -> tuple[pd.DataFrame, pd.Series]:
    """Load and prepare the dataset."""
    try:
        log_message(f"Loading data from {file_path}...")
        df = pd.read_parquet(file_path)

        # TODO: For Kalman Filter, ensure temporal ordering
        # Sort by timestamp if available for sequential modeling
        # if 'timestamp' in df.columns:
        #     df = df.sort_values('timestamp').reset_index(drop=True)
        #     log_message("Data sorted by timestamp for Kalman filtering")

        # Extract features and target, keep only numeric columns
        columns_to_use = [
            col for col in df.columns if df[col].dtype in ("float64", "int64")
        ]
        remove_cols = ["P", "PM1", "PM10", "RH", "T", "VOC"]
        columns_to_use = [col for col in columns_to_use if col not in remove_cols]

        # TODO: For Kalman Filter, add lag features for temporal dependencies
        # Add lagged versions of CO2 and other key features
        # for lag in [1, 2, 3, 5, 10]:
        #     df[f'CO2_lag_{lag}'] = df['CO2'].shift(lag)
        #     columns_to_use.append(f'CO2_lag_{lag}')

        X = df[columns_to_use].dropna()
        y = X.pop("CO2")

        log_message(f"Loaded {len(X)} samples with {X.shape[1]} features")
        return X, y

    except FileNotFoundError:
        log_message(f"Error: {file_path} not found. Please check the file path.")
        raise
