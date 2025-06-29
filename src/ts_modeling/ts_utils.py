# src/modeling/ts_utils.py
"""
Time Series utilities.
Updated utils.py for time series specific functions.
"""

import os
import warnings
from datetime import datetime
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# Create directories for outputs
os.makedirs("metrics", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Global log file
log_file = f"metrics/ts_optimization_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"


def log_message(message: str):
    """Log a message to both console and file."""
    print(message)
    with open(log_file, "a") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")


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


def calculate_time_series_metrics(
    y_true: pd.Series, y_pred: np.ndarray, n_features: int = 0
) -> dict:
    """
    Calculate comprehensive time series evaluation metrics.

    Args:
        y_true: True values
        y_pred: Predicted values
        n_features: Number of features used (for adjusted R²)

    Returns:
        Dictionary of metrics
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    # Basic regression metrics
    mse = mean_squared_error(y_true_arr, y_pred_arr)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    r2 = r2_score(y_true_arr, y_pred_arr)
    adj_r2 = adjusted_r2_score(y_true_arr, y_pred_arr, n_features)

    # Time series specific metrics
    mape = mean_absolute_percentage_error(y_true_arr, y_pred_arr)

    # Directional accuracy (for trend prediction)
    directional_accuracy = calculate_directional_accuracy(y_true, y_pred)

    # Error statistics
    residuals = y_true_arr - y_pred_arr

    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "adj_r2": adj_r2,
        "directional_accuracy": directional_accuracy,
        "residual_mean": np.mean(residuals),
        "residual_std": np.std(residuals),
        "residual_skew": pd.Series(residuals).skew(),
        "residual_kurtosis": pd.Series(residuals).kurtosis(),
    }

    return metrics


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    Handles division by zero by using a small epsilon value.
    """
    epsilon = np.finfo(np.float64).eps
    mape = (
        np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100
    )
    return mape


def calculate_directional_accuracy(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """
    Calculate directional accuracy for time series predictions.

    Measures how often the model correctly predicts the direction of change.
    """
    if len(y_true) < 2:
        return np.nan

    # Calculate actual and predicted changes
    actual_changes = np.diff(y_true.values)
    pred_changes = np.diff(y_pred)

    # Count correct directions (both positive, both negative, or both zero)
    correct_directions = np.sign(actual_changes) == np.sign(pred_changes)

    return np.mean(correct_directions) * 100


def detect_time_series_patterns(y: pd.Series, period_hints: list = None) -> dict:
    """
    Detect patterns in time series data.

    Args:
        y: Time series data
        period_hints: List of expected periodicities to check

    Returns:
        Dictionary with pattern information
    """
    if period_hints is None:
        period_hints = [24, 168, 720, 8760]  # hourly, weekly, monthly, yearly

    patterns = {
        "length": len(y),
        "mean": float(y.mean()),
        "std": float(y.std()),
        "trend": None,
        "seasonality": {},
        "stationarity": None,
    }

    # Simple trend detection
    x = np.arange(len(y))
    correlation = np.corrcoef(x, y.values)[0, 1]
    if correlation > 0.1:
        patterns["trend"] = "increasing"
    elif correlation < -0.1:
        patterns["trend"] = "decreasing"
    else:
        patterns["trend"] = "stable"

    # Basic seasonality detection using autocorrelation
    for period in period_hints:
        if len(y) > period * 2:  # Need at least 2 cycles
            try:
                # Calculate autocorrelation at this lag
                autocorr = y.autocorr(lag=period)
                if autocorr > 0.3:  # Threshold for significant seasonality
                    patterns["seasonality"][f"period_{period}"] = float(autocorr)
            except:
                pass

    # Simple stationarity check (variance ratio test)
    if len(y) > 100:
        first_half_var = y[: len(y) // 2].var()
        second_half_var = y[len(y) // 2 :].var()
        var_ratio = max(first_half_var, second_half_var) / min(
            first_half_var, second_half_var
        )
        patterns["stationarity"] = "stationary" if var_ratio < 2.0 else "non-stationary"

    return patterns


def prepare_exogenous_features(
    df: pd.DataFrame, target_col: str, exclude_cols: list = None, max_features: int = 50
) -> Tuple[pd.DataFrame, pd.Series, dict]:
    """
    Prepare exogenous features for time series modeling.

    Args:
        df: Input DataFrame
        target_col: Name of target column
        exclude_cols: Columns to exclude from features
        max_features: Maximum number of features to keep

    Returns:
        X (features), y (target), feature_info (metadata)
    """
    if exclude_cols is None:
        exclude_cols = []

    # Extract target
    y = df[target_col].copy()

    # Get potential feature columns
    feature_cols = [
        col for col in df.columns if col != target_col and col not in exclude_cols
    ]

    # Filter numeric columns
    numeric_cols = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)

    log_message(f"Found {len(numeric_cols)} numeric feature columns")

    # Basic feature selection based on correlation with target
    if len(numeric_cols) > max_features:
        correlations = {}
        for col in numeric_cols:
            try:
                corr = abs(df[col].corr(y))
                if not np.isnan(corr):
                    correlations[col] = corr
            except:
                pass

        # Select top correlated features
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        selected_cols = [col for col, _ in sorted_features[:max_features]]
        log_message(f"Selected top {len(selected_cols)} features by correlation")
    else:
        selected_cols = numeric_cols

    X = df[selected_cols].copy()

    # Feature information
    feature_info = {
        "n_features": len(selected_cols),
        "feature_names": selected_cols,
        "target_name": target_col,
        "correlations": {},
        "missing_values": {},
    }

    # Calculate correlations and missing value counts
    for col in selected_cols:
        try:
            feature_info["correlations"][col] = float(X[col].corr(y))
        except:
            feature_info["correlations"][col] = 0.0

        feature_info["missing_values"][col] = int(X[col].isna().sum())

    return X, y, feature_info


def create_lag_features(
    df: pd.DataFrame, target_col: str, lags: list = None, rolling_windows: list = None
) -> pd.DataFrame:
    """
    Create lag and rolling window features for time series.

    Args:
        df: Input DataFrame with time series data
        target_col: Name of target column
        lags: List of lag values to create
        rolling_windows: List of window sizes for rolling statistics

    Returns:
        DataFrame with lag features added
    """
    if lags is None:
        lags = [1, 2, 3, 6, 12, 24]  # Default lags

    if rolling_windows is None:
        rolling_windows = [3, 6, 12, 24]  # Default windows

    df_with_lags = df.copy()

    # Create lag features
    for lag in lags:
        lag_col = f"{target_col}_lag_{lag}"
        df_with_lags[lag_col] = df[target_col].shift(lag)

    # Create rolling window features
    for window in rolling_windows:
        # Rolling mean
        mean_col = f"{target_col}_rolling_mean_{window}"
        df_with_lags[mean_col] = df[target_col].rolling(window=window).mean()

        # Rolling std
        std_col = f"{target_col}_rolling_std_{window}"
        df_with_lags[std_col] = df[target_col].rolling(window=window).std()

        # Rolling min/max
        min_col = f"{target_col}_rolling_min_{window}"
        max_col = f"{target_col}_rolling_max_{window}"
        df_with_lags[min_col] = df[target_col].rolling(window=window).min()
        df_with_lags[max_col] = df[target_col].rolling(window=window).max()

    # Create difference features
    diff_col = f"{target_col}_diff_1"
    df_with_lags[diff_col] = df[target_col].diff()

    # Create time-based features if index is datetime
    if isinstance(df.index, pd.DatetimeIndex):
        df_with_lags["hour"] = df.index.hour
        df_with_lags["day_of_week"] = df.index.dayofweek
        df_with_lags["day_of_year"] = df.index.dayofyear
        df_with_lags["month"] = df.index.month
        df_with_lags["quarter"] = df.index.quarter

        # Cyclical encoding for time features
        df_with_lags["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
        df_with_lags["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
        df_with_lags["day_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df_with_lags["day_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        df_with_lags["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
        df_with_lags["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

    log_message(f"Created lag features. Shape: {df.shape} -> {df_with_lags.shape}")

    return df_with_lags


def validate_time_series_data(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Validate time series data for modeling.

    Args:
        X: Exogenous features
        y: Target time series

    Returns:
        Dictionary with validation results and warnings
    """
    validation = {"is_valid": True, "warnings": [], "errors": [], "info": {}}

    # Check basic requirements
    if len(X) != len(y):
        validation["errors"].append("X and y have different lengths")
        validation["is_valid"] = False

    if len(y) < 50:
        validation["warnings"].append("Very short time series (< 50 observations)")

    # Check for missing values
    if y.isna().any():
        validation["errors"].append("Target variable contains missing values")
        validation["is_valid"] = False

    if X.isna().any().any():
        na_cols = X.columns[X.isna().any()].tolist()
        validation["warnings"].append(f"Features with missing values: {na_cols}")

    # Check temporal order
    if not X.index.equals(y.index):
        validation["errors"].append("X and y indices don't match")
        validation["is_valid"] = False

    # Check for duplicate indices
    if X.index.duplicated().any():
        validation["warnings"].append("Duplicate time indices found")

    # Check for constant features
    constant_features = []
    for col in X.columns:
        if X[col].nunique() <= 1:
            constant_features.append(col)

    if constant_features:
        validation["warnings"].append(f"Constant features: {constant_features}")

    # Data info
    validation["info"] = {
        "n_samples": len(y),
        "n_features": X.shape[1],
        "target_range": (float(y.min()), float(y.max())),
        "target_variance": float(y.var()),
        "missing_target": int(y.isna().sum()),
        "missing_features": {
            col: int(X[col].isna().sum()) for col in X.columns if X[col].isna().any()
        },
    }

    return validation


def save_model_summary(
    pipeline, trial_params: dict, results: dict, file_path: str = None
) -> str:
    """
    Save a comprehensive model summary.

    Args:
        pipeline: Fitted pipeline
        trial_params: Optuna trial parameters
        results: Evaluation results
        file_path: Path to save summary (optional)

    Returns:
        Path to saved summary file
    """
    if file_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"metrics/model_summary_{timestamp}.json"

    # Extract model information
    model_info = {
        "model_type": trial_params.get("model_type", "unknown"),
        "parameters": trial_params,
        "performance": {
            "cv_mse": results.get("cv_mse"),
            "cv_r2": results.get("cv_r2"),
            "holdout_test_mse": results.get("holdout_test_mse"),
            "holdout_test_r2": results.get("holdout_test_r2"),
        },
        "feature_info": {
            "n_features_original": results.get("n_features"),
            "n_features_selected": results.get("n_features_selected"),
            "feature_selection_method": trial_params.get("feature_selector_type"),
            "uses_scaling": trial_params.get("use_exog_scaling", False),
        },
        "training_info": {
            "time_taken": results.get("time_taken"),
            "cv_method": results.get("cv_method", "walk_forward"),
            "timestamp": datetime.now().isoformat(),
        },
    }

    # Add model-specific information
    if hasattr(pipeline, "model_type"):
        model_info["model_details"] = {
            "wrapper_type": "TimeSeriesWrapper",
            "base_model": pipeline.model_type,
        }

    # Save summary
    with open(file_path, "w") as f:
        json.dump(model_info, f, indent=2, default=str)

    log_message(f"Model summary saved to {file_path}")
    return file_path
