# src/modeling/ts_cv_handler.py
"""
Time Series Cross Validation Handler.
Replaces cv_handler.py with proper time series validation strategies.
"""

import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

from .utils import adjusted_r2_score, log_message


class TimeSeriesCVHandler:
    """Handles time series cross-validation for hyperparameter optimization."""

    def __init__(
        self,
        n_splits: int = 5,
        test_size: float = 0.2,
        gap: int = 0,
        max_train_size: int = None,
    ):
        """
        Initialize time series CV handler.

        Args:
            n_splits: Number of CV splits
            test_size: Size of final holdout test set
            gap: Number of samples to exclude between train and test sets
            max_train_size: Maximum number of training samples per fold
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.max_train_size = max_train_size

    def temporal_train_test_split(
        self, X: pd.DataFrame, y: pd.Series, test_size: float = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split time series data maintaining temporal order.

        Args:
            X: Feature matrix with temporal index
            y: Target series with temporal index
            test_size: Fraction for test set

        Returns:
            X_train, X_test, y_train, y_test
        """
        if test_size is None:
            test_size = self.test_size

        # Ensure data is sorted by index (temporal order)
        if not X.index.equals(y.index):
            raise ValueError("X and y must have the same index")

        # Sort by index to ensure temporal order
        sort_idx = X.index.argsort()
        X_sorted = X.iloc[sort_idx]
        y_sorted = y.iloc[sort_idx]

        # Calculate split point
        split_idx = int(len(X_sorted) * (1 - test_size))

        # Apply gap if specified
        if self.gap > 0:
            gap_start = split_idx
            gap_end = min(split_idx + self.gap, len(X_sorted))

            X_train = X_sorted.iloc[:gap_start]
            y_train = y_sorted.iloc[:gap_start]
            X_test = X_sorted.iloc[gap_end:]
            y_test = y_sorted.iloc[gap_end:]
        else:
            X_train = X_sorted.iloc[:split_idx]
            y_train = y_sorted.iloc[:split_idx]
            X_test = X_sorted.iloc[split_idx:]
            y_test = y_sorted.iloc[split_idx:]

        return X_train, X_test, y_train, y_test

    def walk_forward_validation(
        self, X: pd.DataFrame, y: pd.Series, pipeline_factory, trial, trial_number: int
    ) -> Dict:
        """
        Perform walk-forward (expanding window) cross-validation.

        This is the gold standard for time series validation as it respects
        temporal dependencies and simulates real-world deployment.
        """
        start_time = time.time()
        n_features = X.shape[1]

        log_message(f"Trial {trial_number}: Starting walk-forward validation...")

        # Ensure temporal ordering
        if not X.index.equals(y.index):
            raise ValueError("X and y must have the same index")

        sort_idx = X.index.argsort()
        X_sorted = X.iloc[sort_idx]
        y_sorted = y.iloc[sort_idx]

        # Create holdout set (most recent data)
        X_dev, X_holdout, y_dev, y_holdout = self.temporal_train_test_split(
            X_sorted, y_sorted, self.test_size
        )

        log_message(
            f"Trial {trial_number}: Development set: {len(X_dev)}, "
            f"Holdout set: {len(X_holdout)}"
        )

        # Time series split for CV
        tscv = TimeSeriesSplit(
            n_splits=self.n_splits,
            gap=self.gap,
            max_train_size=self.max_train_size,
            test_size=None,
        )

        cv_scores = {"mse": [], "mae": [], "r2": [], "adj_r2": []}

        # Walk-forward validation loop
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_dev)):
            log_message(f"Trial {trial_number}: CV fold {fold+1}/{self.n_splits}")

            X_train_fold = X_dev.iloc[train_idx]
            X_val_fold = X_dev.iloc[val_idx]
            y_train_fold = y_dev.iloc[train_idx]
            y_val_fold = y_dev.iloc[val_idx]

            try:
                # Create and fit pipeline
                pipeline = pipeline_factory(trial, X_train_fold, y_train_fold)
                pipeline.fit(X_train_fold, y_train_fold)

                # Predict on validation set
                y_pred_fold = pipeline.predict(X_val_fold)

                # Calculate metrics
                fold_mse = mean_squared_error(y_val_fold, y_pred_fold)
                fold_mae = mean_absolute_error(y_val_fold, y_pred_fold)
                fold_r2 = r2_score(y_val_fold, y_pred_fold)
                fold_adj_r2 = adjusted_r2_score(y_val_fold, y_pred_fold, n_features)

                cv_scores["mse"].append(fold_mse)
                cv_scores["mae"].append(fold_mae)
                cv_scores["r2"].append(fold_r2)
                cv_scores["adj_r2"].append(fold_adj_r2)

                log_message(
                    f"Trial {trial_number}, Fold {fold+1}: "
                    f"MSE={fold_mse:.6f}, R²={fold_r2:.6f}"
                )

                # Report intermediate results for pruning
                trial.report(fold_mse, step=fold)
                if trial.should_prune():
                    log_message(f"Trial {trial_number}: Pruned at fold {fold+1}")
                    raise optuna.exceptions.TrialPruned()

            except optuna.exceptions.TrialPruned:
                raise
            except Exception as e:
                log_message(f"Trial {trial_number}, Fold {fold+1} failed: {str(e)}")
                continue

        # Check if we have successful folds
        if len(cv_scores["mse"]) == 0:
            log_message(f"Trial {trial_number}: No successful CV folds")
            raise Exception("No successful CV folds")

        # Calculate mean CV performance
        mean_cv_mse = np.mean(cv_scores["mse"])
        mean_cv_mae = np.mean(cv_scores["mae"])
        mean_cv_r2 = np.mean(cv_scores["r2"])
        mean_cv_adj_r2 = np.mean(cv_scores["adj_r2"])

        # Final evaluation on holdout set
        try:
            log_message(f"Trial {trial_number}: Final evaluation on holdout set...")

            # Fit final model on entire development set
            final_pipeline = pipeline_factory(trial, X_dev, y_dev)
            final_pipeline.fit(X_dev, y_dev)

            # Predict on holdout set
            y_holdout_pred = final_pipeline.predict(X_holdout)

            # Calculate holdout metrics
            holdout_test_mse = mean_squared_error(y_holdout, y_holdout_pred)
            holdout_test_mae = mean_absolute_error(y_holdout, y_holdout_pred)
            holdout_test_r2 = r2_score(y_holdout, y_holdout_pred)
            holdout_test_adj_r2 = adjusted_r2_score(
                y_holdout, y_holdout_pred, n_features
            )

            # Training set performance
            y_dev_pred = final_pipeline.predict(X_dev)
            holdout_train_mse = mean_squared_error(y_dev, y_dev_pred)
            holdout_train_mae = mean_absolute_error(y_dev, y_dev_pred)
            holdout_train_r2 = r2_score(y_dev, y_dev_pred)
            holdout_train_adj_r2 = adjusted_r2_score(y_dev, y_dev_pred, n_features)

            # Create results DataFrames
            holdout_results = pd.DataFrame(
                {
                    "actual_test": y_holdout.values,
                    "predicted_test": y_holdout_pred,
                    "timestamp_test": y_holdout.index,
                }
            )

            train_results = pd.DataFrame(
                {
                    "actual_train": y_dev.values,
                    "predicted_train": y_dev_pred,
                    "timestamp_train": y_dev.index,
                }
            )

            results = {
                "pipeline": final_pipeline,
                "holdout_results": holdout_results,
                "train_results": train_results,
                "cv_mse": mean_cv_mse,
                "cv_mae": mean_cv_mae,
                "cv_r2": mean_cv_r2,
                "cv_adj_r2": mean_cv_adj_r2,
                "cv_mse_std": np.std(cv_scores["mse"]),
                "cv_r2_std": np.std(cv_scores["r2"]),
                "holdout_test_mse": holdout_test_mse,
                "holdout_test_mae": holdout_test_mae,
                "holdout_test_r2": holdout_test_r2,
                "holdout_test_adj_r2": holdout_test_adj_r2,
                "holdout_train_mse": holdout_train_mse,
                "holdout_train_mae": holdout_train_mae,
                "holdout_train_r2": holdout_train_r2,
                "holdout_train_adj_r2": holdout_train_adj_r2,
                "time_taken": time.time() - start_time,
                "n_features": n_features,
                "n_features_selected": self._get_selected_features_count(
                    final_pipeline
                ),
                "cv_scores_detail": cv_scores,
            }

            log_message(
                f"Trial {trial_number}: Completed in {results['time_taken']:.2f}s"
            )
            log_message(
                f"Trial {trial_number}: CV MSE = {mean_cv_mse:.6f} ± {np.std(cv_scores['mse']):.6f}"
            )
            log_message(f"Trial {trial_number}: Holdout MSE = {holdout_test_mse:.6f}")

            return results

        except Exception as e:
            log_message(f"Trial {trial_number}: Final evaluation failed: {str(e)}")
            raise

    def _get_selected_features_count(self, pipeline) -> int:
        """FIXED: Get number of selected features from pipeline."""
        try:
            if (
                hasattr(pipeline, "feature_selector")
                and pipeline.feature_selector is not None
            ):
                if hasattr(pipeline, "selected_features_"):
                    return np.sum(pipeline.selected_features_)
                elif hasattr(pipeline.feature_selector, "get_support"):
                    return np.sum(pipeline.feature_selector.get_support())
                elif hasattr(pipeline.feature_selector, "n_components_"):
                    return pipeline.feature_selector.n_components_
                elif hasattr(pipeline, "n_components_"):  # PCA case in wrapper
                    return pipeline.n_components_
            return pipeline.n_features_in_ if hasattr(pipeline, "n_features_in_") else 0
        except:
            return 0

    def blocked_time_series_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        pipeline_factory,
        trial,
        trial_number: int,
        block_size: int = None,
    ) -> Dict:
        """
        Blocked time series cross-validation.

        Alternative CV strategy that creates non-overlapping blocks to reduce
        temporal dependence between train/test sets.
        """
        start_time = time.time()
        n_features = X.shape[1]

        if block_size is None:
            block_size = len(X) // (self.n_splits * 2)  # Heuristic

        log_message(
            f"Trial {trial_number}: Starting blocked time series CV with block_size={block_size}"
        )

        # Ensure temporal ordering
        sort_idx = X.index.argsort()
        X_sorted = X.iloc[sort_idx]
        y_sorted = y.iloc[sort_idx]

        # Create holdout set
        X_dev, X_holdout, y_dev, y_holdout = self.temporal_train_test_split(
            X_sorted, y_sorted, self.test_size
        )

        cv_scores = {"mse": [], "mae": [], "r2": [], "adj_r2": []}

        # Create blocks
        n_blocks = len(X_dev) // block_size
        if n_blocks < self.n_splits + 1:
            log_message(
                f"Trial {trial_number}: Not enough blocks, falling back to walk-forward"
            )
            return self.walk_forward_validation(
                X, y, pipeline_factory, trial, trial_number
            )

        # Blocked CV loop
        for fold in range(self.n_splits):
            log_message(
                f"Trial {trial_number}: Blocked CV fold {fold+1}/{self.n_splits}"
            )

            # Calculate block indices
            test_block_start = (fold + 1) * block_size
            test_block_end = min((fold + 2) * block_size, len(X_dev))

            # Training set: all blocks before test block
            train_end = test_block_start

            if train_end < block_size:  # Need minimum training data
                continue

            X_train_fold = X_dev.iloc[:train_end]
            y_train_fold = y_dev.iloc[:train_end]
            X_val_fold = X_dev.iloc[test_block_start:test_block_end]
            y_val_fold = y_dev.iloc[test_block_start:test_block_end]

            if len(X_val_fold) == 0:
                continue

            try:
                # Create and fit pipeline
                pipeline = pipeline_factory(trial, X_train_fold, y_train_fold)
                pipeline.fit(X_train_fold, y_train_fold)

                # Predict on validation set
                y_pred_fold = pipeline.predict(X_val_fold)

                # Calculate metrics
                fold_mse = mean_squared_error(y_val_fold, y_pred_fold)
                fold_mae = mean_absolute_error(y_val_fold, y_pred_fold)
                fold_r2 = r2_score(y_val_fold, y_pred_fold)
                fold_adj_r2 = adjusted_r2_score(y_val_fold, y_pred_fold, n_features)

                cv_scores["mse"].append(fold_mse)
                cv_scores["mae"].append(fold_mae)
                cv_scores["r2"].append(fold_r2)
                cv_scores["adj_r2"].append(fold_adj_r2)

                log_message(
                    f"Trial {trial_number}, Fold {fold+1}: "
                    f"MSE={fold_mse:.6f}, R²={fold_r2:.6f}"
                )

                # Report for pruning
                trial.report(fold_mse, step=fold)
                if trial.should_prune():
                    log_message(f"Trial {trial_number}: Pruned at fold {fold+1}")
                    raise optuna.exceptions.TrialPruned()

            except optuna.exceptions.TrialPruned:
                raise
            except Exception as e:
                log_message(f"Trial {trial_number}, Fold {fold+1} failed: {str(e)}")
                continue

        # Check if we have successful folds
        if len(cv_scores["mse"]) == 0:
            log_message(f"Trial {trial_number}: No successful blocked CV folds")
            raise Exception("No successful blocked CV folds")

        # Calculate mean CV performance
        mean_cv_mse = np.mean(cv_scores["mse"])
        mean_cv_mae = np.mean(cv_scores["mae"])
        mean_cv_r2 = np.mean(cv_scores["r2"])
        mean_cv_adj_r2 = np.mean(cv_scores["adj_r2"])

        # Final evaluation (same as walk-forward)
        try:
            final_pipeline = pipeline_factory(trial, X_dev, y_dev)
            final_pipeline.fit(X_dev, y_dev)

            y_holdout_pred = final_pipeline.predict(X_holdout)
            y_dev_pred = final_pipeline.predict(X_dev)

            # Metrics
            holdout_test_mse = mean_squared_error(y_holdout, y_holdout_pred)
            holdout_test_mae = mean_absolute_error(y_holdout, y_holdout_pred)
            holdout_test_r2 = r2_score(y_holdout, y_holdout_pred)
            holdout_test_adj_r2 = adjusted_r2_score(
                y_holdout, y_holdout_pred, n_features
            )

            holdout_train_mse = mean_squared_error(y_dev, y_dev_pred)
            holdout_train_mae = mean_absolute_error(y_dev, y_dev_pred)
            holdout_train_r2 = r2_score(y_dev, y_dev_pred)
            holdout_train_adj_r2 = adjusted_r2_score(y_dev, y_dev_pred, n_features)

            # Results
            holdout_results = pd.DataFrame(
                {
                    "actual_test": y_holdout.values,
                    "predicted_test": y_holdout_pred,
                    "timestamp_test": y_holdout.index,
                }
            )

            train_results = pd.DataFrame(
                {
                    "actual_train": y_dev.values,
                    "predicted_train": y_dev_pred,
                    "timestamp_train": y_dev.index,
                }
            )

            results = {
                "pipeline": final_pipeline,
                "holdout_results": holdout_results,
                "train_results": train_results,
                "cv_mse": mean_cv_mse,
                "cv_mae": mean_cv_mae,
                "cv_r2": mean_cv_r2,
                "cv_adj_r2": mean_cv_adj_r2,
                "cv_mse_std": np.std(cv_scores["mse"]),
                "cv_r2_std": np.std(cv_scores["r2"]),
                "holdout_test_mse": holdout_test_mse,
                "holdout_test_mae": holdout_test_mae,
                "holdout_test_r2": holdout_test_r2,
                "holdout_test_adj_r2": holdout_test_adj_r2,
                "holdout_train_mse": holdout_train_mse,
                "holdout_train_mae": holdout_train_mae,
                "holdout_train_r2": holdout_train_r2,
                "holdout_train_adj_r2": holdout_train_adj_r2,
                "time_taken": time.time() - start_time,
                "n_features": n_features,
                "n_features_selected": self._get_selected_features_count(
                    final_pipeline
                ),
                "cv_scores_detail": cv_scores,
                "cv_method": "blocked",
                "block_size": block_size,
            }

            log_message(
                f"Trial {trial_number}: Blocked CV completed in {results['time_taken']:.2f}s"
            )
            return results

        except Exception as e:
            log_message(f"Trial {trial_number}: Final evaluation failed: {str(e)}")
            raise

    def evaluate_pipeline(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        pipeline_factory,
        trial,
        trial_number: int,
        cv_method: str = "walk_forward",
    ) -> Dict:
        """
        Main evaluation method that delegates to specific CV strategies.

        Args:
            X: Exogenous variables DataFrame with temporal index
            y: Target time series with temporal index
            pipeline_factory: Function to create pipeline
            trial: Optuna trial
            trial_number: Trial number for logging
            cv_method: CV method ("walk_forward" or "blocked")

        Returns:
            Dictionary with all evaluation results
        """
        if cv_method == "walk_forward":
            return self.walk_forward_validation(
                X, y, pipeline_factory, trial, trial_number
            )
        elif cv_method == "blocked":
            return self.blocked_time_series_cv(
                X, y, pipeline_factory, trial, trial_number
            )
        else:
            raise ValueError(f"Unknown CV method: {cv_method}")

    def forecast_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        pipeline_factory,
        trial,
        trial_number: int,
        forecast_horizon: int = 24,
    ) -> Dict:
        """
        Forecast-specific validation for multi-step ahead prediction.

        This method tests the model's ability to forecast multiple steps into
        the future, which is often the real-world use case.
        """
        start_time = time.time()

        log_message(
            f"Trial {trial_number}: Starting forecast validation with horizon={forecast_horizon}"
        )

        # Ensure temporal ordering
        sort_idx = X.index.argsort()
        X_sorted = X.iloc[sort_idx]
        y_sorted = y.iloc[sort_idx]

        # Reserve last part for multi-step forecasting
        forecast_split = len(X_sorted) - forecast_horizon
        if forecast_split < len(X_sorted) * 0.7:  # Need enough training data
            log_message(
                f"Trial {trial_number}: Insufficient data for forecast validation"
            )
            raise Exception("Insufficient data for forecast validation")

        X_train_full = X_sorted.iloc[:forecast_split]
        y_train_full = y_sorted.iloc[:forecast_split]
        X_forecast = X_sorted.iloc[forecast_split:]
        y_forecast = y_sorted.iloc[forecast_split:]

        # Fit model on training data
        pipeline = pipeline_factory(trial, X_train_full, y_train_full)
        pipeline.fit(X_train_full, y_train_full)

        # Multi-step forecast
        y_pred_forecast = pipeline.predict(X_forecast)

        # Calculate forecast metrics
        forecast_mse = mean_squared_error(y_forecast, y_pred_forecast)
        forecast_mae = mean_absolute_error(y_forecast, y_pred_forecast)
        forecast_r2 = r2_score(y_forecast, y_pred_forecast)

        # Also evaluate on training data
        y_pred_train = pipeline.predict(X_train_full)
        train_mse = mean_squared_error(y_train_full, y_pred_train)
        train_r2 = r2_score(y_train_full, y_pred_train)

        results = {
            "pipeline": pipeline,
            "forecast_mse": forecast_mse,
            "forecast_mae": forecast_mae,
            "forecast_r2": forecast_r2,
            "train_mse": train_mse,
            "train_r2": train_r2,
            "forecast_horizon": forecast_horizon,
            "time_taken": time.time() - start_time,
            "forecast_results": pd.DataFrame(
                {
                    "actual": y_forecast.values,
                    "predicted": y_pred_forecast,
                    "timestamp": y_forecast.index,
                }
            ),
        }

        log_message(
            f"Trial {trial_number}: Forecast validation completed. "
            f"Forecast MSE: {forecast_mse:.6f}, Train MSE: {train_mse:.6f}"
        )

        return results
