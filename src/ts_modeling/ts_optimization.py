# src/modeling/ts_optimization.py
"""
Time Series Optimization Runner.
Updated optimization.py for proper time series modeling with exogenous variables.
"""

import json
import os
import pickle
import time
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import optuna
import pandas as pd
from optuna.exceptions import TrialPruned

from src.ts_modeling.db_handler import DatabaseHandler
from src.ts_modeling.ts_cv_handler import TimeSeriesCVHandler
from src.ts_modeling.ts_pipeline_factory import create_ts_pipeline
from src.ts_modeling.utils import log_message


class TimeSeriesOptimizationRunner:
    """Main class for running time series hyperparameter optimization."""

    def __init__(
        self,
        study_name: str = "co2_ts_prediction",
        storage_type: str = "sqlite",
        mysql_url: Optional[str] = None,
        data_path: str = "data/processed_data/S3-coords.parquet",
        n_splits: int = 5,
        test_size: float = 0.2,
        cv_method: str = "walk_forward",
        gap: int = 0,
        target_column: str = "CO2",
        time_column: Optional[str] = None,
        exclude_columns: Optional[list] = None,
    ):
        """
        Initialize the time series optimization runner.

        Args:
            study_name: Name of the Optuna study
            storage_type: 'sqlite' or 'mysql'
            mysql_url: MySQL connection string (only needed for mysql)
            data_path: Path to the data file
            n_splits: Number of CV splits
            test_size: Size of the holdout test set
            cv_method: CV method ("walk_forward" or "blocked")
            gap: Gap between train/test in CV
            target_column: Name of target column
            time_column: Name of time column (if None, uses index)
            exclude_columns: Columns to exclude from exogenous variables
        """
        self.study_name = study_name
        self.data_path = data_path
        self.target_column = target_column
        self.time_column = time_column
        self.exclude_columns = exclude_columns or []
        self.cv_method = cv_method

        # Initialize handlers
        self.db_handler = DatabaseHandler(study_name, storage_type, mysql_url)
        self.cv_handler = TimeSeriesCVHandler(
            n_splits=n_splits, test_size=test_size, gap=gap
        )

        # Load and prepare data
        self.X, self.y, self.data_info = self._load_and_prepare_data()

        # Worker ID for logging
        self.worker_id = os.environ.get("WORKER_ID", "main")

    def _load_and_prepare_data(self) -> tuple:
        """Load and prepare time series data with exogenous variables."""
        try:
            log_message(f"Loading time series data from {self.data_path}...")
            df = pd.read_parquet(self.data_path)

            log_message(f"Raw data shape: {df.shape}")
            log_message(f"Columns: {list(df.columns)}")

            # Handle time column
            if self.time_column and self.time_column in df.columns:
                df = df.set_index(self.time_column)
                log_message(f"Set {self.time_column} as index")
            elif not isinstance(df.index, pd.DatetimeIndex):
                # If no time column and index is not datetime, create sequential index
                log_message("Creating sequential time index")
                df.index = pd.RangeIndex(len(df))

            # Sort by index to ensure temporal order
            df = df.sort_index()
            log_message("Data sorted by temporal index")

            # Extract target variable
            if self.target_column not in df.columns:
                raise ValueError(
                    f"Target column '{self.target_column}' not found in data"
                )

            y = df[self.target_column].copy()

            # Remove target and excluded columns for exogenous variables
            exclude_cols = [self.target_column] + self.exclude_columns
            available_cols = [col for col in df.columns if col not in exclude_cols]

            # Filter to numeric columns only
            numeric_cols = []
            for col in available_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_cols.append(col)
                else:
                    log_message(f"Excluding non-numeric column: {col}")

            X = df[numeric_cols].copy()

            # Handle missing values
            initial_length = len(X)

            # Drop rows where target is missing
            valid_target_mask = ~y.isna()
            X = X[valid_target_mask]
            y = y[valid_target_mask]

            # Handle missing values in exogenous variables
            if X.isna().any().any():
                log_message("Handling missing values in exogenous variables...")

                # Forward fill then backward fill
                X = X.fillna(method="ffill").fillna(method="bfill")

                # If still missing, drop remaining NaN rows
                complete_mask = ~X.isna().any(axis=1)
                X = X[complete_mask]
                y = y[complete_mask]

            log_message(
                f"After cleaning: {len(X)} samples, {X.shape[1]} exogenous variables"
            )
            log_message(
                f"Removed {initial_length - len(X)} samples due to missing values"
            )

            # Data info for logging
            data_info = {
                "n_samples": len(X),
                "n_exog_features": X.shape[1],
                "target_column": self.target_column,
                "exog_features": list(X.columns),
                "time_range": (X.index.min(), X.index.max()),
                "target_stats": {
                    "mean": float(y.mean()),
                    "std": float(y.std()),
                    "min": float(y.min()),
                    "max": float(y.max()),
                },
            }

            log_message(f"Target statistics: {data_info['target_stats']}")
            log_message(f"Exogenous features: {X.columns.tolist()}")

            return X, y, data_info

        except Exception as e:
            log_message(f"Error loading data: {str(e)}")
            raise

    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function for time series optimization.
        """
        trial_number = trial.number
        log_message(f"Trial {trial_number}: Starting time series optimization...")

        try:
            # Choose CV method for this trial (optional randomization)
            if self.cv_method == "auto":
                cv_method = trial.suggest_categorical(
                    "cv_method", ["walk_forward", "blocked"]
                )
            else:
                cv_method = self.cv_method

            # Run time series cross-validation
            results = self.cv_handler.evaluate_pipeline(
                self.X, self.y, create_ts_pipeline, trial, trial_number, cv_method
            )

            # Save model and results
            self._save_trial_results(trial, results)

            # Set user attributes for detailed tracking
            self._set_trial_attributes(trial, results)

            log_message(
                f"Trial {trial_number}: CV MSE = {results['cv_mse']:.6f} ± {results.get('cv_mse_std', 0):.6f}"
            )
            log_message(
                f"Trial {trial_number}: Holdout Test MSE = {results['holdout_test_mse']:.6f}"
            )

            return results["cv_mse"]  # Optimize based on CV performance

        except TrialPruned:
            log_message(f"Trial {trial_number}: Pruned")
            raise
        except Exception as e:
            log_message(f"Trial {trial_number}: Failed with error: {str(e)}")
            raise TrialPruned()

    def _save_trial_results(self, trial: optuna.Trial, results: Dict):
        """Save model and predictions for a trial."""
        trial_number = trial.number

        # Save the trained pipeline
        model_path = f"models/ts_model_trial_{trial_number}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(results["pipeline"], f)

        # Save holdout results
        results["holdout_results"].to_csv(
            f"metrics/ts_holdout_test_results_trial_{trial_number}.csv", index=False
        )

        # Save training results
        if "train_results" in results:
            results["train_results"].to_csv(
                f"metrics/ts_holdout_train_results_trial_{trial_number}.csv",
                index=False,
            )

        # Save detailed CV scores
        if "cv_scores_detail" in results:
            cv_scores_df = pd.DataFrame(results["cv_scores_detail"])
            cv_scores_df.to_csv(
                f"metrics/ts_cv_scores_trial_{trial_number}.csv", index=False
            )

        log_message(f"Trial {trial_number}: Results saved")

    def _set_trial_attributes(self, trial: optuna.Trial, results: Dict):
        """Set user attributes for trial tracking."""
        for key, value in results.items():
            if key not in [
                "pipeline",
                "holdout_results",
                "train_results",
                "cv_scores_detail",
            ]:  # Skip non-serializable objects
                try:
                    # Ensure value is serializable
                    if isinstance(value, (int, float, str, bool, type(None))):
                        trial.set_user_attr(key, value)
                    elif isinstance(value, (list, tuple)) and len(value) < 10:
                        trial.set_user_attr(key, list(value))
                    elif isinstance(value, dict) and len(str(value)) < 1000:
                        trial.set_user_attr(key, value)
                except:
                    pass  # Skip non-serializable attributes

    def run_optimization(
        self,
        n_trials: int = 20,
        timeout: Optional[int] = None,
        show_progress: bool = True,
    ) -> optuna.Study:
        """
        Run the time series hyperparameter optimization.

        Args:
            n_trials: Number of trials to run
            timeout: Timeout in seconds (optional)
            show_progress: Whether to show progress bar

        Returns:
            The completed Optuna study
        """
        # Configure Optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Create study
        study = self.db_handler.create_study()

        # Log initial study info
        try:
            study_info = self.db_handler.get_study_info(study)
            log_message(f"Study info: {study_info}")
        except Exception as e:
            log_message(f"Could not get study info: {str(e)}")

        log_message(f"Worker ID: {self.worker_id}")
        log_message(f"Data info: {self.data_info}")

        start_time = time.time()

        try:
            # Run optimization
            study.optimize(
                self.objective,
                n_trials=n_trials,
                timeout=timeout,
                gc_after_trial=True,
                show_progress_bar=show_progress,
            )

            # Log completion
            total_time = time.time() - start_time
            log_message(f"Time series optimization completed in {total_time:.2f}s")

            # Save best results only if we have completed trials
            completed_trials = [t for t in study.trials if t.state.name == "COMPLETE"]
            if len(completed_trials) > 0:
                self._save_best_results(study)

                # Log best model info
                best_trial = study.best_trial
                log_message(
                    f"Best model type: {best_trial.params.get('model_type', 'Unknown')}"
                )
                log_message(
                    f"Best feature selection: {best_trial.params.get('use_feature_selection', 'Unknown')}"
                )
                if "n_features_selected" in best_trial.user_attrs:
                    log_message(
                        f"Features selected: {best_trial.user_attrs['n_features_selected']}/{self.data_info['n_exog_features']}"
                    )
            else:
                log_message("No trials completed successfully")

            return study

        except KeyboardInterrupt:
            log_message("Optimization interrupted by user")
            return study
        except Exception as e:
            log_message(f"Optimization failed: {str(e)}")
            raise

    # Quick fix for JSON serialization error in ts_optimization.py

    def _save_best_results(self, study: optuna.Study):
        """FIXED: Save the best trial results with JSON serialization fix."""
        if study.best_trial is None:
            log_message("No successful trials completed")
            return

        trial = study.best_trial
        log_message(f"Best trial: {trial.number}")
        log_message(f"Best CV MSE: {trial.value:.6f}")

        # FIXED: Handle timestamp serialization
        data_info_serializable = self.data_info.copy()
        if "time_range" in data_info_serializable:
            # Convert timestamps to strings
            time_range = data_info_serializable["time_range"]
            data_info_serializable["time_range"] = [
                (
                    time_range[0].isoformat()
                    if hasattr(time_range[0], "isoformat")
                    else str(time_range[0])
                ),
                (
                    time_range[1].isoformat()
                    if hasattr(time_range[1], "isoformat")
                    else str(time_range[1])
                ),
            ]

        # Create comprehensive summary
        result_summary = {
            "trial_number": trial.number,
            "best_cv_mse": trial.value,
            "best_params": trial.params,
            "worker_id": self.worker_id,
            "timestamp": datetime.now().isoformat(),
            "study_info": self.db_handler.get_study_info(study),
            "data_info": data_info_serializable,  # FIXED: Use serializable version
            "cv_method": self.cv_method,
        }

        # Add user attributes with type checking
        for attr_name in [
            "holdout_test_mse",
            "holdout_test_r2",
            "holdout_test_adj_r2",
            "holdout_train_mse",
            "holdout_train_r2",
            "cv_r2",
            "cv_adj_r2",
            "cv_mse_std",
            "cv_r2_std",
            "time_taken",
            "n_features",
            "n_features_selected",
            "cv_method",
        ]:
            if attr_name in trial.user_attrs:
                value = trial.user_attrs[attr_name]
                # FIXED: Ensure JSON serializable
                if isinstance(value, (int, float, str, bool, type(None))):
                    result_summary[attr_name] = value
                else:
                    result_summary[attr_name] = str(value)

        # Save to file
        results_file = f"metrics/ts_best_params_worker_{self.worker_id}.json"
        with open(results_file, "w") as f:
            json.dump(
                result_summary, f, indent=2, default=str
            )  # FIXED: Add default=str

        log_message(f"Best results saved to {results_file}")

        # Log key metrics
        if "holdout_test_mse" in trial.user_attrs:
            log_message(f"Holdout Test MSE: {trial.user_attrs['holdout_test_mse']:.6f}")
        if "holdout_test_r2" in trial.user_attrs:
            log_message(f"Holdout Test R²: {trial.user_attrs['holdout_test_r2']:.6f}")

        log_message("Best parameters:")
        for key, value in trial.params.items():
            log_message(f"  {key}: {value}")

    def run_forecast_evaluation(self, forecast_horizon: int = 24) -> Dict:
        """
        Run forecast evaluation on the best model.

        Args:
            forecast_horizon: Number of steps to forecast

        Returns:
            Forecast evaluation results
        """
        log_message(f"Running forecast evaluation with horizon={forecast_horizon}")

        # Create a dummy trial for the best params (if available)
        try:
            study = self.db_handler.create_study()
            if study.best_trial is None:
                log_message("No best trial available for forecast evaluation")
                return {}

            # Create trial with best parameters
            class DummyTrial:
                def __init__(self, params):
                    self.params = params
                    self.number = -1

                def suggest_categorical(self, name, choices):
                    return self.params.get(name, choices[0])

                def suggest_int(self, name, low, high):
                    return self.params.get(name, (low + high) // 2)

                def suggest_float(self, name, low, high, log=False):
                    return self.params.get(name, (low + high) / 2)

            dummy_trial = DummyTrial(study.best_trial.params)

            # Run forecast validation
            results = self.cv_handler.forecast_validation(
                self.X, self.y, create_ts_pipeline, dummy_trial, -1, forecast_horizon
            )

            # Save forecast results
            forecast_file = f"metrics/ts_forecast_evaluation_{forecast_horizon}h.json"
            forecast_summary = {
                "forecast_horizon": forecast_horizon,
                "forecast_mse": results["forecast_mse"],
                "forecast_mae": results["forecast_mae"],
                "forecast_r2": results["forecast_r2"],
                "train_mse": results["train_mse"],
                "train_r2": results["train_r2"],
                "best_params": study.best_trial.params,
                "timestamp": datetime.now().isoformat(),
            }

            with open(forecast_file, "w") as f:
                json.dump(forecast_summary, f, indent=2)

            # Save detailed forecast results
            results["forecast_results"].to_csv(
                f"metrics/ts_forecast_results_{forecast_horizon}h.csv", index=False
            )

            log_message(
                f"Forecast evaluation completed. MSE: {results['forecast_mse']:.6f}"
            )
            log_message(f"Forecast results saved to {forecast_file}")

            return results

        except Exception as e:
            log_message(f"Forecast evaluation failed: {str(e)}")
            return {}
