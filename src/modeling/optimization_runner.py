"""
Refactored optimization runner with modular design.
"""

import json
import os
import pickle
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import numpy as np
import optuna
import pandas as pd
from optuna.trial import TrialState

from src.modeling.cv_manager import CrossValidationManager
from src.modeling.db_config import DatabaseConfig, DatabaseType, StudyManager
from src.modeling.pipeline_factory import create_pipeline


class OptimizationLogger:
    """Handles logging for optimization runs."""

    def __init__(self, log_file: Optional[str] = None):
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"metrics/optimization_log_{timestamp}.txt"

        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log(self, message: str):
        """Log a message to both console and file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"{timestamp} - {message}"
        print(full_message)

        with open(self.log_file, "a") as f:
            f.write(full_message + "\n")


class OptimizationRunner:
    """Main class for running hyperparameter optimization."""

    def __init__(
        self,
        db_config: DatabaseConfig,
        cv_manager: CrossValidationManager,
        logger: OptimizationLogger,
        data_path: str = "data/processed_data/S3-coords.parquet",
        models_dir: str = "models",
        metrics_dir: str = "metrics",
    ):
        self.db_config = db_config
        self.cv_manager = cv_manager
        self.logger = logger
        self.data_path = data_path
        self.models_dir = models_dir
        self.metrics_dir = metrics_dir

        # Create directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

        # Initialize study manager
        self.study_manager = StudyManager(db_config)

        # Data cache
        self._data_cache = None

    def load_data(self) -> tuple[pd.DataFrame, pd.Series, int]:
        """Load and prepare data."""
        if self._data_cache is not None:
            return self._data_cache

        try:
            self.logger.log(f"Loading data from {self.data_path}")
            df = pd.read_parquet(self.data_path)
        except FileNotFoundError:
            self.logger.log(f"Error: {self.data_path} not found")
            raise

        # Extract features and target
        columns_to_use = [
            col for col in df.columns if df[col].dtype in ("float64", "int64")
        ]
        remove_cols = ["P", "PM1", "PM10", "RH", "T", "VOC"]
        columns_to_use = [col for col in columns_to_use if col not in remove_cols]

        X = df[columns_to_use].dropna()
        y = X.pop("CO2")
        n_features = X.shape[1]

        self.logger.log(f"Data loaded: {len(X)} samples, {n_features} features")
        self._data_cache = (X, y, n_features)
        return X, y, n_features

    def create_objective_function(self) -> Callable:
        """Create the objective function for Optuna."""

        def objective(trial: optuna.Trial) -> float:
            start_time = time.time()

            # Load data
            X, y, n_features = self.load_data()

            # Prepare data splits
            X_dev, X_holdout, y_dev, y_holdout, strata_dev, _ = (
                self.cv_manager.prepare_data(X, y)
            )

            self.logger.log(
                f"Trial {trial.number}: Created holdout set with {len(X_holdout)} samples"
            )

            # Pipeline factory function
            def pipeline_factory(trial_obj, X_data):
                return create_pipeline(trial_obj, X_data)

            # Nested cross-validation
            try:
                outer_metrics = self.cv_manager.nested_cross_validation(
                    X_dev,
                    y_dev,
                    strata_dev,
                    pipeline_factory,
                    trial,
                    n_features,
                    self.logger.log,
                )

                if len(outer_metrics.mse_scores) == 0:
                    self.logger.log(
                        f"Trial {trial.number}: No successful outer folds, pruning trial"
                    )
                    raise optuna.exceptions.TrialPruned()

                # Get mean CV metrics
                cv_metrics = outer_metrics.get_mean_metrics()

                # Final evaluation
                final_results = self.cv_manager.final_evaluation(
                    X_dev,
                    y_dev,
                    X_holdout,
                    y_holdout,
                    pipeline_factory,
                    trial,
                    n_features,
                    self.logger.log,
                )

                # Save model if evaluation was successful
                if final_results["final_pipeline"] is not None:
                    model_path = os.path.join(
                        self.models_dir, f"model_trial_{trial.number}.pkl"
                    )
                    with open(model_path, "wb") as f:
                        pickle.dump(final_results["final_pipeline"], f)
                    self.logger.log(
                        f"Trial {trial.number}: Model saved to {model_path}"
                    )

                    # Save predictions
                    if final_results["predictions"] is not None:
                        predictions_df = pd.DataFrame(final_results["predictions"])
                        predictions_path = os.path.join(
                            self.metrics_dir,
                            f"holdout_results_trial_{trial.number}.csv",
                        )
                        predictions_df.to_csv(predictions_path, index=False)

                # Calculate time taken
                time_taken = time.time() - start_time

                # Log results
                self.logger.log(f"Trial {trial.number}: Completed in {time_taken:.2f}s")
                self.logger.log(
                    f"Trial {trial.number}: Mean CV MSE = {cv_metrics['mean_mse']:.6f}"
                )

                # Store metrics in trial attributes
                trial.set_user_attr("cv_mse", cv_metrics["mean_mse"])
                trial.set_user_attr("cv_mae", cv_metrics["mean_mae"])
                trial.set_user_attr("cv_r2", cv_metrics["mean_r2"])
                trial.set_user_attr("cv_adj_r2", cv_metrics["mean_adj_r2"])
                trial.set_user_attr("time_taken", time_taken)
                trial.set_user_attr("n_features", n_features)

                # Store all CV scores
                all_scores = outer_metrics.get_all_scores()
                for key, scores in all_scores.items():
                    trial.set_user_attr(f"outer_cv_{key}", scores)

                # Store final evaluation metrics
                for key, value in final_results.items():
                    if key != "final_pipeline" and key != "predictions":
                        trial.set_user_attr(key, value)

                return cv_metrics["mean_mse"]

            except optuna.exceptions.TrialPruned:
                raise
            except Exception as e:
                self.logger.log(f"Trial {trial.number}: Failed with error: {str(e)}")
                raise optuna.exceptions.TrialPruned()

        return objective

    def run_optimization(
        self,
        n_trials: int = 10,
        timeout: Optional[int] = None,
        worker_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the optimization process."""

        # Configure Optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        worker_id = worker_id or os.environ.get("WORKER_ID", "unknown")
        self.logger.log(f"Starting optimization with worker ID: {worker_id}")

        # Create objective function
        objective = self.create_objective_function()

        # Run optimization
        study = self.study_manager.optimize(
            objective, n_trials=n_trials, timeout=timeout
        )

        # Get results
        completed_trials = len(
            [t for t in study.trials if t.state == TrialState.COMPLETE]
        )
        self.logger.log(
            f"Optimization finished. Total trials: {len(study.trials)}, Completed: {completed_trials}"
        )

        results = {
            "total_trials": len(study.trials),
            "completed_trials": completed_trials,
            "best_trial": None,
            "worker_id": worker_id,
        }

        if study.best_trial:
            best_trial = study.best_trial
            self.logger.log("Best trial found:")
            self.logger.log(f" Value (Mean CV MSE): {best_trial.value:.6f}")

            # Compile results
            best_results = {
                "best_cv_mse": best_trial.value,
                "best_params": best_trial.params,
                "worker_id": worker_id,
                "trial_number": best_trial.number,
            }

            # Add user attributes
            for key, value in best_trial.user_attrs.items():
                best_results[key] = value

            # Save results
            results_path = os.path.join(
                self.metrics_dir, f"best_params_worker_{worker_id}.json"
            )
            with open(results_path, "w") as f:
                json.dump(best_results, f, indent=2)

            self.logger.log(f"Best parameters saved to {results_path}")
            results["best_trial"] = best_results
        else:
            self.logger.log("No successful trials were completed.")

        return results


def main():
    """Main function for running optimization."""

    # Configuration
    worker_id = os.environ.get("WORKER_ID", "1")
    n_trials = int(os.environ.get("N_TRIALS", "10"))
    timeout = int(os.environ.get("TIMEOUT", "600"))  # 10 minutes default

    # Database configuration
    db_type_str = os.environ.get("DB_TYPE", "sqlite").lower()
    if db_type_str == "mysql":
        db_config = DatabaseConfig(db_type=DatabaseType.MYSQL)
    else:
        db_config = DatabaseConfig(db_type=DatabaseType.SQLITE)

    # Cross-validation configuration
    cv_manager = CrossValidationManager(
        outer_cv_folds=5, inner_cv_folds=3, test_size=0.2, random_state=42
    )

    # Logger
    logger = OptimizationLogger()

    # Runner
    runner = OptimizationRunner(
        db_config=db_config, cv_manager=cv_manager, logger=logger
    )

    try:
        # Run optimization
        results = runner.run_optimization(
            n_trials=n_trials, timeout=timeout, worker_id=worker_id
        )

        logger.log(f"Optimization completed successfully for worker {worker_id}")

    except Exception as e:
        logger.log(f"Error during optimization: {str(e)}")
        logger.log("Please check that:")
        logger.log("1. Database server is running (if using MySQL)")
        logger.log("2. Data file exists at the specified path")
        logger.log("3. Required environment variables are set")
        raise


if __name__ == "__main__":
    main()
