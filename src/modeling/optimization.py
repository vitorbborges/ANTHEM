# src/modeling/optimization.py
import json
import os
import pickle
import time
from datetime import datetime
from typing import Dict, Optional

import optuna
from optuna.exceptions import TrialPruned

from .cv_handler import CrossValidationHandler
from .db_handler import DatabaseHandler
from .pipeline_factory import create_pipeline
from .utils import load_data, log_message


class OptimizationRunner:
    """Main class for running hyperparameter optimization."""

    def __init__(
        self,
        study_name: str = "co2_prediction",
        storage_type: str = "sqlite",
        mysql_url: Optional[str] = None,
        data_path: str = "data/processed_data/S3-coords.parquet",
        outer_folds: int = 5,
        inner_folds: int = 3,
        test_size: float = 0.2,
    ):
        """
        Initialize the optimization runner.

        Args:
            study_name: Name of the Optuna study
            storage_type: 'sqlite' or 'mysql'
            mysql_url: MySQL connection string (only needed for mysql)
            data_path: Path to the data file
            outer_folds: Number of outer CV folds
            inner_folds: Number of inner CV folds
            test_size: Size of the holdout test set
        """
        self.study_name = study_name
        self.data_path = data_path

        # Initialize handlers
        self.db_handler = DatabaseHandler(study_name, storage_type, mysql_url)
        self.cv_handler = CrossValidationHandler(outer_folds, inner_folds, test_size)

        # Load data once
        self.X, self.y = load_data(data_path)

        # Worker ID for logging
        self.worker_id = os.environ.get("WORKER_ID", "main")

    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function with nested cross-validation.
        """
        trial_number = trial.number
        log_message(f"Trial {trial_number}: Starting optimization...")

        try:
            # Run nested cross-validation
            results = self.cv_handler.nested_cv_evaluate(
                self.X, self.y, create_pipeline, trial, trial_number
            )

            # Save model and results
            self._save_trial_results(trial, results)

            # Set user attributes for detailed tracking
            self._set_trial_attributes(trial, results)

            log_message(
                f"Trial {trial_number}: CV MSE = {results['cv_mse']:.6f}, "
                f"Holdout Test MSE = {results['holdout_test_mse']:.6f}"
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

        # Save the trained model
        model_path = f"models/model_trial_{trial_number}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(results["pipeline"], f)

        # Save holdout results (test set)
        results["holdout_results"].to_csv(
            f"metrics/holdout_test_results_trial_{trial_number}.csv", index=False
        )

        # Save training results if available
        if "train_results" in results:
            results["train_results"].to_csv(
                f"metrics/holdout_train_results_trial_{trial_number}.csv", index=False
            )

        log_message(f"Trial {trial_number}: Results saved")

    def _set_trial_attributes(self, trial: optuna.Trial, results: Dict):
        """Set user attributes for trial tracking."""
        for key, value in results.items():
            if key not in [
                "pipeline",
                "holdout_results",
                "train_results",
            ]:  # Skip non-serializable objects
                trial.set_user_attr(key, value)

    def run_optimization(
        self,
        n_trials: int = 10,
        timeout: Optional[int] = None,
        show_progress: bool = True,
    ) -> optuna.Study:
        """
        Run the hyperparameter optimization.

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
            log_message(f"Optimization completed in {total_time:.2f}s")

            # Save best results only if we have completed trials
            if len([t for t in study.trials if t.state.name == "COMPLETE"]) > 0:
                self._save_best_results(study)
            else:
                log_message("No trials completed successfully")

            return study

        except KeyboardInterrupt:
            log_message("Optimization interrupted by user")
            return study
        except Exception as e:
            log_message(f"Optimization failed: {str(e)}")
            raise

    def _save_best_results(self, study: optuna.Study):
        """Save the best trial results."""
        if study.best_trial is None:
            log_message("No successful trials completed")
            return

        trial = study.best_trial
        log_message(f"Best trial: {trial.number}")
        log_message(f"Best CV MSE: {trial.value:.6f}")

        # Create summary of best results
        result_summary = {
            "trial_number": trial.number,
            "best_cv_mse": trial.value,
            "best_params": trial.params,
            "worker_id": self.worker_id,
            "timestamp": datetime.now().isoformat(),
            "study_info": self.db_handler.get_study_info(study),
        }

        # Add user attributes if available
        for attr_name in [
            "holdout_test_mse",
            "holdout_test_r2",
            "holdout_test_adj_r2",
            "holdout_train_mse",
            "holdout_train_r2",
            "cv_r2",
            "cv_adj_r2",
            "time_taken",
            "n_features",
        ]:
            if attr_name in trial.user_attrs:
                result_summary[attr_name] = trial.user_attrs[attr_name]

        # Save to file
        results_file = f"metrics/best_params_worker_{self.worker_id}.json"
        with open(results_file, "w") as f:
            json.dump(result_summary, f, indent=2)

        log_message(f"Best results saved to {results_file}")

        # Log key metrics
        if "holdout_test_mse" in trial.user_attrs:
            log_message(f"Holdout Test MSE: {trial.user_attrs['holdout_test_mse']:.6f}")
        if "holdout_test_r2" in trial.user_attrs:
            log_message(f"Holdout Test R²: {trial.user_attrs['holdout_test_r2']:.6f}")

        log_message("Best parameters:")
        for key, value in trial.params.items():
            log_message(f"  {key}: {value}")
