# src/modeling/db_handler.py
import os
from typing import Optional

import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from .utils import log_message


class DatabaseHandler:
    """Handles Optuna study creation and database connections."""

    def __init__(
        self,
        study_name: str = "co2_prediction",
        storage_type: str = "sqlite",
        mysql_url: Optional[str] = None,
    ):
        """
        Initialize database handler.

        Args:
            study_name: Name of the Optuna study
            storage_type: 'sqlite' or 'mysql'
            mysql_url: MySQL connection string (required if storage_type='mysql')
        """
        self.study_name = study_name
        self.storage_type = storage_type.lower()

        if self.storage_type == "mysql":
            if mysql_url is None:
                mysql_url = os.environ.get(
                    "OPTUNA_STORAGE",
                    "mysql://optuna_user:your_password@localhost/optuna_db",
                )
            self.storage_url = mysql_url
        elif self.storage_type == "sqlite":
            # Use default SQLite storage (optuna creates automatically)
            self.storage_url = None
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")

    def create_study(self) -> optuna.Study:
        """Create or load an Optuna study."""
        try:
            # Configure sampler and pruner
            sampler = TPESampler(multivariate=True, seed=0, n_startup_trials=5)
            pruner = HyperbandPruner(
                min_resource=5, max_resource=15, reduction_factor=3
            )

            if self.storage_type == "sqlite":
                log_message(f"Creating/loading SQLite study '{self.study_name}'")
                study = optuna.create_study(
                    study_name=self.study_name,
                    direction="minimize",
                    sampler=sampler,
                    pruner=pruner,
                    load_if_exists=True,
                )
            else:  # MySQL
                log_message(
                    f"Creating/loading MySQL study '{self.study_name}' at {self.storage_url}"
                )
                study = optuna.create_study(
                    study_name=self.study_name,
                    storage=self.storage_url,
                    direction="minimize",
                    sampler=sampler,
                    pruner=pruner,
                    load_if_exists=True,
                )

            return study

        except Exception as e:
            log_message(f"Error creating study: {str(e)}")
            if self.storage_type == "mysql":
                log_message("Please check that:")
                log_message("1. MySQL server is running")
                log_message("2. Database credentials are correct")
                log_message("3. Database exists and is accessible")
            raise

    def get_study_info(self, study: optuna.Study) -> dict:
        """Get information about the current study."""
        try:
            completed_trials = len(
                [t for t in study.trials if t.state == TrialState.COMPLETE]
            )

            info = {
                "total_trials": len(study.trials),
                "completed_trials": completed_trials,
                "pruned_trials": len(
                    [t for t in study.trials if t.state == TrialState.PRUNED]
                ),
                "failed_trials": len(
                    [t for t in study.trials if t.state == TrialState.FAIL]
                ),
                "storage_type": self.storage_type,
                "study_name": self.study_name,
            }

            if study.best_trial is not None:
                info["best_value"] = study.best_trial.value
                info["best_params"] = study.best_trial.params
            else:
                info["best_value"] = None
                info["best_params"] = None

            return info
        except Exception as e:
            # Return basic info if there's an issue accessing study details
            return {
                "storage_type": self.storage_type,
                "study_name": self.study_name,
                "error": str(e),
            }
