"""Database configuration and connection management for Optuna studies."""

import os
from enum import Enum
from typing import Optional

import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler


class DatabaseType(Enum):
    SQLITE = "sqlite"
    MYSQL = "mysql"


class DatabaseConfig:
    """Configuration class for database connections."""

    def __init__(
        self,
        db_type: DatabaseType = DatabaseType.SQLITE,
        db_path: Optional[str] = None,
        mysql_config: Optional[dict] = None,
        study_name: str = "co2_prediction",
    ):
        self.db_type = db_type
        self.study_name = study_name

        if db_type == DatabaseType.SQLITE:
            self.db_path = db_path or "optuna_study.db"
            self.storage_url = f"sqlite:///{self.db_path}"
        elif db_type == DatabaseType.MYSQL:
            if not mysql_config:
                # Try to get from environment
                mysql_config = {
                    "user": os.environ.get("MYSQL_USER", "optuna_user"),
                    "password": os.environ.get("MYSQL_PASSWORD", "your_password"),
                    "host": os.environ.get("MYSQL_HOST", "localhost"),
                    "database": os.environ.get("MYSQL_DATABASE", "optuna_db"),
                }

            self.storage_url = (
                f"mysql://{mysql_config['user']}:{mysql_config['password']}"
                f"@{mysql_config['host']}/{mysql_config['database']}"
            )
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Create configuration from environment variables."""
        storage_url = os.environ.get("OPTUNA_STORAGE")

        if not storage_url:
            # Default to SQLite
            return cls(db_type=DatabaseType.SQLITE)

        if storage_url.startswith("sqlite"):
            db_path = storage_url.replace("sqlite:///", "")
            return cls(db_type=DatabaseType.SQLITE, db_path=db_path)
        elif storage_url.startswith("mysql"):
            # Parse MySQL URL manually or use the provided URL directly
            config = cls(db_type=DatabaseType.MYSQL)
            config.storage_url = storage_url
            return config
        else:
            raise ValueError(f"Unsupported storage URL: {storage_url}")


class StudyManager:
    """Manages Optuna study creation and connection."""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._study: Optional[optuna.Study] = None

    def get_study(self) -> optuna.Study:
        """Get or create the Optuna study."""
        if self._study is not None:
            return self._study

        try:
            # Try to load existing study
            self._study = optuna.load_study(
                study_name=self.config.study_name, storage=self.config.storage_url
            )
            print(f"Connected to existing study '{self.config.study_name}'")
        except Exception:
            # Create new study
            print(f"Creating new study '{self.config.study_name}'")

            sampler = TPESampler(multivariate=True, seed=0, n_startup_trials=5)
            pruner = HyperbandPruner(
                min_resource=5, max_resource=15, reduction_factor=3
            )

            self._study = optuna.create_study(
                study_name=self.config.study_name,
                storage=self.config.storage_url,
                direction="minimize",
                sampler=sampler,
                pruner=pruner,
                load_if_exists=True,
            )

        return self._study

    def optimize(self, objective, n_trials: int = 10, timeout: Optional[int] = None):
        """Run optimization with the study."""
        study = self.get_study()

        study.optimize(
            objective,
            n_trials=n_trials,
            gc_after_trial=True,
            show_progress_bar=True,
            timeout=timeout,
        )

        return study
