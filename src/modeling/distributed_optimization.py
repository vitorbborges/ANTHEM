import json
import os
import time
from datetime import datetime

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from pipeline_factory import create_pipeline  # Use the faster pipeline factory
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

# Get database connection string from environment variable
DB_CONNECTION = os.environ.get(
    "OPTUNA_STORAGE", "mysql://optuna_user:your_password@localhost/optuna_db"
)
STUDY_NAME = "co2_prediction"  # Use a new study name to avoid conflicts

# Create a directory for metrics and logs
os.makedirs("metrics", exist_ok=True)
log_file = f"metrics/optimization_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"


def log_message(message):
    """Log a message to both console and file"""
    print(message)
    with open(log_file, "a") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")


# Define objective function for the distributed optimization
def objective(trial):
    """
    Optuna objective function that evaluates a pipeline created by the factory.
    Uses 5-fold cross-validation instead of a single validation split.
    """
    start_time = time.time()

    # Load data with a timeout
    try:
        log_message(f"Trial {trial.number}: Loading data...")
        df = pd.read_parquet("data/processed_data/S3-coords.parquet")
    except FileNotFoundError:
        log_message("Error: S3-coords.parquet not found. Please check the file path.")
        raise optuna.exceptions.TrialPruned()

    # Extract features and target, keep only 50% of features to speed up computation
    columns_to_use = [
        col for col in df.columns if df[col].dtype in ("float64", "int64")
    ]

    X = df[columns_to_use].dropna()
    y = X.pop("CO2")

    # Create a proper train/test split
    log_message(f"Trial {trial.number}: Creating train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    try:
        # Create pipeline with the trial parameters
        log_message(f"Trial {trial.number}: Creating pipeline...")
        pipeline = create_pipeline(trial, X_train)

        # Implement 10-fold cross-validation
        log_message(f"Trial {trial.number}: Performing 10-fold cross-validation...")
        kf = KFold(n_splits=10, shuffle=True, random_state=0)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # Fit on training fold
            log_message(f"Trial {trial.number}: Fitting on fold {fold+1}/10...")
            pipeline.fit(X_fold_train, y_fold_train)

            # Evaluate on validation fold
            y_fold_pred = pipeline.predict(X_fold_val)
            fold_mse = mean_squared_error(y_fold_val, y_fold_pred)
            cv_scores.append(fold_mse)

            # Report intermediate result for pruning
            trial.report(fold_mse, step=fold)
            if trial.should_prune():
                log_message(
                    f"Trial {trial.number}: Pruned at fold {fold+1} with MSE = {fold_mse:.6f}"
                )
                raise optuna.exceptions.TrialPruned()

        # Calculate mean cross-validation score
        mean_cv_mse = np.mean(cv_scores)
        log_message(f"Trial {trial.number}: Mean CV MSE = {mean_cv_mse:.6f}")

        # Now fit on full training data
        log_message(f"Trial {trial.number}: Fitting on full training data...")
        pipeline.fit(X_train, y_train)

        # Evaluate on the held-out test set
        y_pred = pipeline.predict(X_test)
        test_mse = mean_squared_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)

        # Calculate time taken
        time_taken = time.time() - start_time

        # Log the results
        log_message(f"Trial {trial.number}: Completed in {time_taken:.2f}s")
        log_message(
            f"Trial {trial.number}: Mean CV MSE = {mean_cv_mse:.6f}, Test MSE = {test_mse:.6f}, R² = {test_r2:.6f}"
        )

        # Store additional metrics in the trial user attributes
        trial.set_user_attr("test_mse", test_mse)
        trial.set_user_attr("test_r2", test_r2)
        trial.set_user_attr("time_taken", time_taken)
        trial.set_user_attr("cv_scores", cv_scores)

        return mean_cv_mse  # Use mean CV score for optimization

    except Exception as e:
        log_message(
            f"Trial {trial.number} failed after {time.time() - start_time:.2f}s due to error: {str(e)}"
        )
        # Prune the trial if an error occurs
        raise optuna.exceptions.TrialPruned()


def create_study():
    """Create a new Optuna study with aggressive pruning."""
    try:
        # Try to load existing study
        study = optuna.load_study(study_name=STUDY_NAME, storage=DB_CONNECTION)
        log_message(f"Connected to existing study '{STUDY_NAME}'")
    except Exception:
        # Create a new study if it doesn't exist
        log_message(f"Creating new study '{STUDY_NAME}'")

        # Use TPESampler with multivariate=True for better performance
        sampler = TPESampler(multivariate=True, seed=0, n_startup_trials=5)

        # Use HyperbandPruner for more aggressive pruning
        pruner = HyperbandPruner(min_resource=5, max_resource=5, reduction_factor=3)

        study = optuna.create_study(
            study_name=STUDY_NAME,
            storage=DB_CONNECTION,
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

    return study


if __name__ == "__main__":
    # Configure Optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # Reduce Optuna's verbosity

    # Set a maximum number of trials per process - more trials but shorter timeout
    n_trials_per_process = 1

    try:
        # Load or create study
        study = create_study()

        # Print current study status
        completed_trials = len(
            [t for t in study.trials if t.state == TrialState.COMPLETE]
        )
        log_message(f"Current number of trials: {len(study.trials)}")
        log_message(f"Completed trials: {completed_trials}")

        # Add worker ID to log messages
        worker_id = os.environ.get("WORKER_ID", "unknown")
        log_message(f"Starting worker ID: {worker_id}")

        # Run the optimization with a shorter timeout
        study.optimize(
            objective,
            n_trials=n_trials_per_process,
            gc_after_trial=True,  # Enable garbage collection after each trial
            show_progress_bar=True,
            timeout=600,  # 10-minute timeout per worker
        )

        # Print best results found so far by this process
        log_message("\nOptimization finished.")
        if study.best_trial:
            log_message("Best trial so far:")
            trial = study.best_trial
            log_message(f" Value (Mean CV MSE): {trial.value:.6f}")
            log_message(
                f" Test MSE: {trial.user_attrs.get('test_mse', 'Not available')}"
            )
            log_message(f" Test R²: {trial.user_attrs.get('test_r2', 'Not available')}")
            log_message(
                f" Time taken: {trial.user_attrs.get('time_taken', 'Not available'):.2f}s"
            )
            log_message(" Params: ")
            for key, value in trial.params.items():
                log_message(f" {key}: {value}")

            # Save local copy of best parameters
            result = {
                "best_mean_cv_mse": trial.value,
                "best_test_mse": trial.user_attrs.get("test_mse", None),
                "best_test_r2": trial.user_attrs.get("test_r2", None),
                "best_params": trial.params,
                "time_taken": trial.user_attrs.get("time_taken", None),
                "cv_scores": trial.user_attrs.get("cv_scores", None),
            }

            with open(f"metrics/best_params_worker_{worker_id}.json", "w") as f:
                json.dump(result, f, indent=2)

            log_message(
                f"\nBest parameters saved to metrics/best_params_worker_{worker_id}.json"
            )
        else:
            log_message("No successful trials were completed.")

    except Exception as e:
        log_message(f"Error during optimization: {str(e)}")
        log_message("\nPlease check that:")
        log_message("1. MySQL server is running")
        log_message("2. The database and study exist")
        log_message(
            "3. The correct credentials are provided in the OPTUNA_STORAGE environment variable"
        )
