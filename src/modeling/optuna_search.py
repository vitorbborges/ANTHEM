import numpy as np
import optuna
import pandas as pd
from pipeline_factory import create_pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler


# Define objective function for the distributed optimization
def objective(trial):
    """
    Optuna objective function that evaluates a pipeline created by the factory.
    """
    # Load data (each process needs to load the data independently)
    try:
        df = pd.read_parquet("data/processed_data/combined_subjects.parquet")
    except FileNotFoundError:
        print("Error: combined_subjects.parquet not found. Please check the file path.")
        raise optuna.exceptions.TrialPruned()

    columns_to_standardize = [
        col for col in df.columns if df[col].dtype in ("float64", "int64")
    ]
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])
    X = df_scaled[columns_to_standardize].dropna()
    y = X.pop("CO2")

    # Define cross-validation scheme
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=0)

    try:
        # Create pipeline with the trial parameters
        pipeline = create_pipeline(trial, X)

        # Use a more conservative n_jobs setting for cross-validation
        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=outer_cv,
            scoring="neg_mean_squared_error",
            n_jobs=1,  # Using single-threaded CV to avoid nested parallelism
            error_score="raise",
        )

        return -np.mean(scores)

    except Exception as e:
        print(f"Trial {trial.number} failed due to error: {str(e)}")
        # Prune the trial if an error occurs
        raise optuna.exceptions.TrialPruned()


if __name__ == "__main__":
    # Configure Optuna logging
    optuna.logging.set_verbosity(optuna.logging.INFO)

    # Load study from the shared database
    # Replace the MySQL connection string with your actual database connection
    study = optuna.load_study(
        study_name="co2_prediction_distributed",
        storage="mysql://optuna_user:anthem1234@localhost/optuna_db",
    )

    print(f"Starting optimization. Current number of trials: {len(study.trials)}")

    # Set a maximum number of trials across all processes
    # Each process will stop once the total number of trials reaches this limit
    max_trials = 1

    # Run the optimization
    study.optimize(
        objective,
        n_trials=max_trials,  # Each process will run up to max_trials
        n_jobs=1,  # No parallelism within this process
        gc_after_trial=True,  # Enable garbage collection after each trial
        show_progress_bar=True,
    )

    # Print best results found so far by this process
    print("\nOptimization finished.")
    if study.best_trial:
        print("Best trial:")
        trial = study.best_trial
        print(f" Value: {trial.value}")
        print(" Params: ")
        for key, value in trial.params.items():
            print(f" {key}: {value}")
    else:
        print("No successful trials were completed.")
