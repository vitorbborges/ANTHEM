# src/modeling/ts_main.py
"""
Time Series Main Entry Point.
Updated main.py for time series CO2 prediction with exogenous variables.

Usage examples:
    # Basic time series optimization
    python -m src.modeling.ts_main --trials 30 --cv-method walk_forward

    # With feature engineering
    python -m src.modeling.ts_main --trials 25 --create-lags --max-features 30

    # Parallel time series optimization
    python -m src.modeling.ts_main --parallel --workers 4 --trials 20

    # Forecast evaluation
    python -m src.modeling.ts_main --forecast-eval --horizon 48

    # MySQL distributed
    python -m src.modeling.ts_main --storage mysql --parallel --workers 6
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd

from src.ts_modeling.ts_optimization import TimeSeriesOptimizationRunner
from src.ts_modeling.ts_utils import (
    create_lag_features,
    log_message,
    validate_time_series_data,
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run time series hyperparameter optimization for CO2 prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Basic configuration
    parser.add_argument(
        "--trials",
        type=int,
        default=25,
        help="Number of trials per worker (default: 25)",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Timeout per worker in seconds (default: 900)",
    )

    # Time series specific options
    parser.add_argument(
        "--cv-method",
        choices=["walk_forward", "blocked", "auto"],
        default="walk_forward",
        help="Cross-validation method (default: walk_forward)",
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of CV splits (default: 5)",
    )

    parser.add_argument(
        "--gap",
        type=int,
        default=0,
        help="Gap between train/test in CV (default: 0)",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Holdout test set size (default: 0.2)",
    )

    # Feature engineering options
    parser.add_argument(
        "--create-lags",
        action="store_true",
        help="Create lag and rolling window features",
    )

    parser.add_argument(
        "--max-features",
        type=int,
        default=40,
        help="Maximum number of exogenous features to use (default: 40)",
    )

    parser.add_argument(
        "--target-column",
        type=str,
        default="CO2",
        help="Name of target column (default: CO2)",
    )

    parser.add_argument(
        "--time-column",
        type=str,
        default=None,
        help="Name of time column (default: use index)",
    )

    parser.add_argument(
        "--exclude-columns",
        type=str,
        nargs="*",
        default=["P", "PM1", "PM10", "RH", "T", "VOC"],
        help="Columns to exclude from features (default: P PM1 PM10 RH T VOC)",
    )

    # Parallel execution
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run in parallel mode with multiple workers",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto-detect CPU count)",
    )

    # Storage configuration
    parser.add_argument(
        "--storage",
        choices=["sqlite", "mysql"],
        default="sqlite",
        help="Storage backend: sqlite for local, mysql for distributed (default: sqlite)",
    )

    parser.add_argument(
        "--mysql-url",
        type=str,
        default=None,
        help="MySQL connection string (only needed for mysql storage)",
    )

    parser.add_argument(
        "--study-name",
        type=str,
        default="co2_ts_prediction",
        help="Name of the Optuna study (default: co2_ts_prediction)",
    )

    # Data configuration
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed_data/S3-coords.parquet",
        help="Path to the data file (default: data/processed_data/S3-coords.parquet)",
    )

    # Evaluation options
    parser.add_argument(
        "--forecast-eval",
        action="store_true",
        help="Run forecast evaluation after optimization",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=24,
        help="Forecast horizon for evaluation (default: 24)",
    )

    parser.add_argument(
        "--validate-data",
        action="store_true",
        help="Run data validation before optimization",
    )

    return parser.parse_args()


def validate_arguments(args):
    """Validate command line arguments."""
    # Check data file exists
    if not Path(args.data_path).exists():
        log_message(f"Error: Data file not found: {args.data_path}")
        sys.exit(1)

    # Validate storage configuration
    if args.storage == "mysql" and args.mysql_url is None:
        if "OPTUNA_STORAGE" not in os.environ:
            log_message(
                "Error: MySQL storage requires --mysql-url or OPTUNA_STORAGE environment variable"
            )
            sys.exit(1)

    # Validate numerical arguments
    if args.trials <= 0:
        log_message("Error: Number of trials must be positive")
        sys.exit(1)

    if args.timeout <= 0:
        log_message("Error: Timeout must be positive")
        sys.exit(1)

    if args.workers is not None and args.workers <= 0:
        log_message("Error: Number of workers must be positive")
        sys.exit(1)

    if not 0 < args.test_size < 1:
        log_message("Error: Test size must be between 0 and 1")
        sys.exit(1)

    if args.n_splits < 2:
        log_message("Error: Number of splits must be at least 2")
        sys.exit(1)

    if args.horizon <= 0:
        log_message("Error: Forecast horizon must be positive")
        sys.exit(1)


def preprocess_data(args):
    """Preprocess data based on arguments."""
    log_message("=" * 50)
    log_message("DATA PREPROCESSING")
    log_message("=" * 50)

    # Load data
    log_message(f"Loading data from {args.data_path}")
    df = pd.read_parquet(args.data_path)
    log_message(f"Original data shape: {df.shape}")

    # Handle time column
    if args.time_column and args.time_column in df.columns:
        df = df.set_index(args.time_column)
        log_message(f"Set {args.time_column} as time index")

    # Create lag features if requested
    if args.create_lags:
        log_message("Creating lag and rolling window features...")
        df = create_lag_features(df, args.target_column)
        log_message(f"Data shape after lag features: {df.shape}")

    # Save preprocessed data
    preprocessed_path = args.data_path.replace(".parquet", "_preprocessed.parquet")
    df.to_parquet(preprocessed_path)
    log_message(f"Preprocessed data saved to {preprocessed_path}")

    return preprocessed_path


def run_data_validation(args, data_path):
    """Run data validation."""
    log_message("=" * 50)
    log_message("DATA VALIDATION")
    log_message("=" * 50)

    # Load data
    df = pd.read_parquet(data_path)

    # Prepare features and target
    exclude_cols = args.exclude_columns or []
    feature_cols = [
        col
        for col in df.columns
        if col != args.target_column and col not in exclude_cols
    ]

    # Filter numeric columns
    numeric_cols = [
        col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])
    ]

    X = df[numeric_cols[: args.max_features]]  # Limit features
    y = df[args.target_column]

    # Run validation
    validation = validate_time_series_data(X, y)

    log_message(
        f"Validation result: {'PASSED' if validation['is_valid'] else 'FAILED'}"
    )

    if validation["errors"]:
        log_message("ERRORS found:")
        for error in validation["errors"]:
            log_message(f"  - {error}")

    if validation["warnings"]:
        log_message("WARNINGS:")
        for warning in validation["warnings"]:
            log_message(f"  - {warning}")

    log_message("Data info:")
    for key, value in validation["info"].items():
        log_message(f"  {key}: {value}")

    if not validation["is_valid"]:
        log_message("❌ Data validation failed. Please fix errors before proceeding.")
        sys.exit(1)
    else:
        log_message("✅ Data validation passed!")


def main():
    """Main entry point."""
    args = parse_arguments()
    validate_arguments(args)

    log_message("=" * 60)
    log_message("CO2 TIME SERIES PREDICTION OPTIMIZATION")
    log_message("=" * 60)

    # Log configuration
    log_message(f"Configuration:")
    log_message(f"  Mode: {'Parallel' if args.parallel else 'Single process'}")
    log_message(f"  Storage: {args.storage}")
    log_message(f"  Study name: {args.study_name}")
    log_message(f"  Data path: {args.data_path}")
    log_message(f"  Target column: {args.target_column}")
    log_message(f"  Trials: {args.trials}")
    log_message(f"  Timeout: {args.timeout}s")
    log_message(f"  CV method: {args.cv_method}")
    log_message(f"  CV splits: {args.n_splits}")
    log_message(f"  Test size: {args.test_size}")
    log_message(f"  Max features: {args.max_features}")
    log_message(f"  Create lags: {args.create_lags}")
    if args.parallel:
        log_message(f"  Workers: {args.workers or 'auto-detect'}")

    try:
        # Preprocess data if needed
        data_path = args.data_path
        if args.create_lags:
            data_path = preprocess_data(args)

        # Run data validation if requested
        if args.validate_data:
            run_data_validation(args, data_path)

        if args.parallel:
            # Import here to avoid circular imports
            from src.modeling.parallel_runner import ParallelOptimizationRunner

            # Note: Parallel runner needs to be adapted for time series
            # For now, run multiple single processes
            log_message("⚠️  Parallel mode not yet fully implemented for time series.")
            log_message("    Running single process optimization instead.")

            runner = TimeSeriesOptimizationRunner(
                study_name=args.study_name,
                storage_type=args.storage,
                mysql_url=args.mysql_url,
                data_path=data_path,
                n_splits=args.n_splits,
                test_size=args.test_size,
                cv_method=args.cv_method,
                gap=args.gap,
                target_column=args.target_column,
                time_column=args.time_column,
                exclude_columns=args.exclude_columns,
            )

            study = runner.run_optimization(
                n_trials=args.trials
                * (args.workers or 1),  # Scale trials for "parallel"
                timeout=args.timeout,
                show_progress=True,
            )

        else:
            # Run in single process mode
            log_message("\nStarting single-process time series optimization...")

            runner = TimeSeriesOptimizationRunner(
                study_name=args.study_name,
                storage_type=args.storage,
                mysql_url=args.mysql_url,
                data_path=data_path,
                n_splits=args.n_splits,
                test_size=args.test_size,
                cv_method=args.cv_method,
                gap=args.gap,
                target_column=args.target_column,
                time_column=args.time_column,
                exclude_columns=args.exclude_columns,
            )

            study = runner.run_optimization(
                n_trials=args.trials, timeout=args.timeout, show_progress=True
            )

        # Check results
        completed_trials = [t for t in study.trials if t.state.name == "COMPLETE"]
        if completed_trials:
            log_message(f"\n🎉 Time Series Optimization Results:")
            log_message(f"  Completed trials: {len(completed_trials)}")
            log_message(f"  Best CV MSE: {study.best_trial.value:.6f}")
            log_message(
                f"  Best model type: {study.best_trial.params.get('model_type', 'Unknown')}"
            )

            # Display best parameters
            log_message(f"  Best parameters:")
            for key, value in study.best_trial.params.items():
                log_message(f"    {key}: {value}")

            # Show feature selection info
            if "n_features_selected" in study.best_trial.user_attrs:
                n_selected = study.best_trial.user_attrs["n_features_selected"]
                n_total = study.best_trial.user_attrs.get("n_features", "Unknown")
                log_message(f"  Features selected: {n_selected}/{n_total}")

        else:
            log_message(f"\n❌ No trials completed successfully")
            failed_count = len([t for t in study.trials if t.state.name == "FAIL"])
            pruned_count = len([t for t in study.trials if t.state.name == "PRUNED"])
            log_message(f"  Total trials attempted: {len(study.trials)}")
            log_message(f"  Failed trials: {failed_count}")
            log_message(f"  Pruned trials: {pruned_count}")

        # Run forecast evaluation if requested
        if args.forecast_eval and completed_trials:
            log_message(
                f"\n📈 Running forecast evaluation (horizon: {args.horizon})..."
            )
            try:
                forecast_results = runner.run_forecast_evaluation(args.horizon)
                if forecast_results:
                    log_message(
                        f"  Forecast MSE: {forecast_results['forecast_mse']:.6f}"
                    )
                    log_message(f"  Forecast R²: {forecast_results['forecast_r2']:.6f}")
                    log_message(f"  Training R²: {forecast_results['train_r2']:.6f}")
                else:
                    log_message("  ⚠️  Forecast evaluation failed")
            except Exception as e:
                log_message(f"  ❌ Forecast evaluation error: {str(e)}")

        # Summary and recommendations
        log_message(f"\n📋 SUMMARY & RECOMMENDATIONS")
        log_message(f"=" * 40)

        if completed_trials:
            best_trial = study.best_trial
            model_type = best_trial.params.get("model_type", "unknown")

            log_message(f"✅ Best performing model: {model_type.upper()}")

            # Model-specific insights
            if model_type == "gaussian_process":
                kernel = best_trial.params.get("gp_kernel", "unknown")
                log_message(f"   → GP kernel: {kernel}")
                log_message(f"   → Excellent for capturing complex CO2 patterns")

            elif model_type == "arima":
                order = (
                    best_trial.params.get("arima_p", 0),
                    best_trial.params.get("arima_d", 0),
                    best_trial.params.get("arima_q", 0),
                )
                log_message(f"   → ARIMA order: {order}")
                log_message(f"   → Good for linear time series patterns")

            elif model_type == "sarimax":
                order = (
                    best_trial.params.get("sarimax_p", 0),
                    best_trial.params.get("sarimax_d", 0),
                    best_trial.params.get("sarimax_q", 0),
                )
                seasonal = best_trial.params.get("sarimax_seasonal", False)
                log_message(f"   → SARIMAX order: {order}")
                log_message(f"   → Seasonal: {seasonal}")
                log_message(f"   → Excellent for incorporating exogenous variables")

            # Feature selection insights
            uses_selection = best_trial.params.get("use_feature_selection", False)
            if uses_selection:
                method = best_trial.params.get("feature_selector_type", "unknown")
                log_message(f"✅ Feature selection: {method}")
                log_message(f"   → Helps reduce noise from irrelevant variables")
            else:
                log_message(f"✅ Uses all available exogenous variables")

            # Performance insights
            cv_mse = best_trial.value
            holdout_mse = best_trial.user_attrs.get("holdout_test_mse")
            if holdout_mse:
                generalization = "Good" if holdout_mse < cv_mse * 1.2 else "Concerning"
                log_message(f"📊 Generalization: {generalization}")
                log_message(f"   → CV MSE: {cv_mse:.6f}")
                log_message(f"   → Holdout MSE: {holdout_mse:.6f}")

        # Next steps
        log_message(f"\n🚀 NEXT STEPS:")
        log_message(f"1. Check detailed results in 'metrics' folder")
        log_message(f"2. Analyze residuals and prediction plots")
        log_message(f"3. Consider ensemble methods for production")
        log_message(f"4. Set up regular model retraining pipeline")
        log_message(f"5. Monitor model performance over time")

        if not completed_trials:
            log_message(f"\n🔧 TROUBLESHOOTING:")
            log_message(f"1. Try reducing the number of features (--max-features)")
            log_message(f"2. Increase timeout (--timeout)")
            log_message(f"3. Check data quality (--validate-data)")
            log_message(f"4. Try different CV method (--cv-method)")

        log_message("\n✅ Time series optimization completed!")
        log_message("📁 Check the 'metrics' and 'models' folders for detailed results.")

    except KeyboardInterrupt:
        log_message("\n⚠️  Optimization interrupted by user")
        sys.exit(1)
    except Exception as e:
        log_message(f"\n❌ Optimization failed with error: {str(e)}")
        log_message(f"\n🔧 Common solutions:")
        log_message(f"1. Check data file exists and is readable")
        log_message(f"2. Verify target column name: '{args.target_column}'")
        log_message(f"3. Check for sufficient data (need >100 samples)")
        log_message(f"4. Ensure numeric features are available")
        log_message(f"5. Try simpler models first (reduce feature complexity)")
        sys.exit(1)


if __name__ == "__main__":
    main()
