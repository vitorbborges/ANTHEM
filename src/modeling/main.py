# src/modeling/main.py
"""
Main entry point for hyperparameter optimization.

This script provides a simple interface to run either single-process or
multi-process hyperparameter optimization with support for both SQLite
(local) and MySQL (distributed) storage backends.

Usage examples:
    # Single process, 50 trials, SQLite storage
    python -m src.modeling.main --trials 50

    # Parallel with 4 workers, 20 trials each, SQLite storage
    python -m src.modeling.main --parallel --workers 4 --trials 20

    # Distributed with MySQL backend
    python -m src.modeling.main --parallel --workers 8 --trials 15 --storage mysql

    # Custom MySQL connection
    python -m src.modeling.main --storage mysql --mysql-url "mysql://user:pass@host/db"
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.modeling.optimization import OptimizationRunner
from src.modeling.parallel_runner import ParallelOptimizationRunner
from src.modeling.utils import log_message


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run hyperparameter optimization for CO2 prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Basic configuration
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of trials per worker (default: 20)",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout per worker in seconds (default: 600)",
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
        default="co2_prediction",
        help="Name of the Optuna study (default: co2_prediction)",
    )

    # Data configuration
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed_data/S3-coords.parquet",
        help="Path to the data file (default: data/processed_data/S3-coords.parquet)",
    )

    # Cross-validation configuration
    parser.add_argument(
        "--outer-folds",
        type=int,
        default=5,
        help="Number of outer CV folds (default: 5)",
    )

    parser.add_argument(
        "--inner-folds",
        type=int,
        default=3,
        help="Number of inner CV folds (default: 3)",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Holdout test set size (default: 0.2)",
    )

    # TODO: Add Kalman Filter specific arguments
    # parser.add_argument(
    #     "--enable-kalman",
    #     action="store_true",
    #     help="Enable Kalman filtering for predictions"
    # )
    #
    # parser.add_argument(
    #     "--kalman-mode",
    #     choices=["post-process", "integrated"],
    #     default="post-process",
    #     help="Kalman filter mode: post-process ML predictions or integrate into pipeline"
    # )

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


def main():
    """Main entry point."""
    args = parse_arguments()
    validate_arguments(args)

    log_message("=" * 60)
    log_message("CO2 Prediction Hyperparameter Optimization")
    log_message("=" * 60)

    # Log configuration
    log_message(f"Configuration:")
    log_message(f"  Mode: {'Parallel' if args.parallel else 'Single process'}")
    log_message(f"  Storage: {args.storage}")
    log_message(f"  Study name: {args.study_name}")
    log_message(f"  Data path: {args.data_path}")
    log_message(f"  Trials: {args.trials}")
    log_message(f"  Timeout: {args.timeout}s")
    if args.parallel:
        log_message(f"  Workers: {args.workers or 'auto-detect'}")
    log_message(f"  CV folds: {args.outer_folds} outer, {args.inner_folds} inner")
    log_message(f"  Test size: {args.test_size}")

    try:
        if args.parallel:
            # Run in parallel mode
            log_message("\nStarting parallel optimization...")

            runner = ParallelOptimizationRunner(
                study_name=args.study_name,
                storage_type=args.storage,
                mysql_url=args.mysql_url,
                data_path=args.data_path,
            )

            results = runner.run_parallel(
                num_workers=args.workers,
                trials_per_worker=args.trials,
                timeout_per_worker=args.timeout,
            )

            # Find and save best overall result
            best_result = runner.find_best_result()

            if best_result:
                log_message(
                    f"\nBest overall MSE: {best_result.get('best_cv_mse', 'N/A')}"
                )
            else:
                log_message("\nNo successful results found")

        else:
            # Run in single process mode
            log_message("\nStarting single-process optimization...")

            runner = OptimizationRunner(
                study_name=args.study_name,
                storage_type=args.storage,
                mysql_url=args.mysql_url,
                data_path=args.data_path,
                outer_folds=args.outer_folds,
                inner_folds=args.inner_folds,
                test_size=args.test_size,
            )

            study = runner.run_optimization(
                n_trials=args.trials, timeout=args.timeout, show_progress=True
            )

            # Check if we have any completed trials
            completed_trials = [t for t in study.trials if t.state.name == "COMPLETE"]
            if completed_trials:
                log_message(f"\nOptimization Results:")
                log_message(f"  Completed trials: {len(completed_trials)}")
                log_message(f"  Best CV MSE: {study.best_trial.value:.6f}")

                # Display best parameters
                log_message(f"  Best parameters:")
                for key, value in study.best_trial.params.items():
                    log_message(f"    {key}: {value}")
            else:
                log_message(f"\nNo trials completed successfully")
                log_message(f"  Total trials attempted: {len(study.trials)}")
                log_message(
                    f"  Failed trials: {len([t for t in study.trials if t.state.name == 'FAIL'])}"
                )
                log_message(
                    f"  Pruned trials: {len([t for t in study.trials if t.state.name == 'PRUNED'])}"
                )

        log_message("\nOptimization completed successfully!")
        log_message("Check the 'metrics' folder for detailed results.")

    except KeyboardInterrupt:
        log_message("\nOptimization interrupted by user")
        sys.exit(1)
    except Exception as e:
        log_message(f"\nOptimization failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
