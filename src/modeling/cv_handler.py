# src/modeling/cv_handler.py
import time
from typing import Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from .utils import adjusted_r2_score, log_message, stratify_continuous


class CrossValidationHandler:
    """Handles nested cross-validation for hyperparameter optimization."""

    def __init__(
        self, outer_folds: int = 5, inner_folds: int = 3, test_size: float = 0.2
    ):
        self.outer_folds = outer_folds
        self.inner_folds = inner_folds
        self.test_size = test_size

    def nested_cv_evaluate(
        self, X: pd.DataFrame, y: pd.Series, pipeline_factory, trial, trial_number: int
    ) -> Dict:
        """
        Performs nested cross-validation with a final holdout evaluation.

        Returns:
            Dict containing all evaluation metrics and scores
        """
        start_time = time.time()
        n_features = X.shape[1]

        # Create stratification bins for continuous target
        log_message(f"Trial {trial_number}: Creating stratification bins...")
        strata = stratify_continuous(y)

        # Create holdout set (never touched during training/CV)
        X_dev, X_holdout, y_dev, y_holdout, strata_dev, _ = train_test_split(
            X, y, strata, test_size=self.test_size, random_state=42, stratify=strata
        )

        log_message(
            f"Trial {trial_number}: Created holdout set with {len(X_holdout)} samples"
        )

        # Outer CV for performance estimation
        outer_cv = StratifiedKFold(
            n_splits=self.outer_folds, shuffle=True, random_state=42
        )
        outer_scores = {"mse": [], "r2": [], "adj_r2": [], "mae": []}

        # Outer CV loop
        for outer_fold, (train_idx, test_idx) in enumerate(
            outer_cv.split(X_dev, strata_dev)
        ):
            log_message(
                f"Trial {trial_number}: Outer fold {outer_fold+1}/{self.outer_folds}"
            )

            X_train, X_test = X_dev.iloc[train_idx], X_dev.iloc[test_idx]
            y_train, y_test = y_dev.iloc[train_idx], y_dev.iloc[test_idx]

            # Inner CV for hyperparameter tuning
            inner_strata = stratify_continuous(y_train)
            inner_cv = StratifiedKFold(
                n_splits=self.inner_folds, shuffle=True, random_state=0
            )

            try:
                pipeline = pipeline_factory(trial, X_train)
                inner_cv_scores = []

                # Inner CV loop
                for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(
                    inner_cv.split(X_train, inner_strata)
                ):
                    X_inner_train = X_train.iloc[inner_train_idx]
                    X_inner_val = X_train.iloc[inner_val_idx]
                    y_inner_train = y_train.iloc[inner_train_idx]
                    y_inner_val = y_train.iloc[inner_val_idx]

                    # Fit and evaluate on inner fold
                    pipeline.fit(X_inner_train, y_inner_train)
                    y_inner_pred = pipeline.predict(X_inner_val)
                    inner_fold_mse = mean_squared_error(y_inner_val, y_inner_pred)
                    inner_cv_scores.append(inner_fold_mse)

                    # Report for pruning
                    step = outer_fold * self.inner_folds + inner_fold
                    trial.report(inner_fold_mse, step=step)
                    if trial.should_prune():
                        log_message(
                            f"Trial {trial_number}: Pruned at outer fold {outer_fold+1}, inner fold {inner_fold+1}"
                        )
                        raise optuna.exceptions.TrialPruned()

                # Refit on entire outer training fold
                pipeline.fit(X_train, y_train)
                y_outer_pred = pipeline.predict(X_test)

                # Calculate outer fold metrics
                outer_fold_mse = mean_squared_error(y_test, y_outer_pred)
                outer_fold_mae = mean_absolute_error(y_test, y_outer_pred)
                outer_fold_r2 = r2_score(y_test, y_outer_pred)
                outer_fold_adj_r2 = adjusted_r2_score(y_test, y_outer_pred, n_features)

                outer_scores["mse"].append(outer_fold_mse)
                outer_scores["mae"].append(outer_fold_mae)
                outer_scores["r2"].append(outer_fold_r2)
                outer_scores["adj_r2"].append(outer_fold_adj_r2)

                log_message(
                    f"Trial {trial_number}, Outer fold {outer_fold+1}: "
                    f"MSE={outer_fold_mse:.6f}, R²={outer_fold_r2:.6f}"
                )

            except Exception as e:
                log_message(
                    f"Trial {trial_number}, Outer fold {outer_fold+1} failed: {str(e)}"
                )
                continue

        # Check if we have any successful folds
        if len(outer_scores["mse"]) == 0:
            log_message(f"Trial {trial_number}: No successful outer folds")
            raise Exception("No successful outer folds")

        # Calculate mean outer CV scores
        mean_outer_mse = np.mean(outer_scores["mse"])
        mean_outer_mae = np.mean(outer_scores["mae"])
        mean_outer_r2 = np.mean(outer_scores["r2"])
        mean_outer_adj_r2 = np.mean(outer_scores["adj_r2"])

        # Final evaluation on holdout set
        try:
            log_message(f"Trial {trial_number}: Final evaluation on holdout set...")
            final_pipeline = pipeline_factory(trial, X_dev)
            final_pipeline.fit(X_dev, y_dev)

            # Holdout test metrics
            y_holdout_pred = final_pipeline.predict(X_holdout)
            holdout_test_mse = mean_squared_error(y_holdout, y_holdout_pred)
            holdout_test_mae = mean_absolute_error(y_holdout, y_holdout_pred)
            holdout_test_r2 = r2_score(y_holdout, y_holdout_pred)
            holdout_test_adj_r2 = adjusted_r2_score(
                y_holdout, y_holdout_pred, n_features
            )

            # Training set metrics
            y_dev_pred = final_pipeline.predict(X_dev)
            holdout_train_mse = mean_squared_error(y_dev, y_dev_pred)
            holdout_train_mae = mean_absolute_error(y_dev, y_dev_pred)
            holdout_train_r2 = r2_score(y_dev, y_dev_pred)
            holdout_train_adj_r2 = adjusted_r2_score(y_dev, y_dev_pred, n_features)

            # Store results for later analysis
            holdout_results = pd.DataFrame(
                {
                    "actual_test": y_holdout.values,
                    "predicted_test": y_holdout_pred,
                }
            )

            # Create separate DataFrame for training results
            train_results = pd.DataFrame(
                {
                    "actual_train": y_dev.values,
                    "predicted_train": y_dev_pred,
                }
            )

            results = {
                "pipeline": final_pipeline,
                "holdout_results": holdout_results,
                "train_results": train_results,
                "cv_mse": mean_outer_mse,
                "cv_mae": mean_outer_mae,
                "cv_r2": mean_outer_r2,
                "cv_adj_r2": mean_outer_adj_r2,
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
                "outer_cv_scores": outer_scores,
            }

            log_message(
                f"Trial {trial_number}: Completed in {results['time_taken']:.2f}s"
            )
            return results

        except Exception as e:
            log_message(f"Trial {trial_number}: Final evaluation failed: {str(e)}")
            raise
