"""Cross-validation utilities and management."""

import time
from typing import Any, Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, train_test_split


def stratify_continuous(y: pd.Series, n_bins: int = 10) -> np.ndarray:
    """
    Create stratification bins for continuous target variable.
    This allows us to use stratified CV with regression problems.
    """
    bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
    return bins


def adjusted_r2_score(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
    """
    Calculate adjusted R² score.
    Adjusted R² = 1 - (1 - R²) * (n - 1) / (n - p - 1)
    """
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)

    if n <= n_features + 1:
        return float("-inf")  # Invalid case

    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return adj_r2


class CVMetrics:
    """Container for cross-validation metrics."""

    def __init__(self):
        self.mse_scores: List[float] = []
        self.mae_scores: List[float] = []
        self.r2_scores: List[float] = []
        self.adj_r2_scores: List[float] = []

    def add_fold_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, n_features: int):
        """Add metrics for a single fold."""
        self.mse_scores.append(mean_squared_error(y_true, y_pred))
        self.mae_scores.append(mean_absolute_error(y_true, y_pred))
        self.r2_scores.append(r2_score(y_true, y_pred))
        self.adj_r2_scores.append(adjusted_r2_score(y_true, y_pred, n_features))

    def get_mean_metrics(self) -> Dict[str, float]:
        """Get mean metrics across all folds."""
        return {
            "mean_mse": np.mean(self.mse_scores),
            "mean_mae": np.mean(self.mae_scores),
            "mean_r2": np.mean(self.r2_scores),
            "mean_adj_r2": np.mean(self.adj_r2_scores),
        }

    def get_all_scores(self) -> Dict[str, List[float]]:
        """Get all individual fold scores."""
        return {
            "mse_scores": self.mse_scores,
            "mae_scores": self.mae_scores,
            "r2_scores": self.r2_scores,
            "adj_r2_scores": self.adj_r2_scores,
        }


class CrossValidationManager:
    """Manages cross-validation procedures for hyperparameter optimization."""

    def __init__(
        self,
        outer_cv_folds: int = 5,
        inner_cv_folds: int = 3,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.outer_cv_folds = outer_cv_folds
        self.inner_cv_folds = inner_cv_folds
        self.test_size = test_size
        self.random_state = random_state

    def prepare_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, np.ndarray, np.ndarray
    ]:
        """
        Prepare data by creating train/test split with stratification.
        Returns: X_dev, X_holdout, y_dev, y_holdout, strata_dev, strata_holdout
        """
        # Create stratification bins
        strata = stratify_continuous(y)

        # Create holdout set
        X_dev, X_holdout, y_dev, y_holdout, strata_dev, strata_holdout = (
            train_test_split(
                X,
                y,
                strata,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=strata,
            )
        )

        return X_dev, X_holdout, y_dev, y_holdout, strata_dev, strata_holdout

    def nested_cross_validation(
        self,
        X_dev: pd.DataFrame,
        y_dev: pd.Series,
        strata_dev: np.ndarray,
        pipeline_factory,
        trial: optuna.Trial,
        n_features: int,
        logger=None,
    ) -> CVMetrics:
        """
        Perform nested cross-validation.
        Returns CVMetrics object with all fold results.
        """
        outer_cv = StratifiedKFold(
            n_splits=self.outer_cv_folds, shuffle=True, random_state=self.random_state
        )

        outer_metrics = CVMetrics()

        for outer_fold, (train_idx, test_idx) in enumerate(
            outer_cv.split(X_dev, strata_dev)
        ):
            if logger:
                logger(
                    f"Trial {trial.number}: Outer fold {outer_fold+1}/{self.outer_cv_folds}"
                )

            X_train, X_test = X_dev.iloc[train_idx], X_dev.iloc[test_idx]
            y_train, y_test = y_dev.iloc[train_idx], y_dev.iloc[test_idx]

            # Inner CV for hyperparameter tuning
            inner_strata = stratify_continuous(y_train)
            inner_cv = StratifiedKFold(
                n_splits=self.inner_cv_folds, shuffle=True, random_state=0
            )

            try:
                # Create pipeline
                pipeline = pipeline_factory(trial, X_train)

                # Inner CV loop
                inner_cv_scores = []
                for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(
                    inner_cv.split(X_train, inner_strata)
                ):
                    X_inner_train = X_train.iloc[inner_train_idx]
                    X_inner_val = X_train.iloc[inner_val_idx]
                    y_inner_train = y_train.iloc[inner_train_idx]
                    y_inner_val = y_train.iloc[inner_val_idx]

                    # Fit and predict
                    pipeline.fit(X_inner_train, y_inner_train)
                    y_inner_pred = pipeline.predict(X_inner_val)
                    inner_fold_mse = mean_squared_error(y_inner_val, y_inner_pred)
                    inner_cv_scores.append(inner_fold_mse)

                    # Report for pruning
                    step = outer_fold * self.inner_cv_folds + inner_fold
                    trial.report(inner_fold_mse, step=step)
                    if trial.should_prune():
                        if logger:
                            logger(
                                f"Trial {trial.number}: Pruned at outer fold {outer_fold+1}, inner fold {inner_fold+1}"
                            )
                        raise optuna.exceptions.TrialPruned()

                # Refit on entire outer training fold
                pipeline.fit(X_train, y_train)
                y_outer_pred = pipeline.predict(X_test)

                # Add metrics for this outer fold
                outer_metrics.add_fold_metrics(y_test, y_outer_pred, n_features)

                if logger:
                    fold_mse = mean_squared_error(y_test, y_outer_pred)
                    logger(
                        f"Trial {trial.number}, Outer fold {outer_fold+1}: MSE = {fold_mse:.6f}"
                    )

            except Exception as e:
                if logger:
                    logger(
                        f"Trial {trial.number}, Outer fold {outer_fold+1} failed: {str(e)}"
                    )
                continue

        return outer_metrics

    def final_evaluation(
        self,
        X_dev: pd.DataFrame,
        y_dev: pd.Series,
        X_holdout: pd.DataFrame,
        y_holdout: pd.Series,
        pipeline_factory,
        trial: optuna.Trial,
        n_features: int,
        logger=None,
    ) -> Dict[str, Any]:
        """
        Train final model and evaluate on holdout set.
        Returns dictionary with train and test metrics.
        """
        try:
            if logger:
                logger(
                    f"Trial {trial.number}: Final evaluation - Training on full development set"
                )

            # Train final model
            final_pipeline = pipeline_factory(trial, X_dev)
            final_pipeline.fit(X_dev, y_dev)

            # Evaluate on holdout test set
            y_holdout_pred = final_pipeline.predict(X_holdout)
            holdout_test_mse = mean_squared_error(y_holdout, y_holdout_pred)
            holdout_test_mae = mean_absolute_error(y_holdout, y_holdout_pred)
            holdout_test_r2 = r2_score(y_holdout, y_holdout_pred)
            holdout_test_adj_r2 = adjusted_r2_score(
                y_holdout, y_holdout_pred, n_features
            )

            # Evaluate on development set (training performance)
            y_dev_pred = final_pipeline.predict(X_dev)
            holdout_train_mse = mean_squared_error(y_dev, y_dev_pred)
            holdout_train_mae = mean_absolute_error(y_dev, y_dev_pred)
            holdout_train_r2 = r2_score(y_dev, y_dev_pred)
            holdout_train_adj_r2 = adjusted_r2_score(y_dev, y_dev_pred, n_features)

            if logger:
                logger(
                    f"Trial {trial.number}: Holdout test MSE = {holdout_test_mse:.6f}, R² = {holdout_test_r2:.6f}"
                )

            return {
                "final_pipeline": final_pipeline,
                "holdout_test_mse": holdout_test_mse,
                "holdout_test_mae": holdout_test_mae,
                "holdout_test_r2": holdout_test_r2,
                "holdout_test_adj_r2": holdout_test_adj_r2,
                "holdout_train_mse": holdout_train_mse,
                "holdout_train_mae": holdout_train_mae,
                "holdout_train_r2": holdout_train_r2,
                "holdout_train_adj_r2": holdout_train_adj_r2,
                "predictions": {
                    "actual_test": y_holdout.values,
                    "predicted_test": y_holdout_pred,
                    "actual_train": y_dev.values,
                    "predicted_train": y_dev_pred,
                },
            }

        except Exception as e:
            if logger:
                logger(f"Trial {trial.number}: Final evaluation failed: {str(e)}")
            return {
                "final_pipeline": None,
                "holdout_test_mse": None,
                "holdout_test_mae": None,
                "holdout_test_r2": None,
                "holdout_test_adj_r2": None,
                "holdout_train_mse": None,
                "holdout_train_mae": None,
                "holdout_train_r2": None,
                "holdout_train_adj_r2": None,
                "predictions": None,
            }
