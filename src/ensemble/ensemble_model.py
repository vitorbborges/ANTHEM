from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error


class EnsembleModel(BaseEstimator, RegressorMixin):
    """
    Ensemble model that trains one optimized model per subject using nested Optuna studies.
    Each subject gets its own hyperparameter optimization via spatial cross-validation.
    """

    def __init__(
        self,
        pipeline_factory,
        spatial_folds=3,
        min_samples_per_subject=10,
        n_trials_per_subject=15,
        optuna_direction="minimize",
        random_state=42,
    ):
        """
        Initialize the ensemble model.

        Args:
            pipeline_factory: Function that creates pipeline given (trial, X)
            spatial_folds: Number of spatial folds for within-subject validation
            min_samples_per_subject: Minimum samples required per subject
            n_trials_per_subject: Number of Optuna trials per subject for hyperparameter optimization
            optuna_direction: Direction for Optuna optimization ("minimize" or "maximize")
            random_state: Random state for reproducibility
        """
        self.pipeline_factory = pipeline_factory
        self.spatial_folds = spatial_folds
        self.min_samples_per_subject = min_samples_per_subject
        self.n_trials_per_subject = n_trials_per_subject
        self.optuna_direction = optuna_direction
        self.random_state = random_state

        # Storage for fitted models and metadata
        self.subject_models = {}
        self.subject_training_data = {}
        self.subject_best_params = {}
        self.subject_optimization_scores = {}
        self.subject_studies = {}
        self.training_subjects = []
        self.is_fitted = False

    def _create_spatial_blocks(self, X_subject):
        """Create spatial blocks within a subject's data for spatial CV"""
        if "x" not in X_subject.columns or "y" not in X_subject.columns:
            # Fallback to sequential blocks
            n_samples = len(X_subject)
            return np.arange(n_samples) % self.spatial_folds

        coords = X_subject[["x", "y"]].values
        if len(coords) < self.spatial_folds:
            return np.arange(len(coords)) % len(coords)

        kmeans = KMeans(
            n_clusters=self.spatial_folds, random_state=self.random_state, n_init=10
        )
        return kmeans.fit_predict(coords)

    def _spatial_cv_objective(self, trial, X_subject, y_subject, subject_id):
        """
        Objective function for individual subject optimization using spatial cross-validation.

        Args:
            trial: Optuna trial for this subject
            X_subject: Subject's training data
            y_subject: Subject's training targets
            subject_id: Subject identifier

        Returns:
            Mean spatial CV score for this trial
        """
        spatial_blocks = self._create_spatial_blocks(X_subject)
        spatial_scores = []

        unique_blocks = np.unique(spatial_blocks)
        for spatial_fold in range(min(self.spatial_folds, len(unique_blocks))):
            if spatial_fold >= len(unique_blocks):
                break

            spatial_train_mask = spatial_blocks != spatial_fold
            spatial_test_mask = spatial_blocks == spatial_fold

            if not spatial_test_mask.any() or not spatial_train_mask.any():
                continue

            X_sp_train = X_subject[spatial_train_mask]
            y_sp_train = y_subject[spatial_train_mask]
            X_sp_test = X_subject[spatial_test_mask]
            y_sp_test = y_subject[spatial_test_mask]

            try:
                # Create pipeline with trial hyperparameters
                pipeline = self.pipeline_factory(trial, X_sp_train)
                pipeline.fit(X_sp_train, y_sp_train)
                y_pred = pipeline.predict(X_sp_test)
                spatial_score = mean_squared_error(y_sp_test, y_pred)
                spatial_scores.append(spatial_score)
            except Exception as e:
                # If this fold fails, return a large penalty
                return float("inf")

        if spatial_scores:
            return np.mean(spatial_scores)
        else:
            return float("inf")

    def _optimize_subject_hyperparameters(self, X_subject, y_subject, subject_id):
        """
        Optimize hyperparameters for a single subject using spatial CV.

        Args:
            X_subject: Subject's training data
            y_subject: Subject's training targets
            subject_id: Subject identifier

        Returns:
            Tuple[bool, optuna.Trial, float]: (success, best_trial, best_score)
        """
        print(
            f"        Optimizing hyperparameters for subject {subject_id} ({len(X_subject)} samples)"
        )

        # Create Optuna study for this subject
        study_name = f"subject_{subject_id}_study"
        study = optuna.create_study(
            direction=self.optuna_direction,
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=self.random_state + subject_id),
        )

        # Define objective function for this subject
        def objective(trial):
            return self._spatial_cv_objective(trial, X_subject, y_subject, subject_id)

        try:
            # Optimize hyperparameters
            study.optimize(
                objective,
                n_trials=self.n_trials_per_subject,
                show_progress_bar=False,
                timeout=None,
                n_jobs=1,  # Keep single-threaded for nested optimization
            )

            # Check if we found any successful trials
            completed_trials = [
                t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
            ]

            if completed_trials and study.best_trial is not None:
                best_score = study.best_value
                best_trial = study.best_trial

                # Store the study for later analysis
                self.subject_studies[subject_id] = study

                print(
                    f"          Subject {subject_id} best spatial CV MSE: {best_score:.6f}"
                )
                print(
                    f"          Subject {subject_id} completed {len(completed_trials)}/{len(study.trials)} trials"
                )

                return True, best_trial, best_score
            else:
                print(f"          Subject {subject_id}: No successful trials")
                return False, None, float("inf")

        except Exception as e:
            print(f"          Subject {subject_id} optimization failed: {e}")
            return False, None, float("inf")

    def fit(self, X, y):
        """
        Fit one optimized model per subject in the training data.
        Each subject gets its own hyperparameter optimization.

        Args:
            X: Training features (must include 'sub' column for subject IDs)
            y: Training targets

        Returns:
            self
        """
        if "sub" not in X.columns:
            raise ValueError("X must contain 'sub' column with subject IDs")

        # Reset state
        self.subject_models = {}
        self.subject_training_data = {}
        self.subject_best_params = {}
        self.subject_optimization_scores = {}
        self.subject_studies = {}
        self.training_subjects = []

        # Get subject groups
        subject_groups = X["sub"].values
        unique_subjects = np.unique(subject_groups)

        print(
            f"      Training ensemble with individual optimization for {len(unique_subjects)} subjects"
        )
        print(
            f"      Each subject will be optimized with {self.n_trials_per_subject} trials"
        )

        successful_subjects = []

        for subject_id in unique_subjects:
            subject_mask = subject_groups == subject_id
            X_subject = X[subject_mask]
            y_subject = y[subject_mask]

            # Check minimum sample requirement
            if len(X_subject) < self.min_samples_per_subject:
                print(
                    f"        Skipping subject {subject_id}: insufficient data ({len(X_subject)} samples)"
                )
                continue

            # Optimize hyperparameters for this specific subject
            optimization_success, best_trial, best_score = (
                self._optimize_subject_hyperparameters(X_subject, y_subject, subject_id)
            )

            if not optimization_success:
                print(f"        Subject {subject_id}: optimization failed")
                continue

            # Train final model with best parameters on all subject data
            try:
                final_pipeline = self.pipeline_factory(best_trial, X_subject)
                final_pipeline.fit(X_subject, y_subject)

                # Store everything
                self.subject_models[subject_id] = final_pipeline
                self.subject_training_data[subject_id] = X_subject.copy()
                self.subject_best_params[subject_id] = dict(best_trial.params)
                self.subject_optimization_scores[subject_id] = best_score
                successful_subjects.append(subject_id)

                print(
                    f"          Subject {subject_id} final model trained successfully"
                )

            except Exception as e:
                print(f"        Subject {subject_id} final training failed: {e}")
                continue

        self.training_subjects = successful_subjects
        self.is_fitted = len(self.subject_models) > 0

        if not self.is_fitted:
            raise ValueError("No subject models could be trained successfully")

        print(
            f"      Successfully trained and optimized {len(self.subject_models)} subject models"
        )

        # Print summary of optimized parameters
        print(f"      Parameter optimization summary:")
        for subject_id in self.training_subjects[:3]:  # Show first 3 subjects
            params = self.subject_best_params[subject_id]
            score = self.subject_optimization_scores[subject_id]
            print(f"        Subject {subject_id} (MSE: {score:.4f}): {params}")
        if len(self.training_subjects) > 3:
            print(f"        ... and {len(self.training_subjects) - 3} more subjects")

        return self

    def predict(self, X):
        """
        Get predictions from all trained subject models.

        Args:
            X: Features to predict on

        Returns:
            Dict[subject_id, predictions]: Dictionary mapping subject IDs to their predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        predictions = {}

        for subject_id, model in self.subject_models.items():
            try:
                pred = model.predict(X)

                # Basic sanity check on predictions
                if len(pred) > 0:
                    pred_mean = np.mean(pred)
                    # If prediction is completely unreasonable (e.g., negative CO2), mark as failed
                    if (
                        pred_mean < -100 or pred_mean > 50000
                    ):  # CO2 shouldn't be negative or extremely high
                        print(
                            f"        Warning: Subject {subject_id} produced unreasonable predictions (mean: {pred_mean:.2f})"
                        )
                        predictions[subject_id] = np.full(len(X), np.nan)
                    else:
                        predictions[subject_id] = pred
                else:
                    predictions[subject_id] = np.full(len(X), np.nan)

            except Exception as e:
                print(f"        Prediction failed for subject {subject_id}: {e}")
                # Return NaN array for failed predictions
                predictions[subject_id] = np.full(len(X), np.nan)

        return predictions

    def get_subject_training_data(self):
        """Get the training data for each subject (for similarity calculations)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.subject_training_data.copy()

    def get_training_subjects(self):
        """Get list of successfully trained subject IDs."""
        return self.training_subjects.copy()

    def get_best_parameters(self):
        """Get the best parameters for each subject."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.subject_best_params.copy()

    def get_optimization_scores(self):
        """Get the optimization scores for each subject."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.subject_optimization_scores.copy()

    def get_optimization_studies(self):
        """Get the Optuna studies for each subject."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.subject_studies.copy()

    def get_model_info(self):
        """Get comprehensive information about the ensemble."""
        if not self.is_fitted:
            return {"fitted": False, "n_models": 0}

        return {
            "fitted": True,
            "n_models": len(self.subject_models),
            "training_subjects": self.training_subjects,
            "best_parameters": self.subject_best_params,
            "optimization_scores": self.subject_optimization_scores,
            "mean_optimization_score": np.mean(
                list(self.subject_optimization_scores.values())
            ),
            "n_trials_per_subject": self.n_trials_per_subject,
        }
