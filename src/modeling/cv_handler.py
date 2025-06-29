from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split

from .env_similarity import UnifiedEnvironmentalSimilarity


def adjusted_r2_score(y_true, y_pred, n_features):
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    if n <= n_features + 1:
        return float("-inf")
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return adj_r2


class SpatioTemporalCV:
    def __init__(self, temporal_folds=4, spatial_folds=3, holdout_size=0.2):
        self.temporal_folds = temporal_folds
        self.spatial_folds = spatial_folds
        self.holdout_size = holdout_size

        # Initialize unified similarity calculator
        self.similarity_calc = UnifiedEnvironmentalSimilarity(method="combined")

    def _create_subject_groups(self, X):
        """Extract subject groups from the 'sub' column"""
        if "sub" not in X.columns:
            raise ValueError(
                "Column 'sub' not found in data. This column should contain subject IDs (1-20)."
            )

        subject_groups = X["sub"].values
        unique_subjects = np.unique(subject_groups)

        print(
            f"Found {len(unique_subjects)} unique subjects: {sorted(unique_subjects)}"
        )

        return subject_groups

    def _create_spatial_blocks(self, X_subject):
        if "x" not in X_subject.columns or "y" not in X_subject.columns:
            # Fallback to sequential blocks
            n_samples = len(X_subject)
            return np.arange(n_samples) % self.spatial_folds

        coords = X_subject[["x", "y"]].values
        if len(coords) < self.spatial_folds:
            return np.arange(len(coords)) % len(coords)

        kmeans = KMeans(n_clusters=self.spatial_folds, random_state=42, n_init=10)
        return kmeans.fit_predict(coords)

    def _calculate_env_similarity(self, test_features, train_features_dict):
        """
        Calculate environmental similarity using unified method for consistency.
        This replaces the old simple euclidean method.
        """
        return self.similarity_calc.calculate_similarity(
            test_features, train_features_dict
        )

    def _create_holdout_split(self, X, y):
        """Create holdout set by subjects to avoid data leakage"""
        subject_groups = self._create_subject_groups(X)
        unique_subjects = np.unique(subject_groups)
        n_holdout_subjects = max(1, int(len(unique_subjects) * self.holdout_size))

        # Randomly select subjects for holdout
        np.random.seed(42)
        holdout_subjects = np.random.choice(
            unique_subjects, n_holdout_subjects, replace=False
        )

        # Create masks
        holdout_mask = np.isin(subject_groups, holdout_subjects)
        dev_mask = ~holdout_mask

        X_dev = X[dev_mask].reset_index(drop=True)
        y_dev = y[dev_mask].reset_index(drop=True)
        X_holdout = X[holdout_mask].reset_index(drop=True)
        y_holdout = y[holdout_mask].reset_index(drop=True)

        print(
            f"Holdout split: {len(X_dev)} dev samples, {len(X_holdout)} holdout samples"
        )
        print(f"Holdout subjects: {sorted(holdout_subjects)}")

        return X_dev, y_dev, X_holdout, y_holdout

    def evaluate(self, X, y, pipeline_factory, trial):
        # Create holdout set first
        X_dev, y_dev, X_holdout, y_holdout = self._create_holdout_split(X, y)

        # Perform cross-validation on development set
        subject_groups = self._create_subject_groups(X_dev)
        temporal_cv = GroupKFold(n_splits=self.temporal_folds)

        temporal_scores = []
        all_subject_models = []

        for fold_idx, (train_idx, test_idx) in enumerate(
            temporal_cv.split(X_dev, y_dev, groups=subject_groups)
        ):
            print(f"  Temporal fold {fold_idx + 1}/{self.temporal_folds}")

            # Split by subjects
            X_train, X_test = X_dev.iloc[train_idx], X_dev.iloc[test_idx]
            y_train, y_test = y_dev.iloc[train_idx], y_dev.iloc[test_idx]

            train_subjects = subject_groups[train_idx]
            unique_train_subjects = np.unique(train_subjects)

            # Train model for each subject using spatial CV
            subject_models = []
            train_env_features = {}

            for subject_id in unique_train_subjects:
                subject_mask = train_subjects == subject_id
                X_subject = X_train[subject_mask]
                y_subject = y_train[subject_mask]

                if len(X_subject) < 10:
                    continue

                # Spatial CV within subject for hyperparameter validation
                spatial_blocks = self._create_spatial_blocks(X_subject)
                spatial_scores = []

                pipeline = pipeline_factory(trial, X_subject)

                # Spatial cross-validation within this subject
                unique_blocks = np.unique(spatial_blocks)
                for fold in range(min(self.spatial_folds, len(unique_blocks))):
                    if fold >= len(unique_blocks):
                        break

                    spatial_train_mask = spatial_blocks != fold
                    spatial_test_mask = spatial_blocks == fold

                    if not spatial_test_mask.any() or not spatial_train_mask.any():
                        continue

                    X_sp_train = X_subject[spatial_train_mask]
                    y_sp_train = y_subject[spatial_train_mask]
                    X_sp_test = X_subject[spatial_test_mask]
                    y_sp_test = y_subject[spatial_test_mask]

                    try:
                        pipeline.fit(X_sp_train, y_sp_train)
                        y_pred = pipeline.predict(X_sp_test)
                        spatial_scores.append(mean_squared_error(y_sp_test, y_pred))
                    except Exception as e:
                        print(
                            f"    Subject {subject_id}, spatial fold {fold} failed: {e}"
                        )
                        continue

                if spatial_scores:
                    # Train final model on all subject data
                    try:
                        pipeline.fit(X_subject, y_subject)
                        subject_models.append(
                            {
                                "subject_id": subject_id,
                                "pipeline": pipeline,
                                "spatial_score": np.mean(spatial_scores),
                                "n_samples": len(X_subject),
                            }
                        )
                        train_env_features[subject_id] = X_subject
                    except Exception as e:
                        print(f"    Subject {subject_id} final training failed: {e}")
                        continue

            if not subject_models:
                print(f"    No successful subject models in fold {fold_idx + 1}")
                continue

            # Store models for final ensemble
            all_subject_models.extend(subject_models)

            # Ensemble prediction with environmental similarity
            test_similarities = self._calculate_env_similarity(
                X_test, train_env_features
            )

            predictions = []
            for _, sample in X_test.iterrows():
                sample_preds = []
                weights = []

                for model_info in subject_models:
                    subject_id = model_info["subject_id"]
                    try:
                        pred = model_info["pipeline"].predict(
                            sample.values.reshape(1, -1)
                        )[0]
                        weight = test_similarities.get(subject_id, 0.1)
                        sample_preds.append(pred)
                        weights.append(weight)
                    except:
                        continue

                if sample_preds:
                    weights = np.array(weights)
                    weights = weights / weights.sum()
                    pred = np.average(sample_preds, weights=weights)
                    predictions.append(pred)
                else:
                    predictions.append(y_test.mean())

            if predictions:
                temporal_score = mean_squared_error(y_test, predictions)
                temporal_scores.append(temporal_score)
                print(f"    Fold {fold_idx + 1} MSE: {temporal_score:.6f}")

        if not temporal_scores:
            print("  No successful temporal folds")
            return float("inf")

        cv_score = np.mean(temporal_scores)
        print(f"  CV MSE: {cv_score:.6f}")

        # Final evaluation on holdout set
        print("  Evaluating on holdout set...")

        # Train final ensemble on all development data
        final_subject_models = {}
        dev_subject_groups = self._create_subject_groups(X_dev)
        unique_dev_subjects = np.unique(dev_subject_groups)

        for subject_id in unique_dev_subjects:
            subject_mask = dev_subject_groups == subject_id
            X_subject = X_dev[subject_mask]
            y_subject = y_dev[subject_mask]

            if len(X_subject) < 10:
                continue

            try:
                # Use the same hyperparameters from the trial
                pipeline = pipeline_factory(trial, X_subject)
                pipeline.fit(X_subject, y_subject)
                final_subject_models[subject_id] = {
                    "pipeline": pipeline,
                    "env_features": X_subject,
                }
            except Exception as e:
                print(f"    Final subject {subject_id} training failed: {e}")
                continue

        if not final_subject_models:
            print("  No final subject models trained")
            return float("inf")

        # Predict on holdout set
        holdout_similarities = self._calculate_env_similarity(
            X_holdout,
            {sid: info["env_features"] for sid, info in final_subject_models.items()},
        )

        holdout_predictions = []
        for _, sample in X_holdout.iterrows():
            sample_preds = []
            weights = []

            for subject_id, model_info in final_subject_models.items():
                try:
                    pred = model_info["pipeline"].predict(sample.values.reshape(1, -1))[
                        0
                    ]
                    weight = holdout_similarities.get(subject_id, 0.1)
                    sample_preds.append(pred)
                    weights.append(weight)
                except:
                    continue

            if sample_preds:
                weights = np.array(weights)
                weights = weights / weights.sum()
                pred = np.average(sample_preds, weights=weights)
                holdout_predictions.append(pred)
            else:
                holdout_predictions.append(y_holdout.mean())

        # Calculate holdout metrics
        holdout_mse = mean_squared_error(y_holdout, holdout_predictions)
        holdout_mae = mean_absolute_error(y_holdout, holdout_predictions)
        holdout_r2 = r2_score(y_holdout, holdout_predictions)
        holdout_adj_r2 = adjusted_r2_score(
            y_holdout, holdout_predictions, X_holdout.shape[1]
        )

        # Calculate development set metrics for comparison
        dev_predictions = []
        dev_similarities = self._calculate_env_similarity(
            X_dev,
            {sid: info["env_features"] for sid, info in final_subject_models.items()},
        )

        for _, sample in X_dev.iterrows():
            sample_preds = []
            weights = []

            for subject_id, model_info in final_subject_models.items():
                try:
                    pred = model_info["pipeline"].predict(sample.values.reshape(1, -1))[
                        0
                    ]
                    weight = dev_similarities.get(subject_id, 0.1)
                    sample_preds.append(pred)
                    weights.append(weight)
                except:
                    continue

            if sample_preds:
                weights = np.array(weights)
                weights = weights / weights.sum()
                pred = np.average(sample_preds, weights=weights)
                dev_predictions.append(pred)
            else:
                dev_predictions.append(y_dev.mean())

        dev_mse = mean_squared_error(y_dev, dev_predictions)
        dev_r2 = r2_score(y_dev, dev_predictions)

        print(f"  Development MSE: {dev_mse:.6f}, R²: {dev_r2:.6f}")
        print(f"  Holdout MSE: {holdout_mse:.6f}, R²: {holdout_r2:.6f}")
        print(
            f"  Overfitting check: {(holdout_mse - dev_mse) / dev_mse * 100:.1f}% increase in MSE"
        )

        # Store additional metrics in trial for analysis
        if hasattr(trial, "set_user_attr"):
            trial.set_user_attr("cv_mse", cv_score)
            trial.set_user_attr("dev_mse", dev_mse)
            trial.set_user_attr("dev_r2", dev_r2)
            trial.set_user_attr("holdout_mse", holdout_mse)
            trial.set_user_attr("holdout_mae", holdout_mae)
            trial.set_user_attr("holdout_r2", holdout_r2)
            trial.set_user_attr("holdout_adj_r2", holdout_adj_r2)
            trial.set_user_attr(
                "overfitting_ratio",
                holdout_mse / dev_mse if dev_mse > 0 else float("inf"),
            )
            trial.set_user_attr("n_final_models", len(final_subject_models))

        # Return CV score for optimization (this is what Optuna minimizes)
        return cv_score
