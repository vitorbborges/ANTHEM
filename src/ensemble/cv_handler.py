from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from .ensemble_model import EnsembleModel
from .env_similarity import UnifiedEnvironmentalSimilarity


def adjusted_r2_score(y_true, y_pred, n_features):
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    if n <= n_features + 1:
        return float("-inf")
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return adj_r2


def safe_mse_calculation(y_true, y_pred):
    """Calculate MSE with numerical stability checks."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Check for NaN or infinite values
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        print(f"      Warning: Invalid predictions detected (NaN/Inf)")
        return float("inf")

    # Calculate MSE
    mse = np.mean((y_true - y_pred) ** 2)

    # Check if MSE is reasonable relative to target variance
    target_var = np.var(y_true)
    target_mean = np.mean(y_true)

    # More lenient check - MSE should be reasonable relative to the data scale
    reasonable_mse_threshold = max(
        1e6, 1000 * target_var
    )  # Allow up to 1000x target variance

    if mse > reasonable_mse_threshold:
        print(f"      Warning: Very large MSE: {mse:.2e}")
        print(f"      Target mean: {target_mean:.2f}, var: {target_var:.2e}")
        print(
            f"      Prediction mean: {np.mean(y_pred):.2f}, var: {np.var(y_pred):.2e}"
        )
        # Don't reject - let's see what's happening

    return float(mse)


class SpatioTemporalCV:
    def __init__(
        self,
        temporal_folds=4,
        spatial_folds=3,
        holdout_size=0.2,
        n_trials_per_subject=15,
    ):
        self.temporal_folds = temporal_folds
        self.spatial_folds = spatial_folds
        self.holdout_size = holdout_size
        self.n_trials_per_subject = n_trials_per_subject

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

    def _create_holdout_split(self, X, y):
        """Create holdout set by subjects to avoid data leakage"""
        subject_groups = self._create_subject_groups(X)
        unique_subjects = np.unique(subject_groups)
        n_holdout_subjects = max(1, int(len(unique_subjects) * self.holdout_size))

        # Randomly select subjects for holdout (4 subjects)
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
        print(
            f"Development subjects: {sorted([s for s in unique_subjects if s not in holdout_subjects])}"
        )

        return X_dev, y_dev, X_holdout, y_holdout

    def _create_similarity_calculator(self, trial):
        """
        Create environmental similarity calculator with trial-optimized parameters.
        This is what the outer Optuna study will optimize.

        Args:
            trial: Optuna trial for similarity hyperparameters

        Returns:
            UnifiedEnvironmentalSimilarity instance
        """
        # Optimize similarity method for environmental features
        similarity_method = trial.suggest_categorical(
            "similarity_method",
            [
                "euclidean",
                "cosine",
                "combined",
                "simple_average",
            ],  # Add simple average option
        )

        # Optimize normalization
        normalize_features = trial.suggest_categorical(
            "normalize_features", [True, False]
        )

        if similarity_method == "simple_average":
            # Return a dummy calculator that will use equal weights
            return None
        else:
            return UnifiedEnvironmentalSimilarity(
                method=similarity_method, normalize_features=normalize_features
            )

    def _ensemble_predict(
        self, ensemble_predictions, test_features, train_features_dict, similarity_calc
    ):
        """
        Combine individual model predictions using environmental similarity weighting.
        """
        # Calculate environmental similarities using the provided calculator
        try:
            if similarity_calc is None:
                # Simple average - equal weights for all subjects
                similarities = {sid: 1.0 for sid in ensemble_predictions.keys()}
                print(f"      Using simple average (equal weights)")
            else:
                similarities = similarity_calc.calculate_similarity(
                    test_features, train_features_dict
                )
        except Exception as e:
            print(
                f"      Warning: Similarity calculation failed: {e}. Using equal weights."
            )
            similarities = {sid: 1.0 for sid in ensemble_predictions.keys()}

        # Get all subject IDs and ensure we have predictions for them
        subject_ids = list(ensemble_predictions.keys())
        if not subject_ids:
            print(f"      Error: No predictions available")
            return np.array([])

        n_samples = len(next(iter(ensemble_predictions.values())))
        final_predictions = []

        for sample_idx in range(n_samples):
            sample_preds = []
            weights = []

            for subject_id in subject_ids:
                pred_array = ensemble_predictions[subject_id]

                # Skip if prediction failed (NaN)
                if np.isnan(pred_array[sample_idx]) or np.isinf(pred_array[sample_idx]):
                    continue

                pred_value = pred_array[sample_idx]

                # Outlier detection: skip predictions that are way off
                # Check if prediction is extremely different from others
                if sample_preds:  # If we already have some predictions
                    current_median = np.median(sample_preds)
                    if abs(pred_value - current_median) > 10 * np.std(
                        sample_preds + [pred_value]
                    ):
                        print(
                            f"        Warning: Outlier prediction from subject {subject_id}: {pred_value:.2f} (median: {current_median:.2f})"
                        )
                        continue

                sample_preds.append(pred_value)
                weight = similarities.get(subject_id, 1.0)  # Default weight of 1.0
                weights.append(weight)

            # Ensemble prediction using weighted average
            if sample_preds:
                weights = np.array(weights)
                # Normalize weights
                if np.sum(weights) > 0:
                    weights = weights / np.sum(weights)
                else:
                    weights = np.ones_like(weights) / len(weights)

                ensemble_pred = np.average(sample_preds, weights=weights)

                # Sanity check on ensemble prediction
                if np.isnan(ensemble_pred) or np.isinf(ensemble_pred):
                    # Fallback to simple average
                    ensemble_pred = np.mean(sample_preds)

                final_predictions.append(ensemble_pred)
            else:
                # Fallback if no models could predict - use mean of all available predictions
                all_valid_preds = []
                for preds in ensemble_predictions.values():
                    for pred in preds:
                        if not (np.isnan(pred) or np.isinf(pred)):
                            all_valid_preds.append(pred)

                if all_valid_preds:
                    fallback_pred = np.mean(all_valid_preds)
                else:
                    fallback_pred = 0.0  # Last resort

                final_predictions.append(fallback_pred)

        result = np.array(final_predictions)

        # Final sanity check
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            print(f"      Warning: Invalid ensemble predictions detected")
            # Replace invalid predictions with median of valid ones
            valid_mask = ~(np.isnan(result) | np.isinf(result))
            if np.any(valid_mask):
                median_val = np.median(result[valid_mask])
                result[~valid_mask] = median_val
            else:
                result = np.zeros_like(result)

        return result

    def evaluate(self, X, y, pipeline_factory, trial):
        """
        Main evaluation function with nested optimization structure:

        OUTER OPTUNA STUDY (this trial):
        - Optimizes environmental similarity method and parameters

        INNER OPTUNA STUDIES (within EnsembleModel):
        - Each of the 12 subjects gets its own hyperparameter optimization
        - Uses spatial CV to find best model parameters for each subject

        CV STRATEGY:
        1. Split 20 subjects into 4 holdout + 16 development
        2. 4-fold temporal CV on 16 subjects (12 train + 4 test per fold)
        3. For each fold:
           a. Train ensemble with 12 individually optimized subject models
           b. Test ensemble aggregation (optimized by outer trial) on 4 validation subjects
        4. Return mean CV score across temporal folds
        """
        # Create holdout set first (4 subjects)
        X_dev, y_dev, X_holdout, y_holdout = self._create_holdout_split(X, y)

        # Create similarity calculator with trial-optimized parameters
        similarity_calc = self._create_similarity_calculator(trial)

        # Perform cross-validation on development set (16 subjects)
        subject_groups = self._create_subject_groups(X_dev)
        temporal_cv = GroupKFold(n_splits=self.temporal_folds)

        temporal_scores = []

        for fold_idx, (train_idx, test_idx) in enumerate(
            temporal_cv.split(X_dev, y_dev, groups=subject_groups)
        ):
            print(f"  Temporal fold {fold_idx + 1}/{self.temporal_folds}")

            # Split by subjects: 12 train subjects + 4 validation subjects
            X_train, X_test = X_dev.iloc[train_idx], X_dev.iloc[test_idx]
            y_train, y_test = y_dev.iloc[train_idx], y_dev.iloc[test_idx]

            train_subjects = subject_groups[train_idx]
            test_subjects = subject_groups[test_idx]
            unique_train_subjects = np.unique(train_subjects)
            unique_test_subjects = np.unique(test_subjects)

            print(
                f"    Train subjects ({len(unique_train_subjects)}): {sorted(unique_train_subjects)}"
            )
            print(
                f"    Test subjects ({len(unique_test_subjects)}): {sorted(unique_test_subjects)}"
            )

            # Train ensemble model with individual subject optimization
            try:
                ensemble_model = EnsembleModel(
                    pipeline_factory=pipeline_factory,
                    spatial_folds=self.spatial_folds,
                    n_trials_per_subject=self.n_trials_per_subject,
                )

                # This will perform nested optimization: one Optuna study per subject
                ensemble_model.fit(X_train, y_train)

                # Get individual predictions from all optimized subject models
                ensemble_predictions = ensemble_model.predict(X_test)

                # Get training data for similarity calculation
                train_features_dict = ensemble_model.get_subject_training_data()

                # Combine predictions using trial-optimized similarity method
                final_predictions = self._ensemble_predict(
                    ensemble_predictions, X_test, train_features_dict, similarity_calc
                )

                # Calculate fold score
                temporal_score = safe_mse_calculation(y_test, final_predictions)

                # Debug information
                print(f"    Fold {fold_idx + 1} debug info:")
                print(f"      Test samples: {len(y_test)}")
                print(
                    f"      Test target range: [{y_test.min():.2f}, {y_test.max():.2f}]"
                )
                print(
                    f"      Prediction range: [{final_predictions.min():.2f}, {final_predictions.max():.2f}]"
                )
                print(f"      Individual model predictions:")
                for sid, preds in ensemble_predictions.items():
                    pred_sample = preds[:3]  # First 3 predictions
                    print(
                        f"        Subject {sid}: [{pred_sample[0]:.2f}, {pred_sample[1]:.2f}, {pred_sample[2]:.2f}]"
                    )

                # Only add valid scores
                if temporal_score != float("inf"):
                    temporal_scores.append(temporal_score)

                    print(f"    Fold {fold_idx + 1} ensemble MSE: {temporal_score:.6f}")
                else:
                    print(f"    Fold {fold_idx + 1} produced invalid MSE, skipping")
                print(
                    f"    Fold {fold_idx + 1} mean subject optimization score: {mean_subject_score:.6f}"
                )

            except Exception as e:
                print(f"    Fold {fold_idx + 1} failed: {e}")
                continue

        # Calculate final CV score
        if not temporal_scores or all(
            score == float("inf") for score in temporal_scores
        ):
            print("  All temporal folds failed")
            return float("inf")

        # Remove any infinite scores and average the rest
        valid_scores = [score for score in temporal_scores if score != float("inf")]
        if not valid_scores:
            return float("inf")

        cv_score = np.mean(valid_scores)
        print(f"  Final CV MSE: {cv_score:.6f}")
        print(
            f"  Similarity method: {trial.params.get('similarity_method', 'unknown')}"
        )
        print(
            f"  Normalize features: {trial.params.get('normalize_features', 'unknown')}"
        )

        # Optional: Final evaluation on holdout set for additional metrics
        try:
            print("  Training final ensemble for holdout evaluation...")

            # Train final ensemble on all development data (16 subjects)
            final_ensemble = EnsembleModel(
                pipeline_factory=pipeline_factory,
                spatial_folds=self.spatial_folds,
                n_trials_per_subject=self.n_trials_per_subject,
            )

            final_ensemble.fit(X_dev, y_dev)

            # Predict on holdout set using optimized similarity method
            holdout_ensemble_predictions = final_ensemble.predict(X_holdout)
            final_train_features_dict = final_ensemble.get_subject_training_data()

            holdout_predictions = self._ensemble_predict(
                holdout_ensemble_predictions,
                X_holdout,
                final_train_features_dict,
                similarity_calc,
            )

            # Calculate holdout metrics
            holdout_mse = mean_squared_error(y_holdout, holdout_predictions)
            holdout_r2 = r2_score(y_holdout, holdout_predictions)

            print(f"  Holdout MSE: {holdout_mse:.6f}, R²: {holdout_r2:.6f}")

            # Store metrics in trial
            if hasattr(trial, "set_user_attr"):
                trial.set_user_attr("cv_mse", cv_score)
                trial.set_user_attr("holdout_mse", holdout_mse)
                trial.set_user_attr("holdout_r2", holdout_r2)
                trial.set_user_attr(
                    "n_final_models", len(final_ensemble.get_training_subjects())
                )
                trial.set_user_attr("successful_folds", len(valid_scores))
                trial.set_user_attr("total_folds", len(temporal_scores))

                # Store some example subject parameters for analysis
                best_params = final_ensemble.get_best_parameters()
                if best_params:
                    example_subject = list(best_params.keys())[0]
                    trial.set_user_attr(
                        f"example_subject_{example_subject}_params",
                        best_params[example_subject],
                    )

        except Exception as e:
            print(f"  Holdout evaluation failed: {e}")

        return cv_score
