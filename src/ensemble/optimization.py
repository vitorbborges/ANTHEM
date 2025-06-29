import json
import os
import pickle
from datetime import datetime
from typing import Any, Dict

import numpy as np
import optuna
import pandas as pd

from .cv_handler import SpatioTemporalCV
from .ensemble_model import EnsembleModel
from .pipeline_factory import create_pipeline


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {convert_numpy_types(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj


def safe_mse_calculation(y_true, y_pred):
    """Calculate MSE with numerical stability checks."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Check for NaN or infinite values
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        print(f"    Warning: Invalid predictions detected (NaN/Inf)")
        return float("inf")

    # Calculate MSE
    mse = np.mean((y_true - y_pred) ** 2)

    # Check if MSE is reasonable
    if mse > 1e10:  # Very large MSE indicates problems
        print(f"    Warning: Very large MSE detected: {mse:.2e}")
        return float("inf")

    return float(mse)


def load_data(file_path):
    """Load and preprocess the dataset with better target scaling."""
    df = pd.read_parquet(file_path)

    # Keep only numeric columns
    numeric_cols = [col for col in df.columns if df[col].dtype in ("float64", "int64")]
    remove_cols = ["PM1", "PM10", "VOC"]
    keep_cols = [col for col in numeric_cols if col not in remove_cols]

    X = df[keep_cols].dropna()
    y = X.pop("CO2")

    print(f"Loaded {len(X)} samples with {X.shape[1]} features")
    print(f"Subjects in dataset: {sorted(X['sub'].unique())}")

    # Check target variable statistics
    print(
        f"CO2 statistics: mean={y.mean():.2f}, std={y.std():.2f}, min={y.min():.2f}, max={y.max():.2f}"
    )

    # Check for outliers in target
    q99 = y.quantile(0.99)
    q01 = y.quantile(0.01)
    print(f"CO2 percentiles: 1%={q01:.2f}, 99%={q99:.2f}")

    # Optional: Remove extreme outliers that might cause numerical issues
    outlier_mask = (y >= q01) & (y <= q99)
    if outlier_mask.sum() < len(y):
        print(f"Removing {len(y) - outlier_mask.sum()} extreme outliers")
        X = X[outlier_mask]
        y = y[outlier_mask]

    # Check for environmental features
    env_features = ["T", "RH", "P", "velocita_vento_medio", "radiazione_globale_medio"]
    available_env = [f for f in env_features if f in X.columns]
    print(f"Available environmental features: {available_env}")

    return X, y


class OptimizationRunner:
    def __init__(
        self,
        data_path,
        temporal_folds=4,
        spatial_folds=3,
        holdout_size=0.2,
        n_trials_per_subject=10,
    ):  # Reduced from 15 for stability
        """
        Initialize the optimization runner.

        Args:
            data_path: Path to the parquet data file
            temporal_folds: Number of temporal folds for CV (4 means 12 train + 4 test subjects)
            spatial_folds: Number of spatial folds within each subject
            holdout_size: Fraction of subjects for holdout (0.2 = 4 out of 20 subjects)
            n_trials_per_subject: Number of optimization trials per subject model
        """
        self.X, self.y = load_data(data_path)
        self.cv = SpatioTemporalCV(
            temporal_folds=temporal_folds,
            spatial_folds=spatial_folds,
            holdout_size=holdout_size,
            n_trials_per_subject=n_trials_per_subject,
        )

        # Create directories for saving models and results
        os.makedirs("models", exist_ok=True)
        os.makedirs("metrics", exist_ok=True)

        # Store trial results for saving models after completion
        self.completed_trials = {}

    def objective(self, trial):
        """
        Objective function for the main Optuna study.

        This optimizes the environmental similarity aggregation method.
        Each trial triggers nested optimization of individual subject models.
        """
        try:
            print(
                f"\n🔍 TRIAL {trial.number}: Testing similarity aggregation parameters"
            )
            print(f"   Parameters: {trial.params}")

            # The CV will optimize similarity method and run nested subject optimizations
            mse = self.cv.evaluate(self.X, self.y, create_pipeline, trial)

            # Validate MSE before storing
            if mse == float("inf") or mse > 1e10 or np.isnan(mse):
                print(f"❌ Trial {trial.number} produced invalid MSE: {mse}")
                return float("inf")

            # Store trial result for later model saving
            self.completed_trials[trial.number] = {
                "trial": trial,
                "mse": float(mse),  # Ensure it's a regular float
                "params": convert_numpy_types(trial.params.copy()),
                "user_attrs": convert_numpy_types(
                    dict(trial.user_attrs) if trial.user_attrs else {}
                ),
            }
            print(f"✅ Trial {trial.number} completed with MSE: {mse:.6f}")

            return float(mse)

        except Exception as e:
            print(f"❌ Trial {trial.number} failed with error: {e}")
            import traceback

            traceback.print_exc()
            return float("inf")

    def _save_trial_model(self, trial_number, trial_data):
        """Save the ensemble model and results for a completed trial."""
        try:
            trial = trial_data["trial"]
            mse = trial_data["mse"]

            print(f"🔄 Training final ensemble for trial {trial_number}...")

            # Train final ensemble on all development data with best parameters
            from .cv_handler import SpatioTemporalCV

            temp_cv = SpatioTemporalCV(
                temporal_folds=self.cv.temporal_folds,
                spatial_folds=self.cv.spatial_folds,
                holdout_size=self.cv.holdout_size,
                n_trials_per_subject=self.cv.n_trials_per_subject,
            )

            # Get development data (excluding holdout)
            X_dev, y_dev, X_holdout, y_holdout = temp_cv._create_holdout_split(
                self.X, self.y
            )

            # Create final ensemble with trial parameters
            final_ensemble = EnsembleModel(
                pipeline_factory=create_pipeline,
                spatial_folds=self.cv.spatial_folds,
                n_trials_per_subject=self.cv.n_trials_per_subject,
            )

            # Train with individual subject optimization
            final_ensemble.fit(X_dev, y_dev)

            # Convert all data to JSON-serializable format
            ensemble_data = {
                "ensemble_model": final_ensemble,
                "similarity_params": convert_numpy_types(trial_data["params"]),
                "trial_number": int(trial_number),
                "trial_value": float(mse),
                "user_attrs": convert_numpy_types(trial_data["user_attrs"]),
                "timestamp": datetime.now().isoformat(),
                "subject_best_params": convert_numpy_types(
                    final_ensemble.get_best_parameters()
                ),
                "subject_optimization_scores": convert_numpy_types(
                    final_ensemble.get_optimization_scores()
                ),
                "training_subjects": [
                    int(x) for x in final_ensemble.get_training_subjects()
                ],
            }

            model_path = f"models/ensemble_trial_{trial_number}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(ensemble_data, f)

            print(f"💾 Saved ensemble model: {model_path}")
            print(
                f"   Ensemble size: {len(final_ensemble.get_training_subjects())} subject models"
            )

            # Save detailed trial summary (JSON-safe)
            trial_summary = convert_numpy_types(
                {
                    "trial_number": trial_number,
                    "trial_value": mse,
                    "similarity_params": trial_data["params"],
                    "user_attrs": trial_data["user_attrs"],
                    "subject_models_info": {
                        "n_subject_models": len(final_ensemble.get_training_subjects()),
                        "training_subjects": final_ensemble.get_training_subjects(),
                        "mean_subject_optimization_score": np.mean(
                            list(final_ensemble.get_optimization_scores().values())
                        ),
                        "subject_optimization_scores": final_ensemble.get_optimization_scores(),
                    },
                    "example_subject_params": {
                        str(sid): params
                        for sid, params in list(
                            final_ensemble.get_best_parameters().items()
                        )[:3]
                    },
                    "timestamp": datetime.now().isoformat(),
                }
            )

            summary_path = f"metrics/trial_{trial_number}_summary.json"
            with open(summary_path, "w") as f:
                json.dump(trial_summary, f, indent=2, default=str)

            return True

        except Exception as e:
            print(f"❌ Failed to save model for trial {trial_number}: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _save_best_model(self, study):
        """Save the best performing model with detailed analysis."""
        if not study.best_trial:
            print("⚠️ No best trial found")
            return

        best_trial = study.best_trial
        print(f"\n💎 Saving best model from trial {best_trial.number}...")

        # Copy the best trial's model to a special location
        source_path = f"models/ensemble_trial_{best_trial.number}.pkl"
        best_path = "models/best_ensemble_model.pkl"

        try:
            if os.path.exists(source_path):
                # Load and re-save with additional metadata
                with open(source_path, "rb") as f:
                    model_data = pickle.load(f)

                # Add best model metadata
                model_data["is_best_model"] = True
                model_data["selection_timestamp"] = datetime.now().isoformat()
                model_data["total_trials"] = len(study.trials)
                model_data["completed_trials"] = len(
                    [
                        t
                        for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE
                    ]
                )

                with open(best_path, "wb") as f:
                    pickle.dump(model_data, f)

                print(f"✅ Best model saved: {best_path}")

                # Create comprehensive best model analysis (JSON-safe)
                best_summary = convert_numpy_types(
                    {
                        "optimization_summary": {
                            "best_trial_number": best_trial.number,
                            "best_cv_mse": best_trial.value,
                            "total_trials": len(study.trials),
                            "completed_trials": len(
                                [
                                    t
                                    for t in study.trials
                                    if t.state == optuna.trial.TrialState.COMPLETE
                                ]
                            ),
                            "selection_timestamp": datetime.now().isoformat(),
                        },
                        "best_similarity_params": best_trial.params,
                        "performance_metrics": (
                            dict(best_trial.user_attrs) if best_trial.user_attrs else {}
                        ),
                        "ensemble_details": {
                            "n_subject_models": len(model_data["training_subjects"]),
                            "training_subjects": model_data["training_subjects"],
                            "mean_subject_optimization_score": np.mean(
                                list(model_data["subject_optimization_scores"].values())
                            ),
                        },
                        "all_subject_model_parameters": model_data[
                            "subject_best_params"
                        ],
                        "all_subject_optimization_scores": model_data[
                            "subject_optimization_scores"
                        ],
                        "subject_model_analysis": self._analyze_subject_models(
                            model_data
                        ),
                        "model_path": best_path,
                    }
                )

                with open("metrics/best_model_analysis.json", "w") as f:
                    json.dump(best_summary, f, indent=2, default=str)

                print(f"✅ Best model analysis: metrics/best_model_analysis.json")

            else:
                print(f"❌ Model file not found: {source_path}")

        except Exception as e:
            print(f"❌ Failed to save best model: {e}")
            import traceback

            traceback.print_exc()

    def _analyze_subject_models(self, model_data) -> Dict[str, Any]:
        """Analyze the individual subject models in the ensemble."""
        try:
            subject_params = model_data["subject_best_params"]
            subject_scores = model_data["subject_optimization_scores"]

            if not subject_params or not subject_scores:
                return {"error": "No subject model data available"}

            analysis = {
                "parameter_diversity": {},
                "performance_statistics": {
                    "best_subject_score": float(min(subject_scores.values())),
                    "worst_subject_score": float(max(subject_scores.values())),
                    "mean_subject_score": float(np.mean(list(subject_scores.values()))),
                    "std_subject_score": float(np.std(list(subject_scores.values()))),
                },
                "top_performing_subjects": [
                    [int(sid), float(score)]
                    for sid, score in sorted(
                        subject_scores.items(), key=lambda x: x[1]
                    )[:5]
                ],
                "parameter_frequency": {},
                "detailed_subject_models": {},
            }

            # Detailed analysis for each subject
            for subject_id, params in subject_params.items():
                score = subject_scores.get(subject_id, float("inf"))
                analysis["detailed_subject_models"][str(subject_id)] = {
                    "optimization_score": float(score),
                    "all_parameters": convert_numpy_types(params),
                    "drift_model": params.get("drift_model_type", "unknown"),
                    "variogram_model": params.get("variogram_model", "unknown"),
                }

            # Analyze parameter diversity
            all_param_keys = set()
            for params in subject_params.values():
                all_param_keys.update(params.keys())

            for param_key in all_param_keys:
                param_values = [
                    params.get(param_key)
                    for params in subject_params.values()
                    if param_key in params
                ]

                if param_values:
                    if isinstance(param_values[0], (int, float)):
                        analysis["parameter_diversity"][param_key] = {
                            "mean": float(np.mean(param_values)),
                            "std": float(np.std(param_values)),
                            "min": float(min(param_values)),
                            "max": float(max(param_values)),
                        }
                    else:
                        # Categorical parameters
                        unique_values = list(set(param_values))
                        analysis["parameter_frequency"][param_key] = {
                            str(val): param_values.count(val) for val in unique_values
                        }

            return convert_numpy_types(analysis)

        except Exception as e:
            return {"error": f"Failed to analyze subject models: {e}"}

    def _save_all_completed_models(self):
        """Save models for all completed trials."""
        print(
            f"\n🔄 Saving models for {len(self.completed_trials)} completed trials..."
        )

        saved_count = 0
        for trial_number, trial_data in self.completed_trials.items():
            if self._save_trial_model(trial_number, trial_data):
                saved_count += 1

        print(
            f"✅ Successfully saved {saved_count}/{len(self.completed_trials)} models"
        )

    def run(self, n_trials=10):  # Reduced from 20 for initial testing
        """
        Run the main optimization process.

        Args:
            n_trials: Number of trials for the main similarity optimization
        """
        print("=" * 80)
        print("STARTING NESTED OPTIMIZATION")
        print("=" * 80)
        print(f"Main optimization: {n_trials} trials for similarity aggregation")
        print(
            f"Nested optimization: {self.cv.n_trials_per_subject} trials per subject model"
        )
        print(f"Target statistics: mean={self.y.mean():.2f}, std={self.y.std():.2f}")
        print("=" * 80)

        # Create main Optuna study for similarity optimization
        study = optuna.create_study(
            direction="minimize",
            study_name="similarity_aggregation_optimization",
            sampler=optuna.samplers.TPESampler(seed=42),
            storage="sqlite:///metrics/optuna-study.db",
            load_if_exists=True,
        )

        # Run optimization
        study.optimize(self.objective, n_trials=n_trials, n_jobs=-1)

        # Save models for all completed trials
        if self.completed_trials:
            self._save_all_completed_models()

        # Display optimization results
        self._print_optimization_results(study)

        return study

    def _print_optimization_results(self, study):
        """Print comprehensive optimization results."""
        print("\n" + "=" * 80)
        print("OPTIMIZATION RESULTS")
        print("=" * 80)

        completed_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]

        if completed_trials:
            print(f"✅ Completed trials: {len(completed_trials)}/{len(study.trials)}")
            print(f"🎯 Best CV MSE: {study.best_trial.value:.6f}")
            print(f"📊 Best trial number: {study.best_trial.number}")

            print(f"\n🔧 Best Similarity Parameters:")
            for key, value in study.best_trial.params.items():
                print(f"   {key}: {value}")

            # Show parameter diversity across all subjects
            best_model_path = f"models/ensemble_trial_{study.best_trial.number}.pkl"
            if os.path.exists(best_model_path):
                try:
                    with open(best_model_path, "rb") as f:
                        model_data = pickle.load(f)

                    all_subject_params = model_data.get("subject_best_params", {})
                    all_subject_scores = model_data.get(
                        "subject_optimization_scores", {}
                    )

                    print(
                        f"\n🔧 Subject Model Parameters (showing key parameters for each subject):"
                    )

                    # Show parameters for all subjects in an organized way
                    for subject_id in sorted(all_subject_params.keys(), key=int):
                        params = all_subject_params[subject_id]
                        score = all_subject_scores.get(subject_id, float("inf"))

                        # Extract key parameters for display
                        key_info = []

                        # Drift model type
                        if "drift_model_type" in params:
                            key_info.append(f"drift={params['drift_model_type']}")

                        # Variogram model
                        if "variogram_model" in params:
                            key_info.append(f"variogram={params['variogram_model']}")

                        # Key hyperparameters (show the most important ones)
                        for key in [
                            "ridge__alpha",
                            "lasso__alpha",
                            "lasso_model__alpha",
                            "random_forest__n_estimators",
                            "svr__C",
                        ]:
                            if key in params:
                                key_info.append(f"{key}={params[key]:.4f}")
                                break  # Show only one main hyperparameter

                        key_info_str = ", ".join(
                            key_info[:3]
                        )  # Limit to 3 key parameters
                        print(
                            f"   Subject {subject_id:2d} (MSE: {score:8.2f}): {key_info_str}"
                        )

                    print(
                        f"\n   📋 Complete parameter sets saved in: metrics/best_model_analysis.json"
                    )

                except Exception as e:
                    print(f"   ⚠️ Could not load detailed parameters: {e}")

            # Display other performance metrics
            if study.best_trial.user_attrs:
                attrs = study.best_trial.user_attrs
                print(f"\n📈 Best Trial Performance Metrics:")
                for key, value in attrs.items():
                    if isinstance(value, float):
                        print(f"   {key}: {value:.6f}")
                    else:
                        print(f"   {key}: {value}")

            # Show trial progression
            print(f"\n📊 Trial Progression (best 5 trials):")
            best_trials = sorted(completed_trials, key=lambda x: x.value)[:5]
            for i, trial in enumerate(best_trials):
                status = "🥇" if i == 0 else f"🏅{i+1}"
                print(f"   {status} Trial {trial.number}: MSE = {trial.value:.6f}")
                print(
                    f"      Similarity: {trial.params.get('similarity_method', 'unknown')}"
                )

            # Save the best model with comprehensive analysis
            self._save_best_model(study)

            print(f"\n💾 Output Files:")
            print(f"   📁 Best model: models/best_ensemble_model.pkl")
            print(f"   📁 Best analysis: metrics/best_model_analysis.json")
            print(f"   📁 All trials: models/ensemble_trial_*.pkl")
            print(f"   📁 Trial summaries: metrics/trial_*_summary.json")

        else:
            failed_trials = len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
            )
            print(f"❌ No trials completed successfully")
            print(f"   Total trials: {len(study.trials)}")
            print(f"   Failed trials: {failed_trials}")
            print(f"\n💡 Troubleshooting suggestions:")
            print(f"   - Check data quality and format")
            print(f"   - Verify 'sub' column contains subject IDs")
            print(f"   - Check for environmental features (T, RH, P, etc.)")
            print(f"   - Consider reducing n_trials_per_subject")
            print(f"   - Check for extreme outliers in CO2 values")

        print("=" * 80)
