import json
import os
import pickle
from datetime import datetime

import optuna
import pandas as pd

from .cv_handler import SpatioTemporalCV
from .pipeline_factory import create_pipeline


def load_data(file_path):
    df = pd.read_parquet(file_path)

    # Keep only numeric columns
    numeric_cols = [col for col in df.columns if df[col].dtype in ("float64", "int64")]
    remove_cols = ["PM1", "PM10", "VOC"]
    keep_cols = [col for col in numeric_cols if col not in remove_cols]

    X = df[keep_cols].dropna()
    y = X.pop("CO2")

    print(f"Loaded {len(X)} samples with {X.shape[1]} features")
    return X, y


class OptimizationRunner:
    def __init__(self, data_path, temporal_folds=4, spatial_folds=3, holdout_size=0.2):
        self.X, self.y = load_data(data_path)
        self.cv = SpatioTemporalCV(temporal_folds, spatial_folds, holdout_size)

        # Create directories for saving models and results
        os.makedirs("models", exist_ok=True)
        os.makedirs("metrics", exist_ok=True)

        # Store trial results for saving models after completion
        self.completed_trials = {}

    def objective(self, trial):
        try:
            mse = self.cv.evaluate(self.X, self.y, create_pipeline, trial)

            # Store trial result for later model saving
            if mse != float("inf"):
                self.completed_trials[trial.number] = {
                    "trial": trial,
                    "mse": mse,
                    "params": trial.params.copy(),
                    "user_attrs": dict(trial.user_attrs) if trial.user_attrs else {},
                }

            return mse
        except Exception as e:
            print(f"Trial failed: {e}")
            return float("inf")

    def _save_trial_model(self, trial_number, trial_data):
        """Save the model and results for a completed trial."""
        try:
            trial = trial_data["trial"]
            mse = trial_data["mse"]

            print(f"🔄 Training final ensemble for trial {trial_number}...")

            # Get subject groups and train model for each subject
            subject_groups = self.cv._create_subject_groups(self.X)
            unique_subjects = list(set(subject_groups))

            final_models = []

            for subject_id in unique_subjects:
                subject_mask = subject_groups == subject_id
                X_subject = self.X[subject_mask]
                y_subject = self.y[subject_mask]

                if len(X_subject) < 10:  # Skip subjects with insufficient data
                    continue

                try:
                    # Create pipeline with trial hyperparameters
                    pipeline = create_pipeline(trial, X_subject)
                    pipeline.fit(X_subject, y_subject)

                    final_models.append(
                        {
                            "subject_id": subject_id,
                            "pipeline": pipeline,
                            "n_samples": len(X_subject),
                        }
                    )

                except Exception as e:
                    print(f"   Failed to train model for subject {subject_id}: {e}")
                    continue

            if final_models:
                # Save ensemble model
                ensemble_data = {
                    "final_models": final_models,
                    "trial_params": trial_data["params"],
                    "trial_number": trial_number,
                    "trial_value": mse,
                    "user_attrs": trial_data["user_attrs"],
                    "timestamp": datetime.now().isoformat(),
                }

                model_path = f"models/ensemble_trial_{trial_number}.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(ensemble_data, f)

                print(f"💾 Saved ensemble model: {model_path}")
                print(f"   Ensemble size: {len(final_models)} subject models")

                # Save trial summary
                trial_summary = {
                    "trial_number": trial_number,
                    "trial_value": mse,
                    "params": trial_data["params"],
                    "user_attrs": trial_data["user_attrs"],
                    "n_subject_models": len(final_models),
                    "subject_ids": [m["subject_id"] for m in final_models],
                    "timestamp": datetime.now().isoformat(),
                }

                summary_path = f"metrics/trial_{trial_number}_summary.json"
                with open(summary_path, "w") as f:
                    json.dump(trial_summary, f, indent=2)

                return True
            else:
                print(f"⚠️ No subject models could be trained for trial {trial_number}")
                return False

        except Exception as e:
            print(f"❌ Failed to save model for trial {trial_number}: {e}")
            return False

    def _save_best_model(self, study):
        """Save the best performing model with special naming."""
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

                # Save best model summary
                best_summary = {
                    "best_trial_number": best_trial.number,
                    "best_cv_mse": best_trial.value,
                    "best_params": best_trial.params,
                    "best_user_attrs": (
                        dict(best_trial.user_attrs) if best_trial.user_attrs else {}
                    ),
                    "total_trials": len(study.trials),
                    "completed_trials": len(
                        [
                            t
                            for t in study.trials
                            if t.state == optuna.trial.TrialState.COMPLETE
                        ]
                    ),
                    "selection_timestamp": datetime.now().isoformat(),
                    "model_path": best_path,
                }

                with open("metrics/best_params_overall.json", "w") as f:
                    json.dump(best_summary, f, indent=2)

                print(f"✅ Best model summary: metrics/best_params_overall.json")

            else:
                print(f"❌ Model file not found: {source_path}")

        except Exception as e:
            print(f"❌ Failed to save best model: {e}")

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

    def run(self, n_trials=20):
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=n_trials)

        # Save models for all completed trials
        if self.completed_trials:
            self._save_all_completed_models()

        # Display optimization results
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

            print(f"\n🔧 Best hyperparameters:")
            for key, value in study.best_trial.params.items():
                print(f"   {key}: {value}")

            # Display detailed metrics if available
            if study.best_trial.user_attrs:
                attrs = study.best_trial.user_attrs
                print(f"\n📈 Detailed Performance Metrics:")

                if "cv_mse" in attrs:
                    print(f"   Cross-Validation MSE: {attrs['cv_mse']:.6f}")
                if "dev_mse" in attrs:
                    print(f"   Development Set MSE: {attrs['dev_mse']:.6f}")
                if "dev_r2" in attrs:
                    print(f"   Development Set R²: {attrs['dev_r2']:.6f}")
                if "holdout_mse" in attrs:
                    print(f"   Holdout Set MSE: {attrs['holdout_mse']:.6f}")
                if "holdout_r2" in attrs:
                    print(f"   Holdout Set R²: {attrs['holdout_r2']:.6f}")
                if "holdout_adj_r2" in attrs:
                    print(f"   Holdout Set Adj. R²: {attrs['holdout_adj_r2']:.6f}")

                if "overfitting_ratio" in attrs and attrs["overfitting_ratio"] != float(
                    "inf"
                ):
                    overfitting_pct = (attrs["overfitting_ratio"] - 1) * 100
                    print(
                        f"   Overfitting: {overfitting_pct:+.1f}% MSE increase on holdout"
                    )

                    if overfitting_pct < 10:
                        print("   🟢 Low overfitting - good generalization")
                    elif overfitting_pct < 25:
                        print("   🟡 Moderate overfitting - acceptable")
                    else:
                        print("   🔴 High overfitting - consider regularization")

                if "n_final_models" in attrs:
                    print(f"   Ensemble Size: {attrs['n_final_models']} subject models")

            # Show trial progression
            print(f"\n📊 Trial Progression (last 5 trials):")
            recent_trials = (
                completed_trials[-5:]
                if len(completed_trials) >= 5
                else completed_trials
            )
            for trial in recent_trials:
                status = "🎯" if trial == study.best_trial else "  "
                print(f"   {status} Trial {trial.number}: MSE = {trial.value:.6f}")

            # Variogram analysis if available
            if any(
                "variogram" in str(param) for param in study.best_trial.params.keys()
            ):
                print(f"\n🗺️  Best Variogram Configuration:")
                variogram_params = {
                    k: v
                    for k, v in study.best_trial.params.items()
                    if "variogram" in k or "kriging" in k or "anisotropy" in k
                }
                for key, value in variogram_params.items():
                    print(f"   {key}: {value}")

            # Drift model analysis
            if "drift_model_type" in study.best_trial.params:
                drift_model = study.best_trial.params["drift_model_type"]
                print(f"\n🏗️  Best Drift Model: {drift_model}")
                drift_params = {
                    k: v
                    for k, v in study.best_trial.params.items()
                    if drift_model.replace("_", "") in k.replace("_", "")
                }
                if drift_params:
                    for key, value in drift_params.items():
                        print(f"   {key}: {value}")

            # Save the best model
            self._save_best_model(study)

            print(f"\n💾 Models saved:")
            print(f"   Best model: models/best_ensemble_model.pkl")
            print(f"   All trials: models/ensemble_trial_*.pkl")
            print(f"   Summaries: metrics/")

        else:
            failed_trials = len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
            )
            pruned_trials = len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
            )

            print(f"❌ No trials completed successfully")
            print(f"   Total trials: {len(study.trials)}")
            print(f"   Failed trials: {failed_trials}")
            print(f"   Pruned trials: {pruned_trials}")
            print(f"\n💡 Suggestions:")
            print(f"   - Check data quality and format")
            print(f"   - Reduce complexity (fewer spatial folds)")
            print(f"   - Check for missing 'x', 'y', 'sub' columns")

        print("=" * 80)

        return study
