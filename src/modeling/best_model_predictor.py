import pickle
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
# Import the unified similarity calculator (same as in cv_handler)
from src.modeling.env_similarity import UnifiedEnvironmentalSimilarity
from src.modeling.optimization import OptimizationRunner


class UniversalKrigingPredictor:
    """
    Predictor class that loads the best Universal Kriging ensemble and
    makes predictions for new coordinates and environmental data.
    """

    def __init__(self, model_path: str = None, trial_number: int = None):
        """
        Initialize the predictor.

        Args:
            model_path: Direct path to saved model file
            trial_number: Trial number to load (will look for ensemble_trial_{trial_number}.pkl)
        """
        self.ensemble_models = None
        self.trial_params = None

        # Initialize unified similarity calculator (same as CV)
        self.similarity_calc = UnifiedEnvironmentalSimilarity(method="combined")

        if model_path:
            self.load_model(model_path)
        elif trial_number is not None:
            self.load_model(f"models/ensemble_trial_{trial_number}.pkl")

    def load_model(self, model_path: str):
        """Load the saved ensemble model."""
        try:
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)

            self.ensemble_models = model_data["final_models"]
            self.trial_params = model_data["trial_params"]
            self.trial_number = model_data["trial_number"]

            print(f"✅ Loaded ensemble with {len(self.ensemble_models)} subject models")
            print(f"📊 Trial parameters: {self.trial_params}")

        except Exception as e:
            raise ValueError(f"Failed to load model from {model_path}: {e}")

    @classmethod
    def load_best_model(cls):
        """Load the best performing model automatically."""
        import json

        try:
            # Try to load best overall result
            with open("metrics/best_params_overall.json", "r") as f:
                best_result = json.load(f)
            trial_number = best_result["best_trial_number"]

            print(f"🎯 Loading best model from trial {trial_number}")
            return cls(trial_number=trial_number)

        except FileNotFoundError:
            # Fallback: find the model with lowest filename number
            import glob

            model_files = glob.glob("models/ensemble_trial_*.pkl")
            if not model_files:
                raise ValueError("No ensemble models found in models/ directory")

            # Extract trial numbers and find the one with best performance
            trial_numbers = [int(f.split("_")[-1].split(".")[0]) for f in model_files]
            trial_number = min(
                trial_numbers
            )  # This is a fallback, not necessarily the best

            print(f"⚠️ Using fallback model from trial {trial_number}")
            return cls(trial_number=trial_number)

    def _calculate_environmental_similarity(
        self, test_features: pd.DataFrame, train_features_dict: Dict
    ) -> Dict[int, float]:
        """
        Calculate environmental similarity using unified method for consistency.
        This now uses the same similarity calculation as the CV handler.
        """
        return self.similarity_calc.calculate_similarity(
            test_features, train_features_dict
        )

    def predict(
        self,
        x: Union[float, List[float], np.ndarray],
        y: Union[float, List[float], np.ndarray],
        environmental_data: Optional[Dict] = None,
        return_uncertainty: bool = False,
        return_individual_predictions: bool = False,
    ) -> Union[float, np.ndarray, Dict]:
        """
        Make CO2 concentration predictions for given coordinates and environmental data.

        Args:
            x: X coordinate(s)
            y: Y coordinate(s)
            environmental_data: Dict with environmental features like
                              {'temperature': 25.0, 'humidity': 60.0, 'wind_speed': 5.0, ...}
                              Can use either English or Italian feature names
            return_uncertainty: Whether to return prediction uncertainty from kriging
            return_individual_predictions: Whether to return predictions from each subject model

        Returns:
            Predictions (float/array) or dict with predictions and additional info
        """
        if self.ensemble_models is None:
            raise ValueError(
                "No model loaded. Call load_model() or load_best_model() first."
            )

        # Convert inputs to arrays for consistent handling
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)

        if len(x) != len(y):
            raise ValueError("x and y must have the same length")

        n_points = len(x)

        # Create input DataFrame
        input_data = pd.DataFrame({"x": x, "y": y})

        # Add environmental data if provided
        if environmental_data:
            for key, value in environmental_data.items():
                if isinstance(value, (int, float)):
                    input_data[key] = [value] * n_points
                elif len(value) == n_points:
                    input_data[key] = value
                else:
                    raise ValueError(
                        f"Environmental data '{key}' must be scalar or have length {n_points}"
                    )

        # Calculate environmental similarities for model weighting
        if environmental_data:
            # Create training features dict from stored model data
            train_features_dict = {}
            for model_info in self.ensemble_models:
                subject_id = model_info["subject_id"]
                # Use the input data as a proxy for training features
                # In a real implementation, you'd want to store actual training features
                train_features_dict[subject_id] = input_data

            similarities = self._calculate_environmental_similarity(
                input_data, train_features_dict
            )
        else:
            # Equal weights if no environmental data
            similarities = {
                model_info["subject_id"]: 1.0 for model_info in self.ensemble_models
            }

        # Get predictions from each subject model
        all_predictions = []
        all_uncertainties = []
        individual_predictions = {}
        weights = []

        for model_info in self.ensemble_models:
            subject_id = model_info["subject_id"]
            pipeline = model_info["pipeline"]

            try:
                # Make prediction
                pred = pipeline.predict(input_data)
                all_predictions.append(pred)
                individual_predictions[subject_id] = pred

                # Get uncertainty if available and requested
                if return_uncertainty:
                    # Try to get kriging uncertainty
                    uk_model = None
                    for step_name, step_obj in pipeline.steps:
                        if hasattr(step_obj, "get_prediction_variance"):
                            uk_model = step_obj
                            break

                    if uk_model:
                        uncertainty = uk_model.get_prediction_variance()
                        if uncertainty is not None:
                            all_uncertainties.append(uncertainty)
                        else:
                            all_uncertainties.append(np.ones_like(pred))
                    else:
                        all_uncertainties.append(np.ones_like(pred))

                # Get similarity weight
                weight = similarities.get(subject_id, 0.1)
                weights.append(weight)

            except Exception as e:
                print(f"⚠️ Model for subject {subject_id} failed: {e}")
                continue

        if not all_predictions:
            raise ValueError("No models could make predictions")

        # Convert to arrays
        all_predictions = np.array(all_predictions)  # Shape: (n_models, n_points)
        weights = np.array(weights)

        # Normalize weights
        weights = weights / weights.sum()

        # Calculate weighted ensemble prediction
        ensemble_predictions = np.average(all_predictions, axis=0, weights=weights)

        # Prepare results
        if n_points == 1:
            ensemble_predictions = float(ensemble_predictions[0])

        if return_uncertainty or return_individual_predictions:
            results = {"predictions": ensemble_predictions}

            if return_individual_predictions:
                results["individual_predictions"] = individual_predictions
                results["weights"] = {
                    model_info["subject_id"]: weights[i]
                    for i, model_info in enumerate(self.ensemble_models)
                }

            if return_uncertainty and all_uncertainties:
                ensemble_uncertainty = np.average(
                    all_uncertainties, axis=0, weights=weights
                )
                if n_points == 1:
                    ensemble_uncertainty = float(ensemble_uncertainty[0])
                results["uncertainty"] = ensemble_uncertainty

            return results

        return ensemble_predictions

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        x_col: str = "x",
        y_col: str = "y",
        environmental_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Make predictions for a DataFrame of coordinates and environmental data.

        Args:
            df: DataFrame with coordinates and environmental data
            x_col: Column name for x coordinates
            y_col: Column name for y coordinates
            environmental_cols: List of column names for environmental features
                              Can use either English or Italian feature names

        Returns:
            DataFrame with added prediction columns
        """
        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(f"DataFrame must contain '{x_col}' and '{y_col}' columns")

        # Prepare environmental data
        env_data = {}
        if environmental_cols:
            for col in environmental_cols:
                if col in df.columns:
                    env_data[col] = df[col].values
                else:
                    print(f"⚠️ Environmental column '{col}' not found in DataFrame")

        # Make predictions
        results = self.predict(
            x=df[x_col].values,
            y=df[y_col].values,
            environmental_data=env_data if env_data else None,
            return_uncertainty=True,
            return_individual_predictions=False,
        )

        # Add results to DataFrame
        result_df = df.copy()
        result_df["co2_prediction"] = results["predictions"]
        if "uncertainty" in results:
            result_df["prediction_uncertainty"] = results["uncertainty"]

        return result_df

    def get_model_info(self) -> Dict:
        """Get information about the loaded ensemble."""
        if self.ensemble_models is None:
            return {"error": "No model loaded"}

        info = {
            "trial_number": getattr(self, "trial_number", "unknown"),
            "n_subject_models": len(self.ensemble_models),
            "subject_ids": [model["subject_id"] for model in self.ensemble_models],
            "trial_parameters": self.trial_params,
            "sample_sizes": [model["n_samples"] for model in self.ensemble_models],
        }

        return info


# Convenience function for quick predictions
def predict_co2(
    x: Union[float, List[float]],
    y: Union[float, List[float]],
    environmental_data: Optional[Dict] = None,
) -> Union[float, np.ndarray]:
    """
    Quick function to make CO2 predictions with the best model.

    Args:
        x: X coordinate(s)
        y: Y coordinate(s)
        environmental_data: Environmental conditions (optional)
                          Can use either English or Italian feature names

    Returns:
        CO2 concentration prediction(s)
    """
    predictor = UniversalKrigingPredictor.load_best_model()
    return predictor.predict(x, y, environmental_data)


# Example usage functions
def example_usage():
    """Show example usage of the predictor."""

    # Load the best model
    predictor = UniversalKrigingPredictor.load_best_model()

    # Single point prediction with Italian feature names (as in training data)
    co2_pred = predictor.predict(
        x=45.4642,  # Example coordinates
        y=9.1900,
        environmental_data={
            "T": 22.5,  # Temperature (Italian name)
            "RH": 65.0,  # Humidity (Italian name)
            "velocita_vento_medio": 3.2,  # Wind speed (Italian name)
            "P": 1013.25,  # Pressure (Italian name)
        },
    )
    print(f"CO2 prediction: {co2_pred:.2f} ppm")

    # Same prediction with English feature names (will be automatically mapped)
    co2_pred_en = predictor.predict(
        x=45.4642,
        y=9.1900,
        environmental_data={
            "temperature": 22.5,  # English name
            "humidity": 65.0,  # English name
            "wind_speed_mean": 3.2,  # English name
            "pressure": 1013.25,  # English name
        },
    )
    print(f"CO2 prediction (English names): {co2_pred_en:.2f} ppm")

    # Multiple points with uncertainty
    results = predictor.predict(
        x=[45.4642, 45.4650, 45.4658],
        y=[9.1900, 9.1905, 9.1910],
        environmental_data={
            "T": [22.5, 23.0, 22.8],  # Italian names
            "RH": [65.0, 63.0, 64.0],
            "velocita_vento_medio": [3.2, 3.5, 3.0],
        },
        return_uncertainty=True,
    )
    print(f"Predictions: {results['predictions']}")
    print(f"Uncertainties: {results['uncertainty']}")

    # DataFrame prediction
    test_df = pd.DataFrame(
        {
            "x": [45.4642, 45.4650, 45.4658],
            "y": [9.1900, 9.1905, 9.1910],
            "T": [22.5, 23.0, 22.8],  # Italian names
            "RH": [65.0, 63.0, 64.0],
        }
    )

    result_df = predictor.predict_dataframe(
        test_df, environmental_cols=["T", "RH"]  # Italian names
    )
    print(result_df[["x", "y", "co2_prediction", "prediction_uncertainty"]])


if __name__ == "__main__":
    example_usage()
