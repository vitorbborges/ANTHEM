from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class UnifiedEnvironmentalSimilarity:
    """
    Unified environmental similarity calculator ensuring consistency
    between CV training and prediction phases.
    """

    # Feature name mapping from Italian (training data) to English (prediction)
    FEATURE_MAPPING = {
        "T": "temperature",
        "RH": "humidity",
        "P": "pressure",
        "velocita_vento_massimo": "wind_speed_max",
        "velocita_vento_medio": "wind_speed_mean",
        "direzzione_vento_massimo": "wind_direction_max",
        "direzzione_vento_medio": "wind_direction_mean",
        "radiazione_globale_medio": "solar_radiation",
        "precipitazione_valore_cumulato": "precipitation",
    }

    # Standard environmental features (in order of importance)
    STANDARD_FEATURES = [
        "temperature",
        "humidity",
        "pressure",
        "wind_speed_mean",
        "wind_direction_mean",
        "solar_radiation",
        "precipitation",
    ]

    # Features that are circular (0-360 degrees)
    CIRCULAR_FEATURES = ["wind_direction_max", "wind_direction_mean"]

    def __init__(self, method: str = "combined", normalize_features: bool = True):
        self.method = method
        self.normalize_features = normalize_features

        # Default weights for features
        self.default_weights = {
            "temperature": 1.0,
            "humidity": 0.8,
            "pressure": 0.6,
            "wind_speed_mean": 0.9,
            "wind_direction_mean": 0.7,
            "solar_radiation": 0.5,
            "precipitation": 0.4,
        }

    def standardize_feature_names(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize feature names to English equivalents."""
        df = data.copy()

        # Rename Italian features to English
        rename_dict = {}
        for italian, english in self.FEATURE_MAPPING.items():
            if italian in df.columns:
                rename_dict[italian] = english

        if rename_dict:
            df = df.rename(columns=rename_dict)

        return df

    def get_available_features(self, data: pd.DataFrame) -> List[str]:
        """Get list of available environmental features in data."""
        # Standardize names first
        df = self.standardize_feature_names(data)

        # Find intersection with standard features
        available = [f for f in self.STANDARD_FEATURES if f in df.columns]

        # Add any other environmental features not in standard list
        other_env = [
            col
            for col in df.columns
            if col not in available
            and col not in ["x", "y", "sub", "location", "regime", "CO2"]
        ]

        return available + other_env

    def calculate_feature_similarity(
        self, test_val: float, train_val: float, feature_name: str
    ) -> float:
        """Calculate similarity for a specific feature."""
        if feature_name in self.CIRCULAR_FEATURES:
            # Handle circular features (wind direction)
            diff = abs(test_val - train_val)
            circular_diff = min(diff, 360 - diff)  # Handle circularity
            similarity = 1.0 - (circular_diff / 180.0)  # Normalize to 0-1
            return max(0.0, similarity)
        else:
            # Standard features
            if test_val == train_val:
                return 1.0

            # Normalized absolute difference
            feature_range = max(abs(test_val), abs(train_val), 1e-8)
            diff = abs(test_val - train_val) / feature_range
            similarity = 1.0 - diff
            return max(0.0, min(1.0, similarity))

    def calculate_similarity(
        self, test_features: pd.DataFrame, train_features_dict: Dict[int, pd.DataFrame]
    ) -> Dict[int, float]:
        """
        Calculate environmental similarity between test and training features.
        """
        # Standardize feature names
        test_std = self.standardize_feature_names(test_features)
        train_std_dict = {
            sid: self.standardize_feature_names(df)
            for sid, df in train_features_dict.items()
        }

        # Get available features
        available_features = self.get_available_features(test_std)

        if not available_features:
            # No environmental features available - return equal weights
            return {sid: 1.0 for sid in train_features_dict.keys()}

        # Calculate mean environmental conditions for test data
        test_env = {}
        for feature in available_features:
            if feature in test_std.columns:
                test_env[feature] = test_std[feature].mean()

        similarities = {}

        for subject_id, train_df in train_std_dict.items():
            # Calculate mean environmental conditions for this subject
            train_env = {}
            for feature in available_features:
                if feature in train_df.columns:
                    train_env[feature] = train_df[feature].mean()

            # Calculate similarity using specified method
            if self.method == "euclidean":
                sim = self._euclidean_similarity(
                    test_env, train_env, available_features
                )
            elif self.method == "cosine":
                sim = self._cosine_similarity(test_env, train_env, available_features)
            elif self.method == "combined":
                sim = self._combined_similarity(test_env, train_env, available_features)
            else:
                # Fallback to simple euclidean for backwards compatibility
                sim = self._simple_euclidean_similarity(
                    test_env, train_env, available_features
                )

            similarities[subject_id] = max(0.0, min(1.0, sim))

        return similarities

    def _simple_euclidean_similarity(
        self, test_env: Dict, train_env: Dict, features: List[str]
    ) -> float:
        """Simple euclidean similarity (original method for compatibility)."""
        test_vals = []
        train_vals = []

        for feature in features:
            if feature in test_env and feature in train_env:
                test_vals.append(test_env[feature])
                train_vals.append(train_env[feature])

        if not test_vals:
            return 1.0

        test_vec = np.array(test_vals)
        train_vec = np.array(train_vals)

        distance = np.linalg.norm(test_vec - train_vec)
        return 1.0 / (1.0 + distance)

    def _euclidean_similarity(
        self, test_env: Dict, train_env: Dict, features: List[str]
    ) -> float:
        """Enhanced euclidean-based similarity with feature-specific handling."""
        distances = []
        weights = []

        for feature in features:
            if feature in test_env and feature in train_env:
                test_val = test_env[feature]
                train_val = train_env[feature]

                # Calculate feature-specific similarity
                sim = self.calculate_feature_similarity(test_val, train_val, feature)
                distances.append(1.0 - sim)  # Convert similarity to distance

                # Get weight for this feature
                weight = self.default_weights.get(feature, 1.0)
                weights.append(weight)

        if not distances:
            return 1.0

        # Weighted euclidean distance
        weights = np.array(weights)
        distances = np.array(distances)
        weighted_distance = np.sqrt(np.sum(weights * distances**2) / np.sum(weights))

        # Convert back to similarity
        return 1.0 / (1.0 + weighted_distance)

    def _cosine_similarity(
        self, test_env: Dict, train_env: Dict, features: List[str]
    ) -> float:
        """Calculate cosine similarity."""
        test_vec = []
        train_vec = []

        for feature in features:
            if feature in test_env and feature in train_env:
                test_vec.append(test_env[feature])
                train_vec.append(train_env[feature])

        if len(test_vec) < 2:
            return 1.0

        # Calculate cosine similarity
        test_vec = np.array(test_vec).reshape(1, -1)
        train_vec = np.array(train_vec).reshape(1, -1)

        # Handle zero vectors
        if np.linalg.norm(test_vec) == 0 or np.linalg.norm(train_vec) == 0:
            return 1.0 if np.array_equal(test_vec, train_vec) else 0.0

        cos_sim = cosine_similarity(test_vec, train_vec)[0, 0]
        return (cos_sim + 1.0) / 2.0  # Convert from [-1,1] to [0,1]

    def _combined_similarity(
        self, test_env: Dict, train_env: Dict, features: List[str]
    ) -> float:
        """Calculate combined similarity using multiple metrics."""
        # Get individual similarities
        eucl_sim = self._euclidean_similarity(test_env, train_env, features)
        cos_sim = self._cosine_similarity(test_env, train_env, features)

        # Calculate feature-wise similarities
        feature_sims = []
        for feature in features:
            if feature in test_env and feature in train_env:
                sim = self.calculate_feature_similarity(
                    test_env[feature], train_env[feature], feature
                )
                feature_sims.append(sim)

        mean_feature_sim = np.mean(feature_sims) if feature_sims else 1.0

        # Combine with weights
        combined = (
            0.4 * eucl_sim  # Distance-based similarity
            + 0.3 * cos_sim  # Directional similarity
            + 0.3 * mean_feature_sim  # Feature-specific similarity
        )

        return combined
