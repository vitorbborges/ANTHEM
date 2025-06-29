from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


class UnifiedEnvironmentalSimilarity:
    """
    Environmental similarity calculator focused only on natural environmental conditions
    that vary between different days/subjects. OSM features are excluded since all subjects
    walked the same route.
    """

    # Natural environmental features that vary between days
    ENVIRONMENTAL_FEATURES = [
        # Italian names (as in training data)
        "T",  # Temperature
        "RH",  # Relative Humidity
        "P",  # Pressure
        "velocita_vento_massimo",  # Maximum wind speed
        "velocita_vento_medio",  # Mean wind speed
        "direzzione_vento_massimo",  # Maximum wind direction
        "direzzione_vento_medio",  # Mean wind direction
        "radiazione_globale_medio",  # Mean solar radiation
        "precipitazione_valore_cumulato",  # Cumulative precipitation
        # English names (backup)
        "temperature",
        "humidity",
        "pressure",
        "wind_speed_max",
        "wind_speed_mean",
        "wind_direction_max",
        "wind_direction_mean",
        "solar_radiation",
        "precipitation",
    ]

    # Features that are circular (0-360 degrees)
    CIRCULAR_FEATURES = [
        "direzzione_vento_massimo",
        "direzzione_vento_medio",
        "wind_direction_max",
        "wind_direction_mean",
    ]

    def __init__(self, method: str = "combined", normalize_features: bool = True):
        self.method = method
        self.normalize_features = normalize_features

        # Scaler for normalization
        self.scaler = StandardScaler()

        print(
            f"    Environmental similarity: method={method}, normalize={normalize_features}"
        )

    def get_environmental_features(self, data: pd.DataFrame) -> List[str]:
        """Get list of available environmental features in data."""
        available_env_features = [
            f for f in self.ENVIRONMENTAL_FEATURES if f in data.columns
        ]

        # Remove coordinate and metadata columns
        exclude_cols = [
            "x",
            "y",
            "sub",
            "CO2",
            "location",
            "regime",
            "timestamp",
            "date",
        ]
        available_env_features = [
            f for f in available_env_features if f not in exclude_cols
        ]

        return available_env_features

    def calculate_feature_similarity(
        self, test_val: float, train_val: float, feature_name: str
    ) -> float:
        """Calculate similarity for a specific environmental feature."""
        if feature_name in self.CIRCULAR_FEATURES:
            # Handle circular features (wind direction)
            diff = abs(test_val - train_val)
            circular_diff = min(diff, 360 - diff)  # Handle circularity
            similarity = 1.0 - (circular_diff / 180.0)  # Normalize to 0-1
            return max(0.0, similarity)
        else:
            # Standard environmental features
            if test_val == train_val:
                return 1.0

            # Avoid division by zero
            if test_val == 0 and train_val == 0:
                return 1.0

            # Normalized absolute difference
            feature_range = max(abs(test_val), abs(train_val), 1e-8)
            diff = abs(test_val - train_val) / feature_range
            similarity = 1.0 - min(diff, 1.0)  # Cap at 1.0
            return max(0.0, similarity)

    def calculate_similarity(
        self, test_features: pd.DataFrame, train_features_dict: Dict[int, pd.DataFrame]
    ) -> Dict[int, float]:
        """
        Calculate environmental similarity between test and training features.
        Only uses natural environmental conditions, not OSM features.
        """
        # Get available environmental features
        available_features = self.get_environmental_features(test_features)

        if not available_features:
            print("    Warning: No environmental features found, 3 equal weights")
            return {sid: 1.0 for sid in train_features_dict.keys()}

        print(
            f"    Using {len(available_features)} environmental features: {available_features[:3]}..."
        )

        # Calculate mean environmental conditions for test data
        test_env = {}
        for feature in available_features:
            if feature in test_features.columns:
                test_env[feature] = test_features[feature].mean()

        similarities = {}

        for subject_id, train_df in train_features_dict.items():
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
                # Fallback to euclidean
                sim = self._euclidean_similarity(
                    test_env, train_env, available_features
                )

            similarities[subject_id] = max(0.0, min(1.0, sim))

        return similarities

    def _euclidean_similarity(
        self, test_env: Dict, train_env: Dict, features: List[str]
    ) -> float:
        """Enhanced euclidean-based similarity with feature-specific handling."""
        distances = []

        for feature in features:
            if feature in test_env and feature in train_env:
                test_val = test_env[feature]
                train_val = train_env[feature]

                # Calculate feature-specific similarity
                sim = self.calculate_feature_similarity(test_val, train_val, feature)
                distances.append(1.0 - sim)  # Convert similarity to distance

        if not distances:
            return 1.0

        # Euclidean distance
        distance = np.sqrt(np.mean(np.array(distances) ** 2))

        # Convert back to similarity
        return 1.0 / (1.0 + distance)

    def _cosine_similarity(
        self, test_env: Dict, train_env: Dict, features: List[str]
    ) -> float:
        """Calculate cosine similarity for environmental features."""
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
