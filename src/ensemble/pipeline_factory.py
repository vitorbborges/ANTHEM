import inspect

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import SVR


class DataFramePreservingTransformer(BaseEstimator):
    """Wrapper that preserves DataFrame structure and coordinate columns"""

    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):
        # Store original column names and coordinate columns
        if isinstance(X, pd.DataFrame):
            self.original_columns_ = X.columns.tolist()
            if "x" in X.columns and "y" in X.columns:
                self.has_coords_ = True
                self.coord_values_ = X[["x", "y"]].copy()
                # Fit transformer on non-coordinate columns
                feature_cols = [col for col in X.columns if col not in ["x", "y"]]
                self.feature_columns_ = feature_cols
                if feature_cols:
                    self.transformer.fit(X[feature_cols], y)
                else:
                    self.transformer = None  # No features to transform
            else:
                self.has_coords_ = False
                self.transformer.fit(X, y)
        else:
            self.has_coords_ = False
            self.transformer.fit(X, y)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame) and self.has_coords_:
            # Transform only non-coordinate features
            if (
                hasattr(self, "feature_columns_")
                and self.feature_columns_
                and self.transformer is not None
            ):
                transformed_features = self.transformer.transform(
                    X[self.feature_columns_]
                )

                # Create new DataFrame with coordinates + transformed features
                if (
                    hasattr(transformed_features, "shape")
                    and len(transformed_features.shape) == 2
                ):
                    n_features = transformed_features.shape[1]
                    feature_names = [f"feature_{i}" for i in range(n_features)]
                else:
                    feature_names = ["feature_0"]
                    transformed_features = transformed_features.reshape(-1, 1)

                result_df = X[["x", "y"]].copy()
                for i, name in enumerate(feature_names):
                    result_df[name] = transformed_features[:, i]

                return result_df
            else:
                # No features to transform, return just coordinates
                return X[["x", "y"]].copy()
        else:
            # No coordinates or not a DataFrame
            return self.transformer.transform(X) if self.transformer is not None else X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class StableKrigingRegressor(BaseEstimator, RegressorMixin):
    """
    Simplified, more stable version of Universal Kriging with better error handling.
    Falls back to drift model only if kriging fails.
    """

    def __init__(self, drift_model=None, use_kriging=True, verbose=False):
        self.drift_model = drift_model
        self.use_kriging = use_kriging
        self.verbose = verbose
        self.fitted_drift_model_ = None
        self.kriging_available_ = False

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "StableKrigingRegressor requires a pandas DataFrame with 'x' and 'y' columns"
            )

        if "x" not in X.columns or "y" not in X.columns:
            raise ValueError("X must contain 'x' and 'y' coordinate columns")

        coords_x = X["x"].values
        coords_y = X["y"].values
        drift_features = X.drop(["x", "y"], axis=1)

        # Store training target statistics for better prediction bounds
        y_array = y.values if hasattr(y, "values") else y
        self.train_y_mean_ = np.mean(y_array)
        self.train_y_std_ = np.std(y_array)

        # Fit drift model
        if self.drift_model is not None and not drift_features.empty:
            self.fitted_drift_model_ = self.drift_model
            self.fitted_drift_model_.fit(drift_features, y)

            # Get residuals for kriging
            if self.use_kriging:
                try:
                    drift_predictions = self.fitted_drift_model_.predict(drift_features)
                    residuals = y - drift_predictions

                    # Simple spatial interpolation based on distance
                    # Store training data for prediction
                    self.train_coords_ = np.column_stack([coords_x, coords_y])
                    self.train_residuals_ = (
                        residuals.values if hasattr(residuals, "values") else residuals
                    )
                    self.kriging_available_ = True

                except Exception as e:
                    if self.verbose:
                        print(f"Kriging setup failed: {e}. Using drift model only.")
                    self.kriging_available_ = False
            else:
                self.kriging_available_ = False
        else:
            # No drift model, use simple mean
            self.mean_y_ = np.mean(y)
            self.fitted_drift_model_ = None
            self.kriging_available_ = False

        return self

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "StableKrigingRegressor requires a pandas DataFrame with 'x' and 'y' columns"
            )

        coords_x = X["x"].values
        coords_y = X["y"].values
        drift_features = X.drop(["x", "y"], axis=1)

        # Get drift predictions
        if self.fitted_drift_model_ is not None and not drift_features.empty:
            try:
                drift_predictions = self.fitted_drift_model_.predict(drift_features)
            except Exception as e:
                if self.verbose:
                    print(f"Drift prediction failed: {e}. Using mean.")
                drift_predictions = np.full(len(X), getattr(self, "mean_y_", 0.0))
        else:
            drift_predictions = np.full(len(X), getattr(self, "mean_y_", 0.0))

        # Add spatial component if available
        if self.kriging_available_:
            try:
                # Simple inverse distance weighting for spatial interpolation
                test_coords = np.column_stack([coords_x, coords_y])
                spatial_predictions = self._inverse_distance_interpolation(test_coords)
                final_predictions = drift_predictions + spatial_predictions
            except Exception as e:
                if self.verbose:
                    print(f"Spatial interpolation failed: {e}. Using drift only.")
                final_predictions = drift_predictions
        else:
            final_predictions = drift_predictions

        # Ensure predictions are reasonable relative to training data
        if hasattr(self, "train_y_mean_") and hasattr(self, "train_y_std_"):
            # Clip to reasonable range around training distribution
            lower_bound = self.train_y_mean_ - 5 * self.train_y_std_
            upper_bound = self.train_y_mean_ + 5 * self.train_y_std_
            final_predictions = np.clip(final_predictions, lower_bound, upper_bound)
        else:
            # Fallback: prevent extreme values
            final_predictions = np.clip(final_predictions, -1000, 10000)

        return final_predictions

    def _inverse_distance_interpolation(self, test_coords, power=2, max_distance=1000):
        """Simple inverse distance weighting interpolation."""
        predictions = np.zeros(len(test_coords))

        for i, test_point in enumerate(test_coords):
            # Calculate distances to all training points
            distances = np.sqrt(np.sum((self.train_coords_ - test_point) ** 2, axis=1))

            # Avoid division by zero
            distances = np.maximum(distances, 1e-10)

            # Inverse distance weights
            weights = 1.0 / (distances**power)

            # Limit influence of very distant points
            mask = distances < max_distance
            if np.any(mask):
                weights = weights * mask

            # Weighted average
            if np.sum(weights) > 0:
                predictions[i] = np.sum(weights * self.train_residuals_) / np.sum(
                    weights
                )
            else:
                predictions[i] = 0.0

        return predictions


# Step Registry
STEP_REGISTRY = {}


def register_step(name: str):
    def decorator(func):
        if name in STEP_REGISTRY:
            raise ValueError(f"Step '{name}' is already registered.")
        STEP_REGISTRY[name] = func
        return func

    return decorator


# Step Creation Functions
@register_step("scaler")
def create_scaler(trial: optuna.Trial, X: pd.DataFrame = None):
    scaler_choice = trial.suggest_categorical(
        "scaler__type", ["standard", "minmax", "robust"]
    )
    if scaler_choice == "standard":
        scaler = StandardScaler()
    elif scaler_choice == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = RobustScaler()
    return ("scaler", DataFramePreservingTransformer(scaler))


@register_step("power_transformer")
def create_power_transformer(trial: optuna.Trial, X: pd.DataFrame = None):
    method = "yeo-johnson"  # Works with any real values
    return (
        "power_trans",
        DataFramePreservingTransformer(PowerTransformer(method=method)),
    )


@register_step("pca")
def create_pca(trial: optuna.Trial, X: pd.DataFrame):
    n_components = trial.suggest_float("pca__n_components", 0.8, 0.99, log=True)
    return ("pca", DataFramePreservingTransformer(PCA(n_components=n_components)))


@register_step("lasso_selection")
def create_lasso_selection(trial: optuna.Trial, X: pd.DataFrame = None):
    alpha = trial.suggest_float("lasso__alpha", 0.01, 50, log=True)
    return (
        "lasso_sel",
        DataFramePreservingTransformer(
            SelectFromModel(
                Lasso(alpha=alpha, random_state=0, max_iter=2000, tol=1e-3),
                threshold="mean",
            )
        ),
    )


@register_step("linear_regression")
def create_linear_regression(trial: optuna.Trial, X: pd.DataFrame = None):
    return ("drift_model", LinearRegression())


@register_step("ridge")
def create_ridge(trial: optuna.Trial, X: pd.DataFrame = None):
    alpha = trial.suggest_float("ridge__alpha", 0.1, 50.0, log=True)
    return ("drift_model", Ridge(alpha=alpha, max_iter=2000, tol=1e-3))


@register_step("lasso")
def create_lasso(trial: optuna.Trial, X: pd.DataFrame = None):
    alpha = trial.suggest_float("lasso_model__alpha", 0.01, 50.0, log=True)
    return ("drift_model", Lasso(alpha=alpha, random_state=0, max_iter=2000, tol=1e-3))


@register_step("random_forest")
def create_random_forest(trial: optuna.Trial, X: pd.DataFrame = None):
    n_estimators = trial.suggest_int("random_forest__n_estimators", 10, 200)
    max_depth = trial.suggest_int("random_forest__max_depth", 1, 20)
    min_samples_split = trial.suggest_int("random_forest__min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("random_forest__min_samples_leaf", 1, 20)
    max_features = trial.suggest_categorical(
        "random_forest__max_features", ["sqrt", "log2", None]
    )
    return (
        "drift_model",
        RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=0,
            n_jobs=1,
        ),
    )


@register_step("svr")
def create_svr(trial: optuna.Trial, X: pd.DataFrame = None):
    C = trial.suggest_float("svr__C", 0.01, 100.0, log=True)
    kernel = trial.suggest_categorical("svr__kernel", ["linear", "rbf", "poly"])

    if kernel == "rbf":
        gamma = trial.suggest_float("svr__gamma", 1e-5, 1e-1, log=True)
        return ("drift_model", SVR(C=C, kernel=kernel, gamma=gamma))
    elif kernel == "poly":
        degree = trial.suggest_int("svr__degree", 2, 5)
        gamma = trial.suggest_float("svr__gamma", 1e-5, 1e-1, log=True)
        return ("drift_model", SVR(C=C, kernel=kernel, degree=degree, gamma=gamma))
    else:
        return ("drift_model", SVR(C=C, kernel=kernel))


@register_step("k_neighbors")
def create_k_neighbors(trial: optuna.Trial, X: pd.DataFrame = None):
    n_neighbors = trial.suggest_int("k_neighbors__n_neighbors", 3, 50)
    weights = trial.suggest_categorical("knn__weights", ["uniform", "distance"])
    metric = trial.suggest_categorical(
        "knn__metric", ["euclidean", "manhattan", "minkowski", "chebyshev"]
    )

    if metric == "minkowski":
        p = trial.suggest_int("knn__p", 1, 5)
    else:
        p = 2

    algorithm = trial.suggest_categorical(
        "knn__algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
    )

    return (
        "drift_model",
        KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            p=p,
            algorithm=algorithm,
        ),
    )


def create_stable_kriging_model(trial: optuna.Trial, X: pd.DataFrame):
    """Create a stable kriging model with better error handling."""

    # Choose drift model
    drift_model_choice = trial.suggest_categorical(
        "drift_model_type",
        ["linear_regression", "ridge", "lasso", "random_forest", "svr", "k_neighbors"],
    )

    # Create drift model
    create_func = STEP_REGISTRY.get(drift_model_choice)
    if create_func:
        _, drift_model = create_func(trial, X)
    else:
        drift_model = LinearRegression()

    # Option to disable kriging for stability
    use_kriging = trial.suggest_categorical("use_kriging", [True, False])

    return StableKrigingRegressor(
        drift_model=drift_model, use_kriging=use_kriging, verbose=False
    )


def create_pipeline(trial: optuna.Trial, X: pd.DataFrame) -> Pipeline:
    """Create a stable pipeline with better error handling."""
    steps = []

    # Always include a scaler first
    steps.append(create_scaler(trial, X))

    # Optional power transformation
    use_power_transform = trial.suggest_categorical(
        "use_power_transform", [True, False]
    )
    if use_power_transform:
        steps.append(create_power_transformer(trial, X))

    # Feature processing options
    feature_step_choice = trial.suggest_categorical(
        "feature_step", ["none", "lasso_selection", "pca"]
    )

    if feature_step_choice != "none":
        create_func = STEP_REGISTRY.get(feature_step_choice)
        if create_func:
            func_params = inspect.signature(create_func).parameters
            if "X" in func_params:
                steps.append(create_func(trial, X))
            else:
                steps.append(create_func(trial))

    # Stable kriging as the final model
    kriging_model = create_stable_kriging_model(trial, X)
    steps.append(("stable_kriging", kriging_model))

    return Pipeline(steps)
