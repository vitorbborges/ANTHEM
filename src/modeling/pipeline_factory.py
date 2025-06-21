# src/modeling/pipeline_factory.py
import inspect
from typing import Optional

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator
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

# --- Step Registry ---
STEP_REGISTRY = {}


def register_step(name: str):
    """Decorator to register a step creation function."""

    def decorator(func):
        if name in STEP_REGISTRY:
            raise ValueError(f"Step '{name}' is already registered.")
        STEP_REGISTRY[name] = func
        return func

    return decorator


# --- Step Creation Functions ---


@register_step("scaler")
def create_scaler(
    trial: optuna.Trial, X: Optional[pd.DataFrame] = None
) -> tuple[str, BaseEstimator]:
    """Creates a scaling step."""
    scaler_choice = trial.suggest_categorical(
        "scaler__type", ["standard", "minmax", "robust"]
    )
    if scaler_choice == "standard":
        scaler = StandardScaler()
    elif scaler_choice == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = RobustScaler()
    return ("scaler", scaler)


@register_step("power_transformer")
def create_power_transformer(
    trial: optuna.Trial, X: Optional[pd.DataFrame] = None
) -> tuple[str, BaseEstimator]:
    """Creates a PowerTransformer step."""
    method = trial.suggest_categorical(
        "power_transform__method", ["yeo-johnson", "box-cox"]
    )
    return ("power_trans", PowerTransformer(method=method))


@register_step("pca")
def create_pca(trial: optuna.Trial, X: pd.DataFrame) -> tuple[str, BaseEstimator]:
    """Creates a PCA step with direct n_components selection."""
    n_components = trial.suggest_float("pca__n_components", 0.8, 0.99, log=True)
    return ("pca", PCA(n_components=n_components))


@register_step("lasso_selection")
def create_lasso_selection(
    trial: optuna.Trial, X: Optional[pd.DataFrame] = None
) -> tuple[str, BaseEstimator]:
    """Creates a SelectFromModel step using Lasso."""
    alpha = trial.suggest_float("lasso__alpha", 0.01, 50, log=True)
    return (
        "lasso_sel",
        SelectFromModel(Lasso(alpha=alpha, random_state=0), threshold="mean"),
    )


@register_step("linear_regression")
def create_linear_regression(
    trial: optuna.Trial, X: Optional[pd.DataFrame] = None
) -> tuple[str, BaseEstimator]:
    """Creates a LinearRegression model step."""
    return ("model", LinearRegression())


@register_step("ridge")
def create_ridge(
    trial: optuna.Trial, X: Optional[pd.DataFrame] = None
) -> tuple[str, BaseEstimator]:
    """Creates a Ridge model step with tunable alpha."""
    alpha = trial.suggest_float("ridge__alpha", 0.1, 50.0, log=True)
    return ("model", Ridge(alpha=alpha))


@register_step("random_forest")
def create_random_forest(
    trial: optuna.Trial, X: Optional[pd.DataFrame] = None
) -> tuple[str, BaseEstimator]:
    """Creates a RandomForestRegressor model step."""
    n_estimators = trial.suggest_int("random_forest__n_estimators", 10, 100)
    max_depth = trial.suggest_int("random_forest__max_depth", 1, 15)
    return (
        "model",
        RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=0,
            n_jobs=1,  # Prevent nested parallelism
        ),
    )


@register_step("svr")
def create_svr(
    trial: optuna.Trial, X: Optional[pd.DataFrame] = None
) -> tuple[str, BaseEstimator]:
    """Creates an SVR model step."""
    C = trial.suggest_float("svr__C", 0.1, 10.0, log=True)
    return ("model", SVR(C=C, kernel="linear"))


@register_step("k_neighbors")
def create_k_neighbors(
    trial: optuna.Trial, X: Optional[pd.DataFrame] = None
) -> tuple[str, BaseEstimator]:
    """Creates a KNeighborsRegressor model step."""
    n_neighbors = trial.suggest_int("k_neighbors__n_neighbors", 3, 50)
    metric = trial.suggest_categorical(
        "knn__metric", ["euclidean", "manhattan", "minkowski"]
    )
    weights = trial.suggest_categorical("knn__weights", ["uniform", "distance"])

    if metric == "minkowski":
        p = trial.suggest_int("knn__p", 1, 5)
    else:
        p = 2

    return (
        "model",
        KNeighborsRegressor(
            n_neighbors=n_neighbors, weights=weights, metric=metric, p=p
        ),
    )


# --- Main Pipeline Creation Function ---


def create_pipeline(trial: optuna.Trial, X: pd.DataFrame) -> Pipeline:
    """
    Creates a machine learning pipeline based on Optuna trial suggestions.
    """
    steps = []

    # Always include a scaler first
    steps.append(create_scaler(trial, X))

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

    # Choose the final model
    model_choice = trial.suggest_categorical(
        "final_model",
        [
            "linear_regression",
            "ridge",
            "random_forest",
            "svr",
            "k_neighbors",
        ],
    )

    create_func = STEP_REGISTRY.get(model_choice)
    if create_func:
        steps.append(create_func(trial, X))

    return Pipeline(steps)
