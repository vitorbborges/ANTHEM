import geopandas as gpd
import numpy as np
import pandas as pd
from fancyimpute import SoftImpute
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder


# --------------------------------------------------
# Imputation Routines
# --------------------------------------------------
class FeatureImputer:
    @staticmethod
    def impute_gdf(
        gdf: gpd.GeoDataFrame, exclude: list[str], max_rank=20, max_iters=100
    ) -> gpd.GeoDataFrame:
        use_cols = [c for c in gdf.columns if c not in exclude]
        X = gdf[use_cols].copy()
        list_cols = [
            c for c in use_cols if X[c].apply(lambda v: isinstance(v, list)).any()
        ]
        for c in list_cols:
            X[c] = X[c].apply(lambda v: v[0] if isinstance(v, list) and v else np.nan)
        for c in X.columns:
            if X[c].dtype == object:
                num = pd.to_numeric(X[c], errors="coerce")
                if num.notna().sum() > len(num) / 2:
                    X[c] = num
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        dummies = pd.get_dummies(X[cat_cols], dummy_na=False)
        M = pd.concat([X[num_cols], dummies], axis=1).astype(float)
        filled = SoftImpute(
            max_rank=max_rank, max_iters=max_iters, verbose=False
        ).fit_transform(M.values)
        Mf = pd.DataFrame(filled, columns=M.columns, index=M.index)
        out = gdf.copy()
        for c in num_cols:
            out[c] = Mf[c]
        for c in cat_cols:
            opts = [col for col in Mf.columns if col.startswith(c + "_")]
            if opts:
                best = Mf[opts].idxmax(axis=1).str[len(c) + 1 :]
                out[c] = best.astype(gdf[c].dtype)
        if "geometry" in gdf.columns:
            out.geometry = gdf.geometry
        return out

    @staticmethod
    def impute_edges(
        edges: pd.DataFrame,
        non_input_cols: list[str] = None,
        categorical_cols: list[str] = None,
        bool_cols: list[str] = None,
        numerical_cols: list[str] = None,
        max_iter: int = 100,
        random_state: int = 0,
    ) -> pd.DataFrame:
        """
        Impute missing values on edge attributes using MICE (IterativeImputer).

        Steps:
        1) Define non_input_cols, categorical_cols, bool_cols, numerical_cols
        2) Encode categoricals and bools via OrdinalEncoder
        3) Impute with IterativeImputer
        4) Clamp numerical >= 0 and restore certain masks
        5) Decode categoricals/bools back and reattach untouched columns
        """
        # Default column sets
        if non_input_cols is None:
            non_input_cols = [
                "osmid",
                "name",
                "ref",
                "geometry",
                "width",
                "bridge",
                "tunnel",
                "junction",
            ]
        if categorical_cols is None:
            categorical_cols = ["highway", "access"]
        if bool_cols is None:
            bool_cols = ["oneway", "reversed"]
        if numerical_cols is None:
            numerical_cols = ["length", "lanes", "maxspeed"]

        input_cols = categorical_cols + bool_cols + numerical_cols
        edges_copy = edges.copy()
        X = edges_copy[input_cols].copy()
        # Mask for footway-like highways
        mask_foot = X["highway"].isin(
            [
                "footway",
                "pedestrian",
                "unclassified",
                "steps",
                "corridor",
                "path",
            ]
        )
        # Encode categoricals and bools
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[categorical_cols + bool_cols] = enc.fit_transform(
            X[categorical_cols + bool_cols]
        )
        # Impute with MICE
        imp = IterativeImputer(max_iter=max_iter, random_state=random_state)
        X_imputed = pd.DataFrame(
            imp.fit_transform(X), columns=input_cols, index=X.index
        )
        # Clamp numericals >= 0
        for col in numerical_cols:
            X_imputed[col] = X_imputed[col].clip(lower=0)
        # Restore footway rows for specific cols
        X_imputed.loc[mask_foot, ["lanes", "maxspeed"]] = np.nan
        # Decode categoricals and bools back
        codes = X_imputed[categorical_cols + bool_cols].round().astype(int)
        X_imputed[categorical_cols + bool_cols] = enc.inverse_transform(codes)
        # Reattach untouched columns
        result = pd.concat([edges_copy[non_input_cols], X_imputed], axis=1)
        return result
