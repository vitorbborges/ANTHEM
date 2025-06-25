import geopandas as gpd
import numpy as np
import pandas as pd
from fancyimpute import SoftImpute
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder


class FeatureImputer:
    """
    Provides routines to impute missing values for GeoDataFrames and tabular edge
    DataFrames. Contains:

    - impute_gdf: soft-matrix completion for spatial attribute tables.
    - impute_edges: MICE-based imputation for graph edge attributes.
    """

    @staticmethod
    def impute_gdf(
        gdf: gpd.GeoDataFrame, exclude: list[str], max_rank=20, max_iters=100
    ) -> gpd.GeoDataFrame:
        """
        Impute missing values in a GeoDataFrame, preserving geometry.

        Parameters
        ----------
        gdf : GeoDataFrame
            Input GeoDataFrame with geometry and attributes.
        exclude : list[str]
            Names of columns to exclude from imputation.
        max_rank : int, default 20
            Maximum rank for SoftImpute factorization.
        max_iters : int, default 100
            Maximum number of iterations for SoftImpute.

        Returns
        -------
        GeoDataFrame
            Copy of input with imputed attributes and original geometry.
        """
        # Select columns to impute
        use_cols = [c for c in gdf.columns if c not in exclude]
        X = gdf[use_cols].copy()
        # Handle list-valued columns by taking first element or NaN
        list_cols = [
            c for c in use_cols if X[c].apply(lambda v: isinstance(v, list)).any()
        ]
        for c in list_cols:
            X[c] = X[c].apply(lambda v: v[0] if isinstance(v, list) and v else np.nan)
        # Convert object columns with mostly numeric values to numeric
        for c in X.columns:
            if X[c].dtype == object:
                num = pd.to_numeric(X[c], errors="coerce")
                if num.notna().sum() > len(num) / 2:
                    X[c] = num
        # Identify categorical and numeric columns
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        # Create dummy variables for categorical fields
        dummies = pd.get_dummies(X[cat_cols], dummy_na=False)
        # Combine numeric and dummy columns into matrix
        M = pd.concat([X[num_cols], dummies], axis=1).astype(float)
        # Apply SoftImpute to fill missing entries
        filled = SoftImpute(
            max_rank=max_rank, max_iters=max_iters, verbose=False
        ).fit_transform(M.values)
        # Reconstruct DataFrame of imputed values
        Mf = pd.DataFrame(filled, columns=M.columns, index=M.index)
        # Prepare output GeoDataFrame
        out = gdf.copy()
        # Assign imputed numeric columns
        for c in num_cols:
            out[c] = Mf[c]
        # Decode categorical columns from dummy matrix
        for c in cat_cols:
            opts = [col for col in Mf.columns if col.startswith(c + "_")]
            if opts:
                # Choose category with highest dummy score
                best = Mf[opts].idxmax(axis=1).str[len(c) + 1 :]
                out[c] = best.astype(gdf[c].dtype)
        # Restore geometry column if present
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

        Workflow:
        1. Define columns to exclude and columns to impute.
        2. Mask footway-like highways to preserve NaNs later.
        3. Encode categorical and boolean columns to numeric codes.
        4. Run IterativeImputer to estimate missing entries.
        5. Clamp numeric columns to non-negative values.
        6. Reinstate NaNs for footway rows in lanes and maxspeed.
        7. Decode categories and booleans back to original labels.
        8. Concatenate untouched columns with imputed data.

        Parameters
        ----------
        edges : DataFrame
            Input DataFrame of edge attributes.
        non_input_cols : list[str], optional
            Columns to exclude from imputation (unchanged).
        categorical_cols : list[str], optional
            Columns to treat as categorical for encoding.
        bool_cols : list[str], optional
            Columns to treat as booleans for encoding.
        numerical_cols : list[str], optional
            Columns to treat as numerical for imputation.
        max_iter : int, default 100
            Maximum iterations for IterativeImputer.
        random_state : int, default 0
            Seed for reproducibility.

        Returns
        -------
        DataFrame
            DataFrame with imputed columns and original non-input columns.
        """
        # Establish defaults if not provided
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

        # Combine all input columns
        imputable_cols = categorical_cols + bool_cols + numerical_cols
        edges_copy = edges.copy()
        X = edges_copy[imputable_cols].copy()

        # Identify rows corresponding to footway-like highways
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
        # Encode categorical and boolean columns to ordinal codes
        encoded_cols = categorical_cols + bool_cols
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[encoded_cols] = enc.fit_transform(X[encoded_cols])

        # Impute missing values using MICE
        imp = IterativeImputer(max_iter=max_iter, random_state=random_state)
        X_imputed = pd.DataFrame(
            imp.fit_transform(X), columns=imputable_cols, index=X.index
        )

        # Ensure numeric columns are non-negative
        for col in numerical_cols:
            X_imputed[col] = X_imputed[col].clip(lower=0)

        # Restore NaNs for footways in lanes and maxspeed
        X_imputed.loc[mask_foot, ["lanes", "maxspeed"]] = np.nan

        # Decode ordinal codes back to original labels
        codes = X_imputed[encoded_cols].round().astype(int)

        # *** FIX START ***
        # Clip the imputed codes to the valid range of categories learned by the encoder.
        # This prevents an IndexError in inverse_transform if the imputer predicts a
        # value outside the range of the original category codes.
        for i, col in enumerate(encoded_cols):
            num_categories = len(enc.categories_[i])
            codes[col] = codes[col].clip(0, num_categories - 1)
        # *** FIX END ***

        # Perform the inverse transformation with the cleaned codes
        X_imputed[encoded_cols] = enc.inverse_transform(codes)

        # Reattach non-input columns unchanged
        result = pd.concat([edges_copy[non_input_cols], X_imputed], axis=1)

        return result
