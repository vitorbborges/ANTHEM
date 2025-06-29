#!/usr/bin/env python3
"""
Residual Analysis Script for Universal Kriging Model

This script loads the best trained model, generates predictions on a holdout set,
and performs comprehensive residual analysis including:
- Residual distribution plots
- Spatial autocorrelation analysis (Moran's I)
- Temporal autocorrelation analysis
- Normality tests
- Heteroscedasticity tests
- QQ plots and other diagnostic plots
"""

import warnings

warnings.filterwarnings("ignore")

import json
import pickle
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from shapely.geometry import Point
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf, ljungbox

# Add src to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Try to import our modules
try:
    from src.modeling.cv_handler import SpatioTemporalCV
    from src.modeling.predictor import UniversalKrigingPredictor
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class ResidualAnalyzer:
    """Comprehensive residual analysis for spatial-temporal models."""

    def __init__(self, model_path: str = None):
        """Initialize with a trained model."""
        self.predictor = None
        self.model_info = None
        self.load_model(model_path)

    def load_model(self, model_path: str = None):
        """Load the best model or a specific model."""
        try:
            if model_path:
                self.predictor = UniversalKrigingPredictor(model_path=model_path)
            else:
                self.predictor = UniversalKrigingPredictor.load_best_model()

            self.model_info = self.predictor.get_model_info()
            print(f"✅ Loaded model from trial {self.model_info['trial_number']}")
            print(
                f"📊 Ensemble size: {self.model_info['n_subject_models']} subject models"
            )

        except Exception as e:
            raise ValueError(f"Failed to load model: {e}")

    def create_holdout_set(self, data_path: str, holdout_size: float = 0.2):
        """Create a holdout set for residual analysis."""
        # Load data
        df = pd.read_parquet(data_path)

        # Keep only numeric columns and remove unwanted ones
        numeric_cols = [
            col for col in df.columns if df[col].dtype in ("float64", "int64")
        ]
        remove_cols = ["PM1", "PM10", "VOC"]
        keep_cols = [col for col in numeric_cols if col not in remove_cols]

        X = df[keep_cols].dropna()
        y = X.pop("CO2")

        # Create subject-wise holdout split
        cv = SpatioTemporalCV(
            temporal_folds=4, spatial_folds=3, holdout_size=holdout_size
        )
        X_dev, y_dev, X_holdout, y_holdout = cv._create_holdout_split(X, y)

        print(
            f"📊 Created holdout set: {len(X_holdout)} samples from {len(X_holdout['sub'].unique())} subjects"
        )

        return X_holdout, y_holdout

    def generate_predictions(self, X_holdout: pd.DataFrame):
        """Generate predictions for the holdout set."""
        # Prepare environmental data if available
        env_features = [
            "velocita_vento_massimo",
            "velocita_vento_medio",
            "direzzione_vento_massimo",
            "direzzione_vento_medio",
            "radiazione_globale_medio",
            "precipitazione_valore_cumulato",
            "P",
            "T",
            "RH",
        ]

        available_env = [f for f in env_features if f in X_holdout.columns]

        if available_env:
            print(f"🌡️ Using environmental features: {available_env}")
            env_data = {col: X_holdout[col].values for col in available_env}
        else:
            print("⚠️ No environmental features found, using coordinates only")
            env_data = None

        # Make predictions with uncertainty
        results = self.predictor.predict(
            x=X_holdout["x"].values,
            y=X_holdout["y"].values,
            environmental_data=env_data,
            return_uncertainty=True,
            return_individual_predictions=True,
        )

        return results

    def calculate_residuals(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calculate various types of residuals."""
        residuals = {
            "raw": y_true - y_pred,
            "standardized": (y_true - y_pred) / np.std(y_true - y_pred),
            "studentized": None,  # Will be calculated if we have uncertainty
        }

        return residuals

    def spatial_autocorrelation_test(
        self,
        residuals: np.ndarray,
        coordinates: np.ndarray,
        max_distance: float = 500.0,
    ):
        """Test for spatial autocorrelation in residuals using Moran's I."""
        from scipy.spatial.distance import pdist, squareform

        # Create distance matrix
        distances = squareform(pdist(coordinates))

        # Create spatial weights matrix (inverse distance with cutoff)
        weights = np.zeros_like(distances)
        mask = (distances > 0) & (distances <= max_distance)
        weights[mask] = 1.0 / distances[mask]

        # Row-normalize weights
        row_sums = weights.sum(axis=1)
        weights = weights / row_sums[:, np.newaxis]
        weights[np.isnan(weights)] = 0

        # Calculate Moran's I
        n = len(residuals)
        z = residuals - np.mean(residuals)

        # Numerator: sum of weighted cross-products
        numerator = np.sum(weights * np.outer(z, z))

        # Denominator: sum of squared deviations
        denominator = np.sum(z**2)

        if denominator == 0:
            return 0, 0, 1  # No variance in residuals

        morans_i = (n / np.sum(weights)) * (numerator / denominator)

        # Expected value and variance under null hypothesis
        expected_i = -1 / (n - 1)
        s0 = np.sum(weights)
        s1 = 0.5 * np.sum((weights + weights.T) ** 2)
        s2 = np.sum(np.sum(weights, axis=1) ** 2)

        var_i = (
            (n * ((n**2 - 3 * n + 3) * s1 - n * s2 + 3 * s0**2))
            - ((n - 1) * (n - 2) * (n - 3) * s0**2)
        ) / ((n - 1) ** 2 * (n - 2) * (n - 3) * s0**2)

        # Z-score and p-value
        z_score = (morans_i - expected_i) / np.sqrt(var_i)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        return morans_i, z_score, p_value

    def temporal_autocorrelation_test(self, residuals: np.ndarray, max_lags: int = 20):
        """Test for temporal autocorrelation in residuals."""
        # Autocorrelation function
        autocorr = acf(residuals, nlags=max_lags, alpha=0.05)

        # Ljung-Box test for autocorrelation
        lb_stat, lb_pvalue = ljungbox(
            residuals, lags=min(10, len(residuals) // 4), return_df=False
        )

        # Durbin-Watson test
        dw_stat = durbin_watson(residuals)

        return autocorr, lb_stat, lb_pvalue, dw_stat

    def normality_tests(self, residuals: np.ndarray):
        """Perform normality tests on residuals."""
        tests = {}

        # Shapiro-Wilk test (for smaller samples)
        if len(residuals) <= 5000:
            tests["shapiro"] = stats.shapiro(residuals)

        # Kolmogorov-Smirnov test
        tests["ks"] = stats.kstest(
            residuals, "norm", args=(np.mean(residuals), np.std(residuals))
        )

        # Anderson-Darling test
        tests["anderson"] = stats.anderson(residuals, dist="norm")

        # Jarque-Bera test
        tests["jarque_bera"] = stats.jarque_bera(residuals)

        return tests

    def heteroscedasticity_tests(
        self, residuals: np.ndarray, fitted_values: np.ndarray
    ):
        """Test for heteroscedasticity in residuals."""
        # Breusch-Pagan test
        # Create a simple linear model: residuals^2 ~ fitted_values
        X = np.column_stack([np.ones(len(fitted_values)), fitted_values])
        y = residuals**2

        try:
            bp_stat, bp_pvalue, _, _ = het_breuschpagan(y, X)
        except:
            bp_stat, bp_pvalue = np.nan, np.nan

        # White test (simplified)
        X_white = np.column_stack(
            [np.ones(len(fitted_values)), fitted_values, fitted_values**2]
        )
        try:
            from sklearn.linear_model import LinearRegression

            reg = LinearRegression().fit(X_white, y)
            r2 = reg.score(X_white, y)
            white_stat = len(fitted_values) * r2
            white_pvalue = 1 - stats.chi2.cdf(white_stat, df=2)
        except:
            white_stat, white_pvalue = np.nan, np.nan

        return {
            "breusch_pagan": (bp_stat, bp_pvalue),
            "white": (white_stat, white_pvalue),
        }

    def create_diagnostic_plots(
        self,
        residuals_dict: dict,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        coordinates: np.ndarray,
        uncertainty: np.ndarray = None,
    ):
        """Create comprehensive diagnostic plots."""
        fig = plt.figure(figsize=(20, 16))

        # 1. Residuals vs Fitted
        plt.subplot(3, 4, 1)
        plt.scatter(y_pred, residuals_dict["raw"], alpha=0.6, s=20)
        plt.axhline(y=0, color="red", linestyle="--", alpha=0.8)
        plt.xlabel("Fitted Values")
        plt.ylabel("Residuals")
        plt.title("Residuals vs Fitted Values")

        # Add LOWESS smoothing line
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess

            smoothed = lowess(residuals_dict["raw"], y_pred, frac=0.3)
            plt.plot(smoothed[:, 0], smoothed[:, 1], "blue", linewidth=2, alpha=0.8)
        except:
            pass

        # 2. QQ Plot
        plt.subplot(3, 4, 2)
        stats.probplot(residuals_dict["standardized"], dist="norm", plot=plt)
        plt.title("Q-Q Plot (Standardized Residuals)")
        plt.grid(True, alpha=0.3)

        # 3. Scale-Location Plot
        plt.subplot(3, 4, 3)
        sqrt_abs_resid = np.sqrt(np.abs(residuals_dict["standardized"]))
        plt.scatter(y_pred, sqrt_abs_resid, alpha=0.6, s=20)
        plt.xlabel("Fitted Values")
        plt.ylabel("√|Standardized Residuals|")
        plt.title("Scale-Location Plot")

        # 4. Histogram of Residuals
        plt.subplot(3, 4, 4)
        plt.hist(
            residuals_dict["raw"], bins=50, density=True, alpha=0.7, edgecolor="black"
        )

        # Overlay normal distribution
        x = np.linspace(residuals_dict["raw"].min(), residuals_dict["raw"].max(), 100)
        plt.plot(
            x,
            stats.norm.pdf(
                x, np.mean(residuals_dict["raw"]), np.std(residuals_dict["raw"])
            ),
            "red",
            linewidth=2,
            label="Normal",
        )
        plt.xlabel("Residuals")
        plt.ylabel("Density")
        plt.title("Histogram of Residuals")
        plt.legend()

        # 5. Spatial Distribution of Residuals
        plt.subplot(3, 4, 5)
        scatter = plt.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=residuals_dict["raw"],
            cmap="RdBu_r",
            s=30,
            alpha=0.7,
        )
        plt.colorbar(scatter, label="Residuals")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("Spatial Distribution of Residuals")

        # 6. Autocorrelation Function
        plt.subplot(3, 4, 6)
        autocorr, _, _, _ = self.temporal_autocorrelation_test(residuals_dict["raw"])
        lags = range(len(autocorr[0]))
        plt.plot(lags, autocorr[0], "bo-", alpha=0.7)
        if autocorr[1] is not None:  # Confidence intervals
            plt.fill_between(lags, autocorr[1][:, 0], autocorr[1][:, 1], alpha=0.3)
        plt.axhline(y=0, color="red", linestyle="--", alpha=0.8)
        plt.xlabel("Lag")
        plt.ylabel("Autocorrelation")
        plt.title("Autocorrelation Function")

        # 7. Observed vs Predicted
        plt.subplot(3, 4, 7)
        plt.scatter(y_true, y_pred, alpha=0.6, s=20)

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot(
            [min_val, max_val], [min_val, max_val], "red", linestyle="--", alpha=0.8
        )

        plt.xlabel("Observed")
        plt.ylabel("Predicted")
        plt.title("Observed vs Predicted")

        # Add R² annotation
        r2 = r2_score(y_true, y_pred)
        plt.text(
            0.05,
            0.95,
            f"R² = {r2:.3f}",
            transform=plt.gca().transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        # 8. Residuals vs Index (if there's a natural order)
        plt.subplot(3, 4, 8)
        plt.plot(range(len(residuals_dict["raw"])), residuals_dict["raw"], alpha=0.7)
        plt.axhline(y=0, color="red", linestyle="--", alpha=0.8)
        plt.xlabel("Index")
        plt.ylabel("Residuals")
        plt.title("Residuals vs Index")

        # 9. Box plot of residuals by subject (if available)
        if "sub" in locals():  # This would need to be passed as parameter
            plt.subplot(3, 4, 9)
            # Would create box plots by subject
            plt.title("Residuals by Subject")
        else:
            plt.subplot(3, 4, 9)
            plt.boxplot(residuals_dict["raw"])
            plt.ylabel("Residuals")
            plt.title("Box Plot of Residuals")

        # 10. Uncertainty vs Absolute Residuals (if uncertainty available)
        plt.subplot(3, 4, 10)
        if uncertainty is not None:
            plt.scatter(uncertainty, np.abs(residuals_dict["raw"]), alpha=0.6, s=20)
            plt.xlabel("Predicted Uncertainty")
            plt.ylabel("|Residuals|")
            plt.title("Uncertainty vs |Residuals|")

            # Add correlation
            corr = np.corrcoef(uncertainty, np.abs(residuals_dict["raw"]))[0, 1]
            plt.text(
                0.05,
                0.95,
                f"Corr = {corr:.3f}",
                transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )
        else:
            plt.text(
                0.5,
                0.5,
                "No uncertainty\navailable",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
                fontsize=12,
            )
            plt.title("Uncertainty vs |Residuals|")

        # 11. Kernel Density Estimate
        plt.subplot(3, 4, 11)
        try:
            from scipy.stats import gaussian_kde

            kde = gaussian_kde(residuals_dict["raw"])
            x_range = np.linspace(
                residuals_dict["raw"].min(), residuals_dict["raw"].max(), 200
            )
            plt.plot(x_range, kde(x_range), label="KDE", linewidth=2)

            # Compare with normal
            normal_pdf = stats.norm.pdf(
                x_range, np.mean(residuals_dict["raw"]), np.std(residuals_dict["raw"])
            )
            plt.plot(x_range, normal_pdf, label="Normal", linewidth=2, linestyle="--")
            plt.xlabel("Residuals")
            plt.ylabel("Density")
            plt.title("Kernel Density Estimate")
            plt.legend()
        except:
            plt.text(
                0.5,
                0.5,
                "KDE failed",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )

        # 12. Leverage vs Residuals (Cook's Distance style)
        plt.subplot(3, 4, 12)
        if uncertainty is not None:
            # Use uncertainty as a proxy for leverage
            leverage = 1.0 / (
                uncertainty + 1e-8
            )  # Inverse uncertainty as leverage proxy
            plt.scatter(leverage, residuals_dict["standardized"], alpha=0.6, s=20)
            plt.xlabel("Leverage (1/Uncertainty)")
            plt.ylabel("Standardized Residuals")
            plt.title("Leverage vs Residuals")
        else:
            plt.text(
                0.5,
                0.5,
                "No leverage\ninformation\navailable",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
                fontsize=12,
            )
            plt.title("Leverage vs Residuals")

        plt.tight_layout()
        return fig

    def run_full_analysis(self, data_path: str, save_plots: bool = True):
        """Run comprehensive residual analysis."""
        print("🔍 Starting comprehensive residual analysis...")

        # Create holdout set
        X_holdout, y_holdout = self.create_holdout_set(data_path)

        # Generate predictions
        print("🎯 Generating predictions...")
        results = self.generate_predictions(X_holdout)
        y_pred = results["predictions"]
        uncertainty = results.get("uncertainty", None)

        # Calculate residuals
        residuals_dict = self.calculate_residuals(y_holdout.values, y_pred)

        # Basic metrics
        mse = mean_squared_error(y_holdout, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_holdout, y_pred)
        r2 = r2_score(y_holdout, y_pred)

        print(f"\n📊 Prediction Performance:")
        print(f"   RMSE: {rmse:.3f}")
        print(f"   MAE:  {mae:.3f}")
        print(f"   R²:   {r2:.3f}")

        # Spatial autocorrelation test
        print("\n🗺️ Spatial Autocorrelation Analysis:")
        coordinates = X_holdout[["x", "y"]].values
        morans_i, z_score, p_value = self.spatial_autocorrelation_test(
            residuals_dict["raw"], coordinates
        )
        print(f"   Moran's I: {morans_i:.4f}")
        print(f"   Z-score:   {z_score:.4f}")
        print(f"   P-value:   {p_value:.4f}")

        if p_value < 0.05:
            if morans_i > 0:
                print("   🔴 Significant positive spatial autocorrelation detected!")
            else:
                print("   🔴 Significant negative spatial autocorrelation detected!")
        else:
            print("   🟢 No significant spatial autocorrelation")

        # Temporal autocorrelation test
        print("\n⏰ Temporal Autocorrelation Analysis:")
        autocorr, lb_stat, lb_pvalue, dw_stat = self.temporal_autocorrelation_test(
            residuals_dict["raw"]
        )
        print(f"   Ljung-Box statistic: {lb_stat:.4f}")
        print(f"   Ljung-Box p-value:   {lb_pvalue:.4f}")
        print(f"   Durbin-Watson:       {dw_stat:.4f}")

        if lb_pvalue < 0.05:
            print("   🔴 Significant temporal autocorrelation detected!")
        else:
            print("   🟢 No significant temporal autocorrelation")

        # Normality tests
        print("\n📈 Normality Tests:")
        norm_tests = self.normality_tests(residuals_dict["raw"])

        for test_name, result in norm_tests.items():
            if test_name == "anderson":
                stat, crit_vals, sig_levels = result
                print(f"   {test_name.title()}: statistic = {stat:.4f}")
                for i, (cv, sl) in enumerate(zip(crit_vals, sig_levels)):
                    if stat > cv:
                        print(f"     🔴 Rejected normality at {sl}% level")
                        break
                else:
                    print(f"     🟢 Cannot reject normality")
            else:
                stat, pval = result
                print(
                    f"   {test_name.title()}: statistic = {stat:.4f}, p-value = {pval:.4f}"
                )
                if pval < 0.05:
                    print(f"     🔴 Rejected normality")
                else:
                    print(f"     🟢 Cannot reject normality")

        # Heteroscedasticity tests
        print("\n📏 Heteroscedasticity Tests:")
        hetero_tests = self.heteroscedasticity_tests(residuals_dict["raw"], y_pred)

        for test_name, (stat, pval) in hetero_tests.items():
            if not np.isnan(stat):
                print(
                    f"   {test_name.title()}: statistic = {stat:.4f}, p-value = {pval:.4f}"
                )
                if pval < 0.05:
                    print(f"     🔴 Heteroscedasticity detected")
                else:
                    print(f"     🟢 Homoscedasticity (constant variance)")

        # Create diagnostic plots
        print("\n📊 Creating diagnostic plots...")
        fig = self.create_diagnostic_plots(
            residuals_dict, y_holdout.values, y_pred, coordinates, uncertainty
        )

        if save_plots:
            plots_dir = Path("analysis_results")
            plots_dir.mkdir(exist_ok=True)

            plot_path = (
                plots_dir
                / f"residual_analysis_trial_{self.model_info['trial_number']}.png"
            )
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"💾 Diagnostic plots saved: {plot_path}")

        plt.show()

        # Summary report
        print("\n" + "=" * 80)
        print("RESIDUAL ANALYSIS SUMMARY")
        print("=" * 80)

        print(f"Model Trial: {self.model_info['trial_number']}")
        print(f"Holdout Set Size: {len(y_holdout)} samples")
        print(f"Prediction RMSE: {rmse:.3f}")
        print(f"Prediction R²: {r2:.3f}")

        # Overall assessment
        issues = []
        if p_value < 0.05:  # Spatial autocorrelation
            issues.append("Spatial autocorrelation in residuals")
        if lb_pvalue < 0.05:  # Temporal autocorrelation
            issues.append("Temporal autocorrelation in residuals")

        norm_rejected = sum(
            1
            for test_name, result in norm_tests.items()
            if (test_name != "anderson" and result[1] < 0.05)
            or (test_name == "anderson" and result[0] > result[1][2])
        )  # 5% level
        if norm_rejected >= 2:  # Majority of tests reject normality
            issues.append("Non-normal residuals")

        hetero_detected = sum(
            1
            for _, (stat, pval) in hetero_tests.items()
            if not np.isnan(pval) and pval < 0.05
        )
        if hetero_detected > 0:
            issues.append("Heteroscedasticity")

        if issues:
            print(f"\n🔴 Issues detected:")
            for issue in issues:
                print(f"   - {issue}")
            print(f"\n💡 Recommendations:")
            if "Spatial autocorrelation" in str(issues):
                print(
                    f"   - Consider larger kriging radius or different variogram model"
                )
            if "Temporal autocorrelation" in str(issues):
                print(f"   - Add temporal features or use temporal kriging")
            if "Non-normal residuals" in str(issues):
                print(f"   - Consider data transformation or robust methods")
            if "Heteroscedasticity" in str(issues):
                print(f"   - Consider weighted regression or variance modeling")
        else:
            print(f"\n🟢 No major issues detected - model residuals look good!")

        print("=" * 80)

        return {
            "residuals": residuals_dict,
            "predictions": y_pred,
            "uncertainty": uncertainty,
            "coordinates": coordinates,
            "metrics": {
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "morans_i": morans_i,
                "spatial_pvalue": p_value,
                "temporal_pvalue": lb_pvalue,
                "normality_tests": norm_tests,
                "heteroscedasticity_tests": hetero_tests,
            },
        }


if __name__ == "__main__":
    # Example usage
    data_path = "data/processed_data/combined_subjects.parquet"

    # Initialize analyzer
    analyzer = ResidualAnalyzer()

    # Run full analysis
    results = analyzer.run_full_analysis(data_path, save_plots=True)

    print("\n✅ Residual analysis complete!")
