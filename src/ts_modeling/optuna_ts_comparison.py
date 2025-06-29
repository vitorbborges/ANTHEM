import logging
import warnings
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

warnings.filterwarnings("ignore")

# ARCH/GARCH models for heteroskedasticity
from arch import arch_model
from arch.unitroot import DFGLS, PhillipsPerron
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace import mlemodel
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter

# Core time series libraries
from statsmodels.tsa.stattools import adfuller, kpss

# Gaussian Process libraries
try:
    import sklearn.gaussian_process as gp
    from sklearn.gaussian_process.kernels import (
        RBF,
        ConstantKernel,
        ExpSineSquared,
        Matern,
        WhiteKernel,
    )
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler

    GP_AVAILABLE = True
except ImportError:
    GP_AVAILABLE = False
    print("Scikit-learn GP not available")

# Advanced signal processing
from scipy import signal
from scipy.stats import jarque_bera, normaltest

# Configure Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaCO2TimeSeriesModeling:
    """
    CO2 Time Series modeling with Optuna hyperparameter optimization
    """

    def __init__(self, data, freq="s", n_trials=100, test_size=0.2):
        self.data = pd.Series(data).dropna()
        self.n = len(self.data)
        self.freq = freq
        self.n_trials = n_trials

        # Split data for validation
        split_idx = int(self.n * (1 - test_size))
        self.train_data = self.data[:split_idx]
        self.test_data = self.data[split_idx:]

        self.best_models = {}
        self.studies = {}

        print(f"Data split: Train={len(self.train_data)}, Test={len(self.test_data)}")

    def comprehensive_diagnostics(self):
        """Enhanced diagnostic suite"""
        print("=== COMPREHENSIVE TIME SERIES DIAGNOSTICS ===\n")

        # Basic statistics
        print("Basic Statistics:")
        print(f"Mean: {self.data.mean():.4f}")
        print(f"Std: {self.data.std():.4f}")
        print(f"Skewness: {self.data.skew():.4f}")
        print(f"Kurtosis: {self.data.kurtosis():.4f}")

        # Stationarity tests
        print("\nStationarity Tests:")
        adf_stat, adf_pval = adfuller(self.data)[:2]
        print(
            f"ADF p-value: {adf_pval:.5f} {'(Stationary)' if adf_pval < 0.05 else '(Non-stationary)'}"
        )

        kpss_stat, kpss_pval = kpss(self.data, regression="c")[:2]
        print(
            f"KPSS p-value: {kpss_pval:.5f} {'(Stationary)' if kpss_pval > 0.05 else '(Non-stationary)'}"
        )

        # Long memory assessment
        self._assess_long_memory()

    def _assess_long_memory(self):
        """Assess long memory characteristics"""
        print("\nLong Memory Assessment:")

        # Calculate ACF decay
        from statsmodels.tsa.stattools import acf

        acf_vals = acf(self.data, nlags=min(200, self.n // 4), fft=True)

        # Find where ACF first crosses 0.1
        decay_point = np.where(np.abs(acf_vals) < 0.1)[0]
        if len(decay_point) > 0:
            print(f"ACF decays below 0.1 at lag: {decay_point[0]}")
        else:
            print("ACF remains above 0.1 for all computed lags - strong long memory")

    def optimize_arima_model(self):
        """Optimize ARIMA model using Optuna"""
        print("\n=== OPTIMIZING ARIMA MODEL WITH OPTUNA ===")

        def objective(trial):
            # Suggest hyperparameters
            p = trial.suggest_int("p", 0, 5)
            d = trial.suggest_int("d", 0, 2)
            q = trial.suggest_int("q", 0, 5)

            # Add seasonal parameters if needed
            include_seasonal = trial.suggest_categorical(
                "include_seasonal", [False, True]
            )
            if include_seasonal:
                P = trial.suggest_int("P", 0, 2)
                D = trial.suggest_int("D", 0, 1)
                Q = trial.suggest_int("Q", 0, 2)
                s = trial.suggest_int("s", 4, 24)  # Seasonal period
                seasonal_order = (P, D, Q, s)
            else:
                seasonal_order = (0, 0, 0, 0)

            try:
                # Fit ARIMA model
                model = ARIMA(
                    self.train_data, order=(p, d, q), seasonal_order=seasonal_order
                )
                fitted_model = model.fit()

                # Calculate validation score using test data
                forecast = fitted_model.forecast(steps=len(self.test_data))
                mse = np.mean((self.test_data - forecast) ** 2)

                # Use negative MSE for maximization
                return -mse

            except Exception as e:
                # Return a large penalty for failed fits
                return -1e10

        # Create and optimize study
        study = optuna.create_study(
            direction="maximize", study_name="arima_optimization"
        )
        study.optimize(objective, n_trials=self.n_trials)

        # Store results
        self.studies["ARIMA"] = study

        # Fit best model on full training data
        best_params = study.best_params
        print(f"Best ARIMA parameters: {best_params}")

        try:
            if best_params["include_seasonal"]:
                seasonal_order = (
                    best_params["P"],
                    best_params["D"],
                    best_params["Q"],
                    best_params["s"],
                )
            else:
                seasonal_order = (0, 0, 0, 0)

            best_model = ARIMA(
                self.train_data,
                order=(best_params["p"], best_params["d"], best_params["q"]),
                seasonal_order=seasonal_order,
            )
            fitted_best = best_model.fit()

            self.best_models["ARIMA"] = fitted_best

            print(f"Best ARIMA validation score: {study.best_value:.4f}")
            print("\nBest ARIMA Model Summary:")
            print(fitted_best.summary())

        except Exception as e:
            print(f"Failed to fit best ARIMA model: {e}")

        return study

    def optimize_garch_model(self):
        """Optimize GARCH model using Optuna"""
        print("\n=== OPTIMIZING GARCH MODEL WITH OPTUNA ===")

        # Prepare data (GARCH needs returns-like data)
        train_diff = self.train_data.diff().dropna()
        test_diff = self.test_data.diff().dropna()

        def objective(trial):
            # Mean model parameters
            mean_model = trial.suggest_categorical("mean", ["Constant", "Zero", "AR"])
            if mean_model == "AR":
                lags = trial.suggest_int("lags", 1, 5)
            else:
                lags = None

            # Volatility model parameters
            vol_model = trial.suggest_categorical("vol", ["GARCH", "EGARCH", "FIGARCH"])
            p = trial.suggest_int("p", 1, 3)
            q = trial.suggest_int("q", 1, 3)

            # EGARCH specific parameter
            if vol_model == "GARCH":
                o = trial.suggest_int("o", 0, 1)  # GJR term
            else:
                o = 0

            # Distribution
            dist = trial.suggest_categorical("dist", ["normal", "t", "skewt"])

            try:
                # Create GARCH model
                if vol_model == "GARCH":
                    model = arch_model(
                        train_diff,
                        mean=mean_model,
                        lags=lags,
                        vol="GARCH",
                        p=p,
                        o=o,
                        q=q,
                        dist=dist,
                    )
                else:
                    model = arch_model(
                        train_diff,
                        mean=mean_model,
                        lags=lags,
                        vol=vol_model,
                        p=p,
                        q=q,
                        dist=dist,
                    )

                fitted_model = model.fit(disp="off")

                # Calculate validation score
                forecast = fitted_model.forecast(horizon=len(test_diff))
                forecast_var = forecast.variance.values[-1]

                # Use log-likelihood or other appropriate metric
                score = fitted_model.loglikelihood

                return score

            except Exception as e:
                return -1e10

        # Create and optimize study
        study = optuna.create_study(
            direction="maximize", study_name="garch_optimization"
        )
        study.optimize(objective, n_trials=self.n_trials)

        # Store results
        self.studies["GARCH"] = study

        # Fit best model
        best_params = study.best_params
        print(f"Best GARCH parameters: {best_params}")

        try:
            # Reconstruct best model
            if best_params["vol"] == "GARCH":
                best_model = arch_model(
                    train_diff,
                    mean=best_params["mean"],
                    lags=best_params.get("lags"),
                    vol="GARCH",
                    p=best_params["p"],
                    o=best_params.get("o", 0),
                    q=best_params["q"],
                    dist=best_params["dist"],
                )
            else:
                best_model = arch_model(
                    train_diff,
                    mean=best_params["mean"],
                    lags=best_params.get("lags"),
                    vol=best_params["vol"],
                    p=best_params["p"],
                    q=best_params["q"],
                    dist=best_params["dist"],
                )

            fitted_best = best_model.fit(disp="off")
            self.best_models["GARCH"] = fitted_best

            print(f"Best GARCH validation score: {study.best_value:.4f}")
            print("\nBest GARCH Model Summary:")
            print(fitted_best.summary())

        except Exception as e:
            print(f"Failed to fit best GARCH model: {e}")

        return study

    def optimize_gaussian_process(self):
        """Optimize Gaussian Process model using Optuna"""
        if not GP_AVAILABLE:
            print("Gaussian Process modeling not available")
            return None

        print("\n=== OPTIMIZING GAUSSIAN PROCESS WITH OPTUNA ===")

        # Prepare data
        X_train = np.arange(len(self.train_data)).reshape(-1, 1)
        y_train = self.train_data.values
        X_test = np.arange(len(self.train_data), len(self.data)).reshape(-1, 1)
        y_test = self.test_data.values

        # Scale data
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        X_test_scaled = scaler_X.transform(X_test)

        def objective(trial):
            # Kernel selection and parameters
            kernel_type = trial.suggest_categorical(
                "kernel_type", ["RBF", "Matern", "RBF_Periodic", "Multi_scale"]
            )

            if kernel_type == "RBF":
                length_scale = trial.suggest_float(
                    "length_scale", 0.01, 100.0, log=True
                )
                kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=0.1)

            elif kernel_type == "Matern":
                length_scale = trial.suggest_float(
                    "length_scale", 0.01, 100.0, log=True
                )
                nu = trial.suggest_categorical("nu", [0.5, 1.5, 2.5])
                kernel = Matern(length_scale=length_scale, nu=nu) + WhiteKernel(
                    noise_level=0.1
                )

            elif kernel_type == "RBF_Periodic":
                rbf_length = trial.suggest_float("rbf_length", 0.01, 100.0, log=True)
                exp_length = trial.suggest_float("exp_length", 0.01, 100.0, log=True)
                periodicity = trial.suggest_float("periodicity", 0.1, 1000.0, log=True)
                kernel = RBF(length_scale=rbf_length) * ExpSineSquared(
                    length_scale=exp_length, periodicity=periodicity
                ) + WhiteKernel(noise_level=0.1)

            else:  # Multi_scale
                ls1 = trial.suggest_float("ls1", 0.01, 100.0, log=True)
                ls2 = trial.suggest_float("ls2", 0.01, 100.0, log=True)
                ls3 = trial.suggest_float("ls3", 0.01, 100.0, log=True)
                kernel = (
                    RBF(length_scale=ls1)
                    + RBF(length_scale=ls2)
                    + RBF(length_scale=ls3)
                    + WhiteKernel(noise_level=0.1)
                )

            # GP parameters
            alpha = trial.suggest_float("alpha", 1e-12, 1e-6, log=True)
            n_restarts = trial.suggest_int("n_restarts", 1, 5)

            try:
                # Subsample for computational efficiency
                n_subsample = min(800, len(X_train_scaled))
                indices = np.linspace(
                    0, len(X_train_scaled) - 1, n_subsample, dtype=int
                )
                X_sub = X_train_scaled[indices]
                y_sub = y_train_scaled[indices]

                # Fit GP
                gp_model = gp.GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=alpha,
                    n_restarts_optimizer=n_restarts,
                    normalize_y=False,
                )

                gp_model.fit(X_sub, y_sub)

                # Predict on test set
                y_pred_scaled = gp_model.predict(X_test_scaled)
                y_pred = scaler_y.inverse_transform(
                    y_pred_scaled.reshape(-1, 1)
                ).ravel()

                # Calculate validation score
                mse = np.mean((y_test - y_pred) ** 2)

                return -mse  # Negative for maximization

            except Exception as e:
                return -1e10

        # Create and optimize study
        study = optuna.create_study(direction="maximize", study_name="gp_optimization")
        study.optimize(objective, n_trials=self.n_trials)

        # Store results
        self.studies["GP"] = study

        # Fit best model
        best_params = study.best_params
        print(f"Best GP parameters: {best_params}")

        try:
            # Reconstruct best kernel
            if best_params["kernel_type"] == "RBF":
                kernel = RBF(length_scale=best_params["length_scale"]) + WhiteKernel(
                    noise_level=0.1
                )
            elif best_params["kernel_type"] == "Matern":
                kernel = Matern(
                    length_scale=best_params["length_scale"], nu=best_params["nu"]
                ) + WhiteKernel(noise_level=0.1)
            elif best_params["kernel_type"] == "RBF_Periodic":
                kernel = RBF(length_scale=best_params["rbf_length"]) * ExpSineSquared(
                    length_scale=best_params["exp_length"],
                    periodicity=best_params["periodicity"],
                ) + WhiteKernel(noise_level=0.1)
            else:  # Multi_scale
                kernel = (
                    RBF(length_scale=best_params["ls1"])
                    + RBF(length_scale=best_params["ls2"])
                    + RBF(length_scale=best_params["ls3"])
                    + WhiteKernel(noise_level=0.1)
                )

            # Fit best GP model
            best_gp = gp.GaussianProcessRegressor(
                kernel=kernel,
                alpha=best_params["alpha"],
                n_restarts_optimizer=best_params["n_restarts"],
                normalize_y=False,
            )

            # Subsample for fitting
            n_subsample = min(800, len(X_train_scaled))
            indices = np.linspace(0, len(X_train_scaled) - 1, n_subsample, dtype=int)
            X_sub = X_train_scaled[indices]
            y_sub = y_train_scaled[indices]

            best_gp.fit(X_sub, y_sub)

            self.best_models["GP"] = (best_gp, scaler_X, scaler_y)

            print(f"Best GP validation score: {study.best_value:.4f}")
            print(f"Optimized kernel: {best_gp.kernel_}")

        except Exception as e:
            print(f"Failed to fit best GP model: {e}")

        return study

    def optimize_arfima_model(self):
        """Optimize ARFIMA-like model using Optuna"""
        print("\n=== OPTIMIZING ARFIMA MODEL WITH OPTUNA ===")

        def fractional_diff(series, d, threshold=0.01):
            """Fractional differencing"""
            weights = [1.0]
            for k in range(1, len(series)):
                weight = -weights[-1] * (d - k + 1) / k
                if abs(weight) < threshold:
                    break
                weights.append(weight)

            weights = np.array(weights)
            return np.convolve(series, weights, mode="valid")

        def objective(trial):
            # Fractional integration parameter
            d = trial.suggest_float("d", 0.1, 0.8)

            # ARMA parameters for fractionally differenced series
            p = trial.suggest_int("p", 0, 3)
            q = trial.suggest_int("q", 0, 3)

            try:
                # Apply fractional differencing
                frac_diff = fractional_diff(self.train_data.values, d)

                if len(frac_diff) < 100:
                    return -1e10

                # Fit ARMA to fractionally differenced series
                arma_model = ARIMA(frac_diff, order=(p, 0, q))
                fitted_model = arma_model.fit()

                # Use AIC as optimization criterion
                return -fitted_model.aic

            except Exception as e:
                return -1e10

        # Create and optimize study
        study = optuna.create_study(
            direction="maximize", study_name="arfima_optimization"
        )
        study.optimize(objective, n_trials=self.n_trials)

        # Store results
        self.studies["ARFIMA"] = study

        # Fit best model
        best_params = study.best_params
        print(f"Best ARFIMA parameters: {best_params}")

        try:
            # Reconstruct best model
            frac_diff = fractional_diff(self.train_data.values, best_params["d"])
            best_arma = ARIMA(frac_diff, order=(best_params["p"], 0, best_params["q"]))
            fitted_best = best_arma.fit()

            self.best_models["ARFIMA"] = (best_params["d"], fitted_best, frac_diff)

            print(f"Best ARFIMA validation score: {study.best_value:.4f}")
            print(f"Optimal d parameter: {best_params['d']:.3f}")
            print("\nBest ARFIMA Model Summary:")
            print(fitted_best.summary())

        except Exception as e:
            print(f"Failed to fit best ARFIMA model: {e}")

        return study

    def compare_optimized_models(self):
        """Compare all optimized models"""
        print("\n=== OPTIMIZED MODEL COMPARISON ===\n")

        if not self.best_models:
            print("No models optimized yet!")
            return

        comparison = []
        forecasts = {}

        for name, model in self.best_models.items():
            try:
                # Calculate test set performance
                if name == "ARIMA":
                    forecast = model.forecast(steps=len(self.test_data))
                    mse = np.mean((self.test_data - forecast) ** 2)
                    aic = model.aic
                    bic = model.bic

                elif name == "GARCH":
                    # For GARCH, we evaluate on differenced data
                    test_diff = self.test_data.diff().dropna()
                    forecast = model.forecast(horizon=len(test_diff))
                    # Use conditional volatility or mean forecast
                    mse = np.mean(
                        (test_diff.values - forecast.mean.values[-len(test_diff) :])
                        ** 2
                    )
                    aic = model.aic
                    bic = model.bic

                elif name == "GP":
                    gp_model, scaler_X, scaler_y = model
                    X_test = np.arange(len(self.train_data), len(self.data)).reshape(
                        -1, 1
                    )
                    X_test_scaled = scaler_X.transform(X_test)
                    y_pred_scaled = gp_model.predict(X_test_scaled)
                    y_pred = scaler_y.inverse_transform(
                        y_pred_scaled.reshape(-1, 1)
                    ).ravel()
                    mse = np.mean((self.test_data.values - y_pred) ** 2)
                    aic = "N/A"
                    bic = "N/A"
                    forecasts[name] = y_pred

                elif name == "ARFIMA":
                    d, arma_model, _ = model
                    # Simple forecast using ARMA model (would need more sophisticated approach for production)
                    # Here we just use the AIC from the ARMA model
                    mse = "N/A"  # Would need more complex implementation
                    aic = arma_model.aic
                    bic = arma_model.bic

                comparison.append(
                    {
                        "Model": name,
                        "MSE": mse if mse != "N/A" else "N/A",
                        "AIC": aic,
                        "BIC": bic,
                        "Optuna_Score": (
                            self.studies[name].best_value
                            if name in self.studies
                            else "N/A"
                        ),
                    }
                )

            except Exception as e:
                print(f"Could not evaluate {name}: {e}")

        if comparison:
            df_comparison = pd.DataFrame(comparison)
            print(df_comparison.to_string(index=False))

            # Find best by MSE (excluding N/A)
            numeric_mse = df_comparison[df_comparison["MSE"] != "N/A"]
            if not numeric_mse.empty:
                best_model = numeric_mse.loc[numeric_mse["MSE"].idxmin()]
                print(
                    f"\nBest model by MSE: {best_model['Model']} (MSE: {best_model['MSE']:.2f})"
                )

        return comparison

    def plot_optimization_histories(self):
        """Plot optimization histories for all models"""
        if not self.studies:
            print("No optimization studies available!")
            return

        n_studies = len(self.studies)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        for i, (name, study) in enumerate(self.studies.items()):
            if i < len(axes):
                optuna.visualization.matplotlib.plot_optimization_history(
                    study, ax=axes[i]
                )
                axes[i].set_title(f"{name} Optimization History")

        # Hide unused subplots
        for j in range(len(self.studies), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.show()

    def run_full_optimization(self):
        """Run optimization for all models"""
        print("=== STARTING FULL OPTUNA OPTIMIZATION SUITE ===\n")

        # Run diagnostics first
        self.comprehensive_diagnostics()

        # Optimize each model type
        self.optimize_arima_model()
        self.optimize_garch_model()

        if GP_AVAILABLE:
            self.optimize_gaussian_process()

        self.optimize_arfima_model()

        # Compare results
        self.compare_optimized_models()

        # Plot optimization histories
        self.plot_optimization_histories()

        print("\n=== OPTIMIZATION COMPLETE ===")
        print("Best models stored in self.best_models")
        print("Optimization studies stored in self.studies")


def main():
    """Main execution function"""
    print("=== CO2 TIME SERIES ANALYSIS WITH OPTUNA OPTIMIZATION ===\n")

    try:
        # Load actual CO2 data
        df = pd.read_parquet("data/processed_data/S3-coords.parquet")
        co2_data = df["CO2"].dropna()
        print(f"Loaded CO2 data: {len(co2_data)} observations")
        print(f"Data range: {co2_data.min():.2f} to {co2_data.max():.2f}")

    except Exception as e:
        print(f"Could not load data: {e}")
        print("Using synthetic CO2-like data for demonstration...")

        # Create realistic synthetic CO2 data
        np.random.seed(42)
        n = 6115
        trend = np.linspace(400, 450, n)
        seasonal = 5 * np.sin(2 * np.pi * np.arange(n) / 365.25)

        # Long memory component
        phi = 0.99
        noise = np.random.normal(0, 1, n)
        long_memory = np.zeros(n)
        for i in range(1, n):
            long_memory[i] = phi * long_memory[i - 1] + noise[i]

        co2_data = pd.Series(trend + seasonal + 2 * long_memory)
        print(f"Generated synthetic CO2 data: {len(co2_data)} observations")

    # Initialize the optimizer
    optimizer = OptunaCO2TimeSeriesModeling(
        co2_data,
        n_trials=50,  # Reduce for faster demo, increase for better results
        test_size=0.2,
    )

    # Run full optimization suite
    optimizer.run_full_optimization()

    print("\n=== USAGE TIPS ===")
    print("1. Increase n_trials for better optimization (e.g., 200-500)")
    print("2. Access best models: optimizer.best_models['MODEL_NAME']")
    print("3. Access optimization studies: optimizer.studies['MODEL_NAME']")
    print("4. Use study.best_params to get optimal hyperparameters")
    print("5. Use optuna.visualization for additional plots")


if __name__ == "__main__":
    main()
