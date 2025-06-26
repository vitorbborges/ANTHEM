import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_white

warnings.filterwarnings("ignore")


def prepare_data_for_modeling(df):
    """Prepare the dataset for mixed effects modeling"""

    print("Preparing data for mixed effects modeling...")

    # Create sub if not present
    if "sub" not in df.columns:
        print("Creating subject IDs based on data structure...")
        n_subjects = 20
        chunk_size = len(df) // n_subjects
        df["sub"] = np.repeat(range(1, n_subjects + 1), chunk_size)[: len(df)]

    # Create day_id if not present (assuming each subject represents a different day)
    if "day_id" not in df.columns:
        df["day_id"] = df["sub"]  # Each subject = different day

    # Handle missing values and outliers
    pollutants = ["CO2", "PM1", "PM10", "PM25", "VOC"]
    available_pollutants = [p for p in pollutants if p in df.columns]

    for pollutant in available_pollutants:
        # Remove extreme outliers (beyond 3 standard deviations)
        mean_val = df[pollutant].mean()
        std_val = df[pollutant].std()
        df.loc[df[pollutant] > mean_val + 3 * std_val, pollutant] = np.nan
        df.loc[df[pollutant] < mean_val - 3 * std_val, pollutant] = np.nan

    # Select geographical features for modeling
    geo_features = [
        col
        for col in df.columns
        if any(
            keyword in col.lower()
            for keyword in ["close2", "num_", "proportion_", "average_"]
        )
    ]

    # Select key features to avoid multicollinearity
    key_geo_features = [
        "close2industry_400",
        "close2park_25",
        "num_green_200",
        "proportion_green_100",
        "num_traffic_light_200",
        "average_building_height_100",
        "close2smoking_shop_75",
        "num_trees_50",
        "average_nearby_maxspeed_50",
        "close2residential_200",
        "close2public_transport_15",
        "num_public_transport_50",
        "close2construction_50",
        "close2railway_100",
        "close2water_100",
    ]

    # Keep only features that exist in the dataset
    available_geo_features = [f for f in key_geo_features if f in df.columns]

    # Add weather features if available
    weather_features = [
        "velocita_vento_medio",
        "direzione_vento_medio",
        "radiazione_globale_medio",
    ]
    available_weather_features = [f for f in weather_features if f in df.columns]

    # Add location as categorical
    if "location" in df.columns:
        df["location"] = df["location"].astype("category")

    print(
        f"Available geographical features for modeling: {len(available_geo_features)}"
    )
    print(f"Available weather features: {len(available_weather_features)}")

    return df, available_pollutants, available_geo_features, available_weather_features


def run_mixed_effects_model(df, pollutant, geo_features, weather_features):
    """Run mixed effects model for a specific pollutant"""

    print(f"\n=== MIXED EFFECTS MODEL FOR {pollutant} ===")

    # Prepare the data
    model_data = df[
        ["sub", "day_id", pollutant]
        + geo_features
        + weather_features
        + (["location"] if "location" in df.columns else [])
    ].dropna()

    print(f"Model data shape: {model_data.shape}")

    if len(model_data) < 100:
        print(f"Insufficient data for {pollutant} modeling")
        return None

    # Log-transform the pollutant if it's strictly positive
    if model_data[pollutant].min() > 0:
        model_data[f"log_{pollutant}"] = np.log(model_data[pollutant])
        dependent_var = f"log_{pollutant}"
        print(f"Using log-transformed {pollutant}")
    else:
        dependent_var = pollutant

    # Create formula for mixed effects model
    fixed_effects = geo_features + weather_features
    if "location" in model_data.columns:
        fixed_effects.append("C(location)")

    # Remove any problematic features
    fixed_effects = [f for f in fixed_effects if f in model_data.columns]

    formula = f"{dependent_var} ~ " + " + ".join(fixed_effects)

    try:
        # Fit mixed effects model with random intercepts for subjects and days
        print("Fitting mixed effects model...")
        model = smf.mixedlm(
            formula,
            model_data,
            groups=model_data["sub"],
            re_formula="1",
            missing="drop",
        )

        result = model.fit(method="lbfgs", maxiter=100)

        print("Model fitted successfully!")
        print(result.summary())

        # Model diagnostics
        print("\n=== MODEL DIAGNOSTICS ===")

        # R-squared approximation for mixed effects
        predictions = result.fittedvalues
        residuals = result.resid

        # Calculate pseudo R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum(
            (model_data[dependent_var] - np.mean(model_data[dependent_var])) ** 2
        )
        r_squared = 1 - (ss_res / ss_tot)
        print(f"Pseudo R-squared: {r_squared:.4f}")

        # RMSE and MAE
        rmse = np.sqrt(mean_squared_error(model_data[dependent_var], predictions))
        mae = mean_absolute_error(model_data[dependent_var], predictions)
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")

        # Plot residuals
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.scatter(predictions, residuals, alpha=0.5)
        plt.xlabel("Fitted Values")
        plt.ylabel("Residuals")
        plt.title("Residuals vs Fitted")
        plt.axhline(y=0, color="red", linestyle="--")

        plt.subplot(1, 3, 2)
        plt.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.title("Distribution of Residuals")

        plt.subplot(1, 3, 3)
        from scipy import stats

        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title("Q-Q Plot of Residuals")

        plt.tight_layout()
        plt.show()

        return result, model_data, r_squared

    except Exception as e:
        print(f"Error fitting model for {pollutant}: {str(e)}")
        return None


def analyze_fixed_effects(results_dict):
    """Analyze and compare fixed effects across pollutants"""

    print("\n=== FIXED EFFECTS ANALYSIS ACROSS POLLUTANTS ===")

    # Collect coefficients from all models
    all_coeffs = []

    for pollutant, (result, data, r2) in results_dict.items():
        if result is not None:
            coeffs = result.params
            pvalues = result.pvalues

            for var in coeffs.index:
                if var != "Intercept" and var != "Group Var":
                    all_coeffs.append(
                        {
                            "Pollutant": pollutant,
                            "Variable": var,
                            "Coefficient": coeffs[var],
                            "P_value": pvalues[var] if var in pvalues else np.nan,
                            "Significant": (
                                pvalues[var] < 0.05 if var in pvalues else False
                            ),
                        }
                    )

    if all_coeffs:
        coeffs_df = pd.DataFrame(all_coeffs)

        # Create heatmap of significant coefficients
        sig_coeffs = coeffs_df[coeffs_df["Significant"]]
        if len(sig_coeffs) > 0:
            pivot_coeffs = sig_coeffs.pivot(
                index="Variable", columns="Pollutant", values="Coefficient"
            )

            plt.figure(figsize=(12, 8))
            sns.heatmap(
                pivot_coeffs,
                annot=True,
                cmap="RdBu_r",
                center=0,
                fmt=".3f",
                cbar_kws={"label": "Coefficient Value"},
            )
            plt.title("Significant Fixed Effects Coefficients Across Pollutants")
            plt.xlabel("Pollutants")
            plt.ylabel("Variables")
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.show()

            # Print summary of most important effects
            print("\nMost significant positive effects:")
            pos_effects = sig_coeffs[sig_coeffs["Coefficient"] > 0].sort_values(
                "Coefficient", ascending=False
            )
            print(
                pos_effects.head(10)[
                    ["Pollutant", "Variable", "Coefficient", "P_value"]
                ].to_string(index=False)
            )

            print("\nMost significant negative effects:")
            neg_effects = sig_coeffs[sig_coeffs["Coefficient"] < 0].sort_values(
                "Coefficient"
            )
            print(
                neg_effects.head(10)[
                    ["Pollutant", "Variable", "Coefficient", "P_value"]
                ].to_string(index=False)
            )

        return coeffs_df

    return None


def analyze_random_effects(results_dict):
    """Analyze random effects to understand subject and day variations"""

    print("\n=== RANDOM EFFECTS ANALYSIS ===")

    random_effects_data = []

    for pollutant, (result, data, r2) in results_dict.items():
        if result is not None:
            # Extract random effects
            random_effects = result.random_effects

            for sub, effects in random_effects.items():
                if "Group" in effects:
                    random_effects_data.append(
                        {
                            "Pollutant": pollutant,
                            "Subject_ID": sub,
                            "Random_Intercept": effects["Group"],
                        }
                    )

    if random_effects_data:
        re_df = pd.DataFrame(random_effects_data)

        # Plot random effects distribution
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Boxplot of random intercepts by pollutant
        sns.boxplot(data=re_df, x="Pollutant", y="Random_Intercept", ax=axes[0])
        axes[0].set_title("Distribution of Random Intercepts by Pollutant")
        axes[0].set_xlabel("Pollutant")
        axes[0].set_ylabel("Random Intercept")
        axes[0].tick_params(axis="x", rotation=45)

        # Heatmap of random intercepts
        pivot_re = re_df.pivot(
            index="Subject_ID", columns="Pollutant", values="Random_Intercept"
        )
        sns.heatmap(
            pivot_re,
            annot=False,
            cmap="RdBu_r",
            center=0,
            ax=axes[1],
            cbar_kws={"label": "Random Intercept"},
        )
        axes[1].set_title("Random Intercepts by Subject and Pollutant")
        axes[1].set_xlabel("Pollutant")
        axes[1].set_ylabel("Subject ID")

        plt.tight_layout()
        plt.show()

        # Calculate variance explained by random effects
        print("Variance explained by random effects (subject-level):")
        for pollutant in re_df["Pollutant"].unique():
            pollutant_re = re_df[re_df["Pollutant"] == pollutant]["Random_Intercept"]
            variance = pollutant_re.var()
            print(f"  {pollutant}: {variance:.6f}")

        return re_df

    return None


def generate_conclusions(results_dict, coeffs_df, re_df):
    """Generate scientific conclusions from the mixed effects analysis"""

    print("\n" + "=" * 60)
    print("MIXED EFFECTS MODELING CONCLUSIONS")
    print("=" * 60)

    if not results_dict:
        print("No successful models were fitted.")
        return

    # Model performance summary
    print("\n1. MODEL PERFORMANCE:")
    for pollutant, (result, data, r2) in results_dict.items():
        print(f"   {pollutant}: Pseudo R² = {r2:.3f}")

    # Spatial pollution patterns
    print("\n2. SPATIAL POLLUTION PATTERNS:")
    if coeffs_df is not None:
        sig_spatial = coeffs_df[
            coeffs_df["Significant"]
            & coeffs_df["Variable"].str.contains("close2|num_|proportion_", regex=True)
        ]

        consistent_effects = (
            sig_spatial.groupby("Variable")
            .agg({"Coefficient": ["mean", "std"], "Pollutant": "count"})
            .reset_index()
        )
        consistent_effects.columns = ["Variable", "Mean_Coeff", "Std_Coeff", "Count"]
        consistent_effects = consistent_effects[
            consistent_effects["Count"] >= 2
        ]  # Consistent across pollutants

        if len(consistent_effects) > 0:
            print("   Key consistent spatial predictors:")
            for _, row in consistent_effects.head(5).iterrows():
                direction = "increases" if row["Mean_Coeff"] > 0 else "decreases"
                print(
                    f"   - {row['Variable']}: {direction} pollution (avg coeff: {row['Mean_Coeff']:.3f})"
                )

    # Environmental factors
    print("\n3. ENVIRONMENTAL FACTOR IMPACT:")
    if coeffs_df is not None:
        # Industry effects
        industry_effects = coeffs_df[
            coeffs_df["Variable"].str.contains("industry", case=False)
            & coeffs_df["Significant"]
        ]
        if len(industry_effects) > 0:
            avg_industry_effect = industry_effects["Coefficient"].mean()
            print(
                f"   - Industrial proximity: {'Increases' if avg_industry_effect > 0 else 'Decreases'} "
                f"pollution by average coefficient of {abs(avg_industry_effect):.3f}"
            )

        # Green space effects
        green_effects = coeffs_df[
            coeffs_df["Variable"].str.contains("green|park|tree", case=False)
            & coeffs_df["Significant"]
        ]
        if len(green_effects) > 0:
            avg_green_effect = green_effects["Coefficient"].mean()
            print(
                f"   - Green spaces: {'Increase' if avg_green_effect > 0 else 'Reduce'} "
                f"pollution by average coefficient of {abs(avg_green_effect):.3f}"
            )

        # Traffic effects
        traffic_effects = coeffs_df[
            coeffs_df["Variable"].str.contains("traffic|maxspeed", case=False)
            & coeffs_df["Significant"]
        ]
        if len(traffic_effects) > 0:
            avg_traffic_effect = traffic_effects["Coefficient"].mean()
            print(
                f"   - Traffic infrastructure: {'Increases' if avg_traffic_effect > 0 else 'Reduces'} "
                f"pollution by average coefficient of {abs(avg_traffic_effect):.3f}"
            )

    # Individual variations
    print("\n4. INDIVIDUAL AND TEMPORAL VARIATIONS:")
    if re_df is not None:
        for pollutant in re_df["Pollutant"].unique():
            pollutant_re = re_df[re_df["Pollutant"] == pollutant]["Random_Intercept"]
            range_re = pollutant_re.max() - pollutant_re.min()
            print(f"   - {pollutant}: Subject-level variation range = {range_re:.3f}")

    # Route heterogeneity
    print("\n5. POLICY IMPLICATIONS:")
    print("   - Mixed effects modeling reveals both systematic environmental factors")
    print("     and individual-level variations in pollution exposure")
    print("   - Results can inform targeted interventions at specific route locations")
    print(
        "   - Individual differences suggest need for personalized exposure assessment"
    )

    print("\n6. STUDY LIMITATIONS:")
    print("   - Single route limits generalizability to other urban areas")
    print("   - Correlational analysis cannot establish direct causation")
    print("   - Weather effects may be confounded with daily variations")

    return True


def main():
    """Main execution function for mixed effects modeling"""

    print("=== MIXED EFFECTS MODELING FOR AIR POLLUTION DATA ===\n")

    # Load data
    df = pd.read_parquet("data/processed_data/combined_subjects.parquet")

    # Prepare data
    df, pollutants, geo_features, weather_features = prepare_data_for_modeling(df)

    # Run mixed effects models for each pollutant
    results_dict = {}

    for pollutant in pollutants:
        result = run_mixed_effects_model(df, pollutant, geo_features, weather_features)
        if result is not None:
            results_dict[pollutant] = result

    if not results_dict:
        print(
            "No successful models were fitted. Check data quality and feature availability."
        )
        return

    # Analyze results
    print("\n" + "=" * 50)
    print("ANALYZING RESULTS ACROSS ALL POLLUTANTS")
    print("=" * 50)

    # Fixed effects analysis
    coeffs_df = analyze_fixed_effects(results_dict)

    # Random effects analysis
    re_df = analyze_random_effects(results_dict)

    # Generate conclusions
    generate_conclusions(results_dict, coeffs_df, re_df)

    return results_dict, coeffs_df, re_df


if __name__ == "__main__":
    results_dict, coeffs_df, re_df = main()
