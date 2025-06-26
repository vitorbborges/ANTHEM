import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")


# Load the dataset
def load_and_explore_data():
    """Load the combined dataset and perform initial exploration"""

    print("Loading dataset...")
    df = pd.read_parquet("data/processed_data/combined_subjects.parquet")

    print(f"Dataset shape: {df.shape}")
    print(
        f"Number of unique subjects: {df['subject_id'].nunique() if 'subject_id' in df.columns else 'Subject ID column not found'}"
    )

    # Identify pollutant columns
    pollutant_cols = ["CO2", "PM1", "PM10", "PM25", "VOC"]
    available_pollutants = [col for col in pollutant_cols if col in df.columns]
    print(f"Available pollutants: {available_pollutants}")

    # Identify geographical features (OSM-derived)
    osm_features = [
        col
        for col in df.columns
        if any(
            keyword in col.lower()
            for keyword in [
                "close2",
                "num_",
                "proportion_",
                "average_",
                "len_nearby",
                "sum_",
            ]
        )
    ]
    print(f"Number of OSM geographical features: {len(osm_features)}")

    # Weather features
    weather_features = [
        col
        for col in df.columns
        if "vento" in col or "radiazione" in col or "precipitazione" in col
    ]
    print(f"Weather features: {weather_features}")

    return df, available_pollutants, osm_features, weather_features


def explore_spatial_temporal_structure(df):
    """Explore the spatial and temporal structure of the data"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Spatial distribution
    if "x" in df.columns and "y" in df.columns:
        axes[0, 0].scatter(df["x"], df["y"], alpha=0.3, s=1)
        axes[0, 0].set_title("Spatial Distribution of Data Points")
        axes[0, 0].set_xlabel("Longitude (x)")
        axes[0, 0].set_ylabel("Latitude (y)")

    # Plot 2: Location distribution
    if "location" in df.columns:
        location_counts = df["location"].value_counts()
        axes[0, 1].bar(location_counts.index, location_counts.values)
        axes[0, 1].set_title("Distribution Across Route Locations")
        axes[0, 1].set_xlabel("Location")
        axes[0, 1].set_ylabel("Number of Observations")
        axes[0, 1].tick_params(axis="x", rotation=45)

    # Plot 3: Regime distribution
    if "regime" in df.columns:
        regime_counts = df["regime"].value_counts()
        axes[1, 0].pie(
            regime_counts.values, labels=regime_counts.index, autopct="%1.1f%%"
        )
        axes[1, 0].set_title("Static vs Dynamic Regime Distribution")

    # Plot 4: Subject distribution (if available)
    if "subject_id" in df.columns:
        subject_counts = df["subject_id"].value_counts()
        axes[1, 1].hist(subject_counts.values, bins=20, edgecolor="black")
        axes[1, 1].set_title("Distribution of Observations per Subject")
        axes[1, 1].set_xlabel("Number of Observations")
        axes[1, 1].set_ylabel("Number of Subjects")

    plt.tight_layout()
    plt.show()

    return df


def analyze_pollutant_distributions(df, pollutants):
    """Analyze the distribution of pollutant measurements"""

    n_pollutants = len(pollutants)
    fig, axes = plt.subplots(2, n_pollutants, figsize=(4 * n_pollutants, 10))

    if n_pollutants == 1:
        axes = axes.reshape(2, 1)

    for i, pollutant in enumerate(pollutants):
        if pollutant in df.columns:
            # Distribution plot
            axes[0, i].hist(
                df[pollutant].dropna(), bins=50, alpha=0.7, edgecolor="black"
            )
            axes[0, i].set_title(f"{pollutant} Distribution")
            axes[0, i].set_xlabel(f"{pollutant} Concentration")
            axes[0, i].set_ylabel("Frequency")

            # Log-scale distribution (if positive values)
            if df[pollutant].min() > 0:
                axes[1, i].hist(
                    np.log(df[pollutant].dropna()),
                    bins=50,
                    alpha=0.7,
                    edgecolor="black",
                )
                axes[1, i].set_title(f"Log({pollutant}) Distribution")
                axes[1, i].set_xlabel(f"Log({pollutant}) Concentration")
                axes[1, i].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

    # Summary statistics
    print("\nPollutant Summary Statistics:")
    print(df[pollutants].describe())

    return df


def check_autocorrelation_structure(df, pollutants, max_lags=50):
    """Check autocorrelation structure for each pollutant by subject"""

    from statsmodels.tsa.stattools import acf

    if "sub" not in df.columns:
        print(
            "Subject ID column not found. Creating dummy subject IDs based on data chunks."
        )
        # Create approximate subject IDs based on data order
        n_subjects = 20  # Known from description
        chunk_size = len(df) // n_subjects
        df["subject_id"] = np.repeat(range(n_subjects), chunk_size)[: len(df)]

    fig, axes = plt.subplots(len(pollutants), 1, figsize=(12, 4 * len(pollutants)))
    if len(pollutants) == 1:
        axes = [axes]

    for i, pollutant in enumerate(pollutants):
        if pollutant in df.columns:
            all_acf = []

            # Calculate ACF for each subject
            for subject in df["subject_id"].unique():
                subject_data = df[df["subject_id"] == subject][pollutant].dropna()
                if len(subject_data) > max_lags:
                    try:
                        acf_vals = acf(subject_data, nlags=max_lags, fft=True)
                        all_acf.append(acf_vals)
                    except:
                        continue

            if all_acf:
                # Plot mean ACF across subjects
                mean_acf = np.mean(all_acf, axis=0)
                std_acf = np.std(all_acf, axis=0)
                lags = range(len(mean_acf))

                axes[i].plot(lags, mean_acf, "b-", linewidth=2, label="Mean ACF")
                axes[i].fill_between(
                    lags,
                    mean_acf - std_acf,
                    mean_acf + std_acf,
                    alpha=0.3,
                    color="blue",
                    label="±1 SD",
                )
                axes[i].axhline(y=0, color="black", linestyle="--", alpha=0.5)
                axes[i].axhline(
                    y=0.05, color="red", linestyle="--", alpha=0.5, label="5% threshold"
                )
                axes[i].axhline(y=-0.05, color="red", linestyle="--", alpha=0.5)
                axes[i].set_title(f"Autocorrelation Function - {pollutant}")
                axes[i].set_xlabel("Lag")
                axes[i].set_ylabel("Autocorrelation")
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return df


def analyze_geographical_feature_importance(df, pollutants, osm_features):
    """Analyze correlation between geographical features and pollutants"""

    # Select a subset of key geographical features for initial analysis
    key_features = [
        f
        for f in osm_features
        if any(
            keyword in f
            for keyword in [
                "close2industry",
                "close2park",
                "num_green",
                "proportion_green",
                "num_traffic_light",
                "average_building_height",
                "close2smoking",
                "num_trees",
                "average_nearby_maxspeed",
            ]
        )
    ][
        :15
    ]  # Limit to 15 key features

    if not key_features:
        key_features = osm_features[:15]  # Fallback to first 15 features

    print(f"Analyzing correlations with {len(key_features)} key geographical features:")
    for feature in key_features:
        print(f"  - {feature}")

    # Calculate correlations
    correlation_data = []
    for pollutant in pollutants:
        if pollutant in df.columns:
            for feature in key_features:
                if feature in df.columns:
                    # Calculate correlation, handling missing values
                    clean_data = df[[pollutant, feature]].dropna()
                    if (
                        len(clean_data) > 10
                    ):  # Minimum data points for meaningful correlation
                        corr = clean_data[pollutant].corr(clean_data[feature])
                        correlation_data.append(
                            {
                                "Pollutant": pollutant,
                                "Feature": feature,
                                "Correlation": corr,
                            }
                        )

    if correlation_data:
        corr_df = pd.DataFrame(correlation_data)

        # Create correlation heatmap
        pivot_corr = corr_df.pivot(
            index="Feature", columns="Pollutant", values="Correlation"
        )

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            pivot_corr,
            annot=True,
            cmap="RdBu_r",
            center=0,
            fmt=".3f",
            cbar_kws={"label": "Correlation Coefficient"},
        )
        plt.title("Correlation Between Geographical Features and Pollutants")
        plt.xlabel("Pollutants")
        plt.ylabel("Geographical Features")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

        # Print strongest correlations
        print("\nStrongest positive correlations:")
        strong_pos = corr_df[corr_df["Correlation"] > 0.1].sort_values(
            "Correlation", ascending=False
        )
        print(strong_pos.head(10).to_string(index=False))

        print("\nStrongest negative correlations:")
        strong_neg = corr_df[corr_df["Correlation"] < -0.1].sort_values("Correlation")
        print(strong_neg.head(10).to_string(index=False))

    return correlation_data


def main():
    """Main execution function"""

    print("=== MIXED EFFECTS MODELING - DATA EXPLORATION ===\n")

    # Load and explore data
    df, pollutants, osm_features, weather_features = load_and_explore_data()

    print(f"\nDataset loaded successfully!")
    print(f"Dimensions: {df.shape}")
    print(f"Pollutants to analyze: {pollutants}")
    print(f"Number of geographical features: {len(osm_features)}")

    # Explore spatial-temporal structure
    print("\n=== SPATIAL-TEMPORAL STRUCTURE ===")
    explore_spatial_temporal_structure(df)

    # Analyze pollutant distributions
    print("\n=== POLLUTANT DISTRIBUTIONS ===")
    analyze_pollutant_distributions(df, pollutants)

    # Check autocorrelation
    print("\n=== AUTOCORRELATION ANALYSIS ===")
    check_autocorrelation_structure(df, pollutants)

    # Analyze geographical correlations
    print("\n=== GEOGRAPHICAL FEATURE ANALYSIS ===")
    correlation_data = analyze_geographical_feature_importance(
        df, pollutants, osm_features
    )

    print("\n=== EXPLORATION COMPLETE ===")
    print("This analysis provides the foundation for mixed effects modeling.")
    print(
        "Next steps: Run the mixed effects modeling script to build predictive models."
    )

    return df, pollutants, osm_features, correlation_data


if __name__ == "__main__":
    df, pollutants, osm_features, correlation_data = main()
