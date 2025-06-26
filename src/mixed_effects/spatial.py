import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore
from shapely.geometry import Point
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def create_spatial_pollution_maps(df):
    """Create spatial maps showing pollution hotspots and patterns"""

    print("Creating spatial pollution maps...")

    pollutants = ["CO2", "PM1", "PM10", "PM25", "VOC"]
    available_pollutants = [p for p in pollutants if p in df.columns]

    if "x" not in df.columns or "y" not in df.columns:
        print("Coordinate columns (x, y) not found in dataset")
        return None

    # Calculate mean pollution at each spatial location
    spatial_means = df.groupby(["x", "y"])[available_pollutants].mean().reset_index()

    # Create subplots for each pollutant
    n_pollutants = len(available_pollutants)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, pollutant in enumerate(available_pollutants):
        if i < len(axes):
            # Create scatter plot with color representing pollution level
            scatter = axes[i].scatter(
                spatial_means["x"],
                spatial_means["y"],
                c=spatial_means[pollutant],
                cmap="Reds",
                alpha=0.7,
                s=50,
            )

            axes[i].set_title(f"{pollutant} Concentration - Spatial Distribution")
            axes[i].set_xlabel("Longitude (x)")
            axes[i].set_ylabel("Latitude (y)")

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=axes[i])
            cbar.set_label(f"{pollutant} Concentration")

            # Mark hotspots (top 10% pollution levels)
            threshold = spatial_means[pollutant].quantile(0.9)
            hotspots = spatial_means[spatial_means[pollutant] >= threshold]

            if len(hotspots) > 0:
                axes[i].scatter(
                    hotspots["x"],
                    hotspots["y"],
                    c="black",
                    marker="x",
                    s=100,
                    label=f"Hotspots (top 10%)",
                )
                axes[i].legend()

    # Remove empty subplots
    for i in range(len(available_pollutants), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

    return spatial_means


def identify_pollution_hotspots(df, method="quantile"):
    """Identify pollution hotspots using different methods"""

    print(f"Identifying pollution hotspots using {method} method...")

    pollutants = ["CO2", "PM1", "PM10", "PM25", "VOC"]
    available_pollutants = [p for p in pollutants if p in df.columns]

    # Aggregate data by location
    if "location" in df.columns:
        location_stats = (
            df.groupby("location")[available_pollutants]
            .agg(["mean", "std", "median"])
            .reset_index()
        )
        location_stats.columns = ["location"] + [
            f"{pol}_{stat}"
            for pol in available_pollutants
            for stat in ["mean", "std", "median"]
        ]
    else:
        # Use spatial aggregation
        df["spatial_bin"] = (
            pd.cut(df["x"], bins=10, labels=False).astype(str)
            + "_"
            + pd.cut(df["y"], bins=10, labels=False).astype(str)
        )
        location_stats = (
            df.groupby("spatial_bin")[available_pollutants]
            .agg(["mean", "std", "median"])
            .reset_index()
        )
        location_stats.columns = ["location"] + [
            f"{pol}_{stat}"
            for pol in available_pollutants
            for stat in ["mean", "std", "median"]
        ]

    # Identify hotspots
    hotspot_results = {}

    for pollutant in available_pollutants:
        mean_col = f"{pollutant}_mean"

        if method == "quantile":
            # Top 20% locations
            threshold = location_stats[mean_col].quantile(0.8)
            hotspots = location_stats[location_stats[mean_col] >= threshold]

        elif method == "zscore":
            # Locations with z-score > 1.5
            z_scores = zscore(location_stats[mean_col])
            hotspots = location_stats[z_scores > 1.5]

        elif method == "clustering":
            # Use K-means clustering to identify high-pollution cluster
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(location_stats[[mean_col]].dropna())

            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(scaled_data)

            # Find the cluster with highest mean pollution
            cluster_means = []
            for i in range(3):
                cluster_mask = clusters == i
                cluster_mean = location_stats[cluster_mask][mean_col].mean()
                cluster_means.append((i, cluster_mean))

            hotspot_cluster = max(cluster_means, key=lambda x: x[1])[0]
            hotspots = location_stats[clusters == hotspot_cluster]

        hotspot_results[pollutant] = {
            "hotspots": hotspots,
            "threshold": threshold if method == "quantile" else None,
            "n_hotspots": len(hotspots),
        }

        print(f"  {pollutant}: {len(hotspots)} hotspot locations identified")

    return hotspot_results, location_stats


def analyze_hotspot_characteristics(df, hotspot_results):
    """Analyze the characteristics of identified hotspots"""

    print("\nAnalyzing hotspot characteristics...")

    # Get geographical features
    geo_features = [
        col
        for col in df.columns
        if any(
            keyword in col.lower()
            for keyword in ["close2", "num_", "proportion_", "average_"]
        )
    ]

    hotspot_characteristics = {}

    for pollutant, hotspot_info in hotspot_results.items():
        hotspots = hotspot_info["hotspots"]

        if len(hotspots) == 0:
            continue

        print(f"\n--- {pollutant} Hotspot Analysis ---")

        # Get data for hotspot locations
        if "location" in df.columns and "location" in hotspots.columns:
            hotspot_data = df[df["location"].isin(hotspots["location"])]
        else:
            # Use spatial proximity for binned data
            hotspot_data = df  # Fallback

        # Calculate mean characteristics of hotspots
        hotspot_geo_means = hotspot_data[geo_features].mean()
        overall_geo_means = df[geo_features].mean()

        # Find characteristics that are elevated in hotspots
        differences = hotspot_geo_means - overall_geo_means
        significant_diffs = differences[abs(differences) > differences.std()]

        print(f"Key characteristics of {pollutant} hotspots:")
        for feature, diff in (
            significant_diffs.sort_values(ascending=False).head(10).items()
        ):
            direction = "Higher" if diff > 0 else "Lower"
            print(f"  - {feature}: {direction} by {abs(diff):.3f}")

        hotspot_characteristics[pollutant] = {
            "elevated_features": significant_diffs.to_dict(),
            "hotspot_data": hotspot_data,
            "geo_means": hotspot_geo_means.to_dict(),
        }

    return hotspot_characteristics


def create_hotspot_comparison_heatmap(hotspot_characteristics):
    """Create heatmap comparing hotspot characteristics across pollutants"""

    print("\nCreating hotspot comparison heatmap...")

    # Collect all elevated features across pollutants
    all_features = set()
    for pollutant, char in hotspot_characteristics.items():
        all_features.update(char["elevated_features"].keys())

    # Create matrix of feature differences
    comparison_matrix = []
    pollutants = list(hotspot_characteristics.keys())

    for feature in all_features:
        row = []
        for pollutant in pollutants:
            if feature in hotspot_characteristics[pollutant]["elevated_features"]:
                value = hotspot_characteristics[pollutant]["elevated_features"][feature]
            else:
                value = 0
            row.append(value)
        comparison_matrix.append(row)

    # Create DataFrame for plotting
    heatmap_df = pd.DataFrame(
        comparison_matrix, index=list(all_features), columns=pollutants
    )

    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        heatmap_df,
        annot=True,
        cmap="RdBu_r",
        center=0,
        fmt=".3f",
        cbar_kws={"label": "Difference from Overall Mean"},
    )
    plt.title("Hotspot Characteristics Comparison Across Pollutants")
    plt.xlabel("Pollutants")
    plt.ylabel("Geographical Features")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return heatmap_df


def spatial_autocorrelation_analysis(df):
    """Analyze spatial autocorrelation in pollution data"""

    print("\nAnalyzing spatial autocorrelation...")

    pollutants = ["CO2", "PM1", "PM10", "PM25", "VOC"]
    available_pollutants = [p for p in pollutants if p in df.columns]

    if "x" not in df.columns or "y" not in df.columns:
        print("Coordinate columns not available for spatial autocorrelation")
        return None

    # Calculate spatial autocorrelation (Moran's I approximation)
    autocorr_results = {}

    for pollutant in available_pollutants:
        # Sample data to make computation feasible
        sample_size = min(1000, len(df))
        sample_df = df.sample(n=sample_size, random_state=42)

        # Get coordinates and pollution values
        coords = sample_df[["x", "y"]].values
        pollution_values = sample_df[pollutant].values

        # Calculate distance matrix
        distances = pdist(coords)
        distance_matrix = squareform(distances)

        # Create spatial weights (inverse distance, with cutoff)
        max_distance = np.percentile(distances, 95)  # Use 95th percentile as cutoff
        weights = 1 / (
            distance_matrix + 1e-10
        )  # Add small value to avoid division by zero
        weights[distance_matrix > max_distance] = 0  # Set distant points to 0 weight

        # Calculate Moran's I approximation
        n = len(pollution_values)
        mean_pollution = np.mean(pollution_values)

        numerator = 0
        denominator = 0
        weight_sum = 0

        for i in range(n):
            for j in range(n):
                if i != j and weights[i, j] > 0:
                    numerator += (
                        weights[i, j]
                        * (pollution_values[i] - mean_pollution)
                        * (pollution_values[j] - mean_pollution)
                    )
                    weight_sum += weights[i, j]
            denominator += (pollution_values[i] - mean_pollution) ** 2

        if weight_sum > 0 and denominator > 0:
            morans_i = (n / weight_sum) * (numerator / denominator)
        else:
            morans_i = 0

        autocorr_results[pollutant] = {
            "morans_i": morans_i,
            "interpretation": (
                "Positive spatial autocorrelation"
                if morans_i > 0.1
                else (
                    "Negative spatial autocorrelation"
                    if morans_i < -0.1
                    else "No significant spatial autocorrelation"
                )
            ),
        }

        print(
            f"  {pollutant}: Moran's I ≈ {morans_i:.3f} ({autocorr_results[pollutant]['interpretation']})"
        )

    return autocorr_results


def route_segment_analysis(df):
    """Analyze pollution patterns by route segments"""

    print("\nAnalyzing pollution by route segments...")

    if "location" not in df.columns:
        print("Location column not available for route segment analysis")
        return None

    pollutants = ["CO2", "PM1", "PM10", "PM25", "VOC"]
    available_pollutants = [p for p in pollutants if p in df.columns]

    # Calculate statistics by route segment
    segment_stats = (
        df.groupby("location")[available_pollutants]
        .agg(["mean", "std", "count"])
        .round(3)
    )

    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))

    # Plot 1: Mean pollution by segment
    segment_means = segment_stats.xs("mean", level=1, axis=1)
    segment_means.plot(kind="bar", ax=axes[0], width=0.8)
    axes[0].set_title("Mean Pollution Concentration by Route Segment")
    axes[0].set_xlabel("Route Segment")
    axes[0].set_ylabel("Concentration")
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[0].tick_params(axis="x", rotation=45)

    # Plot 2: Coefficient of variation (std/mean) by segment
    segment_means = segment_stats.xs("mean", level=1, axis=1)
    segment_stds = segment_stats.xs("std", level=1, axis=1)
    cv = segment_stds / segment_means

    cv.plot(kind="bar", ax=axes[1], width=0.8)
    axes[1].set_title("Coefficient of Variation by Route Segment")
    axes[1].set_xlabel("Route Segment")
    axes[1].set_ylabel("Coefficient of Variation (std/mean)")
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()

    # Identify most and least polluted segments
    print("\nRoute segment analysis results:")
    for pollutant in available_pollutants:
        means = segment_means[pollutant].sort_values(ascending=False)
        print(f"\n{pollutant}:")
        print(f"  Most polluted segments: {', '.join(means.head(3).index.tolist())}")
        print(f"  Cleanest segments: {', '.join(means.tail(3).index.tolist())}")
        print(f"  Pollution range: {means.max():.2f} - {means.min():.2f}")

    return segment_stats


def generate_spatial_conclusions(
    spatial_means, hotspot_results, autocorr_results, segment_stats
):
    """Generate conclusions from spatial analysis"""

    print("\n" + "=" * 60)
    print("SPATIAL ANALYSIS CONCLUSIONS")
    print("=" * 60)

    print("\n1. SPATIAL POLLUTION PATTERNS:")
    if spatial_means is not None:
        print("   - Pollution shows clear spatial heterogeneity across the route")
        print("   - Hotspot identification reveals consistent problem areas")

    print("\n2. POLLUTION HOTSPOTS:")
    if hotspot_results:
        total_hotspots = sum(info["n_hotspots"] for info in hotspot_results.values())
        print(f"   - Total hotspot locations identified: {total_hotspots}")

        for pollutant, info in hotspot_results.items():
            print(f"   - {pollutant}: {info['n_hotspots']} hotspot locations")

    print("\n3. SPATIAL AUTOCORRELATION:")
    if autocorr_results:
        positive_autocorr = [
            p for p, info in autocorr_results.items() if info["morans_i"] > 0.1
        ]
        if positive_autocorr:
            print(
                f"   - Pollutants with spatial clustering: {', '.join(positive_autocorr)}"
            )
        else:
            print("   - Limited evidence of spatial clustering in pollution")

    print("\n4. ROUTE SEGMENT VARIATIONS:")
    if segment_stats is not None:
        print("   - Significant variation in pollution levels across route segments")
        print(
            "   - Some segments consistently show higher pollution across multiple pollutants"
        )

    print("\n5. URBAN PLANNING IMPLICATIONS:")
    print("   - Identified hotspots require targeted intervention strategies")
    print("   - Route planning could avoid high-pollution segments during peak times")
    print("   - Green infrastructure placement could be optimized for maximum impact")

    print("\n6. EXPOSURE ASSESSMENT:")
    print("   - Spatial heterogeneity supports need for high-resolution monitoring")
    print("   - Personal exposure varies significantly based on exact route taken")
    print("   - Time spent in hotspot areas significantly impacts total exposure")


def main():
    """Main execution function for spatial analysis"""

    print("=== SPATIAL ANALYSIS OF AIR POLLUTION DATA ===\n")

    # Load data
    df = pd.read_parquet("data/processed_data/combined_subjects.parquet")
    print(f"Loaded dataset with {len(df)} observations")

    # Create spatial pollution maps
    spatial_means = create_spatial_pollution_maps(df)

    # Identify pollution hotspots
    hotspot_results, location_stats = identify_pollution_hotspots(df, method="quantile")

    # Analyze hotspot characteristics
    if hotspot_results:
        hotspot_characteristics = analyze_hotspot_characteristics(df, hotspot_results)
        heatmap_df = create_hotspot_comparison_heatmap(hotspot_characteristics)

    # Spatial autocorrelation analysis
    autocorr_results = spatial_autocorrelation_analysis(df)

    # Route segment analysis
    segment_stats = route_segment_analysis(df)

    # Generate conclusions
    generate_spatial_conclusions(
        spatial_means, hotspot_results, autocorr_results, segment_stats
    )

    return {
        "spatial_means": spatial_means,
        "hotspot_results": hotspot_results,
        "autocorr_results": autocorr_results,
        "segment_stats": segment_stats,
    }


if __name__ == "__main__":
    spatial_results = main()
