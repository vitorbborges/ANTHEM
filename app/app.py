import argparse
import math
import sys
from pathlib import Path

import folium
import geopandas as gpd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.process_subject_pipeline import ProcessSubjectPipeline

WEST, SOUTH, EAST, NORTH = 9.2257, 45.47162, 9.23768, 45.48537
BBOX = (WEST, SOUTH, EAST, NORTH)


def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points using Haversine formula"""
    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))

    # Radius of earth in kilometers
    r = 6371
    return c * r


def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate the bearing between two points"""
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon_rad = math.radians(lon2 - lon1)

    y = math.sin(dlon_rad) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(
        lat2_rad
    ) * math.cos(dlon_rad)

    bearing_rad = math.atan2(y, x)
    bearing_deg = math.degrees(bearing_rad)
    return (bearing_deg + 360) % 360


# Streamlit app
st.title("Interactive Route Map with Point Selection")

# Initialize session state for selected points
if "selected_points" not in st.session_state:
    st.session_state.selected_points = []

# Get your GeoDataFrame
pipeline = ProcessSubjectPipeline(BBOX)
gdf = pipeline.extractor.loader.extract_kml(
    PROJECT_ROOT / "data" / "raw_data" / "route.kmz"
)

# Ensure the GeoDataFrame is in WGS84
gdf = gdf.to_crs(epsg=4326)

# Create the base map
center_lat = (NORTH + SOUTH) / 2
center_lon = (EAST + WEST) / 2

# Map style options
map_styles = {
    "OpenStreetMap": "OpenStreetMap",
    "Satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tiles/{z}/{y}/{x}",
    "Dark": "CartoDB dark_matter",
    "Light": "CartoDB positron",
    "Terrain": "https://stamen-tiles-{s}.a.ssl.fastly.net/terrain/{z}/{x}/{y}.png",
    "Watercolor": "https://stamen-tiles-{s}.a.ssl.fastly.net/watercolor/{z}/{x}/{y}.jpg",
}

# Style selector
selected_style = st.selectbox(
    "Choose Map Style:", options=list(map_styles.keys()), index=0
)

# Calculate bounds for auto-zoom
bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
center_lat = (bounds[1] + bounds[3]) / 2
center_lon = (bounds[0] + bounds[2]) / 2

# Create folium map with disabled zoom controls
m = folium.Map(
    location=[center_lat, center_lon],
    tiles=(
        map_styles[selected_style]
        if selected_style in ["OpenStreetMap", "Dark", "Light"]
        else None
    ),
    zoom_control=False,
    scrollWheelZoom=False,
    doubleClickZoom=False,
    dragging=True,
)

# Add custom tile layer for non-default styles
if selected_style not in ["OpenStreetMap", "Dark", "Light"]:
    folium.TileLayer(
        tiles=map_styles[selected_style],
        attr="Custom",
        name=selected_style,
        overlay=False,
        control=True,
    ).add_to(m)

# Fit map to bounds with padding
southwest = [bounds[1] - 0.001, bounds[0] - 0.001]
northeast = [bounds[3] + 0.001, bounds[2] + 0.001]
m.fit_bounds([southwest, northeast])

# Add the KML/KMZ data to the map with better styling
route_style = {
    "OpenStreetMap": {
        "color": "#2E86AB",
        "weight": 4,
        "opacity": 0.8,
        "fillOpacity": 0.3,
    },
    "Satellite": {"color": "#F24236", "weight": 4, "opacity": 0.9, "fillOpacity": 0.4},
    "Dark": {"color": "#F18F01", "weight": 4, "opacity": 0.9, "fillOpacity": 0.4},
    "Light": {"color": "#A23B72", "weight": 4, "opacity": 0.8, "fillOpacity": 0.3},
    "Terrain": {"color": "#2E7D32", "weight": 4, "opacity": 0.8, "fillOpacity": 0.3},
    "Watercolor": {"color": "#D32F2F", "weight": 5, "opacity": 0.9, "fillOpacity": 0.4},
}

folium.GeoJson(gdf, style_function=lambda feature: route_style[selected_style]).add_to(
    m
)

# Custom marker styles
marker_styles = {
    "OpenStreetMap": [
        {"color": "red", "icon": "map-pin", "prefix": "fa"},
        {"color": "green", "icon": "map-pin", "prefix": "fa"},
    ],
    "Satellite": [
        {"color": "orange", "icon": "star", "prefix": "fa"},
        {"color": "purple", "icon": "star", "prefix": "fa"},
    ],
    "Dark": [
        {"color": "orange", "icon": "circle", "prefix": "fa"},
        {"color": "lightgreen", "icon": "circle", "prefix": "fa"},
    ],
    "Light": [
        {"color": "red", "icon": "heart", "prefix": "fa"},
        {"color": "blue", "icon": "heart", "prefix": "fa"},
    ],
    "Terrain": [
        {"color": "darkred", "icon": "tree", "prefix": "fa"},
        {"color": "darkgreen", "icon": "tree", "prefix": "fa"},
    ],
    "Watercolor": [
        {"color": "red", "icon": "paint-brush", "prefix": "fa"},
        {"color": "blue", "icon": "paint-brush", "prefix": "fa"},
    ],
}

# Add existing selected points to the map with styled markers
for i, point in enumerate(st.session_state.selected_points):
    style = marker_styles[selected_style][i]

    # Create custom marker with better styling
    marker = folium.Marker(
        [point["lat"], point["lng"]],
        popup=folium.Popup(
            f"""
            <div style='font-family: Arial; font-size: 12px; font-weight: bold;'>
                <div style='color: {style['color']}; margin-bottom: 5px;'>{labels[i]}</div>
                <div>Lat: {point['lat']:.6f}</div>
                <div>Lng: {point['lng']:.6f}</div>
            </div>
            """,
            max_width=200,
        ),
        icon=folium.Icon(
            color=style["color"], icon=style["icon"], prefix=style["prefix"]
        ),
    )
    marker.add_to(m)

# Display the map and capture click events
st.write("🗺️ **Click on the map to select up to 2 points:**")

map_data = st_folium(
    m,
    width=800,
    height=600,
    key="map",
    feature_group_to_add=None,
    returned_objects=["last_object_clicked"],
)

# Handle map clicks
if map_data["last_object_clicked"]:
    clicked_point = map_data["last_object_clicked"]

    # Only process if it's a map click (not a marker click)
    if clicked_point and "lat" in clicked_point and "lng" in clicked_point:
        new_point = {"lat": clicked_point["lat"], "lng": clicked_point["lng"]}

        # Add point if we have less than 2 points
        if len(st.session_state.selected_points) < 2:
            st.session_state.selected_points.append(new_point)
            st.rerun()
        elif len(st.session_state.selected_points) == 2:
            # Replace points in order (first click replaces point 1, second click replaces point 2)
            if len(st.session_state.selected_points) >= 2:
                st.session_state.selected_points = [new_point]
                st.rerun()

# Display selected points and calculations
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Selected Points")
    if len(st.session_state.selected_points) == 0:
        st.info("No points selected yet")
    else:
        for i, point in enumerate(st.session_state.selected_points):
            st.markdown(
                f"""
            **{labels[i]}:**
            📍 Lat: `{point['lat']:.6f}`
            📍 Lng: `{point['lng']:.6f}`
            """
            )

    if st.button("🗑️ Clear All Points", type="secondary"):
        st.session_state.selected_points = []
        st.rerun()

with col2:
    st.subheader("📊 Calculations")

    if len(st.session_state.selected_points) == 2:
        point1 = st.session_state.selected_points[0]
        point2 = st.session_state.selected_points[1]

        # Calculate distance
        distance = calculate_distance(
            point1["lat"], point1["lng"], point2["lat"], point2["lng"]
        )

        # Calculate bearing
        bearing = calculate_bearing(
            point1["lat"], point1["lng"], point2["lat"], point2["lng"]
        )

        # Create metrics with better styling
        col2a, col2b = st.columns(2)
        with col2a:
            st.metric("🏃 Distance", f"{distance:.3f} km")
            st.metric("📐 Bearing", f"{bearing:.1f}°")

        with col2b:
            # Convert to other units
            distance_miles = distance * 0.621371
            st.metric("🏃 Distance", f"{distance_miles:.3f} mi")

            # Cardinal direction
            directions = [
                "N",
                "NNE",
                "NE",
                "ENE",
                "E",
                "ESE",
                "SE",
                "SSE",
                "S",
                "SSW",
                "SW",
                "WSW",
                "W",
                "WNW",
                "NW",
                "NNW",
            ]
            direction_idx = int((bearing + 11.25) / 22.5) % 16
            st.metric("🧭 Direction", directions[direction_idx])

        # Additional calculations in an expander
        with st.expander("📋 Detailed Calculations"):
            st.write("**Coordinate Differences:**")
            st.write(f"📏 Δ Latitude: {abs(point2['lat'] - point1['lat']):.6f}°")
            st.write(f"📏 Δ Longitude: {abs(point2['lng'] - point1['lng']):.6f}°")

            # Midpoint calculation
            mid_lat = (point1["lat"] + point2["lat"]) / 2
            mid_lng = (point1["lng"] + point2["lng"]) / 2
            st.write(f"🎯 Midpoint: ({mid_lat:.6f}, {mid_lng:.6f})")

    else:
        st.info("🎯 Select 2 points on the map to see calculations")

# Optional: Display raw coordinates for debugging
with st.expander("Debug Information"):
    st.write("Selected points:", st.session_state.selected_points)
    st.write("Last clicked:", map_data.get("last_object_clicked"))
