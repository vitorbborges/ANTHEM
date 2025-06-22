import argparse
import sys
from pathlib import Path

import streamlit as st
from streamlit_folium import st_folium

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, List, Optional, Tuple

import folium
import geopandas as gpd
from features.calculations import GeoCalculations
from features.map_handler import MapHandler
from features.ui_components import ClickHandler, UIComponents

from src.data_processing.process_subject_pipeline import ProcessSubjectPipeline

# Constants
WEST, SOUTH, EAST, NORTH = 9.2257, 45.47162, 9.23768, 45.48537
BBOX = (WEST, SOUTH, EAST, NORTH)

st.set_page_config(page_title="Interactive Route Map", layout="wide")

# Custom CSS
st.markdown(
    """
<style>
    iframe { border: none !important; outline: none !important; box-shadow: none !important; }
    .stApp > div > div > div > div > div > section > div > div > div > div > div {
        border: none !important; outline: none !important;
    }
    .folium-map { border: none !important; margin: 0 auto !important; padding: 0 !important; display: block !important; }
    .main .block-container { padding-top: 0.2rem; }
    html { scroll-behavior: auto !important; }
    .stApp { overflow-anchor: auto !important; }
    .stHorizontalBlock {
        align-items: flex-start !important;
        justify-content: center !important;
    }
    div[data-testid="stHorizontalBlock"] > div:nth-child(2) {
        display: flex !important;
        justify-content: center !important;
        align-items: flex-start !important;
        overflow: hidden !important;
    }
    iframe[title="streamlit_folium.st_folium"] {
        display: block !important;
        margin: 0 auto !important;
        vertical-align: top !important;
    }
    @media (max-width: 768px) {
        .stHorizontalBlock { flex-direction: column !important; }
    }
</style>

<script>
    document.addEventListener('DOMContentLoaded', function() {
        let scrollPosition = sessionStorage.getItem('scrollPosition');
        if (scrollPosition) {
            window.scrollTo(0, parseInt(scrollPosition));
            sessionStorage.removeItem('scrollPosition');
        }
        window.addEventListener('beforeunload', function() {
            sessionStorage.setItem('scrollPosition', window.scrollY);
        });
    });
</script>
""",
    unsafe_allow_html=True,
)

# Initialize state
if "selected_points" not in st.session_state:
    st.session_state.selected_points = []
if "last_click_id" not in st.session_state:
    st.session_state.last_click_id = None
if "click_update" not in st.session_state:
    st.session_state.click_update = False


@st.cache_resource
def get_pipeline():
    return ProcessSubjectPipeline(BBOX)


pipeline = get_pipeline()
loader = pipeline.extractor.loader


@st.cache_resource
def load_route_data():
    gdf = loader.extract_kml(PROJECT_ROOT / "data" / "raw_data" / "route.kmz")
    return gdf.to_crs(epsg=4326)


gdf = load_route_data()
map_handler = MapHandler(BBOX)

# Prepare UI layout
bbox_width = EAST - WEST
bbox_height = NORTH - SOUTH
aspect_ratio = bbox_width / bbox_height
if aspect_ratio > 1.5:
    left_col, center_col, right_col = st.columns([0.8, 2.5, 0.8])
elif aspect_ratio < 0.7:
    left_col, center_col, right_col = st.columns([1.2, 1.5, 1.2])
else:
    left_col, center_col, right_col = st.columns([1, 2, 1])

# OSM Layer options with color
osm_layer_options = {
    "Water": {
        "tags": {"natural": ["water", "wetland"], "waterway": True, "water": True},
        "color": "#1f77b4",
    },
    "Smoking Shop": {
        "tags": {"shop": True},
        "color": "#ff7f0e",
    },
    "Chimney": {
        "tags": {"man_made": True},
        "color": "#9467bd",
    },
    "Public Transport": {
        "tags": {"public_transport": True},
        "color": "#d62728",
    },
    "Green Spaces": {
        "tags": {"leisure": ["park", "garden"]},
        "color": "#25a046",
    },
}

# --- Left Column ---
with left_col:
    st.markdown("### 📍 Selected Points")

    if len(st.session_state.selected_points) == 0:
        st.info("No points selected yet")
    else:
        labels = ["Point 1", "Point 2"]
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
        st.session_state.last_click_id = None
        st.session_state.click_update = False
        st.rerun()

    st.markdown("### 🗂️ Show OSM Layers")
    layer_dataframes = {}
    layer_colors = {}

    for label, config in osm_layer_options.items():
        if st.checkbox(label, key=f"layer_{label}"):
            gdf_layer = loader.get_source(config["tags"]).to_crs(epsg=4326)
            layer_dataframes[label] = gdf_layer
            layer_colors[label] = config["color"]

# --- Center Column ---
with center_col:
    map_width = 640
    map_height = int(map_width / aspect_ratio)
    m = map_handler.create_base_map()
    map_handler.add_route_data(m, gdf, show_bbox=False)
    map_handler.add_point_markers(m, st.session_state.selected_points)

    for label, gdf_layer in layer_dataframes.items():
        color = layer_colors.get(label, "orange")
        for _, row in gdf_layer.iterrows():
            geom = row.geometry
            if geom.geom_type == "Point":
                folium.CircleMarker(
                    location=[geom.y, geom.x],
                    radius=4,
                    color=color,
                    fill=True,
                    fill_opacity=0.7,
                    popup=label,
                ).add_to(m)
            elif geom.geom_type == "LineString":
                folium.PolyLine(
                    locations=[[pt[1], pt[0]] for pt in geom.coords],
                    color=color,
                    weight=2,
                    popup=label,
                ).add_to(m)
            elif geom.geom_type == "Polygon":
                folium.GeoJson(
                    geom,
                    name=label,
                    style_function=lambda x, col=color: {
                        "color": col,
                        "weight": 2,
                        "fillOpacity": 0.3,
                    },
                ).add_to(m)

    folium.Rectangle(
        bounds=[[SOUTH, WEST], [NORTH, EAST]],
        color="transparent",
        fill=True,
        fillColor="transparent",
        fillOpacity=0.0,
        weight=0,
        interactive=True,
    ).add_to(m)

    map_data = st_folium(
        m,
        width=map_width,
        height=map_height,
        key="interactive_map",
        returned_objects=["last_object_clicked", "bounds"],
    )

    if map_data and "last_object_clicked" in map_data:
        clicked = map_data["last_object_clicked"]
        if (
            clicked
            and isinstance(clicked, dict)
            and "lat" in clicked
            and "lng" in clicked
            and clicked.get("popup") is None
        ):
            lat, lng = clicked["lat"], clicked["lng"]
            click_id = f"{lat:.6f},{lng:.6f}"
            if click_id != st.session_state.last_click_id:
                st.session_state.last_click_id = click_id
                if SOUTH <= lat <= NORTH and WEST <= lng <= EAST:
                    new_point = {"lat": lat, "lng": lng}
                    if len(st.session_state.selected_points) < 2:
                        st.session_state.selected_points.append(new_point)
                    else:
                        st.session_state.selected_points = [new_point]
                    st.session_state.click_update = True
                    st.markdown(
                        """
                        <script>
                            sessionStorage.setItem('scrollPosition', window.scrollY);
                        </script>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.rerun()

    if st.session_state.click_update:
        st.session_state.click_update = False

# --- Right Column ---
with right_col:
    st.markdown("### 📊 Calculations")

    if len(st.session_state.selected_points) == 2:
        point1, point2 = st.session_state.selected_points
        distance_km = GeoCalculations.calculate_distance(
            point1["lat"], point1["lng"], point2["lat"], point2["lng"]
        )
        bearing = GeoCalculations.calculate_bearing(
            point1["lat"], point1["lng"], point2["lat"], point2["lng"]
        )
        distance_miles = distance_km * 0.621371
        cardinal_direction = GeoCalculations.get_cardinal_direction(bearing)

        st.metric("🏃 Distance", f"{distance_km:.3f} km")
        st.metric("📐 Bearing", f"{bearing:.1f}°")
        st.metric("🏃 Distance", f"{distance_miles:.3f} mi")
        st.metric("🧭 Direction", cardinal_direction)

        with st.expander("📋 Details"):
            delta_lat = abs(point2["lat"] - point1["lat"])
            delta_lng = abs(point2["lng"] - point1["lng"])
            mid_lat = (point1["lat"] + point2["lat"]) / 2
            mid_lng = (point1["lng"] + point2["lng"]) / 2

            st.write("**Coordinate Differences:**")
            st.write(f"📏 Δ Lat: {delta_lat:.6f}°")
            st.write(f"📏 Δ Lng: {delta_lng:.6f}°")
            st.write(f"🎯 Midpoint: ({mid_lat:.6f}, {mid_lng:.6f})")
    else:
        st.info("🎯 Select 2 points to see calculations")
