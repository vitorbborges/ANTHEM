# app/core/config.py - Configuration management
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple


@dataclass
class BoundingBox:
    """Represents a geographic bounding box."""

    west: float
    south: float
    east: float
    north: float

    @property
    def center(self) -> Tuple[float, float]:
        """Get center coordinates (lat, lng)."""
        return (self.north + self.south) / 2, (self.east + self.west) / 2

    @property
    def aspect_ratio(self) -> float:
        """Get aspect ratio (width/height)."""
        return (self.east - self.west) / (self.north - self.south)

    @property
    def bbox_tuple(self) -> Tuple[float, float, float, float]:
        """Get bbox as tuple (west, south, east, north)."""
        return (self.west, self.south, self.east, self.north)


@dataclass
class MapConfig:
    """Map display configuration."""

    width: int = 640
    max_zoom: int = 20
    min_zoom: int = 10
    tile_url: str = (
        "https://cartodb-basemaps-{s}.global.ssl.fastly.net/dark_all/{z}/{x}/{y}.png"
    )
    attribution: str = " "


@dataclass
class LayerConfig:
    """OSM layer configuration."""

    tags: Dict[str, Any]
    color: str
    enabled: bool = False


@dataclass
class AppConfig:
    """Main application configuration."""

    bbox: BoundingBox = field(
        default_factory=lambda: BoundingBox(9.2257, 45.47162, 9.23768, 45.48537)
    )
    map_config: MapConfig = field(default_factory=MapConfig)
    data_path: Path = field(default_factory=lambda: Path("data/raw_data"))
    osm_layers: Dict[str, LayerConfig] = field(default_factory=dict)
    show_kml_points: bool = True

    def __post_init__(self):
        if not self.osm_layers:
            self.osm_layers = {
                "Water": LayerConfig(
                    tags={
                        "natural": ["water", "wetland"],
                        "waterway": True,
                        "water": True,
                    },
                    color="#1f77b4",
                ),
                
                "Green Spaces": LayerConfig(
                    tags={"leisure": ["park", "garden"]}, color="#52cc9b"
                ),

                "Trees": LayerConfig(
                    tags={"natural": "tree"}, color="#52cc9b"
                ),

                "Industry": LayerConfig(tags={"landuse": "industrial", "building": "industrial"}, color="#9FE2BF"),
                "Smoking Shop": LayerConfig(tags={"shop": "tobacco"}, color="#adad85"),
                "Smoking amenity": LayerConfig(tags={"smoking": ["yes", "outside"]}, color="#adad85"),
                #"Chimney": LayerConfig(tags={"man_made": True}, color="#9467bd"),
                "Public Transport": LayerConfig(
                    tags={"public_transport": True}, color="#cbb318"
                ),
                "Railway": LayerConfig(
                    tags={"landuse": "railway"}, color="#cbb318"
                ),

                "Fuel Station": LayerConfig(tags={"amenity": "fuel"}, color="#d62728"),
            }

    @classmethod
    def from_file(cls, path: str) -> "AppConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)
