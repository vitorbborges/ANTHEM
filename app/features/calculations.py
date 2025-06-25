import math
from typing import Dict, Tuple


class GeoCalculations:
    """Helper class for geographical calculations between points."""

    @staticmethod
    def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance between two points using Haversine formula.

        Parameters
        ----------
        lat1, lon1 : float
            Latitude and longitude of first point
        lat2, lon2 : float
            Latitude and longitude of second point

        Returns
        -------
        float
            Distance in kilometers
        """
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

    @staticmethod
    def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the bearing between two points.

        Parameters
        ----------
        lat1, lon1 : float
            Latitude and longitude of first point
        lat2, lon2 : float
            Latitude and longitude of second point

        Returns
        -------
        float
            Bearing in degrees (0-360)
        """
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

    @staticmethod
    def get_cardinal_direction(bearing: float) -> str:
        """
        Convert bearing to cardinal direction.

        Parameters
        ----------
        bearing : float
            Bearing in degrees

        Returns
        -------
        str
            Cardinal direction (N, NE, E, etc.)
        """
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
        return directions[direction_idx]

    @staticmethod
    def calculate_midpoint(
        lat1: float, lon1: float, lat2: float, lon2: float
    ) -> Tuple[float, float]:
        """
        Calculate the midpoint between two coordinates.

        Parameters
        ----------
        lat1, lon1 : float
            Latitude and longitude of first point
        lat2, lon2 : float
            Latitude and longitude of second point

        Returns
        -------
        Tuple[float, float]
            Midpoint coordinates (latitude, longitude)
        """
        mid_lat = (lat1 + lat2) / 2
        mid_lng = (lon1 + lon2) / 2
        return mid_lat, mid_lng

    @classmethod
    def calculate_all_metrics(
        cls, point1: Dict[str, float], point2: Dict[str, float]
    ) -> Dict[str, any]:
        """
        Calculate all metrics between two points.

        Parameters
        ----------
        point1, point2 : Dict[str, float]
            Point dictionaries with 'lat' and 'lng' keys

        Returns
        -------
        Dict[str, any]
            Dictionary containing all calculated metrics
        """
        distance_km = cls.calculate_distance(
            point1["lat"], point1["lng"], point2["lat"], point2["lng"]
        )

        bearing = cls.calculate_bearing(
            point1["lat"], point1["lng"], point2["lat"], point2["lng"]
        )

        mid_lat, mid_lng = cls.calculate_midpoint(
            point1["lat"], point1["lng"], point2["lat"], point2["lng"]
        )

        return {
            "distance_km": distance_km,
            "distance_miles": distance_km * 0.621371,
            "bearing": bearing,
            "cardinal_direction": cls.get_cardinal_direction(bearing),
            "delta_lat": abs(point2["lat"] - point1["lat"]),
            "delta_lng": abs(point2["lng"] - point1["lng"]),
            "midpoint_lat": mid_lat,
            "midpoint_lng": mid_lng,
        }
