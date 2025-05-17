from typing import Any, Dict, Optional, Sequence, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

from src.data_processing.spatial_data_loader import SpatialDataLoader


class SpatialFeatureExtractor:
    """
    Computes spatial features on point data using a SpatialDataLoader.
    """

    def __init__(self, loader: SpatialDataLoader) -> None:
        """
        Initialize the feature extractor.

        Parameters
        ----------
        loader : SpatialDataLoader
            Data loader providing spatial data layers for feature computation.
        """
        # Store the loader for querying spatial layers later
        self.loader = loader

    def extract_static(
        self, df: pd.DataFrame, kml_gdf: gpd.GeoDataFrame
    ) -> pd.DataFrame:
        """
        Attach static regime points from a KML GeoDataFrame to the input DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing a 'regime' column with value 'static'.
        kml_gdf : gpd.GeoDataFrame
            GeoDataFrame of KML points with 'Name' and 'geometry' columns.

        Returns
        -------
        pd.DataFrame
            Subset of df with x, y coordinates joined from KML points.
        """
        # Prepare point lookup: extract location, x, y and drop unwanted columns
        pts = (
            kml_gdf.drop(0)
            .assign(
                location=lambda d: d.Name.str[
                    0
                ],  # first character of Name as location key
                x=lambda d: d.geometry.x,
                y=lambda d: d.geometry.y,
            )
            .drop(columns=["Name", "Description", "geometry"])
            .drop_duplicates("location")
            .set_index("location")
        )
        # Filter only static regimes and join coordinates by location
        return df.query("regime=='static'").join(pts, on="location")

    def extract_dynamic(
        self, df: pd.DataFrame, segments: gpd.GeoDataFrame
    ) -> pd.DataFrame:
        """
        Interpolate dynamic regime points evenly along provided line segments.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing a 'regime' column with value 'dynamic'.
        segments : gpd.GeoDataFrame
            GeoDataFrame of LineString geometries labeled by 'location'.

        Returns
        -------
        pd.DataFrame
            Original rows augmented with computed x, y coordinates along each segment.
        """
        # Select only dynamic regime rows
        dyn = df[df["regime"] == "dynamic"]
        orig_idx = dyn.index
        # Merge with segment geometries to get LineString per location
        merged = dyn.merge(segments, on="location", how="left")
        merged.index = orig_idx
        # Drop entries without valid geometry
        merged = merged.dropna(subset=["geometry"]).copy()

        merged["sample_pt"] = None
        # For each location group, interpolate points at equal intervals
        for loc, grp in merged.groupby("location"):
            seg: LineString = grp.geometry.iloc[0]
            distances = np.linspace(0, seg.length, len(grp))
            merged.loc[grp.index, "sample_pt"] = [seg.interpolate(d) for d in distances]

        # Extract x, y from interpolated points and remove helper columns
        return merged.assign(
            x=lambda d: d.sample_pt.map(lambda p: p.x),
            y=lambda d: d.sample_pt.map(lambda p: p.y),
        ).drop(columns=["sample_pt", "geometry"])

    def is_close_to(
        self,
        features: gpd.GeoDataFrame,
        point: Point,
        threshold: float,
        type_column: Optional[Union[str, Sequence[str]]] = None,
        types: Optional[Union[str, Sequence[Union[str, Sequence[str]]]]] = None,
    ) -> bool:
        """
        Check if any feature lies within a distance threshold of a point.

        Parameters
        ----------
        features : gpd.GeoDataFrame
            GeoDataFrame of candidate geometries (EPSG:4326).
        point : Point
            Query point (EPSG:4326).
        threshold : float
            Distance threshold in meters (after projecting to EPSG:3857).
        type_column : str or sequence of str, optional
            Column(s) for filtering features by category.
        types : str or sequence of str, optional
            Allowed value(s) corresponding to type_column filter.

        Returns
        -------
        bool
            True if at least one feature is within threshold of the point.
        """
        # Filter by provided type(s), if any
        if type_column and types is not None:
            cols = [type_column] if isinstance(type_column, str) else list(type_column)
            vals = types
            first = vals[0] if isinstance(vals, list) else vals
            if not isinstance(first, (list, tuple)):
                vals = [vals] * len(cols)
            vals_list = [v if isinstance(v, (list, tuple)) else [v] for v in vals]
            mask = False
            for col, allowed in zip(cols, vals_list):
                mask = mask | features[col].isin(allowed)
            features = features[mask]
            if features.empty:
                return False

        # Project to metric CRS for accurate distance calculation
        feats_proj = features.to_crs("EPSG:3857")
        feats_proj = feats_proj[
            feats_proj.geometry.notnull() & feats_proj.geometry.is_valid
        ]
        # Project query point as well
        p_proj = gpd.GeoSeries([point], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
        # Compute distances and check threshold
        distances = feats_proj.geometry.distance(p_proj)
        return (distances <= threshold).any()

    def count_nearby(
        self,
        features: gpd.GeoDataFrame,
        point: Point,
        threshold: float,
        type_column: Optional[Union[str, Sequence[str]]] = None,
        types: Optional[Union[str, Sequence[Union[str, Sequence[str]]]]] = None,
    ) -> int:
        """
        Count the number of features within a distance threshold of a point.

        Parameters
        ----------
        features : gpd.GeoDataFrame
            GeoDataFrame of candidate geometries (EPSG:4326).
        point : Point
            Query point (EPSG:4326).
        threshold : float
            Distance threshold in meters.
        type_column : str or sequence of str, optional
            Column(s) for filtering features by category.
        types : str or sequence of str, optional
            Allowed value(s) corresponding to filter columns.

        Returns
        -------
        int
            Count of geometries within threshold distance.
        """
        sub = features
        # Apply same type filtering as in is_close_to
        if type_column and types:
            cols = [type_column] if isinstance(type_column, str) else list(type_column)
            vals = types
            first = vals[0] if isinstance(vals, list) else vals
            if not isinstance(first, (list, tuple)):
                vals = [vals] * len(cols)
            vals_list = [v if isinstance(v, (list, tuple)) else [v] for v in vals]
            mask = False
            for col, allowed in zip(cols, vals_list):
                mask = mask | sub[col].isin(allowed)
            sub = sub.loc[mask]
            if sub.empty:
                return 0

        # Project and count distances
        sub_proj = sub.to_crs("EPSG:3857")
        sub_proj = sub_proj[sub_proj.geometry.notnull() & sub_proj.geometry.is_valid]
        p_proj = gpd.GeoSeries([point], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
        dists = sub_proj.geometry.distance(p_proj)
        return int((dists <= threshold).sum())

    def land_cover_proportion(
        self,
        features: gpd.GeoDataFrame,
        point: Point,
        threshold: float = 100.0,
        type_column: Optional[Union[str, Sequence[str]]] = None,
        types: Optional[Union[str, Sequence[Union[str, Sequence[str]]]]] = None,
    ) -> float:
        """
        Compute the fraction of a buffer around a point covered by features.

        Parameters
        ----------
        features : gpd.GeoDataFrame
            GeoDataFrame of geometries (EPSG:4326).
        point : Point
            Center of buffer (EPSG:4326).
        threshold : float, default=100.0
            Buffer radius in meters.
        type_column : str or sequence of str, optional
            Column(s) for filtering features by category.
        types : str or sequence of str, optional
            Allowed category value(s for filter.

        Returns
        -------
        float
            Proportion of buffer area covered (0.0–1.0).
        """
        sub = features
        # Filter by type/category
        if type_column and types is not None:
            cols = [type_column] if isinstance(type_column, str) else list(type_column)
            vals = types
            first = vals[0] if isinstance(vals, list) else vals
            if not isinstance(first, (list, tuple)):
                vals = [vals] * len(cols)
            vals_list = [v if isinstance(v, (list, tuple)) else [v] for v in vals]
            mask = False
            for col, allowed in zip(cols, vals_list):
                mask = mask | sub[col].isin(allowed)
            sub = sub.loc[mask]
            if sub.empty:
                return 0.0

        # Project to metric CRS and build buffer geometry
        sub_proj = sub.to_crs("EPSG:3857")
        p_proj = gpd.GeoSeries([point], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
        buffer_geom = p_proj.buffer(threshold)
        # Compute intersection area vs buffer area
        inter_area = sub_proj.geometry.intersection(buffer_geom).area.sum()
        buffer_area = buffer_geom.area
        return inter_area / buffer_area if buffer_area > 0 else 0.0

    def add_proximity(
        self,
        df: pd.DataFrame,
        prefix: str,
        source: Union[str, Dict[str, Any]],
        radii: Sequence[float],
        column: Optional[Union[str, Sequence[str]]] = None,
        values: Optional[Union[str, Sequence[Union[str, Sequence[str]]]]] = None,
    ) -> pd.DataFrame:
        """
        Add binary proximity indicators for each radius around points.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with x, y coordinate columns.
        prefix : str
            Prefix for naming new proximity columns.
        source : str or dict
            Key or filter dictionary for loader to select feature layer.
        radii : sequence of float
            Radii (meters) for proximity checks.
        column : str or sequence of str, optional
            Column(s) to filter features by.
        values : str or sequence of str, optional
            Allowed values corresponding to filter columns.

        Returns
        -------
        pd.DataFrame
            Input DataFrame augmented with binary proximity columns.
        """
        # Retrieve the spatial layer from the loader
        layer = self.loader.get_source(source)
        result = df.copy()
        # Loop through each radius to compute proximity flag
        for r in radii:
            col = f"close2{prefix}_{int(r)}"
            result[col] = result.apply(
                lambda row: int(
                    self.is_close_to(layer, Point(row.x, row.y), r, column, values)
                ),
                axis=1,
            )
        return result

    def add_count(
        self,
        df: pd.DataFrame,
        prefix: str,
        source: Union[str, Dict[str, Any]],
        radii: Sequence[float],
        column: Optional[Union[str, Sequence[str]]] = None,
        values: Optional[Union[str, Sequence[Union[str, Sequence[str]]]]] = None,
    ) -> pd.DataFrame:
        """
        Add integer counts of nearby features within specified radii.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'x', 'y' coordinate columns.
        prefix : str
            Prefix for naming new count columns.
        source : str or dict
            Key or filter dict to select feature layer.
        radii : sequence of float
            Radii in meters for counting features.
        column : str or sequence of str, optional
            Column(s) to filter features by category.
        values : str or sequence of str, optional
            Allowed category values for filtering.

        Returns
        -------
        pd.DataFrame
            DataFrame augmented with count columns per radius.
        """
        # Get the spatial layer containing features
        layer = self.loader.get_source(source)
        result = df.copy()
        # For each radius, count features near each point
        for r in radii:
            col = f"num_{prefix}_{int(r)}"
            result[col] = result.apply(
                lambda row: self.count_nearby(
                    layer, Point(row.x, row.y), r, column, values
                ),
                axis=1,
            ).astype(int)
        return result

    def add_proportion(
        self,
        df: pd.DataFrame,
        prefix: str,
        source: Union[str, Dict[str, Any]],
        radii: Sequence[float],
        column: Optional[Union[str, Sequence[str]]] = None,
        values: Optional[Union[str, Sequence[Union[str, Sequence[str]]]]] = None,
    ) -> pd.DataFrame:
        """
        Add proportion of area within buffer covered by features for each radius.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'x', 'y' coordinate columns.
        prefix : str
            Prefix for naming new proportion columns.
        source : str or dict
            Key or filter dict for feature layer selection.
        radii : sequence of float
            Radii in meters for buffer zones.
        column : str or sequence of str, optional
            Column(s) to filter features by category.
        values : str or sequence of str, optional
            Allowed category values for filtering.

        Returns
        -------
        pd.DataFrame
            DataFrame augmented with proportion columns per radius.
        """
        # Retrieve the spatial layer for proportion calculation
        layer = self.loader.get_source(source)
        result = df.copy()
        # Loop radii to compute coverage proportion per row
        for r in radii:
            col = f"proportion_{prefix}_{int(r)}"
            result[col] = result.apply(
                lambda row: self.land_cover_proportion(
                    layer, Point(row.x, row.y), r, column, values
                ),
                axis=1,
            )
        return result

    def add_sum(
        self,
        df: pd.DataFrame,
        prefix: str,
        source: str,  # now ALWAYS a string key into _named_loaders
        radii: Sequence[float],
        column: str,
        _values=None,  # unused
    ) -> pd.DataFrame:
        """
        Add summed values of a specified column from features within each radius.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'x', 'y' coordinate columns.
        prefix : str
            Prefix for naming new sum columns.
        source : str
            Key to select feature layer from loader.
        radii : sequence of float
            Radii in meters for summing values.
        column : str
            Column name in features whose values are summed.

        Returns
        -------
        pd.DataFrame
            DataFrame augmented with sum columns per radius.
        """
        # Load and project feature layer to metric CRS
        layer = self.loader.get_source(source).to_crs(epsg=3857)
        # Keep only valid geometries
        layer = layer[layer.geometry.notnull() & layer.geometry.is_valid]
        # Build GeoSeries of points and project
        pts = gpd.GeoSeries(
            [Point(x, y) for x, y in zip(df.x, df.y)], index=df.index, crs="EPSG:4326"
        ).to_crs(epsg=3857)
        out = df.copy()
        # Sum values in 'column' for all features within each radius
        for r in radii:
            col = f"sum_{prefix}_{int(r)}"
            out[col] = [
                layer.loc[layer.geometry.distance(pt) <= r, column].sum() for pt in pts
            ]
        return out

    def add_mean(
        self,
        df: pd.DataFrame,
        prefix: str,
        source: str,  # string key only
        radii: Sequence[float],
        column: str,
        _values=None,  # unused
    ) -> pd.DataFrame:
        """
        Add mean values of a specified column from features within each radius.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'x', 'y' coordinate columns.
        prefix : str
            Prefix for naming new mean columns.
        source : str
            Key to select feature layer from loader.
        radii : sequence of float
            Radii in meters for averaging values.
        column : str
            Column name in features whose values are averaged.

        Returns
        -------
        pd.DataFrame
            DataFrame augmented with average columns per radius.
        """
        # Load and project feature layer to metric CRS
        layer = self.loader.get_source(source).to_crs(epsg=3857)
        # Filter out invalid geometries
        layer = layer[layer.geometry.notnull() & layer.geometry.is_valid]
        # Build GeoSeries of input points projected to metric CRS
        pts = gpd.GeoSeries(
            [Point(x, y) for x, y in zip(df.x, df.y)], index=df.index, crs="EPSG:4326"
        ).to_crs(epsg=3857)
        out = df.copy()
        # Compute mean of 'column' for all features within each radius
        for r in radii:
            col = f"average_{prefix}_{int(r)}"
            out[col] = [
                layer.loc[layer.geometry.distance(pt) <= r, column].mean() for pt in pts
            ]
        return out
