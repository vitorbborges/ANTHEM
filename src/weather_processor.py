import os
from io import StringIO
from pathlib import Path

import pandas as pd
from scipy.interpolate import PchipInterpolator
from unidecode import unidecode


class WeatherProcessor:
    """
    Processes weather station data files by parsing metadata, reading and cleaning
    raw CSV measurements, and interpolating time series to a uniform frequency.
    """

    @staticmethod
    def parse_metadata(folder: Path) -> dict:
        """
        Parse metadata text files in a directory to map sensor IDs to names.

        This method reads all .txt files in `folder`, extracts lines 12–18,
        loads them into a DataFrame, and constructs a mapping from
        'Id_Sensore' to 'Nome_Sensore'.

        Parameters
        ----------
        folder : Path
            Directory containing metadata .txt files with sensor information.

        Returns
        -------
        dict
            A dictionary mapping each sensor ID (int or str) to its human-readable name.
        """
        meta = {}
        # Iterate over all files in the folder
        for f in os.listdir(folder):
            if f.endswith(".txt"):
                # Read specific lines containing metadata table
                lines = Path(folder / f).read_text().splitlines()[11:18]
                # Load table into DataFrame
                dfm = pd.read_csv(StringIO("\n".join(lines)))
                # Update mapping: Id_Sensore -> Nome_Sensore
                meta.update(dfm.set_index("Id_Sensore")["Nome_Sensore"].to_dict())
        return meta

    @staticmethod
    def read_raw(folder: Path, metadata: dict) -> pd.DataFrame:
        """
        Read and clean raw weather CSV files into a single DataFrame.

        Each .csv in `folder` is read with 'Data-Ora' as index. Missing-code values
        (777, 7777) are set to zero; codes (888, 8888, -999) become NA. Columns are
        renamed using `metadata` and original column names, normalized to lowercase
        with underscores. All sensor series are concatenated by time index.

        Parameters
        ----------
        folder : Path
            Directory containing raw weather .csv files.
        metadata : dict
            Mapping from sensor ID to sensor name, as returned by `parse_metadata`.

        Returns
        -------
        pd.DataFrame
            Combined DataFrame with cleaned sensor readings, indexed by datetime.
        """
        dfs = []
        # Loop through files to process each CSV
        for f in os.listdir(folder):
            if f.endswith(".csv"):
                # Read file into DataFrame, parse dates from 'Data-Ora'
                dft = pd.read_csv(folder / f, index_col="Data-Ora", parse_dates=True)
                # Identify the measurement column (excluding sensor ID)
                orig = [c for c in dft.columns if c != "Id Sensore"][0]
                # Lookup sensor ID and build normalized column name
                sid = dft["Id Sensore"].iat[0]
                name = (
                    unidecode(f"{metadata[sid].strip()} {orig.strip()}")
                    .replace(" ", "_")
                    .lower()
                )
                # Replace placeholder codes with appropriate values
                dft.replace([777, 7777], 0, inplace=True)
                dft.replace([888, 8888, -999], pd.NA, inplace=True)
                # Rename the measurement column and drop the ID column
                dfs.append(
                    dft.rename(columns={orig: name}).drop(columns=["Id Sensore"])
                )
        # Concatenate all sensor series along columns
        return pd.concat(dfs, axis=1)

    @staticmethod
    def interpolate(df: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolate time series to one-second frequency using PCHIP.

        Converts the index of `df` into POSIX seconds, creates a uniformly spaced
        datetime index at 1-second intervals, and applies a piecewise cubic
        Hermite interpolator on each column to fill missing timestamps.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with a DateTimeIndex and numeric columns, possibly
            with irregular sampling.

        Returns
        -------
        pd.DataFrame
            DataFrame reindexed at 1-second intervals with interpolated values for
            each original column.
        """
        # Convert original timestamps to seconds since epoch
        t0 = df.index.view("int64") / 1e9
        # Generate new time index at 1-second frequency
        t1 = pd.date_range(df.index[0], df.index[-1], freq="1s")
        t1s = t1.view("int64") / 1e9
        # Perform PCHIP interpolation for each series
        data = {
            col: PchipInterpolator(t0[df[col].notna()], df[col][df[col].notna()])(t1s)
            for col in df.columns
        }
        # Build new DataFrame with interpolated data
        return pd.DataFrame(data, index=t1)
