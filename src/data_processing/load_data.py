import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# --------------------------------------------------
# Subject Data Loading and Interpolation
# --------------------------------------------------


def read_var(subject_id: int, variable: str):
    """
    Load and return a single variable time series for a subject.

    Reads a parquet file for the given `subject_id` and `variable`, expecting
    a DataFrame with an 'S' matrix containing time and signal columns.

    Args:
        subject_id (int): Identifier for the subject (e.g., 1 for S01 folder).
        variable (str): Variable name (e.g., 'CO2', 'PM25').

    Returns:
        pd.DataFrame: DataFrame indexed by time (datetime.time) with columns:
                      [variable, 'signal'].

    Raises:
        FileNotFoundError: If the expected parquet file does not exist.
        KeyError: If the loaded DataFrame does not contain key 'S'.
    """
    # Construct expected filepath based on subject and variable
    filename = f"data/adapted_raw_data/S{subject_id:02d}/S{subject_id}_{variable}_TableResampled.parquet"
    if not os.path.exists(filename):
        # Inform caller if file is missing
        raise FileNotFoundError(f"{filename} not found.")

    # Load parquet into DataFrame
    mat = pd.read_parquet(filename)
    if "S" not in mat:
        # Ensure expected key is present
        raise KeyError(f"'S' key not found in {filename}. Got keys: {list(mat.keys())}")

    # Rename columns to standardized names
    mat.columns = [variable, "time", "signal"]
    # Convert datetime index and extract only the time component
    mat["time"] = pd.to_datetime(mat["time"]).dt.time
    # Set the time column as index for easy joining
    mat = mat.set_index("time")

    return mat


def load_var(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add location and regime labels to a subject's signal DataFrame.

    Creates a 'location' column by cumulatively summing 'signal', then
    maps integer thresholds to named states. Also adds a 'regime' column
    indicating 'static' or 'dynamic' based on location membership.

    Args:
        data_df (pd.DataFrame): DataFrame with at least 'signal' column.

    Returns:
        pd.DataFrame: Augmented DataFrame with new 'location' and 'regime'.
    """
    # Create a cumulative sum of signal to determine discrete segments
    data_df["location"] = data_df["signal"].cumsum().astype(int).astype(object)

    # Define all possible labeled states in order
    states = [
        "A",
        "AB",
        "B",
        "BC",
        "C",
        "CD",
        "D",
        "DE",
        "E",
        "EF",
        "F",
        "FG",
        "G",
        "GH",
        "H",
    ]
    # Map cumsum multiples of 10000 to their respective state labels
    for i, state in enumerate(states):
        data_df.loc[data_df["location"] == i * 10000, "location"] = state

    # Define static states and assign regime accordingly
    static_states = ["A", "B", "C", "D", "E", "F", "G", "H"]
    data_df["regime"] = np.where(
        data_df["location"].isin(static_states), "static", "dynamic"
    )

    return data_df


def load_subject(subject_id: int) -> pd.DataFrame:
    """
    Load all available variables for a given subject into one DataFrame.

    Iterates through a predefined list of variables, merges each variable's
    DataFrame by time index, handles missing files or formats, and drops
    incomplete rows before labeling.

    Args:
        subject_id (int): Identifier for the subject (1-30).

    Returns:
        pd.DataFrame: Combined DataFrame with measurement columns, 'signal',
                      'location', and 'regime'. Empty if no data.
    """
    # List of variables to attempt loading
    variables = ["CO2", "P", "PM1", "PM10", "PM25", "RH", "T", "VOC"]
    full = None  # Placeholder for combined DataFrame
    read_vars = []  # Track successfully read variables

    for var in variables:
        try:
            # Read each variable's DataFrame
            df_var = read_var(subject_id, var)
            # Merge on time index
            full = (
                df_var
                if full is None
                else full.merge(
                    df_var,
                    left_index=True,
                    right_index=True,
                    how="outer",
                    suffixes=("", f"_{var}"),
                )
            )
            read_vars.append(var)
        except FileNotFoundError as e:
            # Skip subjects missing this variable file
            print(f"  • SKIP (file not found): {e}")
        except KeyError as e:
            # Skip malformed files
            print(f"  • SKIP (bad format): {e}")

    # If no variables loaded, return empty DataFrame
    if full is None:
        return pd.DataFrame()

    # Ensure 'signal' is preserved in column list
    read_vars.append("signal")
    # Filter to only read and 'signal' columns, drop NA rows
    full = full[read_vars]
    full.dropna(inplace=True)

    # Add location and regime labels
    return load_var(full)


def load_all_subjects(variable: str, target_length: int = 300) -> pd.DataFrame:
    """
    Load a specific variable across subjects and resample to uniform length.

    Reads each subject's raw series via `read_var`, then interpolates
    to `target_length` points using cubic or linear interpolation.

    Args:
        variable (str): Name of the variable to load (e.g., 'CO2').
        target_length (int, optional): Number of samples in output series.
                                       Defaults to 300.

    Returns:
        pd.DataFrame: DataFrame of shape (target_length, n_subjects_loaded),
                      columns named 'S{sid}{variable}'.
    """
    interpolated_data = {}

    # Loop over potential subject IDs
    for sid in range(1, 21):
        try:
            df_var = read_var(sid, variable)
        except (FileNotFoundError, KeyError):
            # Skip subjects lacking data for this variable
            print(f"  • Skipping subject {sid} for variable '{variable}'.")
            continue

        # Extract raw measurement values
        values = df_var[variable].to_numpy()
        n = len(values)

        # Choose interpolation method based on number of points
        kind = "cubic" if n >= 4 else "linear"

        # Define original and new index grids
        old_idx = np.linspace(0, n - 1, num=n)
        new_idx = np.linspace(0, n - 1, num=target_length)

        # Create interpolator and compute resampled series
        f_interp = interp1d(
            old_idx, values, kind=kind, bounds_error=False, fill_value="extrapolate"
        )
        interpolated_data[f"S{sid}{variable}"] = f_interp(new_idx)

    # Build and return final DataFrame
    result_df = pd.DataFrame(interpolated_data)
    result_df.index.name = "sample_index"
    return result_df


# Example usage:
# subject_ids = list(range(1, 21))  # subjects 1 to 20
# df_co2 = load_all_subjects('CO2', target_length=100)
# print(df_co2.shape)  # e.g., (100, number_of_subjects_with_CO2)
