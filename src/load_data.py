from scipy.interpolate import interp1d
import pandas as pd
import numpy as np
import os


def read_var(subject_id: int, variable: str):
    """
    Load the 'S' matrix from a single .mat file and return
    a DataFrame with columns [variable'time', 'signal'].
    Raises FileNotFoundError or KeyError if something’s missing.

    Args:
        subject_id (int): The subject ID.
        variable (str): The variable name to read.

    Returns:
        pd.DataFrame: The loaded variable as a pandas DataFrame.
    """
    filename = f"data/adapted_raw_data/S{subject_id:02d}/S{subject_id}_{variable}_TableResampled.parquet"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found.")

    mat = pd.read_parquet(filename)
    if "S" not in mat:
        raise KeyError(f"'S' key not found in {filename}. Got keys: {list(mat.keys())}")

    # rename columns to match the expected format
    mat.columns = [variable, "time", "signal"]
    # Suppose your DataFrame is called df
    mat["time"] = pd.to_datetime(mat["time"]).dt.time  # Extract only the time
    mat = mat.set_index("time")  # Set it as index

    return mat


def load_var(data_df: pd.DataFrame):
    """
    Load a variable from a .mat file and convert the signal column to a categorical indicator of location.

    Args:
        subject_id (int): The subject ID.
        variable (str): The variable name to load.

    Returns:
        pd.DataFrame: The loaded variable as a pandas DataFrame.
    """

    # Convert the signal column to a categorical indicator of location
    data_df["location"] = data_df["signal"].cumsum()
    for i, state in enumerate(
        [
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
    ):
        data_df.loc[data_df["location"] == i * 10000, "location"] = state
    static_list = ["A", "B", "C", "D", "E", "F", "G", "H"]
    data_df["regime"] = np.where(
        data_df["location"].isin(static_list), "static", "dynamic"
    )

    return data_df


def load_subject(subject_id: int):
    """
    Load all variables for a subject, skip any missing or malformed files,
    and return one wide DataFrame joined by index.

    Args:
        subject_id (int): The subject ID.

    Returns:
        pd.DataFrame: A DataFrame containing all variables for the subject.
    """
    import pandas as pd

    variables = ["CO2", "P", "PM1", "PM10", "PM25", "RH", "T", "VOC"]
    full = None

    read = []
    for var in variables:
        print(f"Loading {var} for subject {subject_id}")
        try:
            df_var = read_var(subject_id, var)

            if full is None:
                full = df_var
            else:
                full = full.merge(
                    df_var,
                    left_index=True,
                    right_index=True,
                    how="outer",
                    suffixes=("", f"_{var}"),
                )
            read.append(var)

        except FileNotFoundError as e:
            print(f"  • SKIP (file not found): {e}")
        except KeyError as e:
            print(f"  • SKIP (bad format): {e}")

    if full is None:
        return pd.DataFrame()

    read.append("signal")
    full = full[read]
    full.dropna(inplace=True)

    return load_var(full)


def load_all_subjects(variable: str, target_length=300):
    """
    Load a given variable across multiple subjects, interpolate each subject's data
    to the same length using cubic spline (fallback to linear if too few points).

    Args:
        variable (str): The variable name, e.g. 'CO2', 'PM25', etc.
        target_length (int): Desired common length for all subjects after interpolation.

    Returns:
        pd.DataFrame: A DataFrame of shape (target_length, len(subjects_with_data)),
                      with columns named by subject_id.
    """
    interpolated_data = {}

    for sid in range(1, 21):
        try:
            # Attempt to read the raw variable data (measurement column + 'signal')
            df_var = read_var(sid, variable)
        except (FileNotFoundError, KeyError):
            print(f"  • Skipping subject {sid} for variable '{variable}'.")
            continue

        values = df_var[variable].to_numpy()
        n = len(values)

        # Choose cubic if enough points, else linear
        kind = "cubic" if n >= 4 else "linear"

        # Define original and target grids
        old_idx = np.linspace(0, n - 1, num=n)
        new_idx = np.linspace(0, n - 1, num=target_length)

        # Build interpolator and compute resampled values
        f_interp = interp1d(
            old_idx, values, kind=kind, bounds_error=False, fill_value="extrapolate"
        )
        interpolated_data[f"S{sid}{variable}"] = f_interp(new_idx)

    # Combine into a DataFrame: rows are sample positions, columns are subject IDs
    result_df = pd.DataFrame(interpolated_data)
    result_df.index.name = "sample_index"
    return result_df


# Example usage:
# subject_ids = list(range(1, 31))  # subjects 1 to 30
# df_co2 = load_variable_all_subjects('CO2', subject_ids, target_length=100)
# print(df_co2.shape)  # (100, number_of_subjects_with_CO2)
