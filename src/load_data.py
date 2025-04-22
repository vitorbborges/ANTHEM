from scipy.io import loadmat
import pandas as pd
import numpy as np
import os

def read_var(subject_id: int, variable: str):
    """
    Load the 'S' matrix from a single .mat file and return
    a DataFrame with columns [variable, 'signal'].
    Raises FileNotFoundError or KeyError if something’s missing.

    Args:
        subject_id (int): The subject ID.
        variable (str): The variable name to read.

    Returns:
        pd.DataFrame: The loaded variable as a pandas DataFrame.
    """
    filename = f"../data/S{subject_id:02d}/S{subject_id}{variable}.mat"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found.")
    
    mat = loadmat(filename)
    if "S" not in mat:
        raise KeyError(f"'S' key not found in {filename}. Got keys: {list(mat.keys())}")
    
    arr = mat["S"]
    # first column is the measurement, second is 'signal'
    df = pd.DataFrame(arr, columns=[variable, "signal"])
    return df

def load_var(subject_id: int, variable: str):
    """
    Load a variable from a .mat file and convert the signal column to a categorical indicator of location.

    Args:
        subject_id (int): The subject ID.
        variable (str): The variable name to load.

    Returns:
        pd.DataFrame: The loaded variable as a pandas DataFrame.
    """
    # Read the variable from the .mat file
    data_df = read_var(subject_id, variable)

    # Convert the signal column to a categorical indicator of location
    data_df["location"] = data_df["signal"].cumsum()
    for i, state in enumerate(["A", "AB", "B", "BC", "C", "CD", "D", "DE", "E", "EF", "F", "FG", "G", "GH", "H"]):
        data_df.loc[data_df["location"] == i*10000, "location"] = state
    static_list = ["A", "B", "C", "D", "E", "F", "G", "H"]
    data_df["regime"] = np.where(data_df["location"].isin(static_list), "static", "dynamic")

    return data_df

def load_subject(subject_id: int):
    """
    Load all variables for a subject, skip any missing or malformed files,
    and return one wide DataFrame indexed by the row number.

    Args:
        subject_id (int): The subject ID.

    Returns:
        pd.DataFrame: A DataFrame containing all variables for the subject.
    """
    variables = ["CO2", "P", "PM1", "PM10", "PM25", "RH", "T", "VOC"]
    dfs = []
    
    for var in variables:
        print(f"Loading {var} for subject {subject_id}")
        try:
            df_var = read_var(subject_id, var)
            dfs.append(df_var)
        except FileNotFoundError as e:
            print(f"  • SKIP (file not found): {e}")
        except KeyError as e:
            print(f"  • SKIP (bad format): {e}")
    
    if not dfs:
        # nothing loaded
        return pd.DataFrame()
    
    # Concatenate side‑by‑side; drop duplicate 'signal' columns
    full = pd.concat(dfs, axis=1)
    # If 'signal' appears more than once, keep the first
    full = full.loc[:, ~full.columns.duplicated()]
    
    # Now you can add your location/regime logic here if you want
    # e.g. full['location'] = full['signal'].cumsum() 
    
    return full
    