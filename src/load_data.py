from scipy.io import loadmat
import pandas as pd
import numpy as np

def read_var(subject_id: int, variable: str):
    """
    Read a variable from a .mat file.

    Args:
        subject_id (int): The subject ID.
        variable (str): The variable name to read.

    Returns:
        pd.DataFrame: The loaded variable as a pandas DataFrame.
    """
    # Construct the filename based on the subject ID
    filename = f"../data/S{subject_id:02d}/S{subject_id}{variable}.mat"
    # Load the .mat file
    data = loadmat(filename)
    # Extract the variable from the loaded data
    data_df = pd.DataFrame(data["S"], columns=[f"S{subject_id}{variable}", "signal"])
    return data_df

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
    Load all variables for a given subject.

    Args:
        subject_id (int): The subject ID.

    Returns:
        pd.DataFrame: A DataFrame containing all variables for the subject.
    """
    # Load the variables for the subject
    variables = ["CO2", "P", "PM1", "PM10", "PM25", "RH", "T", "VOC"]
    data = {}
    for variable in variables:
        print(f"Loading {variable} for subject {subject_id}")
        data[variable] = read_var(subject_id, variable)[f"S{subject_id}{variable}"]
    
    # Concatenate all variables into a single DataFrame
    data_df = pd.concat(data.values(), axis=1).drop_duplicates()
    
    return data_df
    