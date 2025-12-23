import h5py
import numpy as np
import pandas as pd

# --------------------- Functions to read the .h5 from the EKF -----------------------
def read_h5_file(filepath):
    """
    Reads an HDF5 (.h5) file and prints its structure and contents.

    Parameters:
    - filepath (str): Path to the .h5 file

    Returns:
    - data (dict): Dictionary containing datasets in the file
    """
    data = {}

    def recursively_extract(name, obj):
        if isinstance(obj, h5py.Dataset):
            data[name] = obj[()]
        elif isinstance(obj, h5py.Group):
            # Groups can contain other groups/datasets
            pass

    with h5py.File(filepath, 'r') as h5file:
        h5file.visititems(recursively_extract)

    return data

def convert_dict_to_dataframe(data_dict):
    """
    Converts a dictionary of 1D NumPy arrays to a pandas DataFrame.
    
    Assumes all arrays have the same length.
    
    Parameters:
    - data_dict (dict): Dictionary with keys as column names and 1D NumPy arrays as values.
    
    Returns:
    - pd.DataFrame
    """
    # Check consistency of array lengths
    lengths = {k: len(v) for k, v in data_dict.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError("Not all arrays are of the same length.")

    # Convert to DataFrame
    df = pd.DataFrame({k: v for k, v in data_dict.items()})
    for col in df.columns:
        df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    return df

def log_wind_profile(z, z_0, v_w_ref, z_ref):
    return v_w_ref * np.log(z / z_0) / np.log(z_ref / z_0)

