"""
A Data Signal Generation Module

This module provides a data source for loading and streaming signal data 
from the 'a_data.csv' file. It mimics the interface of the SinusDataSource 
to allow seamless replacement in existing pipelines.

The `ADataSource` implements a callable interface that yields 
(sample, timestamp) pairs sequentially.

"""

from pathlib import Path
import numpy as np
import pandas as pd

class SensorDataSource:
    """Generator-based wrapper for Sensor data."""
    def __init__(self, signal, fs):
        """
        Initialize the data source.
        
        Args:
            signal (np.ndarray): Array of signal values.
            fs (float): Sampling frequency in Hz.
        """
        self.signal = signal
        self.fs = fs
        self._iterator = self._create_iterator()

    def _create_iterator(self):
        """
        Create a generator that yields (sample, timestamp) pairs.
        
        Iterates through the signal array, computing the timestamp for each
        sample based on its index and the sampling frequency.
        
        Yields:
            tuple: A pair containing (sample_value, timestamp_in_seconds).
        """
        for i, sample in enumerate(self.signal):
            yield sample, i / self.fs

    def __call__(self):
        """
        Retrieve the next sample from the iterator.
        
        Returns:
            tuple|None: (sample, timestamp) if available, None if exhausted.
        """
        try:
            return next(self._iterator)
        except StopIteration:
            return None

def csv_to_arrays(
        file_path: str,
        timestamp: str,
        category: str,
        value: str
    ) -> tuple[np.ndarray, list]:
    """
    Load time-series data from a CSV file (in long-format) and convert it to numpy arrays.
    
    The CSV file is expected to have the following three columns: 'timestamp', 'category', and 'value'.
    
    Args:
        file_path (str): Path to the CSV file containing the data.
        timestamp (str): Name of the column representing timestamps.
        category (str): Name of the column representing categories or parameters.
        value (str): Name of the column representing the values.

    Returns:
        tuple: A tuple containing:
            - records (np.ndarray): 2D array of shape (num_timestamps, num_categories) with the values.
            - category_names (list): List of category names corresponding to the columns in 'records'.

    """
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Create wide-format DataFrame with timestamps as index and categories as columns
    df = df.pivot(
        index= timestamp,
        columns= category,
        values= value
    )

    # Convert the DataFrame to a numpy array
    records = df.to_numpy().T
    category_names = df.columns.tolist()
    # TODO: LOG instead of print
    print(f"Following categories found in the data: {category_names}.")

    return records, category_names

def create_uni_signal(selected_parameter: str, records: np.ndarray, category_names: list, fs: float = 200.0):
    """Select a single signal array corresponding to the chosen parameter."""
    try:
        index = category_names.index(selected_parameter)
        # TODO: LOG instead of print
        print(f"Successfully selected parameter: {category_names[index]}")
    except Exception as e:
        # TODO: LOG instead of print
        print(f"Parameter '{selected_parameter}' could not be selected. Please check `config.yaml` and your data for typo's. Script will continue with first possible category. Here is the error message: {e}")
        index = category_names[0]

    signal = records[index]
    return SensorDataSource(signal, fs)

def get_dataframe(file_path: str = None,) -> pd.DataFrame:
    return pd.read_csv(file_path)


def create_a_data_source(file_path: str = None, fs: float = 200.0) -> SensorDataSource:
    """
    Create data source.

    Returns:
        ADataSource: A data source containing sensor data.

    """
    try:
        data_dir = Path.home() / ".streamsim" / "data"
        df = pd.read_csv(f"{data_dir}/{file_path}")
        signal = df['value'].values
        return SensorDataSource(signal, fs)
    except ImportError:
        # TODO: LOG instead of print
        print("data not found, generate mock data instead")
        t = np.linspace(0, 10, 3600)
        # Simple mock ECG: sine + spikes
        signal = np.sin(2 * np.pi * 1.2 * t) 
        # Add some spikes
        for i in range(0, len(signal), 100):
            if i+10 < len(signal):
                signal[i:i+10] += 2.0
        return SensorDataSource(signal, 360.0)
    

