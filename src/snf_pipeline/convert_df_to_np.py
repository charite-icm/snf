from src.snf_pipeline.constants import EID_NAME


import numpy as np
import pandas as pd


def convert_df_to_np(dfs: tuple[pd.DataFrame]) -> tuple[np.ndarray]:
    """
    Convert a tuple of pandas DataFrames to a tuple of NumPy arrays, excluding a specific column.

    This function takes a tuple of pandas DataFrames, drops the column specified by the global variable `EID_NAME`
    from each DataFrame, and converts the remaining data in each DataFrame to a NumPy array. It returns a tuple
    containing these NumPy arrays in the same order as the input DataFrames.

    Parameters
    ----------
    dfs : tuple of pd.DataFrame
        A tuple containing pandas DataFrames to be converted to NumPy arrays. Each DataFrame must contain
        the column specified by `EID_NAME`, which will be dropped before conversion.

    Returns
    -------
    tuple of np.ndarray
        A tuple containing NumPy arrays corresponding to the input DataFrames, excluding the `EID_NAME` column.

    Raises
    ------
    KeyError
        If the `EID_NAME` column is not present in any of the DataFrames.

    TypeError
        If `dfs` is not a tuple of pandas DataFrames.

    Examples
    --------
    >>> EID_NAME = 'eid'
    >>> df1 = pd.DataFrame({'eid': [1, 2], 'feature1': [10, 20], 'feature2': [100, 200]})
    >>> df2 = pd.DataFrame({'eid': [1, 2], 'feature3': [30, 40], 'feature4': [300, 400]})
    >>> arrays = convert_df_to_np((df1, df2))
    >>> for array in arrays:
    ...     print(array)
    [[ 10 100]
     [ 20 200]]
    [[ 30 300]
     [ 40 400]]

    Notes
    -----
    - The function assumes that `EID_NAME` is a global variable defined elsewhere in your code.
    - The order of the NumPy arrays in the returned tuple corresponds to the order of the input DataFrames.
    - The DataFrames must not contain any non-numeric columns other than `EID_NAME`; otherwise, the resulting
      NumPy arrays may have an object dtype.
    """
    # Validate input is a tuple of DataFrames
    if not isinstance(dfs, tuple):
        raise TypeError(f"'dfs' must be a tuple of pandas DataFrames, got {type(dfs)}.")
    for df in dfs:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"All elements in 'dfs' must be pandas DataFrames, got {type(df)}.")

    # Convert DataFrames to NumPy arrays
    return tuple(df.drop(columns=[EID_NAME]).to_numpy() for df in dfs)