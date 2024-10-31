from src.snf_pipeline.constants import EID_NAME


import pandas as pd


def get_overlapping_modalities(dfs: tuple[pd.DataFrame]) -> tuple[pd.DataFrame]:
    """
    Identify and return the overlapping rows across multiple DataFrames based on a common identifier.

    This function takes a tuple of pandas DataFrames and returns a tuple of DataFrames that contain only
    the rows where the identifier column (specified by `EID_NAME`) overlaps across all input DataFrames.
    It effectively performs an inner join on the identifier column across all DataFrames.

    Parameters
    ----------
    dfs : tuple[pd.DataFrame]
        A tuple containing pandas DataFrames to be processed. Each DataFrame must contain the column specified
        by `EID_NAME`.

    Returns
    -------
    tuple[pd.DataFrame]
        A tuple of pandas DataFrames, each filtered to include only the rows where the `EID_NAME` column
        matches across all input DataFrames.

    Raises
    ------
    ValueError
        If fewer than two DataFrames are provided, or if there are no overlapping identifiers between the DataFrames.

    Examples
    --------
    >>> df1 = pd.DataFrame({'eid': [1, 2, 3], 'data1': [10, 20, 30]})
    >>> df2 = pd.DataFrame({'eid': [2, 3, 4], 'data2': [200, 300, 400]})
    >>> df3 = pd.DataFrame({'eid': [3, 4, 5], 'data3': [3000, 4000, 5000]})
    >>> overlapping_dfs = get_overlapping_modalities((df1, df2, df3))
    >>> for df in overlapping_dfs:
    ...     print(df)
       eid  data1
    2    3     30
       eid  data2
    1    3    300
       eid  data3
    0    3   3000

    Notes
    -----
    - The function assumes that the identifier column `EID_NAME` is present in all DataFrames.
    - The order of DataFrames in the returned tuple corresponds to the order in the input tuple.
    - If there are no overlapping identifiers, a ValueError is raised.
    """
    if len(dfs) < 2:
        raise ValueError(f"Minimum number of DataFrames is 2 ({len(dfs)} given)")

    eid_sets = [set(df[EID_NAME]) for df in dfs]
    eid_intersection = set.intersection(*eid_sets)

    if len(eid_intersection) == 0:
        raise ValueError(f"No overlapping '{EID_NAME}'s between modalities")

    return tuple(df[df[EID_NAME].isin(eid_intersection)].reset_index(drop=True) for df in dfs)