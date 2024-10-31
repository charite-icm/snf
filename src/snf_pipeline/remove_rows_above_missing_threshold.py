import pandas as pd


def remove_rows_above_missing_threshold(df: pd.DataFrame, th_nan: float = 1.0, verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Removes rows from the DataFrame where the proportion of missing values exceeds the threshold.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    th_nan : float, optional (default=1.0)
        A float between 0 and 1 that represents the threshold for the proportion of missing values allowed in each row.
        Rows with a higher proportion of missing values will be removed.
    verbose : bool, optional (default=False)
        If True, prints out how much data was left after filtering out the rows.

    Returns
    -------
    tuple: (pd.DataFrame, pd.DataFrame)
        - A DataFrame with rows removed based on the missing value threshold.
        - A DataFrame containing the percentage of missing values for each row (including removed rows).
    """

    # Check if th_nan is between 0 and 1
    if not (0 <= th_nan <= 1):
        raise ValueError("The threshold (th_nan) must be a float between 0 and 1.")

    # Calculate the proportion (percentage) of missing values for each row
    missing_percentage = df.isna().mean(axis=1) * 100  # Converts the proportion to percentage

    # DataFrame containing the percentage of missing values for each row
    row_missing_percentage = missing_percentage.to_frame(name="missing_percentage")

    # Filter rows where missing values exceed the threshold
    df_cleaned = df[missing_percentage <= th_nan * 100]  # Threshold is compared in percentage

    # If verbose, print how much data was removed and left
    if verbose:
        original_rows = df.shape[0]
        remaining_rows = df_cleaned.shape[0]
        removed_rows = original_rows - remaining_rows
        print(f"Original rows: {original_rows}")
        print(f"Rows remaining after filtering: {remaining_rows}")
        print(f"Rows removed: {removed_rows}")

    return df_cleaned, row_missing_percentage