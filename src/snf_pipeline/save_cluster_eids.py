from src.snf_pipeline.constants import CLUSTER_EIDS_CSV, CLUSTER_NAME, EID_NAME


import pandas as pd


from pathlib import Path


def save_cluster_eids(
    eids: list[int], labels: list[int], save_path: str | Path, verbose: bool = True
) -> None:
    """
    Save a list of entity IDs and their corresponding cluster labels to a CSV file.

    Parameters
    ----------
    eids : list of int
        List of entity IDs to be saved.
    labels : list of int
        List of cluster labels corresponding to each entity ID. Must be the same length as `eids`.
    save_path : str or Path
        Directory path where the CSV file will be saved.
    verbose : bool, optional
        If True, prints a confirmation message after saving. Default is True.

    Raises
    ------
    ValueError
        If the lengths of `eids` and `labels` do not match.
    FileNotFoundError
        If `save_path` does not exist.

    Example
    -------
    >>> save_cluster_eids([1, 2, 3], [0, 1, 1], "/path/to/save", verbose=True)
    "/path/to/save/cluster_eids.csv saved!"
    """

    # Check if `eids` and `labels` have the same length
    if len(eids) != len(labels):
        raise ValueError("The length of `eids` and `labels` must be the same.")

    # Ensure `save_path` exists and is a directory
    save_path = Path(save_path)
    if not save_path.exists() or not save_path.is_dir():
        raise FileNotFoundError(f"The specified directory {save_path} does not exist.")

    # Define full path for the CSV file
    save_csv_path = save_path / CLUSTER_EIDS_CSV

    # Create DataFrame and save to CSV
    df = pd.DataFrame({EID_NAME: eids, CLUSTER_NAME: labels})
    df.to_csv(save_csv_path, index=False)
    if verbose:
        print(f"{save_csv_path} saved!")