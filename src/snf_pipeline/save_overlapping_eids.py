from src.snf_pipeline.constants import EID_NAME, OVERLAPPING_EID_TXT



import pandas as pd


from pathlib import Path


def save_overlapping_eids(dfs: tuple[pd.DataFrame], save_path: str | Path, verbose: bool = True) -> list[int]:
    """
    Compute and save the list of overlapping EIDs from multiple DataFrames to a text file.

    Parameters
    ----------
    dfs : tuple of pd.DataFrame
        A tuple containing pandas DataFrames from which to compute overlapping EIDs.
        Each DataFrame must contain a column named as specified by `EID_NAME`.

    save_path : str or pathlib.Path
        The directory path where the overlapping EIDs text file will be saved.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If fewer than two DataFrames are provided, or if there are no overlapping EIDs.

    IOError
        If the file cannot be written.
    """
    if len(dfs) < 2:
        raise ValueError(f"Minimum number of DataFrames is 2 ({len(dfs)} given)")

    # Compute overlapping EIDs
    eid_sets = [set(df[EID_NAME]) for df in dfs]
    overlapping_eids = set.intersection(*eid_sets)

    if not overlapping_eids:
        raise ValueError(f"No overlapping '{EID_NAME}'s found among the provided DataFrames.")

    lst_eids = sorted(overlapping_eids, key=lambda x: (isinstance(x, str), x))

    # Ensure the save path exists
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Write the overlapping EIDs to a text file
    output_file = save_path / OVERLAPPING_EID_TXT

    _write_list_to_txt(
        file_path=output_file,
        my_list=[str(eid) for eid in lst_eids],
        verbose=verbose
    )
    return list(lst_eids)


def _write_list_to_txt(file_path: str | Path, my_list: list[str], verbose: bool = True) -> None:
    """
    Write a list of strings to a text file, one string per line.

    Parameters
    ----------
    file_path : str or pathlib.Path
        The path to the text file where the list will be written.

    my_list : list of str
        The list of strings to write to the file.

    verbose : bool, optional
        If True, prints a message indicating that the list has been successfully written.
        Default is True.

    Raises
    ------
    TypeError
        If 'my_list' is not a list of strings.

    IOError
        If the file cannot be written.
    """
    from pathlib import Path

    # Validate that my_list is a list of strings
    if not isinstance(my_list, list):
        raise TypeError(f"'my_list' must be a list, got {type(my_list)}.")
    if not all(isinstance(s, str) for s in my_list):
        raise TypeError("All elements in 'my_list' must be strings.")

    # Ensure the directory exists
    file_path = Path(file_path)
    directory = file_path.parent
    if directory != Path('.'):
        directory.mkdir(parents=True, exist_ok=True)

    try:
        with file_path.open("w", encoding='utf-8') as file:
            for string in my_list:
                file.write(string + '\n')
        if verbose:
            print(f"The list has been successfully written to the file {file_path}.")
    except IOError as e:
        raise IOError(f"Could not write to file {file_path}: {e}")
