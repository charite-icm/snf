import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import numbers
from itertools import combinations

from sklearn.cluster import spectral_clustering
from sklearn.metrics import silhouette_score
from upsetplot import plot as upsplot


from typing import Any, Callable

from src.snf_package.compute import DistanceMetric
from src.snf_package.compute import make_affinity, make_affinity_nan, check_symmetric




EID_NAME = "eid"
CLUSTER_NAME = "cluster"

OVERLAPPING_EID_TXT = "overlapping_eids.txt"
CLUSTER_EIDS_CSV = "cluster_eids.csv"
SILHOUETTE_SCORE_FIG_NAME = "silhouette_score"


def _check_validity_loaded_data(dfs: tuple[pd.DataFrame]) -> None:
    """
    Validate a tuple of pandas DataFrames to ensure they meet the required conditions.

    This function checks whether the provided tuple of DataFrames meets the following criteria:
    1. The tuple contains at least two DataFrames.
    2. Each element in the tuple is a pandas DataFrame.
    3. Each DataFrame contains a column named 'eid'.
    4. The 'eid' column in each DataFrame contains unique values (i.e., no duplicates).

    Parameters
    ----------
    dfs : tuple[pd.DataFrame]
        A tuple of pandas DataFrames to be validated.

    Raises
    ------
    ValueError
        If the tuple contains fewer than two DataFrames, if a DataFrame does not contain 
        the 'eid' column, or if the 'eid' column contains duplicate values.
    
    TypeError
        If any element in the tuple is not a pandas DataFrame.
    """

    if not isinstance(dfs, tuple):
        raise TypeError(f"dfs should be of type tuple not {type(dfs)}")
    
    if len(dfs) < 2:
        raise ValueError(f"Minimum number of data modalities is 2. {len(dfs)} provided")
    
    for df in dfs:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Invalid type of data: {type(df)}. It should be pandas.DataFrame")
        if EID_NAME not in df.columns:
            raise ValueError(f"Column with IDs should be named {EID_NAME}")
        if df[EID_NAME].duplicated().any():
            raise ValueError(f"Column {EID_NAME} does not contain unique values.")


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


def plot_row_missing_percentage_histogram(row_missing_percentage: pd.DataFrame, th_nan: float, modality_name: str,
                                          save_path: str, col_name="missing_percentage") -> None:
    """
    Plot a histogram to visualize the percentage of missing data per row in the given DataFrame and save the plot.

    This function creates a histogram showing the percentage of missing data per row for a specified column in the 
    input DataFrame (`row_missing_percentage`). A vertical line is added at the threshold percentage to visualize 
    the threshold used for missing data. The plot is saved to a specified path with the given modality name.

    Parameters
    ----------
    row_missing_percentage : pd.DataFrame
        A DataFrame containing the percentage of missing values per row for each modality. The column name should 
        be specified by the `col_name` parameter.
    
    th_nan : float
        The threshold for the percentage of missing values, represented as a float between 0 and 1 (e.g., 0.3 for 30%).
        A vertical line will be drawn at the corresponding percentage in the histogram to indicate this threshold.
    
    modality_name : str
        The name of the modality being analyzed, which will be included in the plot title and the file name.
    
    save_path : str
        The directory where the generated plot should be saved. The plot will be saved with the file name 
        as `modality_name` in the specified path.
    
    col_name : str, optional
        The name of the column in the `row_missing_percentage` DataFrame that contains the percentage of missing values. 
        The default is "missing_percentage".

    Raises
    ------
    ValueError
        If the specified `col_name` is not present in the `row_missing_percentage` DataFrame.

    Returns
    -------
    None
        The function saves the plot to the specified path and does not return any value.

    Example
    -------
    >>> df = pd.DataFrame({"missing_percentage": [10, 20, 30, 40, 50]})
    >>> plot_row_missing_percentage_histogram(df, 0.3, "Modality1", "/path/to/save", col_name="missing_percentage")
    
    This will create a histogram of the percentages in `df["missing_percentage"]`, draw a vertical line at 30% (since 
    `th_nan=0.3`), and save the plot as "Modality1.jpg" in the `/path/to/save` directory.
    """
    if col_name not in row_missing_percentage.columns:
        raise ValueError(f"{col_name} not in row_missing_percentage data frame")
    
    
    fig, ax = plt.subplots()
    ax.set_title(f"Amount of missing data per row - {modality_name}", fontweight="bold", fontsize=10)
    histplot = sns.histplot(row_missing_percentage, bins=20, kde=False, color='skyblue', ax=ax)
    histplot.set(xlabel="% of missing data")
    ax.bar_label(histplot.containers[0], fmt="%d", label_type="edge", fontsize=8, color="black", weight="bold",
                    labels=[str(v) if v else '' for v in histplot.containers[0].datavalues])

    ax.axvline(x=th_nan*100, color="red", linestyle="--", label="th_nan")
    ax.legend()
    save_figure(fig, os.path.join(save_path, modality_name), plt_close=True)


def save_figure(fig, fig_name: str, plt_close: bool = False, img_formats: tuple[str] = (".jpg", ),
                dpi: int = 300, verbose: bool = True) -> None:
    """
    Save a matplotlib figure in one or more image formats and optionally close the plot.

    This function saves the given `fig` (a matplotlib figure) to the file system in the specified image format(s) with
    the given DPI (dots per inch) resolution. The function can also optionally close the figure after saving it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to save.
    
    fig_name : str
        The base file name (without extension) for the figure. The file extension(s) will be determined by `img_formats`.
    
    plt_close : bool, optional
        Whether to close the figure after saving. If `True`, `plt.close()` will be called. Default is `False`.

    img_formats : tuple of str, optional
        A tuple of strings specifying the file formats in which to save the figure (e.g., ".jpg", ".png"). 
        Default is `(".jpg",)`.

    dpi : int, optional
        The resolution in dots per inch for the saved figure. Default is 300.

    verbose : bool, optional
        Whether to print a message after each file is saved. If `True`, a message will be printed. Default is `True`.

    Raises
    ------
    None

    Returns
    -------
    None
        The function saves the figure to the specified file format(s) and optionally closes the plot.
    
    Example
    -------
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [4, 5, 6])
    >>> save_figure(fig, "plot", img_formats=(".png", ".pdf"), dpi=200, verbose=True, plt_close=True)
    
    This will save the figure as "plot.png" and "plot.pdf" with 200 DPI, print a message for each saved file, and 
    close the figure after saving.
    """
    for img_format in img_formats:
        plt.tight_layout()
        fig_full_name = fig_name + img_format
        fig.savefig(fig_full_name, dpi=dpi)
        if verbose:
            print(f"{fig_full_name} saved!")
        if plt_close:
            plt.close()


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


def set_affinity_matrix_parameters(
    n: int,
    metric: str | list[str] = 'sqeuclidean',
    K: float = 0.1,
    mu: float = 0.5,
    normalize: bool = True,
    th_nan: float = 0.0
) -> dict[str, Any]:
    """
    Set and validate parameters for computing the affinity matrix.

    Parameters
    ----------
    n : int
        The number of cases (samples) in the dataset. Must be a positive integer.

    metric : str or list of str, optional
        Distance metric to compute. Must be one of the available metrics in
        `scipy.spatial.distance.cdist`. If multiple arrays are provided,
        an equal number of metrics may be supplied. Default is 'sqeuclidean'.

    K : float, optional
        Proportion of neighbors to consider when creating the affinity matrix.
        Must be a float between 0.0 and 1.0 (exclusive). The actual number of neighbors,
        `K_actual`, is computed as `int(K * n)`. Default is 0.1.

    mu : float, optional
        Normalization factor to scale the similarity kernel when constructing
        the affinity matrix. Must be between 0.0 and 1.0 (exclusive). Default is 0.5.

    normalize : bool, optional
        Whether to normalize (i.e., z-score) the data before constructing the
        affinity matrix. Each feature (i.e., column) is normalized separately.
        Default is True.

    th_nan : float, optional
        Threshold for handling missing data (NaNs). Must be between 0.0 and 1.0 (inclusive).
        Default is 0.0.

    Returns
    -------
    params : dict
        Dictionary of validated parameters for affinity matrix computation, including 'K_actual'.

    Raises
    ------
    ValueError
        If parameters are not within acceptable ranges or if invalid combinations
        of parameters are provided.

    TypeError
        If parameters are of incorrect types.

    Notes
    -----
    - Different metrics can only be selected if `th_nan` equals 0.0.
      Otherwise, Euclidean distance ('sqeuclidean') is used regardless of the `metric` parameter.
    - The list of valid metrics corresponds to those available in `scipy.spatial.distance.cdist`.
    - The actual number of neighbors used is computed as `K_actual = int(K * n)`.
      `K_actual` must be at least 1 and less than `n`.

    Examples
    --------
    >>> params = set_affinity_matrix_parameters(n=100, metric='cosine', K=0.15, mu=0.7, normalize=False, th_nan=0.0)
    >>> params
    {'n': 100, 'metric': 'cosine', 'K': 0.15, 'K_actual': 15, 'mu': 0.7, 'normalize': False, 'th_nan': 0.0}
    """
    n = _validate_n(n)
    th_nan = _validate_th_nan(th_nan)
    metric = _validate_metric(metric, th_nan)
    K_actual = _validate_K(K, n)
    mu = _validate_mu(mu)
    normalize = _validate_normalize(normalize)
    
    params = {
        'n': n,
        'metric': metric,
        'K': K,
        'K_actual': K_actual,
        'mu': mu,
        'normalize': normalize,
        'th_nan': th_nan
    }

    return params


def _validate_n(n: int) -> int:
    if not isinstance(n, int):
        raise TypeError(f"'n' must be an integer, got {type(n)}.")
    if n <= 0:
        raise ValueError(f"'n' must be a positive integer, got {n}.")
    return n


def _validate_th_nan(th_nan: float) -> float:
    if not isinstance(th_nan, numbers.Number):
        raise TypeError(f"'th_nan' must be a float between 0.0 and 1.0, got {type(th_nan)}.")
    if not (0.0 <= th_nan <= 1.0):
        raise ValueError(f"'th_nan' must be a float between 0.0 and 1.0, got {th_nan}.")
    return th_nan


def _validate_metric(metric: str | list[str], th_nan: float) -> str | list[str]:
    if th_nan != 0.0:
        # If th_nan != 0.0, metric must be 'sqeuclidean'
        if metric != 'sqeuclidean':
            # Warn the user and set metric to 'sqeuclidean'
            print(f"Since 'th_nan' != 0.0, metric is set to 'sqeuclidean'.")
            metric = 'sqeuclidean'
    else:
        # Validate metric against DistanceMetric enum
        valid_metrics = [e.value for e in DistanceMetric]
        if isinstance(metric, str):
            if metric not in valid_metrics:
                raise ValueError(f"Invalid metric '{metric}'. Must be one of {valid_metrics}.")
        elif isinstance(metric, list):
            if not all(isinstance(m, str) for m in metric):
                raise TypeError("All elements in 'metric' list must be strings.")
            invalid_metrics = [m for m in metric if m not in valid_metrics]
            if invalid_metrics:
                raise ValueError(f"Invalid metrics {invalid_metrics}. Must be one of {valid_metrics}.")
        else:
            raise TypeError(f"'metric' must be a string or list of strings, got {type(metric)}.")
    return metric


def _validate_K(K: float, n: int) -> int:
    if not isinstance(K, numbers.Number):
        raise TypeError(f"'K' must be a number between 0.0 and 1.0, got {type(K)}.")
    if not (0.0 < K <= 1.0):
        raise ValueError(f"'K' must be a float between 0.0 and 1.0 (exclusive), got {K}.")
    K_actual = int(K * n)
    if K_actual < 1:
        K_actual = 1
    elif K_actual >= n:
        K_actual = n - 1
    return K_actual


def _validate_mu(mu: float) -> float:
    if not isinstance(mu, numbers.Number):
        raise TypeError(f"'mu' must be a number, got {type(mu)}.")
    if not (0.0 < mu < 1.0):
        raise ValueError(f"'mu' must be a float between 0.0 and 1.0 (exclusive), got {mu}.")
    return mu


def _validate_normalize(normalize: bool) -> bool:
    if not isinstance(normalize, bool):
        raise TypeError(f"'normalize' must be a boolean, got {type(normalize)}.")
    return normalize


def compute_aff_networks(arrs: tuple[np.ndarray], param: dict[str, Any]) -> tuple[np.ndarray]:
    """
    Compute affinity networks based on input arrays using either standard or 
    nan-aware affinity functions, and normalize the resulting affinity matrices.

    Parameters
    ----------
    arrs : tuple of np.ndarray
        A tuple of numpy arrays, where each array represents a set of features 
        for which affinity matrices will be computed. The arrays may represent 
        different data modalities or datasets.
        
    param : dict of {str: Any}
        A dictionary containing parameters for computing the affinity matrices.
        It should include the following keys:
        - 'metric' (str or list of str): The distance metric(s) used to compute the 
          affinity matrices (e.g., 'euclidean', 'cosine'). If multiple arrays are 
          provided in `arrs`, an equal number of metrics can be specified.
        - 'K_actual' (int): The number of nearest neighbors to consider when 
          constructing the affinity matrices.
        - 'mu' (float): A scaling factor to normalize the similarity kernel when 
          constructing the affinity matrix.
        - 'normalize' (bool): Whether to normalize each feature in the input 
          arrays before computing affinity matrices.
        - 'th_nan' (float): A threshold for handling missing values (NaNs). If 
          this value is non-zero, a nan-aware affinity function will be used 
          instead of the standard affinity function.

    Returns
    -------
    tuple of np.ndarray
        A tuple of normalized and symmetric affinity matrices corresponding to 
        each array in `arrs`. Each matrix represents the affinity network for 
        that dataset, computed using the specified parameters.

    Notes
    -----
    - If `th_nan` in the `param` dictionary is non-zero, missing values (NaNs) 
      are handled using `make_affinity_nan`, otherwise `make_affinity` is used.
    - Each affinity matrix is normalized row-wise to ensure that the sum of 
      similarities for each sample is equal to 1. This is achieved by dividing 
      each element in the row by the sum of that row.
    - Symmetry is enforced on each affinity matrix using `check_symmetric`, 
      which ensures that the matrices are symmetric and issues a warning if 
      necessary.

    Example
    -------
    >>> arr1 = np.random.rand(100, 10)
    >>> arr2 = np.random.rand(100, 15)
    >>> param = {
            'metric': 'euclidean',
            'K_actual': 10,
            'mu': 0.5,
            'normalize': True,
            'th_nan': 0.0
        }
    >>> aff_matrices = compute_aff_networks((arr1, arr2), param)
    >>> print(aff_matrices[0].shape)  # Output: (100, 100)

    """
    func_: Callable =  make_affinity
    if param["th_nan"] != 0.0:
        func_ = make_affinity_nan

    affinity_networks = func_(*arrs,
                              metric=param["metric"], K=param["K_actual"],
                              mu=param["mu"], normalize=param["normalize"])

    # Normalize each affinity matrix by the row sum
    affinity_networks = [w / np.nansum(w, axis=1, keepdims=True) for w in affinity_networks]

    # Ensure each matrix is symmetric
    affinity_networks = [check_symmetric(w, raise_warning=False) for w in affinity_networks]

    return tuple(affinity_networks)


def get_optimal_cluster_size(n_clusters: int | None, nb_clusters: list[int]) -> int:
    """
    Determine the number of clusters to use based on input parameters.

    Parameters
    ----------
    n_clusters : int or None
        If an integer is provided, it specifies the number of clusters to use.
        If None, the function will determine the number of clusters based on 
        the `nb_clusters` list derived from eigengap heuristic.

    nb_clusters : list of int
        A list of integers representing possible cluster counts, ordered by 
        relevance. The first element is considered the primary choice.

    Returns
    -------
    int
        The chosen number of clusters based on the provided parameters.

    Raises
    ------
    ValueError
        If `nb_clusters` is empty or does not contain at least two elements 
        when `n_clusters` is None.

    Examples
    --------
    >>> get_optimal_cluster_size(3, [2, 5])
    3

    >>> get_optimal_cluster_size(None, [2, 5])
    2

    >>> get_optimal_cluster_size(None, [1, 5])
    5
    """
    # Case 1: Return n_clusters directly if it is provided
    if n_clusters is not None:
        if not isinstance(n_clusters, int) or n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer if provided.")
        return n_clusters

    # Case 2: Handle the situation when n_clusters is None
    if not isinstance(nb_clusters, list) or len(nb_clusters) < 2:
        raise ValueError("nb_clusters must be a list with at least two elements when n_clusters is None.")

    # Primary selection logic
    if nb_clusters[0] != 1:
        return nb_clusters[0]
    return nb_clusters[1]


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


def plot_silhouette_score(fused_network: np.ndarray, save_path: str | Path, n_clusters_end: int = 20, verbose: bool = True) -> None:
    """
    Plot the silhouette score for a range of clusters based on a fused network.

    Parameters
    ----------
    fused_network : np.ndarray
        The affinity or similarity matrix for clustering.
    save_path : str or Path
        Directory path where the plot will be saved.
    n_clusters_end : int, optional
        The maximum number of clusters to evaluate, starting from 2 up to `n_clusters_end`.
        Default is 20.
    verbose : bool, optional
        If True, prints a confirmation message after saving the plot. Default is True.

    Raises
    ------
    ValueError
        If `n_clusters_end` is less than 2.
    FileNotFoundError
        If `save_path` does not exist.

    Example
    -------
    >>> plot_silhouette_score(fused_network, "/path/to/save", n_clusters_end=15)
    "/path/to/save/silhouette_score.png saved!"
    """
    # Check for valid n_clusters_end
    if n_clusters_end < 2:
        raise ValueError("`n_clusters_end` must be at least 2.")

    # Ensure `save_path` exists and is a directory
    save_path = Path(save_path)
    if not save_path.exists() or not save_path.is_dir():
        raise FileNotFoundError(f"The specified directory {save_path} does not exist.")
    
    x, y = [], []
    fig, ax = plt.subplots()
    aff_matrix = np.array(fused_network)


    for n_clusters in range(2, n_clusters_end):
        fused_labels = spectral_clustering(aff_matrix, n_clusters=n_clusters)
        # Calculate silhouette score
        np.fill_diagonal(aff_matrix, 0)
        silhouette_avg = silhouette_score(aff_matrix, fused_labels, metric="precomputed")

        x.append(n_clusters)
        y.append(silhouette_avg)

    # Plot silhouette scores
    ax.plot(x, y, marker='o')
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score vs. Number of Clusters", fontweight="bold")

    # Define full path for the plot
    save_figure_path = os.path.join(save_path, SILHOUETTE_SCORE_FIG_NAME)
    save_figure(fig, fig_name=save_figure_path, plt_close=True, verbose=verbose)


def plot_ordered_affinity_matrix(network: np.ndarray,
                                 labels: list[int],
                                 figure_path: str | Path,
                                 title: str = None,
                                 dynamic_range_th: tuple[float, float] = (0.1, 0.1),
                                 figsize: tuple[float, float] = (8.0, 8.0),
                                 show_colorbar: bool = False,
                                 plt_close: bool = True,
                                 dynamic_range: tuple[float, float] = None,
                                 return_dynamic_range: bool = False,
                                 show_axis: bool = False,
                                 verbose: bool = True,
                                #  high_quality: bool = False
                                 ) -> None | tuple[float, float]:
    """
    Plot an ordered affinity matrix with optional dynamic range adjustment.

    Parameters
    ----------
    network : np.ndarray
        The affinity matrix to be plotted, a symmetric 2D numpy array.
    labels : list of int
        List of cluster labels for ordering the rows and columns of the matrix.
    figure_path : str or Path
        Path where the plot will be saved.
    title : str, optional
        Title of the plot.
    dynamic_range_th : tuple of float, optional
        Threshold values to define the dynamic range as (lower, upper) percentages of max similarity.
    figsize : tuple of float, optional
        Size of the figure in inches, default is (8.0, 8.0).
    show_colorbar : bool, optional
        If True, a colorbar is displayed alongside the plot.
    plt_close : bool, optional
        If True, the plot is closed after saving.
    dynamic_range : tuple of float, optional
        Directly sets the minimum and maximum values for the color range. Overrides dynamic_range_th if provided.
    return_dynamic_range : bool, optional
        If True, returns the computed dynamic range (vmin, vmax).
    show_axis : bool, optional
        If False, axes are hidden for a cleaner plot.
    verbose : bool, optional
        If True, a confirmation message is printed after saving the plot.

    Returns
    -------
    tuple of float or None
        Returns (vmin, vmax) if `return_dynamic_range` is True, otherwise returns None.

    Raises
    ------
    ValueError
        If `network` is not a 2D square matrix or if the length of `labels` does not match `network`.
    FileNotFoundError
        If `figure_path` directory does not exist.

    Example
    -------
    >>> plot_ordered_affinity_matrix(network, labels, "output_path/affinity_matrix.png", title="Affinity Matrix")
    "output_path/affinity_matrix.png saved!"
    """

    # Check for valid network matrix dimensions
    if network.ndim != 2 or network.shape[0] != network.shape[1]:
        raise ValueError("`network` must be a square 2D matrix.")
    
    # Check for valid labels length
    if len(labels) != network.shape[0]:
        raise ValueError("Length of `labels` must match the dimensions of `network`.")

    # Ensure figure_path exists
    if not Path(figure_path).parent.exists():
        raise FileNotFoundError(f"The specified directory {figure_path.parent} does not exist.")
    
    
    indexing_array = np.argsort(labels)
    fig, ax = plt.subplots(figsize=figsize)

    visualize_network = np.copy(network)
    np.fill_diagonal(visualize_network, 0)
    visualize_network /= np.nansum(visualize_network, axis=1, keepdims=True)

    max_sim = visualize_network.max()
    mean_sim = visualize_network.mean()

    np.fill_diagonal(visualize_network, 1)
    visualize_network_ordered = visualize_network[indexing_array][:, indexing_array]

    vmin, vmax = mean_sim - dynamic_range_th[0] * max_sim, mean_sim + dynamic_range_th[1] * max_sim
    if dynamic_range is not None:
        vmin, vmax = dynamic_range

    ax.imshow(visualize_network_ordered, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    if title is not None:
        ax.set_title(title, fontweight="bold")
    if show_colorbar:
        ax.colorbar()
    if not show_axis:
        ax.axis("off")

    # if high_quality:
        # save_plot_as_vector(fig, format="pdf", dpi=600, output_path=f"{figure_path}.pdf")
        # save_plot_as_tiff(fig, column_type="double", dpi=600, output_path=f"{figure_path}.tiff")
    save_figure(fig, fig_name=figure_path, plt_close=plt_close, verbose=verbose)

    if return_dynamic_range:
        return vmin, vmax


def plot_edge_contribution(labels: list[int],
                           modality_names: list[str],
                           affinity_networks: tuple[np.ndarray],
                           save_path: str | Path,
                           edge_th: float = 1.1,
                           verbose: bool = False) -> None:
    affinity_networks_ordered = _order_affinity_matrices(labels=labels, 
                                                         modality_names=modality_names, 
                                                         affinity_networks=affinity_networks)
    cluster_weights = _get_list_of_edges(labels=labels, 
                                         affinity_networks_ordered=affinity_networks_ordered,
                                         edge_th=edge_th)
    
    list_multi_index_df = _transform_to_upsetplot_format(cluster_weights=cluster_weights,
                                                         modality_names=modality_names,
                                                         verbose=verbose)
    _plot_upset(list_multi_index_df=list_multi_index_df,
                save_path=save_path,
                verbose=verbose)
    
    

def _order_affinity_matrices(labels: list[int], 
                             modality_names: tuple[str], 
                             affinity_networks: tuple[np.ndarray]) -> dict[str, np.ndarray]:
    """
    Order affinity matrices based on the provided cluster labels.

    Parameters
    ----------
    labels : list of int
        List of cluster labels indicating the ordering of nodes.
    modality_names : tuple of str
        Tuple containing the names of each modality corresponding to the affinity networks.
    affinity_networks : tuple of np.ndarray
        Tuple containing the affinity matrices for each modality. Each matrix must be square.

    Returns
    -------
    dict of str, np.ndarray
        Dictionary with modality names as keys and ordered affinity matrices as values.

    Raises
    ------
    ValueError
        If the length of `modality_names` and `affinity_networks` do not match, or if
        any matrix in `affinity_networks` is not square.
    ValueError
        If the length of `labels` does not match the dimension of the affinity matrices.

    Example
    -------
    >>> labels = [2, 0, 1]
    >>> modality_names = ("Mod1", "Mod2")
    >>> affinity_networks = (np.random.rand(3, 3), np.random.rand(3, 3))
    >>> ordered_matrices = _order_affinity_matrices(labels, modality_names, affinity_networks)
    """
    # Validate input lengths
    if len(modality_names) != len(affinity_networks):
        raise ValueError("The number of modality names must match the number of affinity networks.")
    
    for aff in affinity_networks:
        if aff.shape[0] != aff.shape[1]:
            raise ValueError("All affinity networks must be square matrices.")
        if aff.shape[0] != len(labels):
            raise ValueError("The length of `labels` must match the dimension of the affinity matrices.")
    
    # Order the affinity matrices
    indexing_array = np.argsort(labels)
    affinity_networks_ordered: dict[str: np.ndarray] = {}

    for modality, aff in zip(modality_names, affinity_networks):
        network = np.copy(aff)
        np.fill_diagonal(network, 0)
        # We remove the diagonal elements which could greatly influence the normalization factor
        # (they are not constant on the diagonal)
        network /= np.nansum(network, axis=1, keepdims=True)
        np.fill_diagonal(network, 1)

        # Order rows and columns according to `labels`
        network_ordered = network[indexing_array][:, indexing_array]
        affinity_networks_ordered[modality] = network_ordered

    return affinity_networks_ordered


def _get_list_of_edges(labels: list[int], 
                       affinity_networks_ordered: dict[str, np.ndarray], 
                       edge_th: float = 1.1) -> dict[int, list[list[int]]]:
    """
    Generate a list of edges for each cluster based on similarity values 
    from multiple affinity networks and a specified threshold.

    Parameters
    ----------
    labels : list of int
        List of cluster labels indicating the cluster membership of each element.
    affinity_networks_ordered : dict of str, np.ndarray
        Dictionary with modality names as keys and their corresponding ordered affinity matrices.
    edge_th : float, optional
        Threshold for determining significant differences in similarity values. 
        Default is 1.1.

    Returns
    -------
    dict of int, list of list of int
        Dictionary where each key represents a cluster and the value is a list of lists.
        Each inner list contains the indices of modalities where the similarity value
        is the highest. Multiple modalities can be in the list 
        if max_sim_val/mod_sim_val < edge_th

    Raises
    ------
    ValueError
        If `labels` length does not match the dimension of any matrix in `affinity_networks_ordered`.
    ValueError
        If any matrix in `affinity_networks_ordered` is not square.

    """
    # Validate that the length of `labels` matches the dimensions of the matrices
    for modality, matrix in affinity_networks_ordered.items():
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Affinity matrix for {modality} must be square.")
        if len(labels) != matrix.shape[0]:
            raise ValueError("The length of `labels` must match the dimension of the affinity matrices.")

    _, elements_counts = np.unique(labels, return_counts=True)
    end_idxs = list(np.cumsum(elements_counts))
    start_idxs = [0, *end_idxs[:-1]]

    cluster_weights = {}
    for clust, (st, end) in enumerate(zip(start_idxs, end_idxs)):
        cluster_weights[clust] = []
        for i in range(st, end):
            # for j in range(i, end):
            for j in range(i + 1, end):
                sim_values = np.array([affinity_networks_ordered[mod][i][j] for mod in affinity_networks_ordered])

                max_sim_arg = np.argmax(sim_values)
                percentage = sim_values[max_sim_arg] / np.array(sim_values)

                non_zero_indices = np.nonzero(percentage < edge_th)[0]
                non_zero_indices_list = [int(el) for el in non_zero_indices]
                cluster_weights[clust].append(list(non_zero_indices_list))

    return cluster_weights


def _transform_to_upsetplot_format(cluster_weights: dict[int, list[list[int]]],
                                   modality_names: tuple[str],
                                   verbose: bool = False) -> list[pd.MultiIndex]:
    """
    Transform cluster weights into a format suitable for creating UpSet plots.

    Parameters
    ----------
    cluster_weights : dict of int, list of list of int
        Dictionary where each key represents a cluster, and each value is a list of lists.
        Each inner list contains indices of modalities where the similarity value is below a defined threshold.
    modality_names : tuple of str
        Tuple of modality names corresponding to each index in the cluster weights.
    verbose : bool, optional
        If True, prints detailed information about each cluster's data transformation. Default is True.

    Returns
    -------
    list of pd.DataFrame
        A list of Pandas DataFrames, each containing multi-indexed counts of edges for a specific cluster.

    Raises
    ------
    ValueError
        If `modality_names` is empty or if `cluster_weights` contains data that does not match the length of `modality_names`.

    Example
    -------
    >>> cluster_weights = {0: [[0, 1], [1], [0]], 1: [[2], [0, 2]]}
    >>> modality_names = ("Mod1", "Mod2", "Mod3")
    >>> result = _transform_to_upsetplot_format(cluster_weights, modality_names)
    """
    
    
    # Validate input parameters
    if not modality_names:
        raise ValueError("`modality_names` must not be empty.")
    for key, item in cluster_weights.items():
        if any(max(el) > len(modality_names) for el in item):
            raise ValueError(f"Elements in `cluster_weights` for cluster {key} have indices that exceed `modality_names` length.")
    
    
    
    combination_list = []
    for i in range(1, len(modality_names) + 1):
        combination_list.extend(list(combinations(range(len(modality_names)), i)))
    combination_list = [list(el) for el in combination_list]

    list_multi_index_df = []
    for key, item in cluster_weights.items():
        if verbose:
            print_str = f"Cluster {str(key)}\n----------------\n"
            for i in range(len(combination_list)):
                bool_list = [combination_list[i] == list(el) for el in item]
                print_str += f"{[modality_names[el] for el in combination_list[i]]}: {int(100 * np.sum(bool_list) / len(bool_list))} %\n"

    
        # Create boolean arrays representing modality participation in edges
        list_edges = [[count in el for count in range(len(modality_names))] for el in item]
        

        # Cols are modality names, rows represent all points in affinity matrix within a given cluster
        # Boolean - columns containing True are have the higher similarity value for a given point
        df_edges_all = pd.DataFrame(list_edges, columns=modality_names)
        # Count occurrences of each combination
        df_edges_count = df_edges_all.groupby(list(modality_names)).size()

        list_multi_index_df.append(df_edges_count)

        if verbose:
            print(print_str)
    return list_multi_index_df


def _plot_upset(list_multi_index_df: pd.MultiIndex,
                save_path: str | Path,
                verbose: bool = False) -> None:
    fig_list = []
    for i, df in enumerate(list_multi_index_df):
        df_save = pd.DataFrame(df).rename(columns={0: "Number of edges"})
        df_save["Percentage of edges"] = df_save["Number of edges"] / df_save["Number of edges"].sum() * 100
        df_save.to_csv(os.path.join(save_path, f"upsetplot_clust_{i}.csv"), index=True)

        fig = plt.figure(figsize=(10.0, 6.0))
        upsplot(df, fig=fig, show_percentages=True, element_size=None, sort_categories_by="input")
        fig_list.append(fig)
        save_figure(fig, fig_name=os.path.join(save_path, f"upsetplot_clust_{i}"), plt_close=True, verbose=verbose)


    # TODO: save all upset plots in one figure
    # 13, 8
    # big_fig = plt.figure(figsize=(6.0, 6.0))
    # for i, fig in enumerate(fig_list):
    #     ax = big_fig.add_subplot((len(fig_list) + 1) // 2, 2, i + 1)
    #     ax.set_title(f"Cluster {i + 1}", fontweight="bold")

    #     ax.set_xticks([])  # Remove x-axis ticks
    #     ax.set_yticks([])  # Remove y-axis ticks
    #     ax.imshow(fig.canvas.buffer_rgba(), origin="upper")

    #     ax.spines["top"].set_visible(False)
    #     ax.spines["right"].set_visible(False)
    #     ax.spines["bottom"].set_visible(False)
    #     ax.spines["left"].set_visible(False)

    # figure_path = os.path.join(RESULTS_PATH, self.fused_path, FoldersEnum.AFFINITY.value, f"upsetplot")
    # # save_plot_as_vector(big_fig, format="pdf", dpi=600, output_path=f"{figure_path}.pdf")
    # # save_plot_as_tiff(big_fig, column_type="double", dpi=600, output_path=f"{figure_path}.tiff")
    # save_figure(big_fig, fig_name=f"{figure_path}.png", plt_close=True)

