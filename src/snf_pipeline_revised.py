import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


EID_NAME = "eid"
OVERLAPPING_EID_TXT = "overlapping_eid.txt"


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




def _write_list_to_txt(file_path: str, my_list: list[str], verbose: bool = True) -> None:
    """
    Write a list of strings to a text file, one string per line.

    This function opens a file at the specified path and writes each string from the provided list
    to the file, with each string on a new line. If the directories in the path do not exist,
    they are created automatically.

    Parameters
    ----------
    file_path : str
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

    Examples
    --------
    >>> my_strings = ["line 1", "line 2", "line 3"]
    >>> _write_list_to_txt("output.txt", my_strings, verbose=True)
    The list has been successfully written to the file output.txt.

    Notes
    -----
    - If the directory specified in 'file_path' does not exist, it will be created.
    - The file is written using UTF-8 encoding.

    """

    # Validate that my_list is a list of strings
    if not isinstance(my_list, list):
        raise TypeError(f"'my_list' must be a list, got {type(my_list)}.")
    if not all(isinstance(s, str) for s in my_list):
        raise TypeError("All elements in 'my_list' must be strings.")

    # Ensure the directory exists
    directory = os.path.dirname(file_path)
    if directory != '':
        os.makedirs(directory, exist_ok=True)

    try:
        with open(file_path, "w", encoding='utf-8') as file:
            for string in my_list:
                file.write(string + '\n')
        if verbose:
            print(f"The list has been successfully written to the file {file_path}.")
    except IOError as e:
        raise IOError(f"Could not write to file {file_path}: {e}")







def save_overlapping_eids(dfs: tuple[pd.DataFrame]) -> None:
    pass



def prepare_for_clustering():


    # Probaly just converting DataFrame to numpy array

    ## drop eid column, save them in a separate file (.txt)
    ## to numpy ....
    
    ...


def set_affinity_matrix_parameters():
    # metric, K, mu, normalize (value errors: mu between 0 and 1 ....)
    # different metric is only used if no nan values (th_nan = 0.0) (otherwise euclidean distance is computed)
    ... 



def compute_aff_networks():
    ...


def compute_fused_network():
    ...


def get_n_clusters_revised():
    # if number of clusters not preselected, select from the eigengap metric
    ...


def apply_spectral_clustering_on_fused_network():
    ...

def save_cluster_eids():
    ...


def compute_silhouette_score():
    ...


def plot_ordered_affinity_matrix():
    ...


def plot_upset():
    # Ensure the version of upset library (0.9.0)
    ...



# class FoldersEnum(Enum):
#     AFFINITY = "affinity"
#     # VALIDATION = "validation"
#     # OMICS_PROFILE = "omics_profile"
#     # VALIDATION_VERBOSE = "validation_verbose"
#     # QUEST_CONT = DataModalityEnum.QUEST_CONT.value
#     # QUEST_BINOM = DataModalityEnum.QUEST_BINOM.value
#     # QUEST_CATEG = DataModalityEnum.QUEST_CATEG.value
#     # DIAG = DataModalityEnum.DIAG.value
#     # SYMP = DataModalityEnum.SYMP.value
#     # MED_VIT = DataModalityEnum.MED_VIT.value


    # def make_dirs(self) -> None:
    #     self.mod_paths: dict[DataModalityEnum: str] = {}
    #     for modality in self.modalities:
    #         self.mod_paths[modality] = os.path.join(self.result_dir_name, modality.value)
    #     self.fused_path = os.path.join(self.result_dir_name, "fused")

    #     all_paths = [*[path for _, path in self.mod_paths.items()], self.fused_path]
    #     for path in all_paths:
    #         if not os.path.exists(os.path.join(RESULTS_PATH, path)):
    #             os.mkdir(os.path.join(RESULTS_PATH, path))

    #         for folder_enum in FoldersEnum:
    #             plot_path = os.path.join(RESULTS_PATH, path, folder_enum.value)
    #             if not os.path.exists(plot_path):
    #                 os.mkdir(plot_path)


    # def prepare_for_clustering(self) -> None:
    #     self.np_arrays: dict[DataModalityEnum: np.ndarray] = {}
    #     if self.th_nan is None:
    #         self._prepare_for_clustering_no_nan()
    #     else:
    #         self._prepare_for_clustering_with_nan()
    #     print("------------ NP SHAPES ---------------------")
    #     for mod, arr in self.np_arrays.items():
    #         print(f"{mod} shape: {arr.shape}")

    # def _prepare_for_clustering_no_nan(self):
    #     self.before_clust: dict[DataModalityEnum: BeforeClustering] = {}
    #     for modality in self.modalities:
    #         log_scale = False
    #         if "prot" in modality.name.lower():
    #             log_scale = True
    #         before_clust = BeforeClustering(df_train=self.dfs_overlapped[modality],
    #                                         df_val=self.dfs_overlapped[DataModalityEnum.VALIDATION],
    #                                         save_folder=self.mod_paths[modality],
    #                                         val_y_enum=self.val_y_enum, val_x_enum=self.val_x_enum,
    #                                         reduce_dim=None, pca_variance_to_keep=None, log_scale=log_scale)
    #         before_clust.main()
    #         self.before_clust[modality] = before_clust
    #         self.np_arrays[modality] = before_clust.get_np_train()
    #     self.np_arrays[DataModalityEnum.VALIDATION] = before_clust.get_np_val()
    #     self.eids = before_clust.get_eids()
    #     self.val_x_feature_name = self.before_clust[self.modalities[0]].get_val_x_feature_name()
    #     self.val_y_feature_name = self.before_clust[self.modalities[0]].get_val_y_feature_name()
    #     # We only need it for the path to save cluster plots
    #     self.before_clust_concat = BeforeClustering(df_train=self.df_concat,
    #                                                 df_val=self.dfs_overlapped[DataModalityEnum.VALIDATION],
    #                                                 save_folder=self.fused_path,
    #                                                 val_y_enum=self.val_y_enum, val_x_enum=self.val_x_enum,
    #                                                 reduce_dim=None, pca_variance_to_keep=None)
    #     self.before_clust_concat.main()

    # def _prepare_for_clustering_with_nan(self):
    #     for modality in ([*self.modalities, DataModalityEnum.VALIDATION]):
    #         if self.eids is None:
    #             self.eids = self.dfs_overlapped[modality]["eid"].tolist()

    #         data_arr = self.dfs_overlapped[modality].drop(columns=["eid"]).to_numpy()
    #         if "prot" in modality.name.lower():
    #             data_arr = 2**data_arr

    #         self.np_arrays[modality] = np.array(data_arr)
    #     # self.K = 20
    #     self.K = len(self.eids) // 10