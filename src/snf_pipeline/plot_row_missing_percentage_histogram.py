from src.snf_pipeline.save_figure import save_figure


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


import os


def plot_row_missing_percentage_histogram(row_missing_percentage: pd.DataFrame, th_nan: float, modality_name: str,
                                          save_path: str, col_name="missing_percentage", verbose: bool = False) -> None:
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
    save_figure(fig, os.path.join(save_path, modality_name), plt_close=True, verbose=verbose)