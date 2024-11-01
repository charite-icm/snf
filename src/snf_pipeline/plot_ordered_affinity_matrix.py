from src.snf_pipeline.save_figure import save_figure


import matplotlib.pyplot as plt
import numpy as np


from pathlib import Path


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
                                 verbose: bool = False,
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