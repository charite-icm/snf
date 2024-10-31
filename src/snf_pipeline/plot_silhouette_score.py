from src.snf_pipeline.save_figure import save_figure
from src.snf_pipeline.constants import SILHOUETTE_SCORE_FIG_NAME


import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import spectral_clustering
from sklearn.metrics import silhouette_score


import os
from pathlib import Path


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