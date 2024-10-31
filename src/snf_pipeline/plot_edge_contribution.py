import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations


from upsetplot import plot as upsplot
from src.snf_pipeline.save_figure import save_figure


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

