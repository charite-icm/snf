import numpy as np
import os
from pathlib import Path 
import pandas as pd
from sklearn.cluster import spectral_clustering
from typing import Any

from src.snf_pipeline.check_validity_loaded_data import check_validity_loaded_data
from src.snf_pipeline.remove_rows_above_missing_threshold import remove_rows_above_missing_threshold
from src.snf_pipeline.plot_row_missing_percentage_histogram import plot_row_missing_percentage_histogram
from src.snf_pipeline.get_overlapping_modalities import get_overlapping_modalities
from src.snf_pipeline.save_overlapping_eids import save_overlapping_eids
from src.snf_pipeline.convert_df_to_np import convert_df_to_np
from src.snf_pipeline.set_affinity_matrix_parameters import set_affinity_matrix_parameters
from src.snf_pipeline.compute_aff_networks import compute_aff_networks
from src.snf_pipeline.get_optimal_cluster_size import get_optimal_cluster_size
from src.snf_pipeline.save_cluster_eids import save_cluster_eids
from src.snf_pipeline.plot_silhouette_score import plot_silhouette_score
from src.snf_pipeline.plot_ordered_affinity_matrix import plot_ordered_affinity_matrix
from src.snf_pipeline.plot_edge_contribution import plot_edge_contribution

from src.snf_package.compute import DistanceMetric, snf, get_n_clusters_revised, get_n_clusters


DIR_HISTOGRAM = "histogram_missing_percentage"
DIR_FUSED = "fused"
DIR_UPSET = "upset_plots"


class Snf:
    """
    The `Snf` class implements functionality for Similarity Network Fusion (SNF) on multiple dataframes representing 
    different modalities.

    Attributes:
        dfs (tuple[pd.DataFrame]): A tuple of dataframes representing input modalities.
        mod_names (tuple[str]): A tuple of strings representing names of the modalities.
        save_path (str): The directory path for saving results. Default is 'results'.
        plot_missing_percentage (bool): Flag to indicate if missing data percentages should be plotted. Default is True.
        th_nan (float): Threshold for the maximum allowable proportion of missing data. Default is 0.3.
        random_state (int): Random seed for reproducibility. Default is 41.
        n_clusters (int | None): The number of clusters for clustering, if specified. Default is None. If None, the eigengap
        heuristic will be used to calculate the most optimal number of clusters.
        verbose (bool): Flag to print the result of intermediate steps. Default is True.

    Methods:
        set_affinity_matrix_param(K=0.1, mu=0.5, metric=DistanceMetric):
            Sets parameters for constructing affinity matrices.
        set_iterative_algorithm_param(...):
            Configures parameters for the iterative part of the SNF algorithm.
        _create_save_dirs():
            Creates directories for saving outputs if they do not already exist.
        ...
    """
    def __init__(self,
                 dfs: tuple[pd.DataFrame],
                 mod_names: tuple[str],
                 save_path: str = "results",
                 plot_missing_percentage = True, 
                 th_nan: float = 0.3,
                 random_state: int = 41,
                 n_clusters: int | None = None,
                 verbose: bool = True) -> None:
        self.dfs = dfs
        self.mod_names = mod_names
        self.save_path = save_path
        self.plot_missing_percentage = plot_missing_percentage
        self.th_nan = th_nan
        self.random_state = random_state
        self.n_clusters = n_clusters
        self.verbose = verbose

        self.set_affinity_matrix_param()
        self.set_iterative_algorithm_param()
        self._create_save_dirs()
        
    def set_affinity_matrix_param(self,
                                  K: float = 0.1,
                                  mu: float = 0.5,
                                  metric: DistanceMetric = DistanceMetric.SQEUCLIDEAN,
                                  normalize: bool = True) -> None:
        """
        K : (0, N) int, optional
            Number of neighbors to consider when creating affinity matrix. See
            `Notes` of :py:func`snf.compute.affinity_matrix` for more details.
            Default: 20
        mu : (0, 1) float, optional
            Normalization factor to scale similarity kernel when constructing
            affinity matrix. See `Notes` of :py:func`snf.compute.affinity_matrix`
            for more details. Default: 0.5
        metric (DistanceMetric): The distance metric to be used for measuring similarity between data points. 
            The default is SQEUCLIDEAN.
        normalize : bool, optional
            Whether to normalize (i.e., zscore) `arr` before constructing the
            affinity matrix. Each feature (i.e., column) is normalized separately.
            Default: True
        """
        self.K = K
        self.mu = mu
        self.metric = metric.value
        self.normalize = normalize

    def set_iterative_algorithm_param(self,
                                      t: int = 20,
                                      alpha: float = 1.0,
                                      top_k: int = 20,
                                      edge_th: float = 1.1) -> None:
        """
        Configures the parameters for the iterative part of the SNF algorithm, which fuses the networks.

        Parameters:
            t (int): The number of iterations for the network fusion process. Default is 20.
            alpha (float): A parameter that scales the influence of the similarity matrices during iteration. 
                           Default is 1.0.
            top_k (int): The number of top similar connections retained for each data point during fusion. 
                         Default is 20.
            edge_th (float): The threshold for determining whether an edge is significant in the network.
                             Used for plotting upsetplots.
                             Default is 1.1.

        Returns:
            None
        """
        self.t = t
        self.alpha = alpha
        self.top_k = top_k
        self.edge_th = edge_th

    def _create_save_dirs(self) -> None:
        self.save_path_histogram = os.path.join(self.save_path, DIR_HISTOGRAM)
        self.save_path_fused = os.path.join(self.save_path, DIR_FUSED)
        self.save_path_upset = os.path.join(self.save_path, DIR_UPSET)

        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.save_path_histogram, exist_ok=True)
        os.makedirs(self.save_path_fused, exist_ok=True)
        os.makedirs(self.save_path_upset, exist_ok=True)
    
    def _print_line(self) -> None:
        if self.verbose:
            print("-" * 40)

    def main(self) -> None:
        self._validate_data()
        dfs_after_th_nan = self._remove_high_missing_data()
        overlapped_eids, np_arrs = self._process_modalities(dfs_after_th_nan)
        param = self._prepare_affinity_parameters(np_arrs)        
        affinity_networks = self._compute_affinity_networks(np_arrs, param)
        fused_network = self._fuse_affinity_networks(param, affinity_networks)
        n_clusters = self._determine_optimal_clusters(fused_network)
        fused_labels = self._preform_clustering(overlapped_eids, fused_network, n_clusters)
        self._generate_plots_and_results(affinity_networks, fused_network, fused_labels)

    def _validate_data(self) -> None:
        check_validity_loaded_data(dfs=self.dfs)

    def _remove_high_missing_data(self) -> tuple[pd.DataFrame]:
        dfs_after_th_nan = []
        for df, modality_name in zip(self.dfs, self.mod_names):
            df_cleaned, row_missing_percentage = remove_rows_above_missing_threshold(df, th_nan=self.th_nan, verbose=self.verbose)
            dfs_after_th_nan.append(df_cleaned)

            # Plot missing percentage histogram
            if self.plot_missing_percentage:
                plot_row_missing_percentage_histogram(row_missing_percentage, self.th_nan, modality_name, self.save_path_histogram, verbose=self.verbose)
        dfs_after_th_nan = tuple(dfs_after_th_nan)
        return dfs_after_th_nan
    
    def _process_modalities(self, dfs_after_th_nan: tuple[pd.DataFrame]) -> tuple[list[int], tuple[np.ndarray, ...]]:
        dfs_after_modality_intersection = get_overlapping_modalities(dfs_after_th_nan)
        overlapped_eids = save_overlapping_eids(dfs=dfs_after_modality_intersection, save_path=self.save_path_histogram, verbose=self.verbose)    
        np_arrs = convert_df_to_np(dfs_after_modality_intersection)
        return overlapped_eids, np_arrs
    
    def _prepare_affinity_parameters(self, np_arrs: tuple[np.ndarray]) -> dict[str, Any]:
        n = int(np_arrs[0].shape[0])
        param = set_affinity_matrix_parameters(n=n, metric=self.metric, K=self.K, mu=self.mu, normalize=self.normalize, th_nan=self.th_nan)
        self._print_line()
        if self.verbose:
            print(param)
        self._print_line()
        return param
    
    def _compute_affinity_networks(self, np_arrs: tuple[np.ndarray], param: dict[str, Any]) -> list[np.ndarray]:
        affinity_networks = compute_aff_networks(np_arrs, param=param, verbose=self.verbose)
        self._print_line()
        if self.verbose:
            print(f"{len(affinity_networks)} affinity matrices generated")
        return affinity_networks
    
    def _fuse_affinity_networks(self, param: dict[str, Any], affinity_networks: list[np.ndarray]) -> np.ndarray:
        fused_network = snf(affinity_networks, K=param["K_actual"], t=self.t, alpha=self.alpha)
        self._print_line()
        if self.verbose:
            print(f"Fused matrix with shape {fused_network.shape} generated.")
        return fused_network
    
    def _determine_optimal_clusters(self, fused_network: np.ndarray) -> int:
        nb_clusters, eigenvalues, eigenvectors = get_n_clusters_revised(fused_network, plot_path=os.path.join(self.save_path_fused, "eigenvalues.png"), 
                                                                        top_k=self.top_k, verbose=self.verbose)
        # _print_line(self.verbose)
        # if self.verbose:
        #     print(f"Optimal number of clusters: {nb_clusters}") 
        nb_clusters = get_n_clusters(fused_network, n_clusters=range(2, self.top_k))
        if self.verbose:
            print(f"Optimal number of clusters: {nb_clusters}")
        n_clusters = get_optimal_cluster_size(n_clusters=self.n_clusters, nb_clusters=list(nb_clusters))
        
        self._print_line()
        if self.verbose:
            print(f"n_cluster = {n_clusters}")
        self._print_line()
        return n_clusters

    def _preform_clustering(self, overlapped_eids: list[int], fused_network: np.ndarray, n_clusters: int) -> list[int]:
        fused_labels = spectral_clustering(fused_network, n_clusters=n_clusters, random_state=self.random_state)
        if self.verbose:
            print("Spectral clustering on fused matrix.")
            # print(fused_labels)
            print(np.unique(fused_labels, return_counts=True))
            # print(len(fused_labels))
            # print(len(overlapped_eids))
        
        self._print_line()
        save_cluster_eids(eids=overlapped_eids, labels=fused_labels, save_path=self.save_path_fused, verbose=self.verbose)
        self._print_line()
        return fused_labels
    
    def _generate_plots_and_results(self, affinity_networks: tuple[np.ndarray], fused_network: np.ndarray, fused_labels: list[int]) -> None:
        self._print_line()
        plot_silhouette_score(fused_network=fused_network, save_path=self.save_path_fused, verbose=self.verbose)
        self._print_line()

        dynamic_range = plot_ordered_affinity_matrix(network=fused_network,
                                                     labels=fused_labels,
                                                     figure_path=os.path.join(self.save_path_fused, "aff_fused"),
                                                     title=None,
                                                     dynamic_range_th=(0.3, 0.3),
                                                     figsize=(8.0, 8.0),
                                                     show_colorbar=False,
                                                     plt_close=True,
                                                     dynamic_range=None,
                                                     return_dynamic_range=True,
                                                     show_axis=False,
                                                     verbose=self.verbose
                                                     )

        for modality, aff in zip(self.mod_names, affinity_networks):
            plot_ordered_affinity_matrix(network=aff,
                                         labels=fused_labels,
                                         figure_path=os.path.join(self.save_path_fused, f"aff_{modality}"),
                                         title=None,
                                         dynamic_range_th=(0.3, 0.3),
                                         figsize=(8.0, 8.0),
                                         show_colorbar=False,
                                         plt_close=True,
                                         dynamic_range=dynamic_range,
                                         return_dynamic_range=False,
                                         show_axis=False,
                                         verbose=self.verbose
                                         )
        self._print_line()
        plot_edge_contribution(labels=fused_labels,
                               modality_names=self.mod_names,
                               affinity_networks=affinity_networks,
                               save_path=self.save_path_upset,
                               edge_th=self.edge_th,
                               verbose=self.verbose)
        self._print_line()
