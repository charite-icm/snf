import numpy as np
import os
from pathlib import Path 
import pandas as pd
from sklearn.cluster import spectral_clustering

from src.snf_pipeline.check_validity_loaded_data import check_validity_loaded_data
from src.snf_pipeline.remove_rows_above_missing_threshold import remove_rows_above_missing_threshold
from src.snf_pipeline.plot_row_missing_percentage_histogram import plot_row_missing_percentage_histogram
from src.snf_pipeline.get_overlapping_modalities import get_overlapping_modalities
from src.snf_pipeline.save_overlapping_eids import save_overlapping_eids
from src.snf_pipeline.convert_df_to_np import convert_df_to_np
from src.snf_package.compute import DistanceMetric, snf, get_n_clusters_revised, get_n_clusters
from src.snf_pipeline.set_affinity_matrix_parameters import set_affinity_matrix_parameters
from src.snf_pipeline.compute_aff_networks import compute_aff_networks
from src.snf_pipeline.get_optimal_cluster_size import get_optimal_cluster_size
from src.snf_pipeline.save_cluster_eids import save_cluster_eids
from src.snf_pipeline.plot_silhouette_score import plot_silhouette_score
from src.snf_pipeline.plot_ordered_affinity_matrix import plot_ordered_affinity_matrix
from src.snf_pipeline.plot_edge_contribution import plot_edge_contribution


DIR_HISTOGRAM = "histogram_missing_percentage"
DIR_FUSED = "fused"
DIR_UPSET = "upset_plots"


class Snf:
    def __init__(self,
                 data_paths: tuple[str | Path],
                 mod_names: tuple[str],
                 save_path: str = "results",
                 plot_missing_percentage = True, 
                 th_nan: float = 0.3,
                 random_state: int = 41,
                 n_clusters: int | None = 6,
                 verbose: bool = True) -> None:
        self.data_paths = data_paths
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
        self.K = K
        self.mu = mu
        self.metric = metric.value
        self.normalize = normalize

    def set_iterative_algorithm_param(self,
                                      t: int = 20,
                                      alpha: float = 1.0,
                                      top_k: int = 20,
                                      edge_th: float = 1.1) -> None:
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


    def main(self) -> None:
        dfs = tuple([pd.read_feather(path) for path in self.data_paths])
        check_validity_loaded_data(dfs=dfs)

        # Remove rows where the proportion of missing values exceeds the threshold
        dfs_after_th_nan = []
        for df, modality_name in zip(dfs, self.mod_names):
            df_cleaned, row_missing_percentage = remove_rows_above_missing_threshold(df, th_nan=self.th_nan, verbose=self.verbose)
            dfs_after_th_nan.append(df_cleaned)

            # Plot missing percentage histogram
            if self.plot_missing_percentage:
                plot_row_missing_percentage_histogram(row_missing_percentage, self.th_nan, modality_name, self.save_path_histogram, verbose=self.verbose)
        dfs_after_th_nan = tuple(dfs_after_th_nan)


        dfs_after_modality_intersection = get_overlapping_modalities(dfs_after_th_nan)
        overlapped_eids = save_overlapping_eids(dfs=dfs_after_modality_intersection, save_path=self.save_path_histogram, verbose=self.verbose)    
        np_arrs = convert_df_to_np(dfs_after_modality_intersection)

        # TODO: temporary for testing, remove later
        # convert omics to linear scale 
        np_arrs = tuple([2**np_arr if mod == "pro" else np_arr for np_arr, mod in zip(np_arrs, self.mod_names)])
        
        n = int(np_arrs[0].shape[0])
        param = set_affinity_matrix_parameters(n=n, metric=self.metric, K=self.K, mu=self.mu, normalize=self.normalize, th_nan=self.th_nan)
        
        _print_line(self.verbose)
        if self.verbose:
            print(param)
        _print_line(self.verbose)

        affinity_networks = compute_aff_networks(np_arrs, param=param, verbose=self.verbose)
        _print_line(self.verbose)
        if self.verbose:
            print(f"{len(affinity_networks)} affinity matrices generated")


        fused_network = snf(affinity_networks, K=param["K_actual"], t=self.t, alpha=self.alpha) # TODO: add t, alpha to param
        _print_line(self.verbose)
        if self.verbose:
            print(f"Fused matrix with shape {fused_network.shape} generated.")

        nb_clusters, eigenvalues, eigenvectors = get_n_clusters_revised(fused_network, plot_path=os.path.join(self.save_path_fused, "eigenvalues.png"), 
                                                                        top_k=self.top_k, verbose=self.verbose)
        _print_line(self.verbose)
        if self.verbose:
            print(f"Optimal number of clusters: {nb_clusters}") # TODO: check with toydata!!!
        nb_clusters = get_n_clusters(fused_network, n_clusters=range(2, self.top_k))
        if self.verbose:
            print(f"Optimal number of clusters: {nb_clusters}") # TODO: check with toydata!!!

        n_clusters = get_optimal_cluster_size(n_clusters=self.n_clusters, nb_clusters=list(nb_clusters))
        
        _print_line(self.verbose)
        if self.verbose:
            print(f"n_cluster = {n_clusters}")
        _print_line(self.verbose)
            

        fused_labels = spectral_clustering(fused_network, n_clusters=n_clusters, random_state=self.random_state)
        if self.verbose:
            print("Spectral clustering on fused matrix.")
            # print(fused_labels)
            print(np.unique(fused_labels, return_counts=True))
            # print(len(fused_labels))
            # print(len(overlapped_eids))
        
        _print_line(self.verbose)
        save_cluster_eids(eids=overlapped_eids, labels=fused_labels, save_path=self.save_path_fused, verbose=self.verbose)
        _print_line(self.verbose)

        _print_line(self.verbose)
        plot_silhouette_score(fused_network=fused_network, save_path=self.save_path_fused, verbose=self.verbose)
        _print_line(self.verbose)



        dynamic_range = plot_ordered_affinity_matrix(network=fused_network,
                                                     labels=fused_labels,
                                                     figure_path=os.path.join(self.save_path_fused, "aff_fused"),
                                                     title=None,
                                                     dynamic_range_th=(0.1, 0.1),
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
                                         dynamic_range_th=(0.1, 0.1),
                                         figsize=(8.0, 8.0),
                                         show_colorbar=False,
                                         plt_close=True,
                                         dynamic_range=dynamic_range,
                                         return_dynamic_range=False,
                                         show_axis=False,
                                         verbose=self.verbose
                                         )
        _print_line(self.verbose)
        plot_edge_contribution(labels=fused_labels,
                               modality_names=self.mod_names,
                               affinity_networks=affinity_networks,
                               save_path=self.save_path_upset,
                               edge_th=self.edge_th,
                               verbose=self.verbose)
        _print_line(self.verbose)
        

def _print_line(verbose: bool) -> bool:
    if verbose:
        print("-----------------------------------------")