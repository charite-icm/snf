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



class Snf:
    pass





DATA_PATH = "data/hfmodelexport_metab_prot_img_05_15_2024"
MOD_DIRS = ("lab", "metabolomics_marcus_90", "physiology", "proteomics_all")
# MOD_DIRS = ("lab", "metabolomics_marcus_90", "physiology")
FEATHER_NAME = "ever_hfpef_suspect_noimg.feather"


def main() -> None:
    paths = [os.path.join(DATA_PATH, mod_dir, FEATHER_NAME) for mod_dir in MOD_DIRS]
    dfs = tuple([pd.read_feather(path) for path in paths])
    
    # # TODO: temporary for faster testing
    # dfs = tuple([*dfs[:-1], dfs[-1].iloc[:, :200]])
    # print([df.shape for df in dfs])

    save_path = Path("results/test_revised")
    save_path_histogram = os.path.join(save_path, "histogram_missing_percentage")
    save_path_fused = os.path.join(save_path, "fused")
    save_path_upset = os.path.join(save_path, "upset_plots")

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path_histogram, exist_ok=True)
    os.makedirs(save_path_fused, exist_ok=True)
    os.makedirs(save_path_upset, exist_ok=True)
    

    plot_missing_percentage = True 
    verbose = True
    th_nan = 0.3
    random_state = 41
    n_clusters = 6

   

    check_validity_loaded_data(dfs=dfs)

    # Remove rows where the proportion of missing values exceeds the threshold
    dfs_after_th_nan = []
    for df, modality_name in zip(dfs, MOD_DIRS):
        df_cleaned, row_missing_percentage = remove_rows_above_missing_threshold(df, th_nan=th_nan, verbose=verbose)
        dfs_after_th_nan.append(df_cleaned)

        # Plot missing percentage histogram
        if plot_missing_percentage:
            plot_row_missing_percentage_histogram(row_missing_percentage, th_nan, modality_name, save_path_histogram)
    dfs_after_th_nan = tuple(dfs_after_th_nan)

    dfs_after_modality_intersection = get_overlapping_modalities(dfs_after_th_nan)
    overlapped_eids = save_overlapping_eids(dfs=dfs_after_modality_intersection, save_path=save_path_histogram, verbose=verbose)    
    np_arrs = convert_df_to_np(dfs_after_modality_intersection)

    # convert omics to linear scale
    np_arrs = tuple([2**np_arr if mod == "proteomics_all" else np_arr for np_arr, mod in zip(np_arrs, MOD_DIRS)])


    K = 0.1
    mu = 0.5
    metric = f"{DistanceMetric.SQEUCLIDEAN.value}"
    normalize = True
    n = int(np_arrs[0].shape[0])

    param = set_affinity_matrix_parameters(n=n, metric=metric, K=K, mu=mu, normalize=normalize, th_nan=th_nan)
    print("-----------------------------------------")
    print(param)
    print("-----------------------------------------")


    affinity_networks = compute_aff_networks(np_arrs, param=param)
    print("--------------------------------")
    print(f"{len(affinity_networks)} affinity matrices generated")
    
    t = 20
    alpha = 1
    top_k = 20
    edge_th = 1.1

    fused_network = snf(affinity_networks, K=param["K_actual"], t=t, alpha=alpha) # TODO: add t, alpha to param
    print("-----------------------------------------")
    print(f"Fused matrix with shape {fused_network.shape} generated.")

    nb_clusters, eigenvalues, eigenvectors = get_n_clusters_revised(fused_network, plot_path=os.path.join(save_path_fused, "eigenvalues.png"), 
                                                                    top_k=top_k, verbose=verbose)
    print("-----------------------------------------")
    print(f"Optimal number of clusters: {nb_clusters}") # TODO: check with toydata!!!
    # print("eigenvalues: ", eigenvalues)
    # print("eigenvectors: ", eigenvectors)
    nb_clusters = get_n_clusters(fused_network, n_clusters=range(2, top_k))
    print(f"Optimal number of clusters: {nb_clusters}") # TODO: check with toydata!!!

    print("-----------------------------------------")
    n_clusters = get_optimal_cluster_size(n_clusters=n_clusters, nb_clusters=list(nb_clusters))
    print(f"n_cluster = {n_clusters}")
    print("-----------------------------------------")

    fused_labels = spectral_clustering(fused_network, n_clusters=n_clusters, random_state=random_state)
    print("Spectral clustering on fused matrix.")
    # print(fused_labels)
    print(np.unique(fused_labels, return_counts=True))
    # print(len(fused_labels))
    # print(len(overlapped_eids))
    print("-----------------------------------------")
    save_cluster_eids(eids=overlapped_eids, labels=fused_labels, save_path=save_path_fused, verbose=verbose)
    print("-----------------------------------------")
    plot_silhouette_score(fused_network=fused_network, save_path=save_path_fused, verbose=verbose)
    print("-----------------------------------------")


    dynamic_range = plot_ordered_affinity_matrix(network=fused_network,
                                                 labels=fused_labels,
                                                 figure_path=os.path.join(save_path_fused, "aff_fused"),
                                                 title=None,
                                                 dynamic_range_th=(0.1, 0.1),
                                                 figsize=(8.0, 8.0),
                                                 show_colorbar=False,
                                                 plt_close=True,
                                                 dynamic_range=None,
                                                 return_dynamic_range=True,
                                                 show_axis=False,
                                                )

    for modality, aff in zip(MOD_DIRS, affinity_networks):
        plot_ordered_affinity_matrix(network=aff,
                                     labels=fused_labels,
                                     figure_path=os.path.join(save_path_fused, f"aff_{modality}"),
                                     title=None,
                                     dynamic_range_th=(0.1, 0.1),
                                     figsize=(8.0, 8.0),
                                     show_colorbar=False,
                                     plt_close=True,
                                     dynamic_range=dynamic_range,
                                     return_dynamic_range=False,
                                     show_axis=False,
                                    )
    print("-----------------------------------------")
    plot_edge_contribution(labels=fused_labels,
                           modality_names=MOD_DIRS,
                           affinity_networks=affinity_networks,
                           save_path=save_path_upset,
                           edge_th=edge_th,
                           verbose=verbose)
    print("-----------------------------------------")




