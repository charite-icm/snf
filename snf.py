# from src.snf_pipeline import SnfPipeline, DataSelectorEnum, SubgroupEnum, DataModalityEnum, ValFeaturesXEnum, ValFeaturesYEnum
# # Set OMP_NUM_THREADS environment variable to avoid the KMeans memory leak warning
# # os.environ["OMP_NUM_THREADS"] = "3"

# import warnings

# # Suppress the specific warning
# warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.cluster._kmeans")

# DATA_SELECTION = DataSelectorEnum.METAB_PROT_IMG_05_15_2024

# n_clusters = [6]
# subgroups = [SubgroupEnum.HFPEF_SUSPECT_NOIMG]

# for subgroup in subgroups:
#     for n_cluster in n_clusters:
#         snf_pipeline = SnfPipeline(
#             data_selection=DATA_SELECTION,
#             modalities=[
#                 DataModalityEnum.PHYSIO, 
#                 DataModalityEnum.PROT_ALL, 
#                 DataModalityEnum.LAB,
#                 DataModalityEnum.METAB_MARCUS_90,
#             ],
#             subgroup=subgroup,
#             n_clusters=n_cluster,
#             # result_dir_name=f"snf_prot_physio_lab_nan__{subgroup.name.lower()}_{n_cluster}clust",
#             result_dir_name=f"test",
#             val_x_enum=ValFeaturesXEnum.PCA1,
#             val_y_enum=ValFeaturesYEnum.PCA2,
#             th_nan=0.30,
#             snps=False
#         )


#         snf_pipeline.load_data()
#         snf_pipeline.get_overlapped_modalities()
#         snf_pipeline.make_dirs()
#         snf_pipeline.prepare_for_clustering()
#         snf_pipeline.compute_aff_networks()
#         snf_pipeline.compute_fused_network()

#         snf_pipeline.spectral_clustering_modalities()
#         snf_pipeline.spectral_clustering_fused()
#         snf_pipeline.spectral_clustering_internal_validation()
#         snf_pipeline.plot_affinity_matrices()
#         snf_pipeline.order_affinity_matrices()
#         snf_pipeline.edge_contribution()
#         snf_pipeline.save_metadata()

import numpy as np
import os
from pathlib import Path 
import pandas as pd

from src.snf_pipeline_revised import _check_validity_loaded_data
from src.snf_pipeline_revised import remove_rows_above_missing_threshold
from src.snf_pipeline_revised import plot_row_missing_percentage_histogram
from src.snf_pipeline_revised import get_overlapping_modalities
from src.snf_pipeline_revised import save_overlapping_eids
from src.snf_pipeline_revised import convert_df_to_np

from src.snf_package.compute import DistanceMetric, snf, get_n_clusters_revised
from src.snf_pipeline_revised import set_affinity_matrix_parameters
from src.snf_pipeline_revised import compute_aff_networks


DATA_PATH = "data/hfmodelexport_metab_prot_img_05_15_2024"
MOD_DIRS = ("lab", "metabolomics_marcus_90", "physiology", "proteomics_all") 
# MOD_DIRS = ("lab", "metabolomics_marcus_90", "physiology") 
FEATHER_NAME = "ever_hfpef_suspect_noimg.feather"


def main() -> None:
    paths = [os.path.join(DATA_PATH, mod_dir, FEATHER_NAME) for mod_dir in MOD_DIRS]    
    dfs = tuple([pd.read_feather(path) for path in paths])





    save_path = Path("results/test_revised")
    save_path_histogram = os.path.join(save_path, "histogram_missing_percentage")
    save_path_fused = os.path.join(save_path, "fused")

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path_histogram, exist_ok=True)
    os.makedirs(save_path_fused, exist_ok=True)
    

    plot_missing_percentage = True 
    verbose = True
    th_nan = 0.3

    t = 20
    alpha = 1


    _check_validity_loaded_data(dfs=dfs)

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
    save_overlapping_eids(dfs=dfs_after_modality_intersection, save_path=save_path, verbose=verbose)    
    np_arrs = convert_df_to_np(dfs_after_modality_intersection)

    # convert omics to linear scale
    np_arrs = tuple([2**np_arr if mod == "proteomics_all" else np_arr for np_arr, mod in zip(np_arrs, MOD_DIRS)])


    K = 0.1
    mu = 0.5
    metric = f"{DistanceMetric.SQEUCLIDEAN.value}"
    normalize = True
    n = int(np_arrs[0].shape[0])

    param = set_affinity_matrix_parameters(n=n, metric=metric, K=K, mu=mu, normalize=normalize, th_nan=th_nan)
    print(param)

    affinity_networks = compute_aff_networks(np_arrs, param=param)
    print("--------------------------------")
    print(f"{len(affinity_networks)} affinity matrices generated")
    
    fused_network = snf(affinity_networks, K=param["K_actual"], t=t, alpha=alpha) # TODO: add t, alpha to param
    print("-----------------------------------------")
    print(f"Fused matrix with shape {fused_network.shape} generated.")

    nb_clusters, eigenvalues, eigenvectors = get_n_clusters_revised(fused_network, plot_path=os.path.join(save_path_fused, "eigenvalues.png"), 
                                                                    top_k=20, verbose=verbose)
    print(f"Fused matrix generated. Optimal number of clusters: {nb_clusters}") # TODO: check with toydata!!!
    # print("eigenvalues: ", eigenvalues)
    # print("eigenvectors: ", eigenvectors)


if __name__ == "__main__":
    main()
