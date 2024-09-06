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

DATA_PATH = "data/hfmodelexport_metab_prot_img_05_15_2024"
MOD_DIRS = ("lab", "metabolomics_marcus_90", "physiology", "proteomics_all") 
FEATHER_NAME = "ever_hfpef_suspect_noimg.feather"


def main() -> None:
    save_path = Path("results/test_revised")
    paths = [os.path.join(DATA_PATH, mod_dir, FEATHER_NAME) for mod_dir in MOD_DIRS]
    
    dfs = tuple([pd.read_feather(path) for path in paths])
    


    _check_validity_loaded_data(dfs=dfs)

    # Remove rows where the proportion of missing values exceeds the threshold
    dfs_after_th_nan = []
    for df, modality_name in zip(dfs, MOD_DIRS):
        df_cleaned, row_means = remove_rows_above_missing_threshold(df, th_nan=0.3, verbose=True)
        print(row_means)
        dfs_after_th_nan.append(df_cleaned)
    dfs_after_th_nan = tuple(dfs_after_th_nan)



if __name__ == "__main__":
    main()