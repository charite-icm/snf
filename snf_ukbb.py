import os 
import pandas as pd

from src.snf import Snf, DistanceMetric


DATA_PATH = "data/hfmodelexport_metab_prot_img_05_15_2024"
MOD_DIRS = ("lab", "metabolomics_marcus_90", "physiology", "proteomics_all")
# MOD_DIRS = ("lab", "metabolomics_marcus_90", "physiology")
FEATHER_NAME = "ever_hfpef_suspect_noimg.feather"


def log2linear(df: pd.DataFrame) -> pd.DataFrame:
    np_prot = df.to_numpy()
    np_prot_linear = np_prot.copy()
    np_prot_linear[:, 1:] = 2**np_prot[:, 1:]
    df_prot_linear = pd.DataFrame(np_prot_linear, columns=df.columns)

    return df_prot_linear


def main() -> None:
    paths = [os.path.join(DATA_PATH, mod_dir, FEATHER_NAME) for mod_dir in MOD_DIRS]
    mod_names = ("lab", "met", "phe", "pro")
    save_path = "results/test_revised"
    
    dfs = tuple([pd.read_feather(path) for path in paths])
    df_prot_linear = log2linear(dfs[-1])
    dfs = tuple([*dfs[:-1], df_prot_linear])


    snf = Snf(dfs=dfs, mod_names=mod_names, save_path=save_path,
              plot_missing_percentage=True, th_nan=0.3, random_state=41,
              n_clusters=6, verbose=True)
    

    snf.set_affinity_matrix_param(K=0.1, mu=0.5, metric=DistanceMetric.SEUCLIDEAN, normalize=False)
    snf.set_iterative_algorithm_param(t=20, alpha=1.0, top_k=20, edge_th=1.1)

    snf.main()


    


if __name__ == "__main__":
    main()
