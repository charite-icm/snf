import os 
import pandas as pd
from sklearn.metrics import v_measure_score

from src.snf import Snf, DistanceMetric
from src.snf_package import datasets


def main() -> None:
    mod_names = ("0", "1", "2", "3")
    save_path = "results/test_toydata"


    digits = datasets.load_digits()
    print(digits.keys())
    
    dfs = []
    for arr in digits.data:
        df = pd.DataFrame(arr)
        df["eid"] = df.index
        dfs.append(df)
        print(df.shape)
    dfs = tuple(dfs)

    snf = Snf(dfs=dfs, mod_names=mod_names, save_path=save_path,
              plot_missing_percentage=True, th_nan=0.0, random_state=41,
              n_clusters=None, verbose=True)
    snf.set_affinity_matrix_param(K=0.1, mu=0.5, metric=DistanceMetric.EUCLIDEAN, normalize=False)
    snf.set_iterative_algorithm_param(t=20, alpha=1.0, top_k=20, edge_th=1.1)
    snf.main()

    labels = pd.read_csv(os.path.join(save_path, "fused/cluster_eids.csv"))["cluster"]
    print(v_measure_score(labels, digits.labels))
    

if __name__ == "__main__":
    main()
