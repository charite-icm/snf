import os.path

from enum import Enum
import numpy as np
import pandas as pd

from .utils_pandas import UtilsPandas


class EidsClustersColEnum(Enum):
    EID = "eid"
    CLUSTER = "cluster"

EID_NAME = EidsClustersColEnum.EID.value
CLUSTER_NAME = EidsClustersColEnum.CLUSTER.value

class ClusterAlgorithm:
    def __init__(self,
                 np_train: np.ndarray,
                 np_val: np.ndarray,
                 result_folder: str,
                 val_x_name: str,
                 val_y_name: str,
                 plt_close: bool = False,
                 eids: pd.DataFrame = None) -> None:
        self.np_train = np_train
        self.np_val = np_val
        self.result_folder = result_folder
        self.val_x_name = val_x_name
        self.val_y_name = val_y_name
        self.plt_close = plt_close
        self.eids = eids
        # Any other initialization you need can go here

    @staticmethod
    def save_cluster_eids(labels: list[int], eids: list[int],
                          save_path: str, csv_name: str = "cluster_eids") -> None:

        save_csv_path = os.path.join(save_path, f"{csv_name}.csv")
        df = pd.DataFrame({
            EID_NAME: eids,
            CLUSTER_NAME: labels
        })

        UtilsPandas.save_csv(df, save_csv_path)
