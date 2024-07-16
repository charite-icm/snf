import numpy as np

from .dimension_reduction import ReduceDimAlgoPCA, ReduceDimAlgosEnum, PCA
from .utils_pandas import UtilsPandas

from enum import Enum, auto
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

RESULTS_PATH = "results"
if not os.path.exists(RESULTS_PATH):
    os.mkdir(RESULTS_PATH)

class ValFeaturesXEnum(Enum):
    LEFT_EJECT_CATEG = "mri_lvejectionfract_imgvisit"
    LEFT_EJECT_CONT = "mri_lvejectionfractc_imgvisit"
    PCA1 = "PCA1"


class ValFeaturesYEnum(Enum):
    CYSTATINC = "b_cystatinc_initasse"
    AGE = "ageatrecruitment_initasses0"
    GLYC_HEMO = "b_glycatedhaemoglobin_initasse"
    BMI = "bodymassindex_bmi__initasses0"
    NPPBNAT_PEPTIDE = "p_nppbnatriureticpeptidesb"
    PCA2 = "PCA2"


SHORT_FEATURE_NAMES = {
    ValFeaturesXEnum.LEFT_EJECT_CATEG: "lef",
    ValFeaturesXEnum.LEFT_EJECT_CONT: "lef",
    ValFeaturesYEnum.CYSTATINC: "cystanin",
    ValFeaturesYEnum.AGE: "age",
    ValFeaturesYEnum.GLYC_HEMO: "hba1c",
    ValFeaturesYEnum.BMI: "bmi",
    ValFeaturesYEnum.NPPBNAT_PEPTIDE: "nppbnatriu",
    ValFeaturesXEnum.PCA1: "pca1",
    ValFeaturesYEnum.PCA2: "pca2"
}


class BeforeClustering:
    def __init__(self,
                df_train: pd.DataFrame,
                df_val: pd.DataFrame,
                save_folder: str,
                val_y_enum: ValFeaturesYEnum,
                val_x_enum: ValFeaturesXEnum = ValFeaturesXEnum.LEFT_EJECT_CONT,
                reduce_dim: ReduceDimAlgosEnum = None,
                pca_variance_to_keep: float = None,
                log_scale: bool = False) -> None:
        self.df_train = df_train
        self.df_val = df_val
        self.save_folder = save_folder

        self.val_y_enum = val_y_enum
        self.val_x_enum = val_x_enum
        self.val_x_name = None
        self.val_y_name = None

        self.reduce_dim_algos_enum = reduce_dim
        self.pca_variance_to_keep = pca_variance_to_keep
        self.log_scale = log_scale

        self.pca_validation = False
        if self.val_y_enum == ValFeaturesYEnum.PCA2 or self.val_x_enum == ValFeaturesXEnum.PCA1:
            self.pca_validation = True

        self.eids = None

    def main(self) -> None:
        self._drop_eid()
        if self.pca_validation:
            self._prepare_data_pca_validation()
        else:
            self._prepare_data_no_nans()
        self._reduce_dimensions()
        self._to_numpy()
        self._scale()
        self._create_save_folder()

    def _drop_eid(self) -> None:
        if not all(self.df_train["eid"] == self.df_val["eid"]):
            raise ValueError(f"Eids are not the same between df_train and df_val")

        self.df_train = self.df_train.dropna()
        self.eids = self.df_train["eid"]

        self.df_train = UtilsPandas.drop_columns(self.df_train, ["eid"])
        self.df_val = UtilsPandas.drop_columns(self.df_val, ["eid"])


    def _prepare_data_pca_validation(self) -> None:
        self.val_x_enum = ValFeaturesXEnum.PCA1
        self.val_y_enum = ValFeaturesYEnum.PCA2

        pca = PCA(n_components=2)
        numpy_arr = pca.fit_transform(self.df_train)
        self.df_val = pd.DataFrame(data=numpy_arr)
        self.df_val.rename(columns={
            self.df_val.columns[0]: self.val_x_enum.value,
            self.df_val.columns[1]: self.val_y_enum.value,
        }, inplace=True)

        self.val_x_name = (f"{self.val_x_enum.value}: "
                           f"{int(pca.explained_variance_ratio_[0]*100)} % variance")
        self.val_y_name = (f"{self.val_y_enum.value}: "
                           f"{int(pca.explained_variance_ratio_[1]*100)} % variance")


    def _prepare_data_no_nans(self) -> None:

        merged_df = pd.merge(self.df_val[[self.val_x_enum.value, self.val_y_enum.value]],
                             self.df_train,
                             left_index=True, right_index=True)


        x_name = self.val_x_enum.value
        y_name = self.val_y_enum.value
        if x_name in self.df_train.columns.values and x_name in self.df_val.columns.values:
            x_name = f"{x_name}_x"
        if y_name in self.df_train.columns.values and y_name in self.df_val.columns.values:
            y_name = f"{y_name}_x"


        # testing_plot_df = merged_df[[feature_1, f"{feature_2}_x"]]
        # training_df = merged_df.drop([feature_1, f"{feature_2}_x"], axis=1)

        self.df_val = merged_df[[x_name, y_name]]
        self.df_train = merged_df.drop([x_name, y_name], axis=1)

        # if self.val_x_name in [ValFeaturesXEnum.LEFT_EJECT_CONT, ValFeaturesXEnum.LEFT_EJECT_CATEG]:
        #     self.val_x_name = f"{self.val_x_enum.value} [%]"

        print(f"df_validation shape: {self.df_val.shape}")
        print(f"df_training shape: {self.df_train.shape}")


    def _reduce_dimensions(self) -> None:
        if self.reduce_dim_algos_enum == ReduceDimAlgosEnum.PCA:
            pca = ReduceDimAlgoPCA(df=self.df_train, n_components=self.pca_variance_to_keep)
            pca.run()
            self.df_train = pca.df.copy()
            self.save_folder = f"{self.save_folder}_{pca.get_short_name()}"

    def _to_numpy(self) -> None:
        self.np_train = self.df_train.to_numpy()
        if self.log_scale:
            self.np_train = 2**self.np_train
        self.np_val = self.df_val.to_numpy()

    def _scale(self) -> None:
        scaler = StandardScaler()
        self.np_train = scaler.fit_transform(self.np_train)

    def _create_save_folder(self) -> None:
        self.save_path = os.path.join(RESULTS_PATH, self.save_folder)
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        # Another folder based on validation features
        name = (f"x_{SHORT_FEATURE_NAMES[self.val_x_enum]}_"
                f"y_{SHORT_FEATURE_NAMES[self.val_y_enum]}")
        self.save_path = os.path.join(self.save_path, name)
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

    def get_df_train(self) -> pd.DataFrame:
        return self.df_train

    def get_df_val(self) -> pd.DataFrame:
        return self.df_val

    def get_np_train(self) -> np.ndarray:
        return self.np_train

    def get_np_val(self) -> np.ndarray:
        return self.np_val

    def get_save_path(self) -> str:
        return self.save_path

    def get_val_x_feature_name(self) -> str:
        if self.val_x_name is None:
            return self.val_x_enum.value
        else:
            return self.val_x_name
    def get_val_y_feature_name(self) -> str:
        if self.val_y_name is None:
            return self.val_y_enum.value
        else:
            return self.val_y_name

    def get_eids(self) -> pd.DataFrame:
        return self.eids
