import os
from enum import Enum, auto
import numpy as np
import pandas as pd
from typing import Union

from .utils import Utils
from .utils_pandas import UtilsPandas
# from .image_format import *


DATA_FOLDER = "data"
TXT_FOLDER = os.path.join(DATA_FOLDER, "txt")
SHORT2LONG_PATH = "data/do/short2long.csv"
SHORT2LONG_PATH_PROT = "data/do/short2long_all_proteins.csv"

# CLUSTER_PATH = "key_results//snf_prot_physio_lab_14_12_2023//fused//eids_4_clusters.csv"
CLUSTER_PATH = "key_results//hfpef_suspects__phy_lab_prot_metab_4//fused//eids_4_clusters.csv"
CLUSTER_PATH_NEW = "key_results//hfpef_suspects__phy_lab_prot_metab_6//fused//eids_6_clusters.csv"
CLUSTER_BIG_PATH = "key_results//hfpef__phy_lab_metab_3//fused//eids_3_clusters.csv"

TOPS_SNPS_PATH = "supervised_learning/hfpef_suspects_noimg_vs_healthy_controls/pgrm_snps/intersection_top_snps.txt"



class DataSelectorEnum(Enum):
    # METAB_PROT_IMG_03_11_2024 = "hfmodelexport_metab_prot_img_03_11_2024"
    METAB_PROT_IMG_05_15_2024 = "hfmodelexport_metab_prot_img_05_15_2024"



class DataModalityEnum(Enum):
    ALL = "all"
    PROT_P1 = "proteomics_p1"
    PROT_P2 = "proteomics_p2"
    PROT_ALL = "proteomics_all"
    PROT_SELECTED = "proteomics_selected"
    PROT_NO_BNP = auto()
    PROT_DS_0_1 = auto()
    PROT_DS_0_2 = auto()
    PROT_DS_0_3 = auto()
    METAB = "metabolomics"
    METAB_MARCUS = "metabolomics_marcus"
    METAB_MARCUS_90 = "metabolomics_marcus_90"
    VALIDATION = "validation"
    PHYSIO = "physiology"
    LAB = "lab"
    QUEST_CONT = "quest_cont"
    QUEST_BINOM = "quest_binom"
    QUEST_CATEG = "quest_categ"
    DIAG = "diagnoses"
    SYMP = "symptoms"
    MED_VIT = "medications_vitamins"
    MED = "medications"
    VIT = "vitamins"
    SNPS_10K = "snps_10k"
    SNPS_PGRM = "snps_pgrm"
    PRS = "prs"
    AIR_QUALITY = "air_quality"
    TELOMERE = "telomere"
    WATER_MINERALS = "water_minerals"
    MRI = "mri"



# TODO: add Value error if duplicated eids
class SubgroupEnum(Enum):
    HF = "hf"
    HF_RV = "hf_rv"
    NO_HF_RV = "no_hf_rv"
    NO_HF = "no_hf"
    HF_EVER = "hf_ever"
    HFPEF_MREF = "hfpef_mref"
    HFPEF_SUSPECT_IMG = "ever_hfpef_suspect"
    HFPEF_SUSPECT_NOIMG = "ever_hfpef_suspect_noimg"
    NO_HFPEF_SUSPECT_NOIMG = "no_hfpef_suspect_noimg"
    CONTROL_DIAB_F = "control_diab_f"
    CONTROL_DIAB_M = "control_diab_m"
    CONTROL_OBES_F = "control_obes_f"
    CONTROL_OBES_M = "control_obes_m"
    CONTROL_F = "control_f"
    CONTROL_M = "control_m"
    HFPEF = "hfpef"
    NO_HFPEF = "no_hfpef"
    HFREF = "hfref"
    CLUST_1 = "cluster_1"
    NO_CLUST_1 = "no_cluster_1"
    CLUST_2 = "cluster_2"
    NO_CLUST_2 = "no_cluster_2"
    CLUST_3 = "cluster_3"
    NO_CLUST_3 = "no_cluster_3"
    CLUST_4 = "cluster_4"
    NO_CLUST_4 = "no_cluster_4"
    CLUST_NEW_1 = "cluster_new_1"
    NO_CLUST_NEW_1 = "no_cluster_new_1"
    CLUST_NEW_2 = "cluster_new_2"
    NO_CLUST_NEW_2 = "no_cluster_new_2"
    CLUST_NEW_3 = "cluster_new_3"
    NO_CLUST_NEW_3 = "no_cluster_new_3"
    CLUST_NEW_4 = "cluster_new_4"
    NO_CLUST_NEW_4 = "no_cluster_new_4"
    CLUST_NEW_5 = "cluster_new_5"
    NO_CLUST_NEW_5 = "no_cluster_new_5"
    CLUST_NEW_6 = "cluster_new_6"
    NO_CLUST_NEW_6 = "no_cluster_new_6"
    CLUST_BIG_1 = "cluster_big_1"
    NO_CLUST_BIG_1 = "no_cluster_big_1"
    CLUST_BIG_2 = "cluster_big_2"
    NO_CLUST_BIG_2 = "no_cluster_big_2"
    CLUST_BIG_3 = "cluster_big_3"
    NO_CLUST_BIG_3 = "no_cluster_big_3"



SUBGROUP_ENUM_2_NAMES = {
    SubgroupEnum.CONTROL_DIAB_M: "Diabetes\nMales",
    SubgroupEnum.CONTROL_DIAB_F: "Diabetes\nFemales",
    SubgroupEnum.CONTROL_OBES_M: "Obese\nMales",
    SubgroupEnum.CONTROL_OBES_F: "Obese\nFemales",
    SubgroupEnum.CONTROL_M: "Healthy\nMales",
    SubgroupEnum.CONTROL_F: "Healthy\nFemales",
    SubgroupEnum.HFPEF_MREF: "HFpEF /\nHFmrEF",
    SubgroupEnum.HFPEF_SUSPECT_NOIMG: "HFpEF\nSuspects",
    SubgroupEnum.HF_EVER: "HF Ever",
    SubgroupEnum.HFPEF: "HFpEF",
    SubgroupEnum.HFREF: "HFrEF"
}


class EidMissingError(Exception):
    def __init__(self, path: str):
        self.message = f"'eid' is missing in the list of features. Please add to {path}"
        super().__init__(self.message)


class AddColEnum(Enum):
    CONT_LEFT_EJECTION_FRACTION = auto()
    CONT_AGE = auto()


class DataLoaderSnps10k:
    MEMMAP_PATH = os.path.join(DATA_FOLDER, "snps", "all_chrs_all.PGS001790_heart_failure-10k.X.npy")
    EID_PATH = os.path.join(DATA_FOLDER, "snps", "all_chrs_all.PGS001790_heart_failure-10k.obs.csv")
    def __init__(self,
                 data_selection: DataSelectorEnum,
                 subgroup: SubgroupEnum) -> None:
        self.data_selection = data_selection
        self.subgroup = subgroup

        self.val_subgroup_dir = os.path.join(DATA_FOLDER, data_selection.value, DataModalityEnum.VALIDATION.value)
        self.save_dir = os.path.join(DATA_FOLDER, data_selection.value, DataModalityEnum.SNPS_10K.value)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.save_path = os.path.join(self.save_dir, f"{self.subgroup.value}.feather")

    def load(self) -> None:
        if os.path.exists(self.save_path):
            self._read_feather()
        else:
            self._convert_to_df()
            self._filter_out_missing_eids()
            self._merge_eids()
            self._save()

    def _read_feather(self) -> None:
        print(f"{self.save_path} already exists!")
        self.df = UtilsPandas.read_feather(self.save_path)

    def _convert_to_df(self) -> None:
        x = np.memmap(self.MEMMAP_PATH, mode="r", shape=(487163, 10000), dtype="float16")
        self.df = pd.DataFrame(x)
        print(self.df)
        df_eids = UtilsPandas.read_csv(self.EID_PATH)
        self.df_eid = pd.DataFrame(df_eids)
        print(df_eids)

    def _filter_out_missing_eids(self) -> None:
        validation_path = os.path.join(self.val_subgroup_dir, f"{self.subgroup.value}.feather")
        if not os.path.exists(validation_path):
            print(f"{validation_path} does not exist!")
        else:
            print(f"{validation_path} exists!")

        data_loader = DataLoader(
            data_selection=self.data_selection,
            data_modality=DataModalityEnum.VALIDATION,
            subgroup=self.subgroup
        )
        data_loader.load()
        df_eids_val = data_loader.get_main_df()

        missing_eids = set(df_eids_val["eid"]) - set(self.df_eid["eid"])
        print(f"missing_eids: {len(missing_eids)}/{len(df_eids_val['eid'])}")

        df_filtered_eids = self.df_eid[self.df_eid["eid"].isin(df_eids_val["eid"])]
        self.df_filtered = self.df.loc[df_filtered_eids.index]
        print(self.df_filtered)

    def _merge_eids(self) -> None:
        self.df = self.df_eid.merge(self.df_filtered, left_index=True, right_index=True)
        print(self.df)

    def _save(self) -> None:
        UtilsPandas.save_feather(self.df, self.save_path)

    def get_main_df(self) -> pd.DataFrame:
        return self.df.copy()


class DataLoaderSnpsPgrm:
    FEATHER_PATH = os.path.join(DATA_FOLDER, "snps", "231201_pgrm_snps.feather")
    def __init__(self,
                 data_selection: DataSelectorEnum,
                 subgroup: SubgroupEnum = None) -> None:
        self.data_selection = data_selection
        self.subgroup = subgroup

        self.val_subgroup_dir = os.path.join(DATA_FOLDER, data_selection.value, DataModalityEnum.VALIDATION.value)
        self.save_dir = os.path.join(DATA_FOLDER, data_selection.value, DataModalityEnum.SNPS_PGRM.value)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)


    def _get_feather_file_name(self, subgroup: SubgroupEnum | None) -> str:
        feather_file = "all"
        if subgroup is not None:
            feather_file = subgroup.value
        return feather_file

    def load(self) -> None:
        self.save_path = os.path.join(self.save_dir, f"{self._get_feather_file_name(self.subgroup)}.feather")
        if os.path.exists(self.save_path):
            self._read_feather()
        else:
            self._load_to_df()
            self._filter_out_missing_eids(self.subgroup)
            self._save()

    def save_all(self, subgroups: list[SubgroupEnum]) -> None:
        self._load_to_df()
        for subgroup in subgroups:
            # feather_file = "all"
            # if subgroup is not None:
            #     feather_file = subgroup.value
            feather_file = self._get_feather_file_name(subgroup)
            self.save_path = os.path.join(self.save_dir, f"{feather_file}.feather")
            if not os.path.exists(self.save_path):
                self._filter_out_missing_eids(subgroup, feather_file)
                self._save()
            else:
                print(f"{self.save_path} already exist!")

    def _read_feather(self) -> None:
        print(f"{self.save_path} already exists!")
        self.df_final = UtilsPandas.read_feather(self.save_path)

    def _load_to_df(self) -> None:
        self.df = UtilsPandas.read_feather(self.FEATHER_PATH).rename(columns={"index": "eid"})

    def _filter_out_missing_eids(self, subgroup: SubgroupEnum | None, feather_file: str) -> None:
        validation_path = os.path.join(self.val_subgroup_dir, f"{feather_file}.feather")
        if not os.path.exists(validation_path):
            print(f"{validation_path} does not exist!")
        else:
            print(f"{validation_path} exists!")

        data_loader = DataLoader(
            data_selection=self.data_selection,
            data_modality=DataModalityEnum.VALIDATION,
            subgroup=subgroup
        )
        data_loader.load()
        df_eids_val = data_loader.get_main_df()

        missing_eids = set(df_eids_val["eid"]) - set(self.df["eid"])
        print(f"missing_eids: {len(missing_eids)}/{len(df_eids_val['eid'])}")

        self.df_final = self.df[self.df["eid"].isin(df_eids_val["eid"])]
        # print(self.df)

    def _save(self) -> None:
        UtilsPandas.save_feather(self.df_final, self.save_path)

    def get_main_df(self) -> pd.DataFrame:
        return self.df_final.copy()



class DataLoader:
    def __init__(self,
                 data_selection: DataSelectorEnum,
                 data_modality: DataModalityEnum,
                 subgroup: Union[None, SubgroupEnum]) -> None:
        self.data_selection = data_selection
        self.path = os.path.join(DATA_FOLDER, data_selection.value, data_modality.value)
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        self.data_modality: DataModalityEnum = data_modality
        self.subgroup = subgroup
        self.all_loaded = False
        # self.all_subgroup_loaded = False

        self._check_data_modality_validity()
        self._set_path_all()

    def _check_data_modality_validity(self) -> None:
        if self.data_modality == DataModalityEnum.SNPS_10K:
            raise ValueError(
                f"data_modality should not be SNPS when using DataLoader class. "
                f"Use DataLoaderSnps class instead.")

    def _set_path_all(self) -> None:
        self.csv_path_all = os.path.join(DATA_FOLDER, self.data_selection.value,
                                         DataModalityEnum.ALL.value, DataModalityEnum.ALL.value + ".csv")
        self.feather_path_all = os.path.join(DATA_FOLDER, self.data_selection.value,
                                             DataModalityEnum.ALL.value, DataModalityEnum.ALL.value + ".feather")
        # csv_path = os.path.join(self.path, self.data_subselection.value + ".csv")

        if not os.path.exists(self.csv_path_all) and not os.path.exists(self.feather_path_all):
            raise ValueError(f"{self.csv_path_all} and {self.feather_path_all} do not exist!")

    def load(self) -> None:
        if self.subgroup is None and self.data_modality == DataModalityEnum.ALL:
            self._load_all()
            self._add_cols()
            self._save_headers_all()
            self.df = self.df_all
        elif self.subgroup is None and self.data_modality != DataModalityEnum.ALL:
            self._load_filtered()
        else:
            self._load_subclass(self.subgroup)

    def load_all(self, subgroups: list[SubgroupEnum]) -> None:
        for subgroup in subgroups:
            if subgroup is None and self.data_modality == DataModalityEnum.ALL:
                raise ValueError(f"load_all method does not work for subgroup: {subgroup}, data_modality: {self.data_modality}")
            elif subgroup is None and self.data_modality != DataModalityEnum.ALL:
                self._load_filtered()
            else:
                self._load_subclass(subgroup, open_if_exists=False)



    def _load_all(self) -> None:
        if self.all_loaded:
            pass
        elif not os.path.exists(self.feather_path_all):
            self.df_all = UtilsPandas.read_csv(path=self.csv_path_all)
            UtilsPandas.save_feather(self.df_all, self.feather_path_all)
            self.all_loaded = True
        else:
            self.df_all = UtilsPandas.read_feather(self.feather_path_all)
            self.all_loaded = True


    def _add_cols(self) -> None:
        if AddColEnum.CONT_LEFT_EJECTION_FRACTION:
            self._add_col_left_ejection_fraction()
        if AddColEnum.CONT_AGE:
            self._add_col_age()

        UtilsPandas.save_feather(self.df_all, self.feather_path_all)

    def _add_col_left_ejection_fraction(self) -> None:
        # Add continous left ventricle ejection fraction
        LV_DIAS_VOL = "mri_lvenddiastolicv_imgvisit"
        LC_SYS_VOL = "mri_lvendsystolicvo_imgvisit"
        # LV_EJECT_FRACT_CATEG = "mri_lvejectionfract_imgvisit"

        LV_EJECT_FRACT_CONT = "mri_lvejectionfractc_imgvisit"

        diastolic_vol = self.df_all[LV_DIAS_VOL]
        sistolic_vol = self.df_all[LC_SYS_VOL]
        # lve_categ = self.df_all[LV_EJECT_FRACT_CATEG]

        # Calculating continuous left ventricle ejection fraction
        lve_cont = (diastolic_vol - sistolic_vol) / diastolic_vol

        self.df_all[LV_EJECT_FRACT_CONT] = lve_cont

    def _add_col_age(self) -> None:
        # Add continuous age recruitment
        # TODO:
        pass

    def _save_headers_all(self) -> None:
        self.header_names_all = list(self.df_all.columns)
        path_txt = os.path.join(TXT_FOLDER, DataModalityEnum.ALL.value + ".txt")

        Utils.write_list_to_txt(file_path=path_txt, my_list=self.header_names_all)

    def _load_filtered(self) -> None:
        txt_path = os.path.join(TXT_FOLDER, self.data_modality.value + ".txt")
        feather_path = os.path.join(self.path, DataModalityEnum.ALL.value + ".feather")

        if not os.path.exists(txt_path):
            raise ValueError(f"{txt_path} does not exist!")

        if os.path.exists(feather_path):
            print(f"{feather_path} already exists! Loading ...")
            self.df = UtilsPandas.read_feather(path=feather_path)
        else:
            self._load_all()
            self.header_names = Utils.read_txt_to_list(file_path=txt_path)
            if "eid" not in self.header_names:
                raise EidMissingError(path=txt_path)
            # Pick only headers from the text file
            self.df = self.df_all[self.header_names]
            UtilsPandas.save_feather(df=self.df, path=feather_path)

    def _load_subclass(self, subgroup: SubgroupEnum, open_if_exists: bool = True) -> None:
        feather_all_subclass_path = os.path.join(DATA_FOLDER, self.data_selection.value, DataModalityEnum.ALL.value,
                                                 f"{subgroup.value}.feather")
        # print(feather_all_subclass_path)
        feather_path = os.path.join(self.path, f"{subgroup.value}.feather")
        txt_path = os.path.join(TXT_FOLDER, f"{self.data_modality.value}.txt")

        if os.path.exists(feather_path):
            if open_if_exists:
                print(f"{feather_path} already exists! Loading ...")
                self.df = UtilsPandas.read_feather(path=feather_path)
            else:
                print(f"{feather_path} already exists!")

        else:
            # if not os.path.exists(feather_all_subclass_path):
            self._load_all()
            self.df = self.filter_out_subclass(subgroup=subgroup, df=self.df_all)
            # else:
            #     self.df = UtilsPandas.read_feather(path=feather_all_subclass_path)

            # if not os.path.exists(feather_all_subclass_path):
            #     self._load_all()
            #     self.df = self.df_all
            #     self.df = self._filter_out_subclass()
            # else:
            #     self.df = UtilsPandas.read_feather(path=feather_all_subclass_path)

            self.header_names = Utils.read_txt_to_list(file_path=txt_path)
            if "eid" not in self.header_names:
                raise EidMissingError(path=txt_path)
            # Pick only headers from the text file
            df_result = self.df[self.header_names]
            UtilsPandas.save_feather(df=df_result, path=feather_path)

    @staticmethod
    def filter_out_subclass(subgroup: SubgroupEnum, df: pd.DataFrame) -> pd.DataFrame:
        if subgroup in [SubgroupEnum.HF, SubgroupEnum.HF_RV, SubgroupEnum.HFPEF_MREF,
                        SubgroupEnum.HFPEF_SUSPECT_IMG, SubgroupEnum.HFPEF_SUSPECT_NOIMG,
                        SubgroupEnum.CONTROL_DIAB_F, SubgroupEnum.CONTROL_DIAB_M,
                        SubgroupEnum.CONTROL_OBES_F, SubgroupEnum.CONTROL_OBES_M,
                        SubgroupEnum.CONTROL_F, SubgroupEnum.CONTROL_M,
                        SubgroupEnum.HFPEF, SubgroupEnum.HFREF]:
            df_result = UtilsPandas.filter_out(df=df, values_to_filter=[0], header_name=subgroup.value)
        elif subgroup == SubgroupEnum.NO_HF:
            df_result = UtilsPandas.filter_out(df=df, values_to_filter=[1], header_name=SubgroupEnum.HF.value)
        elif subgroup == SubgroupEnum.NO_HFPEF:
            df_result = UtilsPandas.filter_out(df=df, values_to_filter=[1], header_name=SubgroupEnum.HFPEF.value)
        elif subgroup == SubgroupEnum.NO_HFPEF_SUSPECT_NOIMG:
            df_result = UtilsPandas.filter_out(df=df, values_to_filter=[1], header_name=SubgroupEnum.HFPEF_SUSPECT_NOIMG.value)
        elif subgroup == SubgroupEnum.NO_HF_RV:
            df_result = UtilsPandas.filter_out(df=df, values_to_filter=[1], header_name=SubgroupEnum.HF_RV.value)
        elif subgroup == SubgroupEnum.HF_EVER:
            # 0 represents cases with no heart failure
            df_result = UtilsPandas.filter_out(df=df, values_to_filter=[0], header_name="ever_dx_cv_heartfailure_all")

        elif "no_clust" in subgroup.name.lower():
            df_clusters = UtilsPandas.read_csv(CLUSTER_PATH)
            if "new" in subgroup.name.lower():
                df_clusters = UtilsPandas.read_csv(CLUSTER_PATH_NEW)
            elif "big" in subgroup.name.lower():
                df_clusters = UtilsPandas.read_csv(CLUSTER_BIG_PATH)

            eids = list(df_clusters.loc[df_clusters["cluster"] == (int(subgroup.value.split("_")[-1]) - 1), "eid"])
            df_result = df[~df["eid"].isin(eids)]

        elif "clust" in subgroup.name.lower():
            df_clusters = UtilsPandas.read_csv(CLUSTER_PATH)
            if "new" in subgroup.name.lower():
                df_clusters = UtilsPandas.read_csv(CLUSTER_PATH_NEW)
            elif "big" in subgroup.name.lower():
                df_clusters = UtilsPandas.read_csv(CLUSTER_BIG_PATH)

            eids = list(df_clusters.loc[df_clusters["cluster"] == (int(subgroup.value.split("_")[-1]) - 1), "eid"])
            df_result = df[df["eid"].isin(eids)]
        else:
            raise ValueError(f"{subgroup} does not exist")
        return df_result

    def get_main_df(self) -> pd.DataFrame:
        return self.df.copy()



def load_short2panel_dict() -> dict[str: str]:
    df = UtilsPandas.read_csv(SHORT2LONG_PATH_PROT)
    short2panel_dic = df.set_index("short_names").to_dict(orient="dict")["protein_panel"]
    return short2panel_dic

def load_short2long_dict() -> dict[str: str]:
    short2long = UtilsPandas.read_csv(SHORT2LONG_PATH)
    short2long_dic = short2long.set_index("short_names").to_dict(orient="dict")["long_names"]
    return short2long_dic


def reformat_short2long_dict(s2l: dict[str: str],
                             col_names: list[str],
                             mod: DataModalityEnum) -> dict[str: str]:
    split_sign = "!!!!!!!!!!"
    st_index = 0
    split_index = 0
    if mod in (DataModalityEnum.METAB_MARCUS_90, DataModalityEnum.PHYSIO, DataModalityEnum.LAB):
        split_sign = "_"
    if "prot" in mod.name.lower():
        # split_sign = ";"
        split_sign = "!!!!!!!!!!"
        st_index = 2
    if "quest" in mod.name.lower():
        split_sign = "_"
        split_index = -2
    if mod in (DataModalityEnum.DIAG, DataModalityEnum.SYMP):
        split_sign = "_"
        split_index = -1

    for short_name in col_names:
        if short_name in s2l:
            s2l[short_name] = s2l[short_name].split(split_sign)[split_index][st_index:]
        else:
            s2l[short_name] = short_name

    return s2l