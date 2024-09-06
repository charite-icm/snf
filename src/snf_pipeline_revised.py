import pandas as pd


def _check_validity_loaded_data(dfs: tuple[pd.DataFrame]) -> None:
    """
    Validate a tuple of pandas DataFrames to ensure they meet the required conditions.

    This function checks whether the provided tuple of DataFrames meets the following criteria:
    1. The tuple contains at least two DataFrames.
    2. Each element in the tuple is a pandas DataFrame.
    3. Each DataFrame contains a column named 'eid'.
    4. The 'eid' column in each DataFrame contains unique values (i.e., no duplicates).

    Parameters
    ----------
    dfs : tuple[pd.DataFrame]
        A tuple of pandas DataFrames to be validated.

    Raises
    ------
    ValueError
        If the tuple contains fewer than two DataFrames, if a DataFrame does not contain 
        the 'eid' column, or if the 'eid' column contains duplicate values.
    
    TypeError
        If any element in the tuple is not a pandas DataFrame.
    """

    if not isinstance(dfs, tuple):
        raise TypeError(f"dfs should be of type tuple not {type(dfs)}")
    
    if len(dfs) < 2:
        raise ValueError(f"Minimum number of data modalities is 2. {len(dfs)} provided")
    
    for df in dfs:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Invalid type of data: {type(df)}. It should be pandas.DataFrame")
        if "eid" not in df.columns:
            raise ValueError("Column with IDs should be named 'eid'")
        if df["eid"].duplicated().any():
            raise ValueError("Column 'eid' does not contain unique values.")



# def remove

# def _get_overlapped_modalities_with_nan(self) -> None:
#     # TODO: It should be based on the eids not indices!!!
#     print("------------- GETTING OVERLAPPED MODALITIES WITH MISSING DATA --------------------")
#     valid_index: list[list] = []
#     for modality in self.modalities:
#         df = self.dfs[modality].copy()
#         mean_nan_per_row = df.isna().mean(axis=1)
#         index_under_th = list(df[mean_nan_per_row < self.th_nan].index)
#         valid_index.append(index_under_th)
#         print(f"{modality.value}: {len(index_under_th)}/{df.shape[0]} data "
#                 f"with <{int(self.th_nan*100)} % NaN values")


#         fig, ax = plt.subplots()
#         ax.set_title(f"Amount of missing data per row - {modality.name.lower()}",
#                         fontweight="bold")
#         histplot = sns.histplot(mean_nan_per_row*100, bins=20, kde=False, color='skyblue', ax=ax)
#         histplot.set(xlabel="% of missing data")
#         ax.bar_label(histplot.containers[0], fmt="%d", label_type="edge", fontsize=8, color="black", weight="bold",
#                         labels=[str(v) if v else '' for v in histplot.containers[0].datavalues])

#         ax.axvline(x=self.th_nan*100, color="red", linestyle="--", label="threshold")
#         ax.legend()
#         save_figure(fig, os.path.join(RESULTS_PATH, self.result_dir_name, f"missing_data_{modality.name.lower()}"),
#                     plt_close=True)









# def _load_snps(self) -> None:
#     data_loader_snps = DataLoaderSnps10k(data_selection=self.data_selection,
#                                             subgroup=self.subgroup)
#     data_loader_snps.load()
#     df_snps = data_loader_snps.get_main_df().sort_values(by="eid", ascending=True).reset_index(drop=True)
#     self.dfs[DataModalityEnum.SNPS_10K] = df_snps
#     self.modalities.append(DataModalityEnum.SNPS_10K)

#     snps_eids = df_snps["eid"].tolist()
#     for modality, df in self.dfs.items():
#         self.dfs[modality] = df[df["eid"].isin(snps_eids)].sort_values(by="eid", ascending=True).reset_index(drop=True)

# def get_overlapped_modalities(self) -> None:
#     self.dfs_overlapped: dict[DataModalityEnum: pd.DataFrame] = {}
#     if self.th_nan is None:
#         self._get_overlapped_modalities_no_nan()
#     else:
#         self._get_overlapped_modalities_with_nan()

#     print("------------- DF OVERLAPPED SHAPES --------------------")
#     for modality, df in self.dfs.items():
#         self.dfs_overlapped[modality] = df.loc[self.intersection_index]
#         print(f"{modality.value} shape: {df.loc[self.intersection_index].shape}")


# def _get_overlapped_modalities_no_nan(self) -> None:
#     self.df_concat, self.intersection_index = UtilsPandas.get_overlapping_indices(
#         dfs=[self.dfs[modality] for modality in self.modalities],
#         on="eid"
#     )

# def _get_overlapped_modalities_with_nan(self) -> None:
#     # TODO: It should be based on the eids not indices!!!
#     print("------------- GETTING OVERLAPPED MODALITIES WITH MISSING DATA --------------------")
#     valid_index: list[list] = []
#     for modality in self.modalities:
#         df = self.dfs[modality].copy()
#         mean_nan_per_row = df.isna().mean(axis=1)
#         index_under_th = list(df[mean_nan_per_row < self.th_nan].index)
#         valid_index.append(index_under_th)
#         print(f"{modality.value}: {len(index_under_th)}/{df.shape[0]} data "
#                 f"with <{int(self.th_nan*100)} % NaN values")


#         fig, ax = plt.subplots()
#         ax.set_title(f"Amount of missing data per row - {modality.name.lower()}",
#                         fontweight="bold")
#         histplot = sns.histplot(mean_nan_per_row*100, bins=20, kde=False, color='skyblue', ax=ax)
#         histplot.set(xlabel="% of missing data")
#         ax.bar_label(histplot.containers[0], fmt="%d", label_type="edge", fontsize=8, color="black", weight="bold",
#                         labels=[str(v) if v else '' for v in histplot.containers[0].datavalues])

#         ax.axvline(x=self.th_nan*100, color="red", linestyle="--", label="threshold")
#         ax.legend()
#         save_figure(fig, os.path.join(RESULTS_PATH, self.result_dir_name, f"missing_data_{modality.name.lower()}"),
#                     plt_close=True)

#     print("------------- OVERLAPPED MISSING DATA --------------------")
#     # Find the intersection of elements
#     intersection_result = set(valid_index[0])
#     for l in valid_index[1:]:
#         intersection_result = intersection_result.intersection(l)
#     # Convert the result back to a list if needed
#     self.intersection_index = list(intersection_result)
#     print(f"{len(self.intersection_index)}/{self.dfs[self.modalities[0]].shape[0]} intersected data!")