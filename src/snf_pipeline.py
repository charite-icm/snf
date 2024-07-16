from enum import Enum
from itertools import combinations
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.cluster import spectral_clustering
from sklearn.utils.validation import check_symmetric
from sklearn.metrics import silhouette_score

from upsetplot import plot as upsplot

from src.data_loader import DataLoader, DataSelectorEnum, DataModalityEnum, SubgroupEnum
from src.data_loader import DataLoaderSnps10k
from src.image_format import *
from src.utils_pandas import UtilsPandas
from src.utils import Utils
from src.before_clustering import BeforeClustering, ValFeaturesXEnum, ValFeaturesYEnum 
from src.cluster_algorithm import ClusterAlgorithm
from src.snf.compute import make_affinity, DistanceMetric, snf, get_n_clusters, make_affinity_nan, get_n_clusters_revised


from src.ordered_affinity_matrix import plot_ordered_affinity_matrix


RESULTS_PATH = "results"


class FoldersEnum(Enum):
    AFFINITY = "affinity"
    VALIDATION = "validation"
    OMICS_PROFILE = "omics_profile"
    VALIDATION_VERBOSE = "validation_verbose"
    QUEST_CONT = DataModalityEnum.QUEST_CONT.value
    QUEST_BINOM = DataModalityEnum.QUEST_BINOM.value
    QUEST_CATEG = DataModalityEnum.QUEST_CATEG.value
    DIAG = DataModalityEnum.DIAG.value
    SYMP = DataModalityEnum.SYMP.value
    MED_VIT = DataModalityEnum.MED_VIT.value

METADATA_CSV = "metadata.csv"


class MetadataValuesEnum(Enum):
    DATA_SELECTION = "data_selection_member"
    SUBGROUP = "subgroup_member"
    N_CLUSTERS = "n_clusters"



class SnfPipeline:
    def __init__(self, data_selection: DataSelectorEnum, modalities: list[DataModalityEnum], subgroup: SubgroupEnum,
                 n_clusters: int | None, result_dir_name: str,
                 val_x_enum: ValFeaturesXEnum, val_y_enum: ValFeaturesYEnum,
                 th_nan: float = None,
                 snps: bool = False,
                 random_state: int = 42) -> None:
        self.data_selection = data_selection
        self.modalities = modalities
        self.subgroup = subgroup
        self.n_clusters = n_clusters
        self.result_dir_name = result_dir_name
        self.val_x_enum = val_x_enum
        self.val_y_enum = val_y_enum
        self.th_nan = th_nan
        self.snps = snps
        self.random_state = random_state

        self.eids = None

        self._validate_init_param()
        self._mk_result_dir()

    def _validate_init_param(self) -> None:
        if len(self.modalities) < 2:
            raise ValueError(f"Minimum number of modalities: 2 ({len(self.modalities)} given)")

        if self.n_clusters is not None:
            if self.n_clusters < 2:
                raise ValueError(f"Minimum number of clusters: 2 ({self.n_clusters} given)")

    def _mk_result_dir(self) -> None:
        if not os.path.exists(os.path.join(RESULTS_PATH, self.result_dir_name)):
            os.makedirs(os.path.join(RESULTS_PATH, self.result_dir_name), exist_ok=True)

    def load_data(self) -> None:
        self.dfs: dict[DataModalityEnum: pd.DataFrame] = {}
        all_modalities = [*self.modalities, DataModalityEnum.VALIDATION]

        for modality in all_modalities:
            data_loader = DataLoader(
                data_selection=self.data_selection,
                data_modality=modality,
                subgroup=self.subgroup
            )
            data_loader.load()
            self.dfs[modality] = data_loader.get_main_df()

        if self.snps:
            self._load_snps()

        print("----------- DF SHAPES ----------------")
        for modality, df in self.dfs.items():
            print(f"{modality.value} shape: {df.shape}")


    def _load_snps(self) -> None:
        data_loader_snps = DataLoaderSnps10k(data_selection=self.data_selection,
                                             subgroup=self.subgroup)
        data_loader_snps.load()
        df_snps = data_loader_snps.get_main_df().sort_values(by="eid", ascending=True).reset_index(drop=True)
        self.dfs[DataModalityEnum.SNPS_10K] = df_snps
        self.modalities.append(DataModalityEnum.SNPS_10K)

        snps_eids = df_snps["eid"].tolist()
        for modality, df in self.dfs.items():
            self.dfs[modality] = df[df["eid"].isin(snps_eids)].sort_values(by="eid", ascending=True).reset_index(drop=True)

    def get_overlapped_modalities(self) -> None:
        self.dfs_overlapped: dict[DataModalityEnum: pd.DataFrame] = {}
        if self.th_nan is None:
            self._get_overlapped_modalities_no_nan()
        else:
            self._get_overlapped_modalities_with_nan()

        print("------------- DF OVERLAPPED SHAPES --------------------")
        for modality, df in self.dfs.items():
            self.dfs_overlapped[modality] = df.loc[self.intersection_index]
            print(f"{modality.value} shape: {df.loc[self.intersection_index].shape}")


    def _get_overlapped_modalities_no_nan(self) -> None:
        self.df_concat, self.intersection_index = UtilsPandas.get_overlapping_indices(
            dfs=[self.dfs[modality] for modality in self.modalities],
            on="eid"
        )

    def _get_overlapped_modalities_with_nan(self) -> None:
        # TODO: It should be based on the eids not indices!!!
        print("------------- GETTING OVERLAPPED MODALITIES WITH MISSING DATA --------------------")
        valid_index: list[list] = []
        for modality in self.modalities:
            df = self.dfs[modality].copy()
            mean_nan_per_row = df.isna().mean(axis=1)
            index_under_th = list(df[mean_nan_per_row < self.th_nan].index)
            valid_index.append(index_under_th)
            print(f"{modality.value}: {len(index_under_th)}/{df.shape[0]} data "
                  f"with <{int(self.th_nan*100)} % NaN values")


            fig, ax = plt.subplots()
            ax.set_title(f"Amount of missing data per row - {modality.name.lower()}",
                         fontweight="bold")
            histplot = sns.histplot(mean_nan_per_row*100, bins=20, kde=False, color='skyblue', ax=ax)
            histplot.set(xlabel="% of missing data")
            ax.bar_label(histplot.containers[0], fmt="%d", label_type="edge", fontsize=8, color="black", weight="bold",
                         labels=[str(v) if v else '' for v in histplot.containers[0].datavalues])

            ax.axvline(x=self.th_nan*100, color="red", linestyle="--", label="threshold")
            ax.legend()
            save_figure(fig, os.path.join(RESULTS_PATH, self.result_dir_name, f"missing_data_{modality.name.lower()}"),
                        plt_close=True)

        print("------------- OVERLAPPED MISSING DATA --------------------")
        # Find the intersection of elements
        intersection_result = set(valid_index[0])
        for l in valid_index[1:]:
            intersection_result = intersection_result.intersection(l)
        # Convert the result back to a list if needed
        self.intersection_index = list(intersection_result)
        print(f"{len(self.intersection_index)}/{self.dfs[self.modalities[0]].shape[0]} intersected data!")


    def make_dirs(self) -> None:
        self.mod_paths: dict[DataModalityEnum: str] = {}
        for modality in self.modalities:
            self.mod_paths[modality] = os.path.join(self.result_dir_name, modality.value)
        self.fused_path = os.path.join(self.result_dir_name, "fused")

        all_paths = [*[path for _, path in self.mod_paths.items()], self.fused_path]
        for path in all_paths:
            if not os.path.exists(os.path.join(RESULTS_PATH, path)):
                os.mkdir(os.path.join(RESULTS_PATH, path))

            for folder_enum in FoldersEnum:
                plot_path = os.path.join(RESULTS_PATH, path, folder_enum.value)
                if not os.path.exists(plot_path):
                    os.mkdir(plot_path)

    def prepare_for_clustering(self) -> None:
        self.np_arrays: dict[DataModalityEnum: np.ndarray] = {}
        if self.th_nan is None:
            self._prepare_for_clustering_no_nan()
        else:
            self._prepare_for_clustering_with_nan()
        print("------------ NP SHAPES ---------------------")
        for mod, arr in self.np_arrays.items():
            print(f"{mod} shape: {arr.shape}")

    def _prepare_for_clustering_no_nan(self):
        self.before_clust: dict[DataModalityEnum: BeforeClustering] = {}
        for modality in self.modalities:
            log_scale = False
            if "prot" in modality.name.lower():
                log_scale = True
            before_clust = BeforeClustering(df_train=self.dfs_overlapped[modality],
                                            df_val=self.dfs_overlapped[DataModalityEnum.VALIDATION],
                                            save_folder=self.mod_paths[modality],
                                            val_y_enum=self.val_y_enum, val_x_enum=self.val_x_enum,
                                            reduce_dim=None, pca_variance_to_keep=None, log_scale=log_scale)
            before_clust.main()
            self.before_clust[modality] = before_clust
            self.np_arrays[modality] = before_clust.get_np_train()
        self.np_arrays[DataModalityEnum.VALIDATION] = before_clust.get_np_val()
        self.eids = before_clust.get_eids()
        self.val_x_feature_name = self.before_clust[self.modalities[0]].get_val_x_feature_name()
        self.val_y_feature_name = self.before_clust[self.modalities[0]].get_val_y_feature_name()
        # We only need it for the path to save cluster plots
        self.before_clust_concat = BeforeClustering(df_train=self.df_concat,
                                                    df_val=self.dfs_overlapped[DataModalityEnum.VALIDATION],
                                                    save_folder=self.fused_path,
                                                    val_y_enum=self.val_y_enum, val_x_enum=self.val_x_enum,
                                                    reduce_dim=None, pca_variance_to_keep=None)
        self.before_clust_concat.main()

    def _prepare_for_clustering_with_nan(self):
        for modality in ([*self.modalities, DataModalityEnum.VALIDATION]):
            if self.eids is None:
                self.eids = self.dfs_overlapped[modality]["eid"].tolist()

            data_arr = self.dfs_overlapped[modality].drop(columns=["eid"]).to_numpy()
            if "prot" in modality.name.lower():
                data_arr = 2**data_arr

            self.np_arrays[modality] = np.array(data_arr)
        # self.K = 20
        self.K = len(self.eids) // 10

    def compute_aff_networks(self) -> None:
        if self.th_nan is None:
            affinity_networks = make_affinity([self.np_arrays[mod] for mod in self.modalities],
                                              metric=DistanceMetric.SQEUCLIDEAN.value, K=self.K,
                                              mu=0.5, normalize=True)
        else:
            affinity_networks = make_affinity_nan([self.np_arrays[mod] for mod in self.modalities],
                                                  metric=DistanceMetric.SQEUCLIDEAN.value, K=self.K,
                                                  mu=0.5, normalize=True)

        affinity_networks = [W / np.nansum(W, axis=1, keepdims=True) for W in affinity_networks]
        affinity_networks = [check_symmetric(W, raise_warning=False) for W in affinity_networks]

        self.affinity_networks = {mod: aff for mod, aff in zip(self.modalities, affinity_networks)}
        print("--------------------------------")
        print(f"{len(self.affinity_networks)} affinity matrices generated")

        # for i in range(len(affinity_networks)):
        #     print(np.sum(affinity_networks[i], axis=1))
        #     print(np.sum(affinity_networks[i], axis=0))

    def compute_fused_network(self) -> None:
        self.fused_network = snf([aff for _, aff in self.affinity_networks.items()], K=self.K)
        best, second = get_n_clusters(self.fused_network)
        if self.n_clusters is None:
            self.n_clusters = best

        self.csv_name = f"eids_{self.n_clusters}_clusters"

        print("-----------------------------------------")
        print(f"Fused matrix generated. Suggested number of clusters: ({best}, {second})")

        nb_clusters, eigenvalues, eigenvectors = get_n_clusters_revised(self.fused_network)
        print()
        print(nb_clusters)
        print(eigenvalues)
        print(eigenvectors)
        print()

    def spectral_clustering_modalities(self) -> None:
        self.mod_labels: dict[DataModalityEnum: list[int]] = {}
        self.eids_clusters_csv_paths: dict[DataModalityEnum: str] = {}
        for modality in self.modalities:
            labels = spectral_clustering(self.affinity_networks[modality], n_clusters=self.n_clusters, random_state=self.random_state)
            self.mod_labels[modality] = labels

            # clustering_plotter = ClusteringPlotter(n_clusters=self.n_clusters,
            #                                        feature1_name=self.val_x_feature_name,
            #                                        feature2_name=self.val_y_feature_name,
            #                                        title=f"Spectral - {modality.value.capitalize()}",
            #                                        save_path=self.before_clust[modality].get_save_path(),
            #                                        testing_plot_np=self.np_arrays[DataModalityEnum.VALIDATION],
            #                                        labels=labels,
            #                                        plt_close=True)
            # clustering_plotter.plot()
            ClusterAlgorithm.save_cluster_eids(
                labels=list(labels),
                eids=list(self.eids),
                save_path=os.path.join(RESULTS_PATH, self.mod_paths[modality]),
                csv_name=self.csv_name
            )

    def spectral_clustering_fused(self, rearange_dict: dict[int, int] = None) -> None:
        self.fused_labels = spectral_clustering(self.fused_network, n_clusters=self.n_clusters, random_state=self.random_state)
        # cls_plotter = ClusteringPlotter(n_clusters=self.n_clusters,
        #                                 feature1_name=self.val_x_feature_name,
        #                                 feature2_name=self.val_y_feature_name,
        #                                 title="Spectral - Fused",
        #                                 save_path=self.before_clust_concat.get_save_path(),
        #                                 testing_plot_np=self.np_arrays[DataModalityEnum.VALIDATION],
        #                                 labels=self.fused_labels,
        #                                 plt_close=True)
        # cls_plotter.plot()

        if rearange_dict is not None:
            self.fused_labels = [rearange_dict[val] for val in self.fused_labels]

        ClusterAlgorithm.save_cluster_eids(
            labels=list(self.fused_labels),
            eids=list(self.eids),
            save_path=os.path.join(RESULTS_PATH, self.fused_path),
            csv_name=self.csv_name
        )

    def spectral_clustering_internal_validation(self) -> None:
        x, y, y_diag = [], [], []
        fig, ax = plt.subplots()


        for n_clusters in range(2, 20):
            aff_matrix = np.array(self.fused_network)
            aff_matrix_diag = np.array(self.fused_network)
            np.fill_diagonal(aff_matrix_diag, 0)

            fused_labels = spectral_clustering(aff_matrix, n_clusters=n_clusters)
            fused_labels_diag = spectral_clustering(aff_matrix_diag, n_clusters=n_clusters)
            # Calculate silhouette score
            np.fill_diagonal(aff_matrix, 0)
            silhouette_avg = silhouette_score(aff_matrix, fused_labels, metric='precomputed')
            silhouette_avg_diag = silhouette_score(aff_matrix_diag, fused_labels_diag, metric='precomputed')

            x.append(n_clusters)
            y.append(silhouette_avg)
            y_diag.append(silhouette_avg_diag)

        plt.plot(x, y, label="orig")
        plt.plot(x, y_diag, label="diag_zero")
        ax.legend()
        ax.set_xlabel("Clusters")
        ax.set_ylabel("Silhouette score")
        ax.set_title("Internal Validation", fontweight="bold")
        save_figure(fig, fig_name=os.path.join(RESULTS_PATH, self.fused_path, FoldersEnum.AFFINITY.value,
                                               "silhouette_score"), plt_close=True)



    def plot_affinity_matrices(self, high_quality: bool = False) -> None:
        all_labels = [*[self.mod_labels[mod] for mod in self.modalities], self.fused_labels]
        all_paths = [*[self.mod_paths[mod] for mod in self.modalities], self.fused_path]

        for labels, path in zip(all_labels, all_paths):
            dynamic_range = plot_ordered_affinity_matrix(network=self.fused_network,
                                                         labels=labels,
                                                         figure_path=os.path.join(RESULTS_PATH, path, FoldersEnum.AFFINITY.value, "aff_fused"),
                                                         # title="Fused",
                                                         title=None,
                                                         dynamic_range_th=(0.1, 0.1),
                                                         figsize=(8.0, 8.0),
                                                         show_colorbar=False,
                                                         plt_close=True,
                                                         dynamic_range=None,
                                                         return_dynamic_range=True,
                                                         show_axis=False,
                                                         high_quality=high_quality)

            for modality in self.modalities:
                plot_ordered_affinity_matrix(network=self.affinity_networks[modality],
                                             labels=labels,
                                             figure_path=os.path.join(RESULTS_PATH, path, FoldersEnum.AFFINITY.value,
                                                                      f"aff_{modality.value}"),
                                             # title=f"{modality.value.split('_')[0].capitalize()}",
                                             title=None,
                                             dynamic_range_th=(0.1, 0.1),
                                             figsize=(8.0, 8.0),
                                             show_colorbar=False,
                                             plt_close=True,
                                             dynamic_range=dynamic_range,
                                             return_dynamic_range=False,
                                             show_axis=False,
                                             high_quality=high_quality)

    def order_affinity_matrices(self) -> None:
        indexing_array = np.argsort(self.fused_labels)
        self.affinity_networks_ordered: dict[DataModalityEnum: np.ndarray] = {}

        for modality in self.modalities:
            network = np.copy(self.affinity_networks[modality])
            np.fill_diagonal(network, 0)
            # We remove the diagonal elements which could greatly influence the normalization factor
            # (they are not constant on the diagonal)
            network /= np.nansum(network, axis=1, keepdims=True)
            np.fill_diagonal(network, 1)

            network_ordered = network[indexing_array][:, indexing_array]
            self.affinity_networks_ordered[modality] = network_ordered

    # TODO: Separate class, more detailed subfunctions, it should be more clear what is happening here
    def edge_contribution(self):
        self._get_list_of_edges()
        self._transform_to_upsetplot_format()
        self._plot_upset()

    def _get_list_of_edges(self) -> None:
        EDGE_TH = 1.10
        unique_elements, elements_counts = np.unique(self.fused_labels, return_counts=True)
        end_idxs = list(np.cumsum(elements_counts))
        start_idxs = [0, *end_idxs[:-1]]

        cluster_weights = {}
        for clust, (st, end) in enumerate(zip(start_idxs, end_idxs)):
            cluster_weights[clust] = []
            for i in range(st, end):
                for j in range(i, end):
                    sim_values = np.array([self.affinity_networks_ordered[mod][i][j] for mod in self.modalities])

                    max_sim_arg = np.argmax(sim_values)
                    percentage = sim_values[max_sim_arg] / np.array(sim_values)

                    non_zero_indices = np.nonzero(percentage < EDGE_TH)[0]
                    cluster_weights[clust].append(list(non_zero_indices))

        self.cluster_weights = cluster_weights

    def _transform_to_upsetplot_format(self) -> None:
        source_names = [mod.value.split("_")[0] for mod in self.modalities]

        combination_list = []
        for i in range(1, len(source_names) + 1):
            combination_list.extend(list(combinations(range(len(source_names)), i)))
        combination_list = [list(el) for el in combination_list]

        self.list_of_df_edges = []
        for key, item in self.cluster_weights.items():
            print_str = f"Cluster {str(key)}\n----------------\n"
            for i in range(len(combination_list)):
                bool_list = [combination_list[i] == list(el) for el in item]
                print_str += f"{[source_names[el] for el in combination_list[i]]}: {int(100 * np.sum(bool_list) / len(bool_list))} %\n"

            list_edges = []
            for el in item:
                add_list = [count in el for count in range(len(source_names))]
                list_edges.append(add_list)

            self.list_of_df_edges.append(pd.DataFrame(list_edges, columns=source_names))
        self.source_names = source_names

    def _plot_upset(self) -> None:
        fig_list = []
        for i, df in enumerate(self.list_of_df_edges):

            df = df.groupby(self.source_names).size()

            df_save = pd.DataFrame(df).rename(columns={0: "Number of edges"})
            df_save["Percentage of edges"] = df_save["Number of edges"] / df_save["Number of edges"].sum() * 100
            UtilsPandas.save_csv(df_save, os.path.join(RESULTS_PATH, self.fused_path, FoldersEnum.AFFINITY.value, f"upsetplot_clust_{i}.csv"), index=True)

            df = df.rename_axis(index={"physiology": "phenomics"})
            # percentage_sizes = (df / df.sum()) * 100
            fig = plt.figure(figsize=(10.0, 6.0))
            upsplot(df, fig=fig, show_percentages=True, element_size=None, sort_categories_by="input")
            fig_list.append(fig)
            save_figure(fig,
                        fig_name=os.path.join(RESULTS_PATH, self.fused_path, FoldersEnum.AFFINITY.value, f"upsetplot_clust_{i}"),
                        plt_close=True)
        # 13, 8
        big_fig = plt.figure(figsize=(6.0, 6.0))
        for i, fig in enumerate(fig_list):
            ax = big_fig.add_subplot((len(fig_list) + 1) // 2, 2, i + 1)
            ax.set_title(f"Cluster {i + 1}", fontweight="bold")

            ax.set_xticks([])  # Remove x-axis ticks
            ax.set_yticks([])  # Remove y-axis ticks
            ax.imshow(fig.canvas.buffer_rgba(), origin="upper")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)

        figure_path = os.path.join(RESULTS_PATH, self.fused_path, FoldersEnum.AFFINITY.value, f"upsetplot")
        # save_plot_as_vector(big_fig, format="pdf", dpi=600, output_path=f"{figure_path}.pdf")
        # save_plot_as_tiff(big_fig, column_type="double", dpi=600, output_path=f"{figure_path}.tiff")
        save_figure(big_fig, fig_name=f"{figure_path}.png", plt_close=True)

    def save_metadata(self) -> None:
        metadata = {
            MetadataValuesEnum.DATA_SELECTION.value: [self.data_selection.name],
            MetadataValuesEnum.SUBGROUP.value: [self.subgroup.name],
            MetadataValuesEnum.N_CLUSTERS.value: [self.n_clusters]
        }
        UtilsPandas.save_csv(pd.DataFrame(metadata), os.path.join(RESULTS_PATH, self.result_dir_name, METADATA_CSV))

    # def prepare_data_for_result_plots(self) -> None:
    #     self.all_labels = [self.fused_labels, *[self.mod_labels[mod] for mod in self.modalities]]

    #     all_paths = [self.fused_path, *[self.mod_paths[mod] for mod in self.modalities]]
    #     self.all_paths = [os.path.join(RESULTS_PATH, path) for path in all_paths]

    #     self.dfs_overlapped_mod = {key: self.dfs_overlapped[key] for key in self.modalities}

