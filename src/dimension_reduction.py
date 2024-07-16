import datetime
from enum import Enum, auto
from sklearn.decomposition import PCA
import os
import pandas as pd


class ReduceDimAlgosEnum(Enum):
    PCA = auto()


class ReduceDimAlgoPCA:
    def __init__(self, df: pd.DataFrame, n_components: float) -> None:
        self.df = df.copy()
        self.n_components = n_components

    def run(self) -> None:
        print(f"Shape before PCA: {self.df.shape}")

        pca = PCA(n_components=self.n_components)
        numpy_arr = pca.fit_transform(self.df)
        self.df = pd.DataFrame(data=numpy_arr)

        print(f"Shape after PCA: {self.df.shape}")
        print(f"explained variance ratio: {pca.explained_variance_ratio_}")


    def get_short_name(self) -> str:
        return f"pca_{self.n_components}"




    # def high_correlation_filter(self) -> None:
    #     # Pearson correlation matrix, VIF (Variance Inflation Factor)
    #     # Remove variables with VIF > 5 ?
    #     pass
    #
    # def low_variance_filter(self) -> None:
    #     # Data must be scaled first!!!
    #     pass
    #
    # def run_autoencoder(self) -> None:
    #     pass









