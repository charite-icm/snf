import os
import pandas as pd


class WrongFileExtension(Exception):
    def __init__(self, wrong_ext: str, correct_ext) -> None:
        self.message = f"Wrong file extension: {wrong_ext}. It should be: {correct_ext}"
        super().__init__(self.message)

# import os
#
# file_path = "path/to/your/file.txt"
#
# # Get the file extension, .extension or an empty string in case of no file extension
# file_extension = os.path.splitext(file_path)[1]
#
# # Print the file extension
# print("File extension:", file_extension)


class UtilsPandas:
    @staticmethod
    def remove_duplicated_columns(df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[:, ~df.columns.duplicated()]


    @staticmethod
    def read_feather(path: str) -> pd.DataFrame:
        # file_extension = os.path.splitext(path)[1]

        if not path.endswith(".feather"):
            path = f"{path}.feather"

        df = pd.read_feather(path)
        print(f"{path} loaded!")
        print(f"shape: {df.shape}")
        return df

    @staticmethod
    def save_feather(df: pd.DataFrame, path: str) -> None:
        df = UtilsPandas.remove_duplicated_columns(df)
        df.to_feather(path)
        print(f"{path} saved!")
        print(f"shape: {df.shape}")


    @staticmethod
    def read_csv(path: str, sep: str = ",") -> pd.DataFrame:
        if not path.endswith(".csv"):
            path = f"{path}.csv"
        df = pd.read_csv(path, encoding="utf-8", sep=sep)
        print(f"{path} loaded!")
        print(f"shape: {df.shape}")
        return df

    @staticmethod
    def save_csv(df: pd.DataFrame, path: str, index: bool = False) -> None:
        df.to_csv(path, index=index)
        print(f"{path} saved!")
        print(f"shape: {df.shape}")

    @staticmethod
    def filter_out_multiple(df: pd.DataFrame, header_names: list[str], values_to_filter: list[list]) -> pd.DataFrame:
        if len(header_names) != len(values_to_filter):
            raise ValueError(f"Length of header_names (f{len(header_names)}) "
                             f"and values_to_filter ({len(values_to_filter)}) must be the same!")
        # # Create a boolean mask based on the condition
        # mask = df[header_names[0]].isin(values_to_filter[0])
        # for i in range(1, len(header_names)):
        #     mask |= df[header_names[i]].isin(values_to_filter[i])
        #
        # # Use the mask to filter out rows
        # filtered_df = df[~mask]
        # filtered_df = filtered_df.reset_index(drop=True)
        filtered_df = df.copy()
        for header_name, values in zip(header_names, values_to_filter):
            filtered_df = UtilsPandas.filter_out(df=filtered_df, values_to_filter=values, header_name=header_name)

        return filtered_df

    @staticmethod
    def filter_out(df: pd.DataFrame,
                   values_to_filter: list,
                   header_name: str) -> pd.DataFrame:
        # Create a boolean mask based on the condition
        mask = df[header_name].isin(values_to_filter)

        # Use the mask to filter out rows
        filtered_df = df[~mask]
        filtered_df = filtered_df.reset_index(drop=True)

        return filtered_df

    @staticmethod
    def drop_columns(df: pd.DataFrame, columns_to_drop: list[str]) -> pd.DataFrame:
        return df.drop(columns=columns_to_drop)

    @staticmethod
    def add_row(df: pd.DataFrame, new_row: dict[str: str]) -> None:
        df.loc[len(df)] = new_row

    @staticmethod
    def get_overlapping_indices(dfs: list[pd.DataFrame], on: str = "eid") -> tuple[pd.DataFrame, pd.Index]:
        if len(dfs) < 2:
            raise ValueError(f"Minimum number of dfs: 2 ({len(dfs)} given)")

        df_concat = dfs[0].copy()
        for df in dfs[1:]:
            df_concat = df_concat.merge(df, on=on)
        df_concat.dropna(inplace=True)

        return df_concat, df_concat.index

    @staticmethod
    def count_duplicated_rows(df: pd.DataFrame, subset: list[str] = None) -> int:
        """

        :param df:
        :param subset: list of columns to look for duplicates, if None all columns are included
        :return:
        """
        duplicated_count = df.duplicated().sum()
        if subset is not None:
            duplicated_count = df.duplicated(subset=subset).sum()
        return duplicated_count

    @staticmethod
    def drop_duplicates(df: pd.DataFrame, subset: list[str] = None,  keep: str | bool = "first") -> pd.DataFrame:
        """

        :param df:
        :param subset: list of columns to look for duplicates, if None all columns are included
        :param keep: "first", "last" or False
        :return:
        """
        df_results = df.drop_duplicates(keep=keep)
        if subset is not None:
            df_results = df.drop_duplicates(subset=subset, keep=keep)
        return df_results


    @staticmethod
    def check_string_in_lists(df: pd.DataFrame, column_name: str, string_to_check: str) -> list[bool]:
        """
        Check if a string is present in each list in a specified column of a DataFrame.

        Parameters:
        - df: DataFrame
        - column_name: str, the column containing lists
        - string_to_check: str, the string to check in each list

        Returns:
        - DataFrame with a new column indicating whether the string is present in each list
        """
        df_copy = df.copy()  # Create a copy to avoid modifying the original DataFrame
        result = df_copy[column_name].apply(lambda x: string_to_check in x)
        return result

    @staticmethod
    def reset_and_drop_index(df: pd.DataFrame) -> pd.DataFrame:
        df_result = df.copy()
        df_result.reset_index(inplace=True)
        df_result.drop("index", axis=1, inplace=True)
        return df_result

    @staticmethod
    def df2dict(df, column_key: str, column_item: str) -> dict:
        """
        Convert DataFrame to dictionary using specified columns for keys and items.

        Parameters:
            df (DataFrame): Input DataFrame.
            column_key (str): Name of the column to be used as keys in the dictionary.
            column_item (str): Name of the column to be used as items in the dictionary.

        Returns:
            dict: Dictionary where keys are taken from the specified column_key and values
                  are taken from the specified column_item.
        """
        return dict(zip(df[column_key], df[column_item]))



