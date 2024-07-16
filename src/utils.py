import datetime
from enum import Enum
import json
import logging
import os
from typing import Any

import matplotlib.pyplot as plt
from pathlib import Path

class Utils():
    @staticmethod
    def create_path(host: str, params: list[str]) -> str:
        """
        Create a new path by joining the host directory path with a list of subdirectories.

        Args:
            host (str): The base directory path.
            params (list of str): List of subdirectories or path components.

        Returns:
            str: The newly created path by joining the host and subdirectories.
        """
        path = host
        for param in params:
            path = os.path.join(path, param)
        return path 

    @staticmethod
    def remove_items_from_list(input_list: list[Any], items_to_remove: list[Any]) -> None:
        """
        Remove specific items from a list.

        Args:
            input_list (list): The input list from which items will be removed.
            items_to_remove (list): List of items to be removed from the input list.

        Returns:
            None. The input list is modified in-place.
        """
        for item in items_to_remove:
            if item in input_list:
                input_list.remove(item)

    @staticmethod
    def keep_items_in_list(input_list: list[Any], items_to_keep: list[Any]) -> list[Any]:
        """
        Create a new list containing only specific items from the input list.

        Args:
            input_list (list): The input list from which items will be selected.
            items_to_keep (list): List of items to be kept in the new list.

        Returns:
            list: A new list containing only the selected items from the input list.
        """
        output = []
        for item in items_to_keep:
            if item in input_list:
                output.append(item)
        return output

    @staticmethod
    def write_list_to_txt(file_path: str, my_list: list[Any]) -> None:
        # Öffne die Datei im Schreibmodus
        with open(file_path, 'w') as file:
            # Schreibe jede Zeichenkette in eine separate Zeile
            for string in my_list:
                file.write(string + '\n')
        print(f"Die Liste wurde erfolgreich in die Datei {file_path} geschrieben.")

    @staticmethod
    def read_txt_to_list(file_path: str):
        result_list = []
        # try:
        # Öffne die Datei im Lesemodus
        with open(file_path, 'r') as file:
            # Lese jede Zeile der Datei und füge sie zur Ergebnisliste hinzu
            for line in file:
                result_list.append(
                    line.strip())  # .strip() entfernt Leerzeichen und Zeilenumbrüche am Anfang und Ende der Zeile
        return result_list
        # except Exception as e:
        #     print(f'Fehler beim Lesen der Datei: {str(e)}')
        #     return None

    @staticmethod
    def find_enum_member(input_str: str, enum_class) -> None | Enum:
        for member in enum_class.__members__:
            if enum_class[member].value == input_str:
                return enum_class[member]
        return None

    @staticmethod
    def get_file_name_without_extension(file_path: str) -> str:
        """
        Extracts the name of the file without its extension from the given file path.

        Args:
            file_path (str): The path to the file.

        Returns:
            str: The file name without extension.

        Example:
            >>> file_path = "/path/to/your/file/example.txt"
            >>> result = get_file_name_without_extension(file_path)
            >>> print(result)
            'example'
        """
        base_name = os.path.basename(file_path)
        file_name_without_extension, _ = os.path.splitext(base_name)
        return file_name_without_extension

    @staticmethod
    def get_file_extension(file_path: str) -> str:
        """
        Extracts the name of the file without its extension from the given file path.

        Args:
            file_path (str): The path to the file.

        Returns:
            str: The file name without extension.

        Example:
            >>> file_path = "/path/to/your/file/example.txt"
            >>> result = get_file_name_without_extension(file_path)
            >>> print(result)
            'example'
        """
        base_name = os.path.basename(file_path)
        _, extension = os.path.splitext(base_name)
        return extension

    @staticmethod
    def get_current_date_string() -> str:
        current_datetime = datetime.datetime.now()
        date_time_string = current_datetime.strftime("%Y_%m_%d__%H_%M_%S")
        return date_time_string

    @staticmethod
    def create_logger(logfile_path: str) -> logging:
        # Create a new logger
        logger = logging.getLogger("feature_selector_logger")
        logger.setLevel(logging.INFO)
        # Create a file handler and set the level to INFO
        fh = logging.FileHandler(logfile_path)
        fh.setLevel(logging.INFO)
        # Create a formatter and add it to the handler
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        # Add the handler to the logger
        logger.addHandler(fh)

        return logger

    @staticmethod
    def save_dict_to_json(dictionary: dict, file_path: str) -> None:
        """
        Save a dictionary to a JSON file.

        Args:
        dictionary (dict): The dictionary to be saved.
        file_path (str): The path to the JSON file.

        Returns:
        None
        """
        with open(file_path, "w") as json_file:
            json.dump(dictionary, json_file, indent=4)

    @staticmethod
    def read_dict_from_json(file_path: str) -> dict:
        """
        Read a JSON file and return its contents as a dictionary.

        Args:
        file_path (str): The path to the JSON file.

        Returns:
        dict: The contents of the JSON file as a dictionary.
        """
        with open(file_path, "r") as json_file:
            dictionary = json.load(json_file)
        return dictionary

    @staticmethod
    def save_figure(fig: plt.figure, path: Path, fig_name: str, img_format: str = "jpg",
                    plt_close: bool = False, dpi: int = 300) -> None:
        plt.tight_layout()
        fig_full_name = f"{fig_name}.{img_format}"
        fig.savefig(os.path.join(path, fig_full_name), dpi=dpi)
        print(f"{fig_full_name} saved!")
        if plt_close:
            plt.close()

    @staticmethod
    def map_enum_names(enum_class) -> dict[str: Enum]:
        """
        Map all possible names of enum class attributes to enum class attributes.

        Args:
        - enum_class: The enum class.

        Returns:
        - A dictionary mapping names to enum class attributes.
        """
        return {member.name: member for member in enum_class}


