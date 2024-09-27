import os
import pytest

from src.snf_pipeline_revised import _write_list_to_txt

def test_write_list_to_txt_valid_input(tmp_path):
    """
    Test that the function writes the list to a file successfully with valid input.
    """
    file_path = os.path.join(tmp_path, "test_output.txt")
    my_list = ["First line", "Second line", "Third line"]

    _write_list_to_txt(file_path, my_list, verbose=False)

    # Read the file and check contents
    with open(file_path, "r", encoding='utf-8') as file:
        lines = file.read().splitlines()
    assert lines == my_list

def test_write_list_to_txt_invalid_my_list_type():
    """
    Test that the function raises a TypeError when 'my_list' is not a list.
    """
    file_path = "test_output.txt"
    my_list = "This is a string, not a list"

    with pytest.raises(TypeError, match="'my_list' must be a list"):
        _write_list_to_txt(file_path, my_list, verbose=False)

def test_write_list_to_txt_non_string_elements():
    """
    Test that the function raises a TypeError when elements in 'my_list' are not strings.
    """
    file_path = "test_output.txt"
    my_list = ["First line", 2, "Third line"]

    with pytest.raises(TypeError, match="All elements in 'my_list' must be strings"):
        _write_list_to_txt(file_path, my_list, verbose=False)

def test_write_list_to_txt_create_directories(tmp_path):
    """
    Test that the function creates directories if they do not exist.
    """
    nested_dir = os.path.join(tmp_path, "nested", "dir")
    file_path = os.path.join(nested_dir, "test_output.txt")
    my_list = ["Line 1", "Line 2"]

    _write_list_to_txt(file_path, my_list, verbose=False)

    assert os.path.exists(file_path)

# def test_write_list_to_txt_io_error(monkeypatch):
#     """
#     Test that the function raises an IOError when the file cannot be written.
#     """
#     file_path = "/root/test_output.txt"  # Typically inaccessible path
#     my_list = ["Line 1", "Line 2"]

#     with pytest.raises(IOError, match="Could not write to file"):
#         _write_list_to_txt(file_path, my_list, verbose=False)

def test_write_list_to_txt_verbose_output(capfd, tmp_path):
    """
    Test that the function prints the success message when 'verbose' is True.
    """
    file_path = os.path.join(tmp_path, "test_output.txt")
    my_list = ["First line", "Second line"]

    _write_list_to_txt(file_path, my_list, verbose=True)

    out, err = capfd.readouterr()
    assert f"The list has been successfully written to the file {file_path}." in out

def test_write_list_to_txt_empty_list(tmp_path):
    """
    Test that the function writes an empty file when 'my_list' is empty.
    """
    file_path = os.path.join(tmp_path, "test_output.txt")
    my_list = []

    _write_list_to_txt(file_path, my_list, verbose=False)

    # Read the file and check contents
    with open(file_path, "r", encoding='utf-8') as file:
        content = file.read()
    assert content == ""
