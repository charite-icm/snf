import pandas as pd
import pytest

from src.snf_pipeline_revised import _check_validity_loaded_data

def test_valid_data():
    """
    Test that the function passes without raising any exceptions
    when provided with valid data: a tuple of two or more DataFrames,
    each containing an 'eid' column with unique values.
    """
    df1 = pd.DataFrame({"eid": [1, 2, 3], "value": [10, 20, 30]})
    df2 = pd.DataFrame({"eid": [4, 5, 6], "value": [40, 50, 60]})

    # Should not raise any exceptions
    _check_validity_loaded_data((df1, df2))

def test_less_than_two_dataframes():
    """
    Test that the function raises a ValueError when provided with fewer 
    than two DataFrames in the tuple.
    """
    df1 = pd.DataFrame({"eid": [1, 2, 3], "value": [10, 20, 30]})

    with pytest.raises(ValueError, match="Minimum number of data modalities is 2"):
        _check_validity_loaded_data((df1,))

def test_invalid_dataframe_type():
    """
    Test that the function raises a TypeError when any item in the tuple 
    is not a pandas DataFrame.
    """
    df1 = pd.DataFrame({"eid": [1, 2, 3], "value": [10, 20, 30]})
    invalid_data = {"eid": [4, 5, 6], "value": [40, 50, 60]}  # Not a DataFrame

    with pytest.raises(TypeError, match="Invalid type of data:"):
        _check_validity_loaded_data((df1, invalid_data))

def test_missing_eid_column():
    """
    Test that the function raises a ValueError when any DataFrame 
    in the tuple does not contain the required 'eid' column.
    """
    df1 = pd.DataFrame({"eid": [1, 2, 3], "value": [10, 20, 30]})
    df2 = pd.DataFrame({"id": [4, 5, 6], "value": [40, 50, 60]})  # Missing 'eid' column

    with pytest.raises(ValueError, match="Column with IDs should be named 'eid'"):
        _check_validity_loaded_data((df1, df2))

def test_duplicate_eid_column():
    """
    Test that the function raises a ValueError when the 'eid' column 
    in any DataFrame contains duplicate values.
    """
    df1 = pd.DataFrame({"eid": [1, 2, 2], "value": [10, 20, 30]})  # Duplicate 'eid' values
    df2 = pd.DataFrame({"eid": [4, 5, 6], "value": [40, 50, 60]})

    with pytest.raises(ValueError, match="Column 'eid' does not contain unique values."):
        _check_validity_loaded_data((df1, df2))

def test_input_not_a_tuple():
    """
    Test that the function raises a TypeError when the input is not a tuple.
    """
    df1 = pd.DataFrame({"eid": [1, 2, 3], "value": [10, 20, 30]})
    df2 = pd.DataFrame({"eid": [4, 5, 6], "value": [40, 50, 60]})

    # Using a list instead of a tuple
    with pytest.raises(TypeError, match="dfs should be of type tuple not"):
        _check_validity_loaded_data([df1, df2])  # Passing a list instead of a tuple
