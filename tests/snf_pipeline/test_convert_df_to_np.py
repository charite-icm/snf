import pytest
import pandas as pd
import numpy as np

from src.snf_pipeline.convert_df_to_np import convert_df_to_np
from src.snf_pipeline.constants import EID_NAME

def test_convert_df_to_np_valid_input():
    """
    Test that the function correctly converts DataFrames to NumPy arrays, excluding the EID_NAME column.
    """
    df1 = pd.DataFrame({
        EID_NAME: [1, 2],
        'feature1': [10, 20],
        'feature2': [100, 200]
    })
    df2 = pd.DataFrame({
        EID_NAME: [1, 2],
        'feature3': [30, 40],
        'feature4': [300, 400]
    })
    
    arrays = convert_df_to_np((df1, df2))
    
    expected_array1 = np.array([[10, 100], [20, 200]])
    expected_array2 = np.array([[30, 300], [40, 400]])
    
    np.testing.assert_array_equal(arrays[0], expected_array1)
    np.testing.assert_array_equal(arrays[1], expected_array2)

def test_convert_df_to_np_missing_eid_column():
    """
    Test that the function raises a KeyError when the EID_NAME column is missing.
    """
    df1 = pd.DataFrame({
        'feature1': [10, 20],
        'feature2': [100, 200]
    })
    df2 = pd.DataFrame({
        EID_NAME: [1, 2],
        'feature3': [30, 40],
        'feature4': [300, 400]
    })
    
    with pytest.raises(KeyError):
        convert_df_to_np((df1, df2))

def test_convert_df_to_np_non_numeric_data():
    """
    Test that the function correctly handles DataFrames with non-numeric data after dropping EID_NAME.
    """
    df1 = pd.DataFrame({
        EID_NAME: [1, 2],
        'feature1': ['a', 'b'],
        'feature2': ['c', 'd']
    })
    
    arrays = convert_df_to_np((df1,))
    
    expected_array = np.array([['a', 'c'], ['b', 'd']], dtype=object)
    
    np.testing.assert_array_equal(arrays[0], expected_array)

def test_convert_df_to_np_empty_dataframe():
    """
    Test that the function handles empty DataFrames correctly.
    """
    df1 = pd.DataFrame(columns=[EID_NAME, 'feature1', 'feature2'])
    df2 = pd.DataFrame(columns=[EID_NAME, 'feature3', 'feature4'])
    
    arrays = convert_df_to_np((df1, df2))
    
    expected_array1 = np.empty((0, 2))
    expected_array2 = np.empty((0, 2))
    
    np.testing.assert_array_equal(arrays[0], expected_array1)
    np.testing.assert_array_equal(arrays[1], expected_array2)

def test_convert_df_to_np_single_dataframe():
    """
    Test that the function works correctly with a single DataFrame.
    """
    df1 = pd.DataFrame({
        EID_NAME: [1, 2],
        'feature1': [10, 20],
        'feature2': [100, 200]
    })
    
    arrays = convert_df_to_np((df1,))
    
    expected_array = np.array([[10, 100], [20, 200]])
    
    np.testing.assert_array_equal(arrays[0], expected_array)

def test_convert_df_to_np_incorrect_input_type():
    """
    Test that the function raises a TypeError when input is not a tuple of DataFrames.
    """
    df1 = pd.DataFrame({
        EID_NAME: [1, 2],
        'feature1': [10, 20]
    })
    df2 = pd.DataFrame({
        EID_NAME: [1, 2],
        'feature2': [100, 200]
    })
    
    with pytest.raises(TypeError, match="'dfs' must be a tuple"):
        convert_df_to_np([df1, df2])  # Passing a list instead of a tuple

def test_convert_df_to_np_different_column_order():
    """
    Test that the function correctly handles DataFrames with different column orders.
    """
    df1 = pd.DataFrame({
        EID_NAME: [1, 2],
        'feature1': [10, 20],
        'feature2': [100, 200]
    })
    df2 = pd.DataFrame({
        'feature3': [30, 40],
        EID_NAME: [1, 2],
        'feature4': [300, 400]
    })
    
    arrays = convert_df_to_np((df1, df2))
    
    expected_array1 = np.array([[10, 100], [20, 200]])
    expected_array2 = np.array([[30, 300], [40, 400]])
    
    np.testing.assert_array_equal(arrays[0], expected_array1)
    np.testing.assert_array_equal(arrays[1], expected_array2)
