import pandas as pd
import pytest
from src.snf_pipeline.remove_rows_above_missing_threshold import remove_rows_above_missing_threshold

def test_remove_rows_above_missing_threshold_default():
    """
    Test the function with the default threshold (1.0), which should not remove any rows.
    """
    data = {'A': [1, 2, None, 4], 'B': [None, 2, 3, 4], 'C': [1, None, 3, None]}
    df = pd.DataFrame(data)

    df_cleaned, row_missing_percentage = remove_rows_above_missing_threshold(df)

    # Assert no rows were removed (default threshold is 1.0)
    assert df_cleaned.shape[0] == df.shape[0]

    # Check that the missing_percentage contains the correct values
    expected_missing = pd.Series([33.33, 33.33, 33.33, 33.33], name='missing_percentage', index=[0, 1, 2, 3]).round(2)
    pd.testing.assert_series_equal(row_missing_percentage['missing_percentage'].round(2), expected_missing)

def test_remove_rows_above_missing_threshold_50_percent():
    """
    Test the function with a threshold of 0.5, which should remove rows with more than 50% NaN.
    """
    data = {'A': [1, 2, None, 4], 'B': [None, 2, 3, None], 'C': [1, None, 3, None]}
    df = pd.DataFrame(data)

    df_cleaned, row_missing_percentage = remove_rows_above_missing_threshold(df, th_nan=0.5)

    # Assert that only one row was removed (the row with index 3 has more than 50% NaN)
    assert df_cleaned.shape[0] == 3

    # Check that the missing_percentage contains the correct values
    expected_missing = pd.Series([33.33, 33.33, 33.33, 66.67], name='missing_percentage', index=[0, 1, 2, 3]).round(2)
    pd.testing.assert_series_equal(row_missing_percentage['missing_percentage'].round(2), expected_missing)

def test_remove_rows_above_missing_threshold_all_removed():
    """
    Test the function with a threshold of 0.0, which should remove all rows with any NaN values.
    """
    data = {'A': [1, 2, None, 4], 'B': [1, 2, 3, 4], 'C': [1, None, 3, None]}
    df = pd.DataFrame(data)

    df_cleaned, row_missing_percentage = remove_rows_above_missing_threshold(df, th_nan=0.0)

    # Assert that only the fully complete row (index 0) remains
    assert df_cleaned.shape[0] == 1
    assert df_cleaned.index[0] == 0

    # Check that the missing_percentage contains the correct values
    expected_missing = pd.Series([0.0, 33.33, 33.33, 33.33], name='missing_percentage', index=[0, 1, 2, 3]).round(2)
    pd.testing.assert_series_equal(row_missing_percentage['missing_percentage'].round(2), expected_missing)

def test_remove_rows_above_missing_threshold_invalid_threshold():
    """
    Test that the function raises a ValueError when the threshold is not between 0 and 1.
    """
    data = {'A': [1, 2, None, 4], 'B': [None, 2, 3, 4], 'C': [1, None, 3, None]}
    df = pd.DataFrame(data)

    with pytest.raises(ValueError, match="The threshold .* must be a float between 0 and 1"):
        remove_rows_above_missing_threshold(df, th_nan=1.5)

def test_remove_rows_above_missing_threshold_verbose(capfd):
    """
    Test the function with the verbose flag set to True.
    """
    data = {'A': [1, 2, None, 4], 'B': [None, 2, 3, 4], 'C': [1, None, 3, None]}
    df = pd.DataFrame(data)

    remove_rows_above_missing_threshold(df, th_nan=0.5, verbose=True)

    # Capture the printed output
    captured = capfd.readouterr()

    # Check if the correct output was printed
    assert "Original rows: 4" in captured.out
    assert "Rows remaining after filtering: 4" in captured.out
    assert "Rows removed: 0" in captured.out
