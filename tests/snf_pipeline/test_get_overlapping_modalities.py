import pandas as pd
import pytest

from src.snf_pipeline.get_overlapping_modalities import get_overlapping_modalities
from src.snf_pipeline.constants import EID_NAME


def test_overlapping_modalities():
    """
    Test that the function correctly identifies overlapping 'eid's and returns the filtered DataFrames.
    """
    df1 = pd.DataFrame({EID_NAME: [1, 2, 3, 4], 'data1': [10, 20, 30, 40]})
    df2 = pd.DataFrame({EID_NAME: [3, 4, 5, 6], 'data2': [300, 400, 500, 600]})
    df3 = pd.DataFrame({EID_NAME: [4, 5, 6, 7], 'data3': [4000, 5000, 6000, 7000]})

    # Expected overlapping EIDs are [4]
    overlapping_dfs = get_overlapping_modalities((df1, df2, df3))
    expected_eids = [4]

    for df in overlapping_dfs:
        assert df[EID_NAME].tolist() == expected_eids

def test_no_overlap():
    """
    Test that the function raises a ValueError when there is no overlap between DataFrames.
    """
    df1 = pd.DataFrame({EID_NAME: [1, 2, 3], 'data1': [10, 20, 30]})
    df_no_overlap = pd.DataFrame({EID_NAME: [4, 5, 6], 'data2': [40, 50, 60]})

    with pytest.raises(ValueError, match=f"No overlapping '{EID_NAME}'s between modalities"):
        get_overlapping_modalities((df1, df_no_overlap))

def test_single_dataframe():
    """
    Test that the function raises a ValueError when only one DataFrame is provided.
    """
    df1 = pd.DataFrame({EID_NAME: [1, 2, 3], 'data1': [10, 20, 30]})

    with pytest.raises(ValueError, match=f"Minimum number of DataFrames is 2 ..."):
        get_overlapping_modalities((df1,))

def test_empty_dataframes():
    """
    Test that the function raises a ValueError when provided with empty DataFrames.
    """
    df_empty = pd.DataFrame({EID_NAME: [], 'data': []})

    with pytest.raises(ValueError, match=f"No overlapping '{EID_NAME}'s between modalities"):
        get_overlapping_modalities((df_empty, df_empty))

def test_reset_index():
    """
    Test that the returned DataFrames have their indices reset after filtering.
    """
    df1 = pd.DataFrame({EID_NAME: [1, 2, 3, 4], 'data1': [10, 20, 30, 40]})
    df2 = pd.DataFrame({EID_NAME: [3, 4, 5, 6], 'data2': [300, 400, 500, 600]})

    overlapping_dfs = get_overlapping_modalities((df1, df2))
    expected_eids = [3, 4]

    for df in overlapping_dfs:
        assert df[EID_NAME].tolist() == expected_eids
        # Check if index is reset
        assert df.index.tolist() == [0, 1]

# def test_different_eid_column(monkeypatch):
#     """
#     Test that the function correctly handles a different 'eid' column name.
#     """
#     eid_name = 'id'
#     df1 = pd.DataFrame({eid_name: [1, 2, 3], 'data1': [10, 20, 30]})
#     df2 = pd.DataFrame({eid_name: [3, 4, 5], 'data2': [300, 400, 500]})

#     # Use monkeypatch to temporarily set EID_NAME to 'id' in the module
#     from src.snf_pipeline import get_overlapping_modalities
#     monkeypatch.setattr(get_overlapping_modalities, 'EID_NAME', eid_name)

#     overlapping_dfs = get_overlapping_modalities((df1, df2))
#     expected_eids = [3]

#     for df in overlapping_dfs:
#         assert df[eid_name].tolist() == expected_eids

def test_duplicate_eid_values():
    """
    Test that the function correctly handles DataFrames with duplicate 'eid' values.
    """
    df1 = pd.DataFrame({EID_NAME: [1, 2, 2, 3], 'data1': [10, 20, 20, 30]})
    df2 = pd.DataFrame({EID_NAME: [2, 3, 4], 'data2': [200, 300, 400]})

    overlapping_dfs = get_overlapping_modalities((df1, df2))
    # Expected overlapping EIDs are [2, 2, 3] in df1 and [2, 3] in df2
    expected_eids_df1 = [2, 2, 3]
    expected_eids_df2 = [2, 3]

    assert overlapping_dfs[0][EID_NAME].tolist() == expected_eids_df1
    assert overlapping_dfs[1][EID_NAME].tolist() == expected_eids_df2
