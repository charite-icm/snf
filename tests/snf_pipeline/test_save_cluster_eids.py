from src.snf_pipeline_revised import save_cluster_eids, EID_NAME, CLUSTER_NAME, CLUSTER_EIDS_CSV

import pytest
import pandas as pd


def test_save_cluster_eids(tmp_path):
    # Basic test case
    eids = [1, 2, 3]
    labels = [0, 1, 1]
    save_cluster_eids(eids, labels, tmp_path, verbose=False)
    
    # Check if the CSV file is saved correctly
    csv_path = tmp_path / CLUSTER_EIDS_CSV
    assert csv_path.exists()

    # Verify contents
    df = pd.read_csv(csv_path)
    assert list(df[EID_NAME]) == eids
    assert list(df[CLUSTER_NAME]) == labels

def test_save_cluster_eids_length_mismatch():
    # Test for length mismatch between eids and labels
    with pytest.raises(ValueError):
        save_cluster_eids([1, 2, 3], [0, 1], "/nonexistent_path")

def test_save_cluster_eids_invalid_path():
    # Test for invalid save_path
    with pytest.raises(FileNotFoundError):
        save_cluster_eids([1, 2, 3], [0, 1, 1], "/nonexistent_path")
