import pytest
import numpy as np

from src.snf_pipeline_revised import _get_list_of_edges

def test_get_list_of_edges_valid():
    labels = [0, 0, 1, 1]
    affinity_networks_ordered = {
        "Mod1": np.array([[1, 0.8, 0, 0], [0.8, 1, 0, 0], [0, 0, 1, 0.9], [0, 0, 0.9, 1]]),
        "Mod2": np.array([[1, 0.7, 0, 0], [0.7, 1, 0, 0], [0, 0, 1, 0.85], [0, 0, 0.85, 1]])
    }

    result = _get_list_of_edges(labels, affinity_networks_ordered, edge_th=1.1)
    assert isinstance(result, dict)
    assert all(isinstance(key, int) and isinstance(value, list) for key, value in result.items())
    assert result == {0: [[0]], 1: [[0, 1]]}

def test_get_list_of_edges_label_length_mismatch():
    labels = [0, 0, 1, 1]
    affinity_networks_ordered = {
        "Mod1": np.array([[1, 0.8, 0], [0.8, 1, 0], [0, 0, 1]]),
        "Mod2": np.array([[1, 0.7, 0], [0.7, 1, 0], [0, 0, 1]])
    }

    with pytest.raises(ValueError, match="The length of `labels` must match the dimension of the affinity matrices."):
        _get_list_of_edges(labels, affinity_networks_ordered)

def test_get_list_of_edges_non_square_matrix():
    labels = [0, 0, 1, 1]
    affinity_networks_ordered = {
        "Mod1": np.array([[1, 0.8, 0], [0.8, 1, 0], [0, 0, 1]]),  # Non-square matrix
        "Mod2": np.array([[1, 0.7, 0, 0], [0.7, 1, 0, 0], [0, 0, 1, 0.85], [0, 0, 0.85, 1]])
    }

    with pytest.raises(ValueError):
        _get_list_of_edges(labels, affinity_networks_ordered)

def test_get_list_of_edges_empty_inputs():
    labels = []
    affinity_networks_ordered = {}
    
    result = _get_list_of_edges(labels, affinity_networks_ordered)
    assert result == {}
