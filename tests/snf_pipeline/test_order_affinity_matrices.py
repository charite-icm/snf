import pytest
import numpy as np

from src.snf_pipeline_revised import _order_affinity_matrices

def test_order_affinity_matrices_valid():
    labels = [2, 0, 1]
    modality_names = ("Mod1", "Mod2")
    affinity_networks = (
        np.array([[0.1, 0.2, 0.3], [0.2, 0.4, 0.5], [0.3, 0.5, 0.6]]),
        np.array([[0.5, 0.1, 0.4], [0.1, 0.3, 0.7], [0.4, 0.7, 0.8]])
    )
    
    result = _order_affinity_matrices(labels, modality_names, affinity_networks)
    assert isinstance(result, dict)
    assert all(isinstance(key, str) and isinstance(value, np.ndarray) for key, value in result.items())
    assert set(result.keys()) == {"Mod1", "Mod2"}


def test_order_affinity_matrices_length_mismatch():
    labels = [2, 0, 1]
    modality_names = ("Mod1",)
    affinity_networks = (np.array([[0.1, 0.2, 0.3], [0.2, 0.4, 0.5], [0.3, 0.5, 0.6]]),)

    with pytest.raises(ValueError):
        _order_affinity_matrices(labels, modality_names * 2, affinity_networks)

def test_order_affinity_matrices_non_square_matrix():
    labels = [0, 1, 2]
    modality_names = ("Mod1",)
    affinity_networks = (np.array([[0.1, 0.2], [0.2, 0.3]]),)

    with pytest.raises(ValueError):
        _order_affinity_matrices(labels, modality_names, affinity_networks)

def test_order_affinity_matrices_label_length_mismatch():
    labels = [0, 1]
    modality_names = ("Mod1",)
    affinity_networks = (np.array([[0.1, 0.2, 0.3], [0.2, 0.4, 0.5], [0.3, 0.5, 0.6]]),)

    with pytest.raises(ValueError):
        _order_affinity_matrices(labels, modality_names, affinity_networks)
