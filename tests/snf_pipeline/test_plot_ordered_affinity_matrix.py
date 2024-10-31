import pytest
import numpy as np
from pathlib import Path
import os


from src.snf_pipeline import plot_ordered_affinity_matrix


def test_plot_ordered_affinity_matrix_valid(tmp_path):
    # Create a sample affinity matrix and labels
    network = np.random.rand(10, 10)
    network = (network + network.T) / 2  # Symmetrize
    labels = [0, 1, 1, 0, 2, 2, 0, 1, 2, 0]

    # Run the function
    plot_ordered_affinity_matrix(network, labels, os.path.join(tmp_path, "affinity_matrix"), verbose=False)

    # Check if the plot file is saved correctly
    assert Path(os.path.join(tmp_path, "affinity_matrix.jpg")).exists()

def test_plot_ordered_affinity_matrix_invalid_network():
    # Test with a non-square network matrix
    network = np.random.rand(10, 5)
    labels = [0] * 10
    with pytest.raises(ValueError):
        plot_ordered_affinity_matrix(network, labels, "/some_path/affinity_matrix")

def test_plot_ordered_affinity_matrix_label_length_mismatch():
    # Test with labels length mismatch
    network = np.random.rand(10, 10)
    labels = [0] * 9  # Length mismatch
    with pytest.raises(ValueError):
        plot_ordered_affinity_matrix(network, labels, "/some_path/affinity_matrix")


