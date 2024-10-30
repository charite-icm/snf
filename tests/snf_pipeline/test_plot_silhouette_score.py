import pytest
import numpy as np
import os
from pathlib import Path

from src.snf_pipeline_revised import plot_silhouette_score, SILHOUETTE_SCORE_FIG_NAME



def test_plot_silhouette_score_valid(tmp_path):
    # Create a sample fused network (similarity matrix)
    fused_network = np.random.rand(10, 10)
    fused_network = (fused_network + fused_network.T) / 2  # Symmetrize
    np.fill_diagonal(fused_network, 1)  # Diagonal elements as self-similarity

    # Run the function
    plot_silhouette_score(fused_network, save_path=tmp_path, n_clusters_end=5, verbose=False)

    # Check if the plot file is saved correctly
    plot_path = Path(os.path.join(tmp_path, SILHOUETTE_SCORE_FIG_NAME + ".jpg"))
    assert plot_path.exists()

def test_plot_silhouette_score_invalid_n_clusters_end():
    # Test with invalid n_clusters_end (less than 2)
    fused_network = np.random.rand(10, 10)
    with pytest.raises(ValueError):
        plot_silhouette_score(fused_network, save_path="/some_path", n_clusters_end=1)

def test_plot_silhouette_score_invalid_save_path():
    # Test with an invalid save_path
    fused_network = np.random.rand(10, 10)
    with pytest.raises(FileNotFoundError):
        plot_silhouette_score(fused_network, save_path="/nonexistent_path")
