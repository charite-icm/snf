import pytest
import pandas as pd
import numpy as np

from src.snf_pipeline_revised import _transform_to_upsetplot_format


def test_transform_to_upsetplot_format_valid():
    cluster_weights = {0: [[0, 1], [1], [0]], 1: [[2], [0, 2]]}
    modality_names = ("Mod1", "Mod2", "Mod3")
    
    result = _transform_to_upsetplot_format(cluster_weights, modality_names, verbose=False)
    assert isinstance(result, list)
    assert all(isinstance(df, pd.Series) for df in result)

def test_transform_to_upsetplot_format_empty_modality_names():
    cluster_weights = {0: [[0, 1], [1]]}
    modality_names = ()
    
    with pytest.raises(ValueError):
        _transform_to_upsetplot_format(cluster_weights, modality_names)

def test_transform_to_upsetplot_format_invalid_index_in_weights():
    cluster_weights = {0: [[0, 4], [1]]}  # 4 is out of bounds for modality_names of length 3
    modality_names = ("Mod1", "Mod2", "Mod3")
    
    with pytest.raises(ValueError):
        _transform_to_upsetplot_format(cluster_weights, modality_names)

def test_transform_to_upsetplot_format_no_edges():
    cluster_weights = {}
    modality_names = ("Mod1", "Mod2", "Mod3")
    
    result = _transform_to_upsetplot_format(cluster_weights, modality_names, verbose=False)
    assert result == []

def test_transform_to_upsetplot_format_verbose_output(capfd):
    cluster_weights = {0: [[0], [1]]}
    modality_names = ("Mod1", "Mod2")
    
    _transform_to_upsetplot_format(cluster_weights, modality_names, verbose=True)
    captured = capfd.readouterr()
    assert "Cluster 0" in captured.out
