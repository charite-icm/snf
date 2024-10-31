import pytest
from src.snf_pipeline import get_optimal_cluster_size


# Test cases for get_n_clusters function
def test_get_optimal_cluster_size_with_explicit_value():
    # Test when n_clusters is provided
    assert get_optimal_cluster_size(3, [2, 5]) == 3
    assert get_optimal_cluster_size(1, [1, 2]) == 1
    assert get_optimal_cluster_size(10, [3, 7, 10]) == 10

def test_get_optimal_cluster_size_with_nb_clusters_fallback():
    # Test when n_clusters is None and nb_clusters[0] != 1
    assert get_optimal_cluster_size(None, [2, 5]) == 2
    assert get_optimal_cluster_size(None, [4, 10]) == 4
    # Test when nb_clusters[0] == 1, should return nb_clusters[1]
    assert get_optimal_cluster_size(None, [1, 5]) == 5
    assert get_optimal_cluster_size(None, [1, 3]) == 3

def test_get_optimal_cluster_size_invalid_n_clusters():
    # Test with invalid n_clusters (negative or non-integer)
    with pytest.raises(ValueError):
        get_optimal_cluster_size(-1, [2, 5])
    with pytest.raises(ValueError):
        get_optimal_cluster_size(0, [2, 5])

def test_get_optimal_cluster_size_invalid_nb_clusters():
    # Test when nb_clusters has less than two elements and n_clusters is None
    with pytest.raises(ValueError):
        get_optimal_cluster_size(None, [])
    with pytest.raises(ValueError):
        get_optimal_cluster_size(None, [1])
    
    # Test when nb_clusters is not a list
    with pytest.raises(ValueError):
        get_optimal_cluster_size(None, "not a list")
    with pytest.raises(ValueError):
        get_optimal_cluster_size(None, 123)

def test_get_optimal_cluster_size_edge_cases():
    # Test when both n_clusters and nb_clusters have valid values but nb_clusters has 1 as the first element
    assert get_optimal_cluster_size(4, [1, 3, 4]) == 4  # n_clusters should take precedence
