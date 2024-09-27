import pytest
from src.snf_pipeline_revised import set_affinity_matrix_parameters

def test_set_affinity_matrix_parameters_defaults():
    """
    Test the function with default parameters.
    """
    params = set_affinity_matrix_parameters(n=100)
    assert params['metric'] == "sqeuclidean"
    assert params['K'] == 0.1
    assert params['K_actual'] == 10
    assert params['mu'] == 0.5
    assert params['normalize'] is True
    assert params['th_nan'] == 0.0
    assert params['n'] == 100

def test_set_affinity_matrix_parameters_valid_custom():
    """
    Test the function with valid custom parameters.
    """
    params = set_affinity_matrix_parameters(n=200, metric='cosine', K=0.1, mu=0.7, normalize=False, th_nan=0.0)
    assert params['metric'] == 'cosine'
    assert params['K'] == 0.1
    assert params['K_actual'] == 20
    assert params['mu'] == 0.7
    assert params['normalize'] is False
    assert params['th_nan'] == 0.0
    assert params['n'] == 200

def test_set_affinity_matrix_parameters_th_nan_nonzero():
    """
    Test that when th_nan != 0.0, metric is set to 'sqeuclidean'.
    """
    params = set_affinity_matrix_parameters(n=100, metric='cosine', th_nan=0.1)
    assert params['metric'] == 'sqeuclidean'
    assert params['th_nan'] == 0.1

def test_set_affinity_matrix_parameters_invalid_metric():
    """
    Test that an invalid metric raises ValueError.
    """
    with pytest.raises(ValueError):
        set_affinity_matrix_parameters(n=100, metric='invalid_metric')

def test_set_affinity_matrix_parameters_invalid_K():
    """
    Test that invalid K values raise ValueError or TypeError.
    """
    with pytest.raises(ValueError, match="'K' must be a float between 0.0 and 1.0"):
        set_affinity_matrix_parameters(n=100, K=0.0)

    with pytest.raises(ValueError, match="'K' must be a float between 0.0 and 1.0"):
        set_affinity_matrix_parameters(n=100, K=1.1)

    with pytest.raises(TypeError, match="'K' must be a number"):
        set_affinity_matrix_parameters(n=100, K='0.1')

def test_set_affinity_matrix_parameters_K_actual_limits():
    """
    Test that K_actual is computed correctly and within [1, n - 1].
    """
    # Test K resulting in K_actual less than 1
    params = set_affinity_matrix_parameters(n=100, K=0.0001)
    assert params['K_actual'] == 1

    # Test K resulting in K_actual >= n
    params = set_affinity_matrix_parameters(n=100, K=1.0)
    assert params['K_actual'] == 99

def test_set_affinity_matrix_parameters_invalid_mu():
    """
    Test that invalid mu values raise ValueError or TypeError.
    """
    with pytest.raises(ValueError, match="'mu' must be a float between 0.0 and 1.0"):
        set_affinity_matrix_parameters(n=100, mu=-0.1)

    with pytest.raises(ValueError, match="'mu' must be a float between 0.0 and 1.0"):
        set_affinity_matrix_parameters(n=100, mu=1.0)

    with pytest.raises(ValueError, match="'mu' must be a float between 0.0 and 1.0"):
        set_affinity_matrix_parameters(n=100, mu=0.0)

    with pytest.raises(TypeError, match="'mu' must be a number"):
        set_affinity_matrix_parameters(n=100, mu='0.5')

def test_set_affinity_matrix_parameters_invalid_normalize():
    """
    Test that invalid normalize values raise TypeError.
    """
    with pytest.raises(TypeError, match="'normalize' must be a boolean"):
        set_affinity_matrix_parameters(n=100, normalize='yes')

def test_set_affinity_matrix_parameters_invalid_th_nan():
    """
    Test that invalid th_nan values raise ValueError or TypeError.
    """
    with pytest.raises(ValueError, match="'th_nan' must be a float between 0.0 and 1.0"):
        set_affinity_matrix_parameters(n=100, th_nan=-0.1)

    with pytest.raises(ValueError, match="'th_nan' must be a float between 0.0 and 1.0"):
        set_affinity_matrix_parameters(n=100, th_nan=1.1)

    with pytest.raises(TypeError, match="'th_nan' must be a float"):
        set_affinity_matrix_parameters(n=100, th_nan='0.0')

def test_set_affinity_matrix_parameters_invalid_n():
    """
    Test that invalid n values raise ValueError or TypeError.
    """
    with pytest.raises(ValueError, match="'n' must be a positive integer"):
        set_affinity_matrix_parameters(n=0)

    with pytest.raises(ValueError, match="'n' must be a positive integer"):
        set_affinity_matrix_parameters(n=-10)

    with pytest.raises(TypeError, match="'n' must be an integer"):
        set_affinity_matrix_parameters(n=100.5)

def test_set_affinity_matrix_parameters_valid_metric_list():
    """
    Test that a valid list of metrics is accepted.
    """
    metrics_list = ['cosine', 'euclidean']
    params = set_affinity_matrix_parameters(n=100, metric=metrics_list, th_nan=0.0)
    assert params['metric'] == metrics_list

def test_set_affinity_matrix_parameters_invalid_metric_in_list():
    """
    Test that invalid metrics in the list raise ValueError.
    """
    metrics_list = ['cosine', 'invalid_metric']
    with pytest.raises(ValueError, match="Invalid metrics"):
        set_affinity_matrix_parameters(n=100, metric=metrics_list, th_nan=0.0)

def test_set_affinity_matrix_parameters_metric_list_th_nan_nonzero():
    """
    Test that when th_nan != 0.0, metric is set to 'sqeuclidean' even if metric is a list.
    """
    metrics_list = ['cosine', 'euclidean']
    params = set_affinity_matrix_parameters(n=100, metric=metrics_list, th_nan=0.5)
    assert params['metric'] == 'sqeuclidean'

def test_set_affinity_matrix_parameters_metric_type_error():
    """
    Test that passing an incorrect type for 'metric' raises TypeError.
    """
    with pytest.raises(TypeError, match="'metric' must be a string or list of strings"):
        set_affinity_matrix_parameters(n=100, metric=123)

def test_set_affinity_matrix_parameters_metric_list_with_non_string():
    """
    Test that a list containing non-string elements raises TypeError.
    """
    metrics_list = ['cosine', 123]
    with pytest.raises(TypeError, match="All elements in 'metric' list must be strings"):
        set_affinity_matrix_parameters(n=100, metric=metrics_list, th_nan=0.0)
