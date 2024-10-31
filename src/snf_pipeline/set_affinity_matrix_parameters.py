from typing import Any
import numbers

from src.snf_package.compute import DistanceMetric


def set_affinity_matrix_parameters(
    n: int,
    metric: str | list[str] = 'sqeuclidean',
    K: float = 0.1,
    mu: float = 0.5,
    normalize: bool = True,
    th_nan: float = 0.0
) -> dict[str, Any]:
    """
    Set and validate parameters for computing the affinity matrix.

    Parameters
    ----------
    n : int
        The number of cases (samples) in the dataset. Must be a positive integer.

    metric : str or list of str, optional
        Distance metric to compute. Must be one of the available metrics in
        `scipy.spatial.distance.cdist`. If multiple arrays are provided,
        an equal number of metrics may be supplied. Default is 'sqeuclidean'.

    K : float, optional
        Proportion of neighbors to consider when creating the affinity matrix.
        Must be a float between 0.0 and 1.0 (exclusive). The actual number of neighbors,
        `K_actual`, is computed as `int(K * n)`. Default is 0.1.

    mu : float, optional
        Normalization factor to scale the similarity kernel when constructing
        the affinity matrix. Must be between 0.0 and 1.0 (exclusive). Default is 0.5.

    normalize : bool, optional
        Whether to normalize (i.e., z-score) the data before constructing the
        affinity matrix. Each feature (i.e., column) is normalized separately.
        Default is True.

    th_nan : float, optional
        Threshold for handling missing data (NaNs). Must be between 0.0 and 1.0 (inclusive).
        Default is 0.0.

    Returns
    -------
    params : dict
        Dictionary of validated parameters for affinity matrix computation, including 'K_actual'.

    Raises
    ------
    ValueError
        If parameters are not within acceptable ranges or if invalid combinations
        of parameters are provided.

    TypeError
        If parameters are of incorrect types.

    Notes
    -----
    - Different metrics can only be selected if `th_nan` equals 0.0.
      Otherwise, Euclidean distance ('sqeuclidean') is used regardless of the `metric` parameter.
    - The list of valid metrics corresponds to those available in `scipy.spatial.distance.cdist`.
    - The actual number of neighbors used is computed as `K_actual = int(K * n)`.
      `K_actual` must be at least 1 and less than `n`.

    Examples
    --------
    >>> params = set_affinity_matrix_parameters(n=100, metric='cosine', K=0.15, mu=0.7, normalize=False, th_nan=0.0)
    >>> params
    {'n': 100, 'metric': 'cosine', 'K': 0.15, 'K_actual': 15, 'mu': 0.7, 'normalize': False, 'th_nan': 0.0}
    """
    n = _validate_n(n)
    th_nan = _validate_th_nan(th_nan)
    metric = _validate_metric(metric, th_nan)
    K_actual = _validate_K(K, n)
    mu = _validate_mu(mu)
    normalize = _validate_normalize(normalize)

    params = {
        'n': n,
        'metric': metric,
        'K': K,
        'K_actual': K_actual,
        'mu': mu,
        'normalize': normalize,
        'th_nan': th_nan
    }

    return params


def _validate_n(n: int) -> int:
    if not isinstance(n, int):
        raise TypeError(f"'n' must be an integer, got {type(n)}.")
    if n <= 0:
        raise ValueError(f"'n' must be a positive integer, got {n}.")
    return n


def _validate_th_nan(th_nan: float) -> float:
    if not isinstance(th_nan, numbers.Number):
        raise TypeError(f"'th_nan' must be a float between 0.0 and 1.0, got {type(th_nan)}.")
    if not (0.0 <= th_nan <= 1.0):
        raise ValueError(f"'th_nan' must be a float between 0.0 and 1.0, got {th_nan}.")
    return th_nan


def _validate_metric(metric: str | list[str], th_nan: float) -> str | list[str]:
    if th_nan != 0.0:
        # If th_nan != 0.0, metric must be 'sqeuclidean'
        if metric != 'sqeuclidean':
            # Warn the user and set metric to 'sqeuclidean'
            print(f"Since 'th_nan' != 0.0, metric is set to 'sqeuclidean'.")
            metric = 'sqeuclidean'
    else:
        # Validate metric against DistanceMetric enum
        valid_metrics = [e.value for e in DistanceMetric]
        if isinstance(metric, str):
            if metric not in valid_metrics:
                raise ValueError(f"Invalid metric '{metric}'. Must be one of {valid_metrics}.")
        elif isinstance(metric, list):
            if not all(isinstance(m, str) for m in metric):
                raise TypeError("All elements in 'metric' list must be strings.")
            invalid_metrics = [m for m in metric if m not in valid_metrics]
            if invalid_metrics:
                raise ValueError(f"Invalid metrics {invalid_metrics}. Must be one of {valid_metrics}.")
        else:
            raise TypeError(f"'metric' must be a string or list of strings, got {type(metric)}.")
    return metric


def _validate_K(K: float, n: int) -> int:
    if not isinstance(K, numbers.Number):
        raise TypeError(f"'K' must be a number between 0.0 and 1.0, got {type(K)}.")
    if not (0.0 < K <= 1.0):
        raise ValueError(f"'K' must be a float between 0.0 and 1.0 (exclusive), got {K}.")
    K_actual = int(K * n)
    if K_actual < 1:
        K_actual = 1
    elif K_actual >= n:
        K_actual = n - 1
    return K_actual


def _validate_mu(mu: float) -> float:
    if not isinstance(mu, numbers.Number):
        raise TypeError(f"'mu' must be a number, got {type(mu)}.")
    if not (0.0 < mu < 1.0):
        raise ValueError(f"'mu' must be a float between 0.0 and 1.0 (exclusive), got {mu}.")
    return mu


def _validate_normalize(normalize: bool) -> bool:
    if not isinstance(normalize, bool):
        raise TypeError(f"'normalize' must be a boolean, got {type(normalize)}.")
    return normalize
