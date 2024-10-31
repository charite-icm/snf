from src.snf_package.compute import check_symmetric, make_affinity, make_affinity_nan


import numpy as np


from typing import Any, Callable


def compute_aff_networks(arrs: tuple[np.ndarray], param: dict[str, Any]) -> tuple[np.ndarray]:
    """
    Compute affinity networks based on input arrays using either standard or 
    nan-aware affinity functions, and normalize the resulting affinity matrices.

    Parameters
    ----------
    arrs : tuple of np.ndarray
        A tuple of numpy arrays, where each array represents a set of features 
        for which affinity matrices will be computed. The arrays may represent 
        different data modalities or datasets.

    param : dict of {str: Any}
        A dictionary containing parameters for computing the affinity matrices.
        It should include the following keys:
        - 'metric' (str or list of str): The distance metric(s) used to compute the 
          affinity matrices (e.g., 'euclidean', 'cosine'). If multiple arrays are 
          provided in `arrs`, an equal number of metrics can be specified.
        - 'K_actual' (int): The number of nearest neighbors to consider when 
          constructing the affinity matrices.
        - 'mu' (float): A scaling factor to normalize the similarity kernel when 
          constructing the affinity matrix.
        - 'normalize' (bool): Whether to normalize each feature in the input 
          arrays before computing affinity matrices.
        - 'th_nan' (float): A threshold for handling missing values (NaNs). If 
          this value is non-zero, a nan-aware affinity function will be used 
          instead of the standard affinity function.

    Returns
    -------
    tuple of np.ndarray
        A tuple of normalized and symmetric affinity matrices corresponding to 
        each array in `arrs`. Each matrix represents the affinity network for 
        that dataset, computed using the specified parameters.

    Notes
    -----
    - If `th_nan` in the `param` dictionary is non-zero, missing values (NaNs) 
      are handled using `make_affinity_nan`, otherwise `make_affinity` is used.
    - Each affinity matrix is normalized row-wise to ensure that the sum of 
      similarities for each sample is equal to 1. This is achieved by dividing 
      each element in the row by the sum of that row.
    - Symmetry is enforced on each affinity matrix using `check_symmetric`, 
      which ensures that the matrices are symmetric and issues a warning if 
      necessary.

    Example
    -------
    >>> arr1 = np.random.rand(100, 10)
    >>> arr2 = np.random.rand(100, 15)
    >>> param = {
            'metric': 'euclidean',
            'K_actual': 10,
            'mu': 0.5,
            'normalize': True,
            'th_nan': 0.0
        }
    >>> aff_matrices = compute_aff_networks((arr1, arr2), param)
    >>> print(aff_matrices[0].shape)  # Output: (100, 100)

    """
    func_: Callable =  make_affinity
    if param["th_nan"] != 0.0:
        func_ = make_affinity_nan

    affinity_networks = func_(*arrs,
                              metric=param["metric"], K=param["K_actual"],
                              mu=param["mu"], normalize=param["normalize"])

    # Normalize each affinity matrix by the row sum
    affinity_networks = [w / np.nansum(w, axis=1, keepdims=True) for w in affinity_networks]

    # Ensure each matrix is symmetric
    affinity_networks = [check_symmetric(w, raise_warning=False) for w in affinity_networks]

    return tuple(affinity_networks)