def get_optimal_cluster_size(n_clusters: int | None, nb_clusters: list[int]) -> int:
    """
    Determine the number of clusters to use based on input parameters.

    Parameters
    ----------
    n_clusters : int or None
        If an integer is provided, it specifies the number of clusters to use.
        If None, the function will determine the number of clusters based on 
        the `nb_clusters` list derived from eigengap heuristic.

    nb_clusters : list of int
        A list of integers representing possible cluster counts, ordered by 
        relevance. The first element is considered the primary choice.

    Returns
    -------
    int
        The chosen number of clusters based on the provided parameters.

    Raises
    ------
    ValueError
        If `nb_clusters` is empty or does not contain at least two elements 
        when `n_clusters` is None.

    Examples
    --------
    >>> get_optimal_cluster_size(3, [2, 5])
    3

    >>> get_optimal_cluster_size(None, [2, 5])
    2

    >>> get_optimal_cluster_size(None, [1, 5])
    5
    """
    # Case 1: Return n_clusters directly if it is provided
    if n_clusters is not None:
        if not isinstance(n_clusters, int) or n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer if provided.")
        return n_clusters

    # Case 2: Handle the situation when n_clusters is None
    if not isinstance(nb_clusters, list) or len(nb_clusters) < 2:
        raise ValueError("nb_clusters must be a list with at least two elements when n_clusters is None.")

    # Primary selection logic
    if nb_clusters[0] != 1:
        return nb_clusters[0]
    return nb_clusters[1]