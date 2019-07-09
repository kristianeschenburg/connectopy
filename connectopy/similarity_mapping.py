import numpy as np
import scipy
from connectopy import l1_median as l1

def cross_similarity(x, y, k=5, metric='correlation'):

    """
    Method to compute top K most similar neighbors between two
    cortical regions.

    Parameters:
    - - - - -
    x: float, array
        source region
    y: float, array
        target region
    k: int
        neighborhood size
    metric: string
        similarity metric to use
        any value from "sklearn.spatial.distance.cdist" is valid

    Returns:
    - - - -
    neighbors: int, array
        Top K nearest neighbors in target region to source region
        
        Each row in ```neighbors``` corresponds to a sample in ```x```.
        Each index (i, j) in ```neighbors``` corresponds to sample ```j``` in
        ```y``` to which sample ```i``` in ```x``` is most similar, where
        ```j``` ranges in value from [0, y.shape[0]].

    Example:
    - - - -

    x = np.asarray([[1, 2, 3], 
                    [4, 3, 2]])
    y = np.asarray([[2, 3, 4], 
                    [5, 4, 3]])

    print(cross_similarity(x, y, k=2, metric='correlation'))
    [[0, 1],
    [1, 0]]

    """

    cross = scipy.spatial.distance.cdist(x, y, metric=metric)

    neighbors = np.argsort(cross, axis=1)
    neighbors = neighbors[:, 0:k]

    return neighbors

def cross_mapping(y, neighbors, statistic = "mean"):

    """
    Method to compute summary measure of top K nearest target 
    neighbors.

    Parameters:
    - - - - -
    y: float, array
        target region feature vectors
    neighbors: int array
        K nearest target vertices for each source region point
    statistic: string
        "mean" or "median"
    """

    if np.ndim(y) == 1:
        y = y[:, None]

    assert statistic in ['mean', 'median']

    y_features = y[:, :, None][neighbors]
    if np.ndim(neighbors) == 1:
        return y_features.squeeze()

    if statistic == "mean":

        y_summary = y[:, :, None][neighbors].mean(axis=1)
    
    else:
        L = l1.L1()
        y_summary = []
        
        for k in y_features:

            temp_k = k.squeeze()
            L.fit(temp_k)
            y_summary.append(L.median_)
        
        y_summary = np.row_stack(y_summary)
    
    return y_summary.squeeze()