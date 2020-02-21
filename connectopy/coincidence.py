import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans


class SpectralClustering(object):

    def __init__(self, n_clusters=2, n_components=3):
        
        """
        Perform Spectral Clustering of similarity matrix.
        """

        self.n_clusters = n_clusters
        self.n_components = n_components
    
    def fit(self, X):
        
        """
        Parameters:
        - - - - -
        X: float, array
            dissimilarity matrix
        """

        self.labels_ = self.cluster(X)
        
    def cluster(self, X):
        
        """
        Perform Spectral Clustering of similarity matrix.

        Parameters:
        - - - - -
        X: float, array
            dissimilarity matrix
        
        Returns:
        - - - -
        labels: int, array
            cluster labels for each sample
        """
        
        D = np.diag(X.sum(0))
        L = np.subtract(D, X)
        [l, y] = eigh(a=L, b=D, eigvals=(1, self.n_components+1))
        
        self.components_ = y

        K = KMeans(n_clusters=self.n_clusters, precompute_distances=True)
        K.fit(y)
        
        labels = K.labels_ + 1
        
        return labels

    
class CoClustering(object):
    
    """
    Generate co-clustering matrix of similarity matrix.
    Clusters similarity matrix using Spectral Clustering.
    """
    
    def __init__(self, n_clusters=4, n_components=3):
        
        self.n_clusters = n_clusters
        self.n_components = n_components
    
    def fit(self, X):
        
        """
        Fit the clustering, and generate co-clustering matrix.
        
        Parameters:
        - - - - -
        X: float, array
            dissimilarity matrix
        """
        
        S = SpectralClustering(n_clusters = self.n_clusters, 
                               n_components = self.n_components)
        S.fit(X)
        self.labels_ = S.labels_
        self.y_ = S.components_
        self.coclust_ = self.cocluster(S.labels_)
        
    def cocluster(self, labels):
        
        """

        Convert labeling into co-clustering matrix.

        Parameters:
        - - - - -
        labels: int, array
            label assigned to each sample

        Returns:
        - - - -
        z: int, array
            Symmetric coindincidence matrix.
            Each index (i,j) contains 1 if regions (i,j) 
            belong to same cluster and 0 if they belong 
            to different clusters.
        """

        z = np.zeros((labels.shape[0], labels.shape[0]))
        for l in np.unique(labels):
            idx = np.where(labels == l)[0]
            pairs = [(u, v) for u in idx for v in idx]
            for p in pairs:
                z[p[0], p[1]] = 1

        return z