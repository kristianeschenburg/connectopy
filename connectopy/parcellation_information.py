import numpy as np
from statistics import information
from dask import delayed


class ConnectivityMutualInformation(object):

    """
    Class to compute the mutual information of a label map, using a
    parameterized connectivity pattern.

    Paramters:
    - - - - -
    label: ndarray
        2D array, where each row is a vertex, and each column
        is the sub-label it belongs to
    sub_regions: dictionary
        nested dictionary mapping original label ID to sub-region IDs and
        sub-region indices
    num_labels: int
        expected number of labels in label map (not guaranteed
        to be the same across subjects).  Label is expected
    conn: ndarray
        connectivity information between all vertices
        in the label map
    Returns:
    - - - -
    mutual_info: ndarray
        pairwise mutual information array
    """

    def __init__(self, label, sub_regions, num_labels, conn):

        self.label = label
        self.sub_regions = sub_regions
        self.n_labels = num_labels
        self.conn = conn

        def fit(self):

            """
            Wrapper to compute the mutual information between all
            pairs of labels in a label map.
            """

            n = self.n_labels
            parcels = self.sub_regions
            unq = np.arange(1, n+1)

            mutual = []
            for s, slab in unq:
                for t, tlab in unq:

                    pairs = pair_data(
                        conn, parcels.get(slab), parcels.get(tlab))
                    mi = pair_mi(pairs)

                    mutual.append(mi)

            mutual = np.asarray(mutual).reshape(n, n)

        @delayed
        def pair_data(self, conn, src_idx=None, trg_idx=None):

            """
            Return single-pair count matrix.

            Parameters:
            - - - - -
            conn: array
                connectivity matrix
            src_idx, trg_idx: dictionary
                sub-region ID to index mapping
            Returns:
            - - - -
            counts: array
                streamline counts between sub-regions
            """

            if not src_idx or not trg_idx:
                counts = np.asarray([0])

            else:
                sl = len(src_idx)
                tl = len(trg_idx)

                counts = np.zeros((sl, tl))

                for i, idxs in enumerate(src_idx.values()):
                    for j, idxt in enumerate(trg_idx.values()):
                        counts[i, j] = (conn[idxs, :][:, idxt]).sum()

            return counts

        @delayed
        def pair_mi(self, counts):

            """
            Compute mutual information of single-pair count matrix.

            Parameters:
            - - - - -
            counts: array
                streamline counts between sub-regions
            Returns:
            - - - -
            mi: float
                mutual information of streamline count matrix
            """

            if not np.any(counts):
                mi = 0
            else:
                mi = information.mutual_information(counts)

            return mi
