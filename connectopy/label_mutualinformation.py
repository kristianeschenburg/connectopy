import numpy as np
from fragmenter import Fragment
from fragmenter import RegionExtractor
from statistics import information


class ConnectivityMutualInformation(object):

    def __init__(self, label, parcel_mapping, n_regions, connectivity):

        """
        Class to compute the mutual information of a label map, using a
        parameterized connectivity pattern.

        Paramters:
        - - - - -
        label: ndarray
            hyper-parcellated cortical map
        parcel_mapping: dictionary
            nested dictionary mapping region names to sub-region indices
            region names are strings
        n_regions: int
            expected number of labels in label map (not guaranteed
            to be the same across subjects).  Label is expected
        connectivity: array
            connectivity information between all vertices
            in the label map
        Returns:
        - - - -
        mutual_info: array
            n-labels by n-labels array containing mutual
            information value between pairs of labelsa
        """

        self.label = label
        self.parcel_mapping = parcel_mapping
        self.num_labels = n_regions
        self.connectivity = connectivity

        def fit(self):

            """
            Working function to compute the mutual information between all
            pairs of labels in a label map.

            """

            nl = self.n_regions
            label_max = nl+1

            mutual_info = np.zeros((nl, nl))

            for slabel in np.arange(1, label_max):
                for tlabel in np.arange(1, label_max):

                    mutual_info[slabel-1, tlabel-1] = self._pair(
                        slabel, tlabel)

            self.mutual_info = mutual_info

        def _pair(self, source, target):

            """
            Compute mutual information between single label pair.

            Parameters:
            - - - - -
            source: int
                source label
            target: int
                target label
            """



            if sn == 0 or tn == 0:
                return np.nan

            else:

                count_matrix = np.zeros((sn, tn))

                for s, slab in enumerate(source_parcels):
                    sidx = np.where(smap == slab)[0]

                    for t, tlab in enumerate(target_parcels):
                        tidx = np.where(tmap == tlab)[0]

                        counts = self.connectivity[sidx, :][:, tidx]
                        count_matrix[s, t] = counts.sum()

                mi = MI(count_matrix)

                return mi
