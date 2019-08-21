from connectopy import utilities as uti

class DualConn(object):

    """
    Class to perform dual connectopy analyses.
    """

    def __init__(self, subject_id, hemisphere, region_map):

        self.subject_id = subject_id
        self.hemisphere = hemisphere
        self.region_map = region_map

    def fit(self, features, connectopy_dir, data_dir):

        """
        Meta-method for processing dual connectopy.
        """

        print('Computing source to target mappings.')
        self.mappings(features, connectopy_dir)

        print('Computing regional shortest paths.')
        self.paths(data_dir)

        print('Computing source-to-target correlations.')
        self.correlations(features, connectopy_dir)

        print('Computing source-to-target dispersions.')
        self.dispersions(connectopy_dir, data_dir)

    def mappings(self, features, connectopy_dir):

        """
        Meta-method to compute source-to-target mappings.
        """

        print('Computing single pair maps.')
        uti.s2t_mappings(self.subject_id, self.region_map, self.hemisphere, 
                       features, connectopy_dir)
        
        print('Aggregating pair maps.')
        uti.st2_mappingcounts(self.subject_id, self.region_map, self.hemisphere, 
                             connectopy_dir)

    def correlations(self, features, connectopy_dir):

        """
        Meta-method to compute source-to-target correlations.
        """

        print('Computing single pair correlation maps.')
        uti.s2t_correlations(self.subject_id, self.region_map, self.hemisphere, 
                         features, connectopy_dir)

        print('Aggregating pair maps.')
        uti.s2t_correlations_aggregate(self.subject_id, self.region_map, self.hemisphere,
                                   connectopy_dir)

    def paths(self, data_dir):

        """
        Meta-method to compute regional distance matrices.
        """

        uti.regional_shortest_path(self.subject_id, self.region_map, self.hemisphere, 
                                  data_dir)
    
    def dispersions(self, connectopy_dir, data_dir):

        """
        Meta-method to compute source-to-target disperions.
        """

        uti.s2t_dispersion(self.subject_id, self.region_map, self.hemisphere, 
                         data_dir, connectopy_dir)
