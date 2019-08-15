import os
import numpy as np
from niio import loaded, write

import connectopy
from connectopy import similarity_mapping
from fragmenter import RegionExtractor as re

from scipy.spatial.distance import cdist
import scipy.io as sio
from scipy import sparse

import pandas as pd

import networkx as nx

# GLOBAL VARIABLES
HEMI_MAP = {'L': 'LEFT',
                'R': 'RIGHT'}
CORTEX_MAP = {'L': 'CortexLeft',
              'R': 'CortexRight'}
ISIZE=50


def plot_pairwise(sim_features, region_map, connectopy_dir,
                  out_dir, out_base,
                  subject_id, source_id, target_id,
                  max_neighborhood=50, plot=False, plot_neighborhood_sizes=[]):
    
    """
    Method to save and plot scatterplot matrices of source-to-target connectopy mappings.
    
    Parameters:
    - - - - -
    sim_features: float, array
        features to use in order to find most similar samples between source and target regions
        i.e. resting-state time series matrix, streamline connectivity matrix
    region_map: dict
        a mapping between region names and region indices
        can be computed using '''fragmenter.RegionExtractor.Extractor'''
    connectopy_dir: str
        main directory where regional connectopies exist
    out_dir: str
        directory where to save mapped coordinate files as csv
    out_base: str
        'Subject_ID.Hemisphere.'
    subject_id: str
        name of subject to process
    source_id: str
        source region name
    target_id: str
        target region name
    max_neighborhood: int
        largest neighborhood size to compute
    plot: boolean
        whether to generate plots
    plot_neighborhood_sizes: list of ints
        which neighborhood sizes to plot
        largest value must be <= '''max_neighborhood'''
    """
    
    source_indices = region_map[source_id]
    target_indices = region_map[target_id]

    source_evecs_file = '{:}/{:}/{:}.L.{:}.2.brain.Evecs.func.gii'.format(connectopy_dir, subject_id, subject_id, source_id)
    source_evecs = loaded.load(source_evecs_file)
    source_features = source_evecs[region_map[source_id], :]
    
    s_x = source_features[:, 0]
    s_y = source_features[:, 1]
    
    df = pd.DataFrame({'s_e1': s_x, 's_e2': s_y})
    
    target_evecs_file = '{:}/{:}/{:}.L.{:}.2.brain.Evecs.func.gii'.format(connectopy_dir, subject_id, subject_id, target_id)
    target_evecs = loaded.load(target_evecs_file)
    target_features = target_evecs[region_map[target_id], :]
    
    source_samples = sim_features[source_indices, :]
    target_samples = sim_features[target_indices, :]

    neighbors = similarity_mapping.cross_similarity(source_samples, 
                                                target_samples, 
                                                k=max_neighborhood, metric='correlation')
    
    neighbor_coords = {}.fromkeys(plot_neighborhood_sizes)
    for n in plot_neighborhood_sizes:
        
        neighbor_coords[n] = {'e1': None, 'e2': None}
        
        print('Neighborhood size: {:}'.format(n))
        
        centroids = similarity_mapping.cross_mapping(target_features, neighbors[:, 0:n])
        neighbor_coords[n]['e1'] = centroids[:, 0]
        neighbor_coords[n]['e2'] = centroids[:, 1]
        
        df = df.merge(pd.DataFrame({'t{:}_{:}'.format(n, eigen): neighbor_coords[n][eigen] for eigen in neighbor_coords[n].keys()}),
                     left_index=True, right_index=True)
        
        columns = ['s_e1', 's_e2'] + ['t{:}_{:}'.format(n, s) for s in ['e1', 'e2']]
        temp = df[columns]
        temp = temp.sort_values(by=['s_e1'])
        colors = np.arange(1, len(s_x)+1)
        
        if plot:
        
            A = pd.plotting.scatter_matrix(temp, diagonal='hist', c=colors, figsize=(15, 15), marker='.', alpha=0.5,
                                      hist_kwds=dict(alpha=0.5, density=True));

            for ax in A.ravel():
                ax.set_xlabel(ax.get_xlabel(), fontsize = 15)
                ax.set_xticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
                ax.set_xticklabels(labels=[0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=12)

                ax.set_ylabel(ax.get_ylabel(), fontsize = 15)
                ax.set_yticklabels(labels=[0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=12)
                plt.tight_layout()

            plt.yticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
            plt.show()
        
    out_csv = '{:}/{:}.{:}.2.{:}.csv'.format(out_dir, 
                                             out_base,
                                             source_id, 
                                             target_id)
    df.to_csv(out_csv, header=True, index=False)
    
    N = {'neighbors': target_indices[neighbors].squeeze()}
    out_neighbors = '{:}/{:}.{:}.2.{:}.Neighbors.mat'.format(out_dir, 
                                                             out_base,
                                                             source_id,
                                                             target_id)
    sio.savemat(file_name=out_neighbors, mdict=N, appendmat=False)
    
    return df


def mapping_counts(neighborhood):
    
    """
    Compute the number of times a given target vertex is mapped to by all source vertices, 
    as a function of neighborhood size.
    
    Parameters:
    - - - - -
    source_id: str
        source region ID
    target_id: str
        target region ID
    neighborhood: int, array
        neighborhood matrix -- i.e. mapping from source region to target region
        Each row is source vertex.  Each column is a target vertex.  Column order
        is from most similar to least similar to source vertex.
    roi_map: dict
        mapping from region names to indices in mesh
    """
    
    z = np.zeros((32492, neighborhood.shape[1]))
    
    y = np.zeros((32492,))
    for i in np.arange(neighborhood.shape[1]):
        
        for j in neighborhood[:, i]:
            y[j] += 1
        
        z[:,i] = y
    
    return z

def weightedadjacency(V, F):
    
    """
    Compute Euclidean distance matrix of all pairs of directly adjacency vertices in a surface mesh.
    
    Parameters:
    - - - - -
    V: float, array
        vertex coordinates
    F: int, array
        list of triangles in mesh
        
    Returns:
    - - - -
    W: float, array
        pairwise Euclidean distance matrix
    """

    n,p = V.shape

    weights = np.sqrt(np.concatenate([((V[F[:, 0], :] - V[F[:, 1], :])**2).sum(1),
                                     ((V[F[:, 0], :] - V[F[:, 2], :])**2).sum(1),
                                     ((V[F[:, 1], :] - V[F[:, 0], :])**2).sum(1),
                                     ((V[F[:, 1], :] - V[F[:, 2], :])**2).sum(1),
                                     ((V[F[:, 2], :] - V[F[:, 0], :])**2).sum(1),
                                     ((V[F[:, 2], :] - V[F[:, 1], :])**2).sum(1),]))

    eps = 1e-6

    rows = np.concatenate([F[:, 0], F[:, 0],
                         F[:, 1], F[:, 1],
                         F[:, 2], F[:, 2]])
    cols = np.concatenate([F[:, 1], F[:, 2],
                          F[:, 0], F[:, 2],
                          F[:, 0], F[:, 1]])

    combos = np.column_stack([rows, cols])
    [rc, idx] = np.unique(combos, axis=0, return_index=True)
    weights = weights[idx]

    W = sparse.csr_matrix((weights, (rc[:, 0], rc[:, 1])), shape=(n, n))
    W = (W + W.transpose())/2

    return W

def adjacency_2_nx(adjacency, inds):
    
    """
    Convert adjacency to networkx graph.
    
    Parameters:
    - - - - -
    adjacency: float, array
        adjacency matrix of whole-brain surface mesh
        Indices between neighboring pairs of vertices contain Euclidean distance,
        rather than 1 or 0.
    inds: int, list
        list of indices to compute networkx graph from
        
    Returns:
    - - - -
    g: networkx Graph
    
    """
    
    adjacency = adjacency[inds, :][:, inds]
    
    g = nx.from_scipy_sparse_matrix(adjacency)
    
    return g

def intraregional_distance(nx_graph):
    
    """
    Compute intraregional pairwise Euclidean distance matrix.
    
    Parameters:
    - - - - -
    nx_graph: nx.Graph
        Networkx graph object for adjacency matrix of single region.
        Indices between neighboring pairs of vertices contain Euclidean distance,
        rather than 1 or 0.
    
    Returns:
    - - - -
    apsp: float, array
        all-pairs shortest path length matrix
    """
    
    apsp = nx.floyd_warshall_numpy(nx_graph, weight='weight')
    
    return apsp

def knearest_distance(d, inds, ioi):
    
    """
    Compute the mean distance between all pairs of indices in a list.
    
    Parameters:
    - - - - -
    d: float, array
        
        All-pairs shortest path length matrix.
        In this case, used for intra-regional distnces.
        
    inds: dictionary
    
        A mapping from ROI indices to 0 : d.shape[1]
        
        Usage example:
            
            # initialize region extractor
            R = RegionExtractor.Extractor(label_file)
            
            # get indices for each region in map
            region_map = R.map_regions()
            
            # get indices for precentral gyrus
            inds = region_map['precentral']
            
            # create mapping from indices to order
            D = dict(zip(inds, np.arange(len(inds))))
             
    ioi: int, list
        
        List of indices (index from 0 : d.shape[1]) for
        which to compute mean distance between, in the
        original index coordinates, not in the remapped
        coordiantes.
    
    """
    
    remapped = []
    for i in ioi:
        remapped.append(inds[i])
    
    d = d[remapped, :][:, remapped]
    distances = d[np.triu_indices(d.shape[0], k=1)]
    
    return distances.mean()
    
def knearest_crosscorr(d, k):
    
    """
    Compute the mean cross-correlation between source vertex and k-most similar neighbors.
    
    Parameters:
    - - - - -
    d: float, array
        cross-correlation matrix between source and target regions
    k: int
        k-nearest neighbors
    """
    
    dsort = -1*np.sort(-1*d, axis=1)
    
    dmeans = np.zeros((d.shape[0], k))
    dmedians = np.zeros((d.shape[0], k))
    for i in np.arange(k):
        dmeans[:, i] = dsort[:, 0:(i+1)].mean(1).squeeze()
        dmedians[:, i] = np.median(dsort[:, 0:(i+1)], axis=1).squeeze()
    
    return [dmeans, dmedians]


####################################################################
def s2t_mappings(subject_id, region_map, hemisphere, features, connectopy_dir):
    
    """
    Sub-method for computing source-to-target neighborhood mappings.

    """
    for j, source_region in enumerate(list(region_map.keys())):
        if source_region not in ['corpuscallosum']:
            
            print('Source Map ID: {:}, {:}'.format(j, source_region))
            
            for target_region in region_map.keys():
                if target_region not in ['corpuscallosum']:
                    
                    out_dir = '{:}NeighborMappings/{:}'.format(connectopy_dir, subject_id)
                    if not os.path.exists(out_dir):
                        os.mkdir(out_dir)
                    
                    D = plot_pairwise(features, region_map, 
                    connectopy_dir='{:}Desikan'.format(connectopy_dir), 
                    out_dir=out_dir,
                    out_base='{:}.{:}'.format(subject_id, hemisphere),
                    subject_id=subject_id, 
                    source_id=source_region, 
                    target_id=target_region)



def st2_mappingcounts(subject_id, region_map, hemisphere, connectopy_dir):
    
    """
    Sub-method for aggregating source-to-target neighborhood mappings
    as total count scalar maps.
    
    """
    for j, source_region in enumerate(list(region_map.keys())):
        if source_region not in ['corpuscallosum']:
            
            z = np.zeros((32492, 50))
            
            for target_region in region_map.keys():
                if target_region not in ['corpuscallosum']:
                    
                    neighbor_file = '{:}NeighborMappings/{:}/{:}.{:}.{:}.2.{:}.Neighbors.mat'.format(connectopy_dir,
                                                                                            subject_id,
                                                                                            subject_id,
                                                                                            hemisphere,
                                                                                            source_region,
                                                                                            target_region)

                    neighbors = loaded.load(neighbor_file)
                    t = mapping_counts(neighbors)

                    n, p = t.shape
                    z[:, :p] += t
                    
            out_dir = '{:}NeighborCounts/{:}'.format(connectopy_dir, subject_id)
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)

            write.save(z, '{:}/{:}.{:}.{:}.2.Brain.Counts.func.gii'.format(out_dir,
                                                                           subject_id,
                                                                           hemisphere,
                                                                           source_region), CORTEX_MAP[hemisphere])



def s2t_correlations(subject_id, region_map, hemisphere, features, connectopy_dir):
    
    """
    Sub-method for computing correlations between source and target regions.
    """

    for j, source_region in enumerate(list(region_map.keys())[:-1]):
        if source_region not in ['corpuscallosum']:
            
            print('Source Map ID: {:}, {:}'.format(j, source_region))
            
            source_inds = region_map[source_region]
            source_features = features[source_inds, :]

            for i, target_region in enumerate(list(region_map.keys())[j:]):
                if target_region not in ['corpuscallosum']:

                    target_inds = region_map[target_region]
                    target_features = features[target_inds, :]

                    pairwise = 1-cdist(source_features, target_features, metric='correlation')
                    
                    # Compute nearest source-to-target correlations
                    [tmu, tmed] = knearest_crosscorr(pairwise, ISIZE)
                    z = np.zeros((32492, ISIZE))
                    z[source_inds, :] = tmu
                    
                    out_dir = '{:}NeighborFunctional/{:}'.format(connectopy_dir, subject_id)
                    if not os.path.exists(out_dir):
                        os.mkdir(out_dir)

                    write.save(z, '{:}/{:}.{:}.{:}.2.{:}.mean_knn.func.gii'.format(out_dir,
                                                                                   subject_id,
                                                                                   hemisphere,
                                                                                   source_region,
                                                                                   target_region), CORTEX_MAP[hemisphere])

                    # Compute nearest target-to-source correlations
                    [tmu, tmed] = knearest_crosscorr(pairwise.T, ISIZE)
                    z = np.zeros((32492, ISIZE))
                    z[target_inds, :] = tmu

                    write.save(z, '{:}/{:}.{:}.{:}.2.{:}.mean_knn.func.gii'.format(out_dir,
                                                                                   subject_id,
                                                                                   hemisphere,
                                                                                   target_region,
                                                                                   source_region), CORTEX_MAP[hemisphere])
                    


def s2t_correlations_aggregate(subject_id, region_map, hemisphere, connectopy_dir):
    
    """
    Sub-method to aggregate source-to-target correlation maps.
    """
    for j, target_region in enumerate(list(region_map.keys())[:-1]):
        if target_region not in ['corpuscallosum']:
            
            print('Source Map ID: {:}, {:}'.format(j, target_region))
            target_inds = region_map[target_region]

            z = np.zeros((32492, 50))
            for i, source_region in enumerate(list(region_map.keys())[j:]):
                if source_region not in ['corpuscallosum']:
                    source_inds = region_map[source_region]

                    knn_file = '{:}NeighborFunctional/{:}/{:}.{:}.{:}.2.{:}.mean_knn.func.gii'.format(connectopy_dir,
                                                                                                      subject_id,
                                                                                                      subject_id,
                                                                                                      hemisphere,
                                                                                                      source_region,
                                                                                                      target_region)
                    knn = loaded.load(knn_file)
                    z += knn

            z[target_inds, :] = np.nan
            out_dir = '{:}NeighborFunctional/{:}'.format(connectopy_dir, subject_id)
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
                
            write.save(z, '{:}/{:}.{:}.knn_mean.2.{:}.func.gii'.format(out_dir,
                                                                       subject_id,
                                                                       hemisphere,
                                                                       target_region), CORTEX_MAP[hemisphere])
            
            

def s2t_dispersion(subject_id, region_map, hemisphere, dat_dir, connectopy_dir):
    
    """
    Sub-method to compute distance between mapped neighbors.
    """
    for j, target_region in enumerate(list(region_map.keys())):
        if target_region not in ['corpuscallosum']:
            
            print('Source Map ID: {:}, {:}'.format(j, target_region))

            target_inds = region_map[target_region]
            target_maps = dict(zip(target_inds, np.arange(len(target_inds))))

            dist_file = '{:}RegionalDistances/{:}/{:}.{:}.{:}.DistanceMap.mat'.format(dat_dir,
                                                                                       subject_id,
                                                                                       subject_id,
                                                                                       hemisphere,
                                                                                       target_region)
            
            dmat = loaded.load(dist_file)

            z = np.zeros((32492, ISIZE))
            for i, source_region in enumerate(list(region_map.keys())):
                if source_region not in ['corpuscallosum']:
                    source_inds = region_map[source_region]

                    neighbor_file = '{:}NeighborMappings/{:}/{:}.{:}.{:}.2.{:}.Neighbors.mat'.format(connectopy_dir,
                                                                                            subject_id,
                                                                                            subject_id, 
                                                                                            hemisphere, 
                                                                                            source_region, 
                                                                                            target_region)
                    neighbors = loaded.load(neighbor_file)
                    n, p = neighbors.shape
                    for j in np.arange(n):
                        for k in np.arange(p):
                            z[source_inds[j], k] = knearest_distance(dmat, target_maps, neighbors[j, 0:(k+1)])
                            
            out_dir = '{:}NeighborDistances/{:}'.format(connectopy_dir, subject_id)
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
                            
            out_func = '{:}/{:}.{:}.Distance.2.{:}.func.gii'.format(out_dir,
                                                                    subject_id,
                                                                    hemisphere,
                                                                    target_region)
            write.save(z, out_func, CORTEX_MAP[hemisphere])
            
            

def regional_shortest_path(subject_id, region_map, hemisphere, dat_dir):
    
    """
    Sub-method to compute distance matrices for each cortical region in a label map.
    """
    
    midthick_surface_file = '{:}Surfaces/{:}.{:}.midthickness.32k_fs_LR.acpc_dc.surf.gii'.format(dat_dir, subject_id, hemisphere)
    [vertices, faces] = loaded.loadSurf(midthick_surface_file)
    sadj = weightedadjacency(V = vertices, F = faces)
    
    out_dir = '{:}RegionalDistances/{:}'.format(dat_dir, subject_id)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    for j, source_region in enumerate(region_map.keys()):
        
        print('Source Map ID: {:}, {:}'.format(j, source_region))
        
        out_dist = '{:}/{:}.{:}.{:}.DistanceMap'.format(out_dir,
                                                        subject_id,
                                                        hemisphere,
                                                        source_region)
        
        rinds = region_map[source_region]
        g = adjacency_2_nx(sadj, rinds)
        sps = intraregional_distance(g)
        sps = {'dist': sps}
        
        sio.savemat(file_name=out_dist, mdict=sps)
