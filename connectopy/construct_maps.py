import numpy as np


def mappings(sinds, tinds, coordinates, connectivity):

    """
    Map 3-D source coordinate to 3-D target coordinate and streamline count
    between them.

    Parameters:
    - - - - -
        sinds : source indices
        tinds : target indices
        coordinates : 2-D array of 3-D coordinates
        connectivity : |sinds| x |tinds| array of streamline counts

    Returns:
    - - - -
        source : 2-D array of 3-D source coordinates
        target : 2-D array of 3-D source coordinates
        count : 1-D array of counts between source and target coordinates
    """

    source = []
    target = []
    fibers = []

    for s, s_idx in enumerate(sinds):

        for t, t_idx in enumerate(tinds):

            count = connectivity[s, t]

            if count > 0:

                s_coord = coordinates[s_idx, :][np.newaxis, :]
                t_coord = coordinates[t_idx, :][np.newaxis, :]

                source.append(s_coord)
                target.append(t_coord)
                fibers.append(count)

    if len(source) > 0 and len(target) > 0 and len(fibers) > 0:
        source = np.row_stack(source)
        target = np.row_stack(target)
    else:
        source = np.asarray(source)
        target = np.asarray(target)

    fibers = np.asarray(fibers).astype(np.int32)

    return [source, target, fibers]


def stack(source, target, counts):

    """
    Repeat source and target coordinate pairs, as many times as there are
    streamlines between these two endpoints.

    Use array instead of list-concatenation because streamline counts.

    Parameters:
    - - - - -
        source: 2-D array of 3-D source coordinates
        target: 2-D array of 3-D target coordinates
        counts: 1-D array of counts between source and target points

    Returns:
    - - - -
        sources : repeated source coordinates
        targets : repeated target coordinates
    """

    assert source.shape[0] == target.shape[0] == counts.shape[0]

    totalfibers = np.sum(counts)

    sources = np.zeros(shape=(totalfibers, 3))
    targets = np.zeros(shape=(totalfibers, 3))

    curr = 0

    for c in np.arange(len(counts)):

        update = counts[c]

        s_coord = source[c, :][np.newaxis, :]
        t_coord = target[c, :][np.newaxis, :]

        sources[(curr):(curr+update), :] = np.repeat(s_coord, update, axis=0)
        targets[(curr):(curr+update), :] = np.repeat(t_coord, update, axis=0)

        curr += update

    return [sources, targets]
