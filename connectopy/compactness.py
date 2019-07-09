import numpy as np
import niio
import surface_utilities as sul

import networkx as nx

def top2circ(p):
    
    """
    Return area of circle with same perimeter as arbitrary shape.
    
    Parameters:
    - - - - -
    p: float
        perimeter of arbitary shape
    
    Returns:
    - - - -
    a: float
        area of circle
    r: float
        radius of circle
    """
    
    radius = p / (2*np.pi)
    area = np.pi*(radius**2)
    
    return [radius, area]

def skew(coordinates, neighbors):
    
    """
    Return skew of coordinate axis.
    
    Parameters:
    - - - - -
    coordinates: float, array
        features to compute skew of
    neighbors: int, array
        points to include in skew computation
    
    Returns:
    - - - -
    s: float, array
        skew value for each input point
    """
    
    coordinates = coordinates[neighbors]
    s = stats.skew(coordinates, axis=1)
    
    return s

def kurtosis(coordinates, neighbors):
    
    """
    Compute kutosis of coorinate axis.
    
    Parameters:
    - - - - -
    coordinates: float, array
        features to compute kurtosis of
    neighbors: int, array
        points in include in kurtosis computation
    
    Returns:
    - - - -
    k: float, array
        kurtosis value for each input point
    """
    
    coordinates = coordinates[neighbors]
    k = stats.kurtosis(coordinates, axis=1)
    
    return k