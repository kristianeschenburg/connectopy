# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 19:24:48 2016

@author: kristianeschenburg
"""

import numpy as np
from numpy import linalg as LA

class L1(object):

    """
    Compute the L1 median of a set of points.
    """

    def __init__(self, eps=1e-15, max_iter=200, tol=100):

        self.eps = eps
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, samples):

        """
        Fitting procedure.

        Parameters:
        - - - - -
        samples: float, array
            data to compute geometric median of
        """

        med = np.median(samples, axis=0)[None, :]

        k = 0
        while k <= self.max_iter:

            # if current median exists in samples
            # randomly perturb
            if self.equality(samples, med):

                med = med + np.random.normal(med.std(), med.mean(), np.shape(med))

            update = self.weiszfeld(samples, med)

            tol = LA.norm(med-update)
            self.tol = tol
            med = update[None, :]

            if (self.tol < self.eps):
                break
            k += 1

        self.median_ = med

    def weiszfeld(self, samples, current):

        """
        Updates current median using Weiszfeld's algorithm
        
        Parameter:
        - - - - -
            data: float, array
                array of samples
            current: float, array
                current median
        """

        x, y = np.shape(samples)
        tiled = np.tile(current, [x, 1])

        dist = LA.norm(samples-tiled, axis=1)

        numer = samples / (np.tile(dist, [y, 1]).T)
        denom = np.ones(shape=[x, y]) / (np.tile(dist, [y, 1]).T)

        update = numer.sum(axis=0) / denom.sum(axis=0)

        return update

    def equality(self, samples, current):

        """
        Compute whether current median is equivalent to any points in the set
        
        Parameters:
        - - - - -
            samples: float, array
                samples to compare t
            current: float, array
                comparison sample
        """

        n, p = samples.shape
        reps = np.repeat(current, n, axis=0)

        comp = (reps == samples)
        comp = comp.sum(axis=1) == p

        return np.any(comp)