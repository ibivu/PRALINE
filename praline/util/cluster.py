"""Hierarchical clustering classes and support methods.

.. moduleauthor:: Maurits Dijkstra <mauritsdijkstra@gmail.com>

"""
from __future__ import division, absolute_import, print_function

import numpy as np
import six
from six.moves import range

INFINITY = 2**32


class HierarchicalClusteringAlgorithm(object):
    """Implementation of a hierarchical clustering algorithm with support
    for multiple linkage methods.

    :params distance_matrix: a two-dimensional array containing pairwise
        distances for all objects

    """

    def __init__(self, distance_matrix):
        self.distance_matrix = distance_matrix

    def merge_order(self, linkage='single'):
        """ Yield the merge order for the pairwise distance matrix given
        according to a given linkage method.

        :param linkage: the linkage method to use
        """
        linkage_fun = LINKAGES[linkage]

        clusters = {i: {i} for i in range(self.distance_matrix.shape[0])}
        while len(clusters) > 1:
            a = np.empty((len(clusters), len(clusters)), dtype=float)
            id_map = {}

            np.fill_diagonal(a, INFINITY)

            for i, (c_one_id, c_one) in enumerate(six.iteritems(clusters)):
                id_map[i] = c_one_id
                for j, (c_two_id, c_two) in enumerate(six.iteritems(clusters)):
                    if c_one_id != c_two_id:
                        a[i, j] = linkage_fun(
                            self.distance_matrix, c_one, c_two)


            merge_idxs = np.unravel_index(a.argmin(), a.shape)
            merge_one_idx, merge_two_idx = merge_idxs
            merge_one_id = id_map[merge_one_idx]
            merge_two_id = id_map[merge_two_idx]

            clusters[merge_one_id] |= clusters[merge_two_id]
            del clusters[merge_two_id]
            yield merge_one_id, merge_two_id


def _single_linkage(distance_matrix, c_one, c_two):
    """Single linkage method. Calculates a distance between two given
    clusters.

    :param distance_matrix: the distance matrix
    :param c_one: list of objects in cluster one
    :param c_two: list of objects in cluster two
    :returns: distance between clusters according to linkage method

    """
    a = np.empty((len(c_one), len(c_two)), dtype=float)
    for i, c_one_id in enumerate(c_one):
        for j, c_two_id in enumerate(c_two):
            a[i, j] = distance_matrix[c_one_id, c_two_id]

    return a.min()


def _complete_linkage(distance_matrix, c_one, c_two):
    """Complete linkage method. Calculates a distance between two given
    clusters.

    :param distance_matrix: the distance matrix
    :param c_one: list of objects in cluster one
    :param c_two: list of objects in cluster two
    :returns: distance between clusters according to linkage method

    """
    a = np.empty((len(c_one), len(c_two)), dtype=float)
    for i, c_one_id in enumerate(c_one):
        for j, c_two_id in enumerate(c_two):
            a[i, j] = distance_matrix[c_one_id, c_two_id]

    return a.max()


def _average_linkage(distance_matrix, c_one, c_two):
    """Average linkage method. Calculates a distance between two given
    clusters.

    :param distance_matrix: the distance matrix
    :param c_one: list of objects in cluster one
    :param c_two: list of objects in cluster two
    :returns: distance between clusters according to linkage method

    """
    a = np.empty((len(c_one), len(c_two)), dtype=float)
    for i, c_one_id in enumerate(c_one):
        for j, c_two_id in enumerate(c_two):
            a[i, j] = distance_matrix[c_one_id, c_two_id]

    return a.mean()

LINKAGES = {'single': _single_linkage, 'complete':
            _complete_linkage, 'average': _average_linkage}
