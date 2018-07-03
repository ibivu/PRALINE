"""Container type to store and operate on an alignment between sequences.

.. moduleauthor:: Maurits Dijkstra <mauritsdijkstra@gmail.com>

"""
from __future__ import division, absolute_import, print_function

import numpy as np
from six.moves import range

from praline.core import *
from praline.util import window

class Alignment(Container):
    """Alignment container type. This container is used to store an alignment
    between two or more sequences. A method is also provided to merge two
    alignments. Internally the alignment stores the alignment as a path
    through the dynamic programming matrix.

    :param items: the sequences which are aligned in this container
    :param path: a two-dimensional array containing a path through the
        alignment matrix
    """
    tid = "praline.container.Alignment"

    def __init__(self, items, path):
        self.items = items
        self.path = path

    def merge(self, alignment, path):
        """Merge this alignment with another one. The resulting alignment
        will contain all the sequences from this one and the merge target.
        The merge is not performed in-place, meaning that this alignment and
        the target remain intact and a new alignment is created.

        :param alignment: the alignment to merge with
        :param path: an alignment path to guide the merge
        :returns: an alignment object representing the merged alignments

        """
        size_one = (path.shape[0], self.path.shape[1])
        size_two = (path.shape[0], alignment.path.shape[1])
        path_component_one = np.zeros(size_one, dtype=int)
        path_component_two = np.zeros(size_two, dtype=int)

        for i in range(path.shape[0]):
            idx_one = path[i, 0]
            if idx_one >= 0:
                path_component_one[i, :] = self.path[idx_one, :]
            else:
                path_component_one[i, :] = (-1)

            idx_two = path[i, 1]
            if idx_two >= 0:
                path_component_two[i, :] = alignment.path[path[i, 1], :]
            else:
                path_component_two[i, :] = (-1)
        merged_path = np.hstack([path_component_one, path_component_two])
        merged_sequences = self.items+alignment.items

        return Alignment(merged_sequences, merged_path)
