"""Sequence tree container type.

.. moduleauthor:: Maurits Dijkstra <mauritsdijkstra@gmail.com>

"""
from __future__ import division, absolute_import, print_function

from praline.core import *

class SequenceTree(Container):
    """The SequenceTree container is, as the name implies, a tree with
    sequence objects representing the various nodes. Sequence trees are
    constructed and used to store the merge order between the input sequences
    in various multiple sequence alignment scenarios.

    :param sequences: the sequences mapping to nodes of the tree
    :param merge_orders: a list of 2-tuples containing two indices which
    should be merged when walking the tree from bottom to top.

    """
    tid = "praline.container.SequenceTree"

    def __init__(self, sequences, merge_orders):
        self.merge_orders = merge_orders
        self.sequences = sequences
