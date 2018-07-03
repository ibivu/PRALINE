"""Legacy code that will be eventually removed.

.. moduleauthor:: Maurits Dijkstra <mauritsdijkstra@gmail.com>

"""

from __future__ import division, absolute_import, print_function

import numpy as np
from six.moves import range

MINUS_INFINITY = -(2**32)

TYPE_SEQUENCE = 1
TYPE_PROFILE = 2

TRACEBACK_UP = 1 << 1
TRACEBACK_LEFT = 1 << 2
TRACEBACK_DIAG = 1 << 3

class AlignmentAlgorithm(object):
    """Superclass implementing the common logic of dynamic programming
    alignment algorithms. Subclasses implement the various alignment
    scenario's, such as local alignment or semiglobal alignment. This class,
    however, contains most of the heavy lifting.

    :param items: the items to align, can be either one-dimensional arrays
        containing sequences or two-dimensional arrays containing
        profiles
    :param score_matrix: the score matrix to use during alignment, this
        should be an array containing as many dimensions as supplied items.
    :param gap_one_scores: two-dimensional array containing gap penalties
        which should be assigned when a gap would be aligned against a
        certain position in sequence one.
    :param gap_two_scores: two-dimensional array containing gap penalties
        which should be assigned when a gap would be aligned against a
        certain position in sequence two.
    :param local: do a local alignment -- this determines whether the value
        zero is included in the evaluation of the maximum scoring value
        for a cell
    :param zero_idxs: a set of indices to lock to zero -- this is used by the
        Waterman-Eggert nonoptimal alignment algorithm to exclude a given
        range in iterations after the first

    """

    def __init__(self, item_sets, score_matrices, gap_one_scores,
                 gap_two_scores, local, zero_idxs = None):
        self._item_sets = item_sets
        self._type_sets = []
        for item_set in self._item_sets:
            type_set = []
            for item in item_set:
                if item.ndim == 2:
                    type_set.append(TYPE_PROFILE)
                else:
                    type_set.append(TYPE_SEQUENCE)
            self._type_sets.append(type_set)

        x = self._item_sets[0][0].shape[0]+1
        y = self._item_sets[0][1].shape[0]+1
        self.shape = (x, y)

        self._resolved_idxs = set()
        self.gap_one_scores = self.scores_to_acc_scores(gap_one_scores)
        self.gap_two_scores = self.scores_to_acc_scores(gap_two_scores)

        self.score_matrices = score_matrices
        self.local = local

        self.matrix = np.zeros(self.shape, dtype=np.float32)
        self.traceback = np.zeros(self.shape, dtype=np.uint8)
        self.zero_mask = np.zeros(self.shape, dtype=bool)

        if zero_idxs is not None:
            for idx in zero_idxs:
                self.zero_mask[idx] = True

    def scores_to_acc_scores(self, gap_scores):
        max_len = gap_scores.shape[1]
        gap_cumulative_scores = np.empty(gap_scores.shape[0], dtype=np.float32)
        acc = 0.0
        for i in range(gap_cumulative_scores.shape[0]):
            if i < max_len:
                gap_score = gap_scores[0, i]
            else:
                gap_score = gap_scores[0, max_len - 1]

            acc += gap_score
            gap_cumulative_scores[i] = acc

        return gap_cumulative_scores

    def indices(self):
        """Yields the two-dimensional indices corresponding to
        the cells of the dynamic programming matrix."""
        n, m = self.shape
        for x in range(n):
            for y in range(m):
                yield (x, y)

    def resolve_all(self):
        """Resolve all the cells in the dynamic programming matrix.

        """
        a = np.empty((4,), dtype=np.float32)

        for n, m in self.indices():
            if self.zero_mask[n, m] or (n, m) == (0, 0):
                self.matrix[n, m] = 0.0
                continue

            # If we're not at the topmost row we have a row above it to check.
            if n > 0:
                match_scores = self.matrix[0:n, m]
                gap_idxs = np.arange(n-1, -1, -1, dtype=np.intp)
                gap_scores = self.gap_one_scores[gap_idxs]

                a[0] = np.max(gap_scores + match_scores)
            else:
                a[0] = MINUS_INFINITY

            # If we're not at the leftmost column we have a column to the left
            # of it to check.
            if m > 0:
                match_scores = self.matrix[n, 0:m]
                gap_idxs = np.arange(m-1, -1, -1, dtype=np.intp)
                gap_scores = self.gap_two_scores[gap_idxs]

                a[1] = np.max(gap_scores + match_scores)
            else:
                a[1] = MINUS_INFINITY

            # If we're not at the topmost row and leftmost column, we
            # have a topleft cell to check.
            if n > 0 and m > 0:
                # Iterative reduction of the scoring matrix to the score. This
                # is probably the most important bit of all of PRALINE. We
                # loop through each profile/sequence we've been asked to
                # align in reverse order. For each step we reduce the dimension
                # of (a copy of) the score matrix by one, until we end up with
                # a dimensionality of 0, or one value. How we reduce is
                # dependent on whether we're dealing with a sequence or
                # profile. Both cases are described below:
                # * Sequence: simply get a slice of the matrix where every
                #             dimension is included in full except the target,
                #             of which only the sequence index is retained
                # * Profile: calculate the product of the probability vector
                #            and the matrix for the profile at the target
                #            position, then sum along the dimension to reduce
                #            it.
                match_score = 0.0
                for k, item_set in enumerate(self._item_sets):
                    types = self._type_sets[k]
                    score_matrix = self.score_matrices[k]
                    mat = np.array(score_matrix, dtype=np.float32)
                    ndim = mat.ndim
                    for x in range(ndim):
                        dim = ndim - x - 1
                        item = item_set[dim]
                        if dim % 2:
                            idx = m - 1
                        else:
                            idx = n - 1

                        if types[dim] == TYPE_SEQUENCE:
                            aop = [slice(None) for d in range(dim)]
                            aop.append(item[idx])

                            mat = mat[tuple(aop)]
                        else:
                            v = item[idx, :]
                            aop = [np.newaxis for d in range(dim)]
                            aop.append(slice(None))
                            mat = (v[tuple(aop)] * mat).sum(axis=dim)
                    match_score += mat
                a[2] = self.matrix[n-1, m-1] + match_score
            else:
                a[2] = MINUS_INFINITY


            if self.local:
                # If we're doing a local alignment, we should always include zero
                # as a possible max score.
                a[3] = 0
            else:
                a[3] = MINUS_INFINITY

            a_max = a.max()
            self.matrix[n, m] = a_max

            tb = 0
            if a[0] == a_max:
                tb |= TRACEBACK_UP
            if a[1] == a_max:
                tb |= TRACEBACK_LEFT
            if a[2] == a_max:
                tb |= TRACEBACK_DIAG

            self.traceback[n, m] = tb

    def get_score(self):
        """Get the alignment score. This corresponds to the score of the
        bottom rightmost cell. Subclasses which have different ways of
        scoring alignments are supposed to override this method.

        :returns: the score of the alignment
        """

        return self.matrix[(self.shape[0]-1, self.shape[1]-1)]


class NeedlemanWunschAlgorithm(AlignmentAlgorithm):
    """Class with an implementation of a standard Needleman-Wunsch
    global alignment strategy.

    :param items: the items to align, can be either one-dimensional arrays
        containing sequences or two-dimensional arrays containing
        profiles
    :param score_matrix: the score matrix to use during alignment, this
        should be an array containing as many dimensions as supplied items.
    :param gap_series: a one-dimensional array with gap penalties for a gap
        of length l -- if the gap length exceeds the size of this array the
        last element is used to determine the penalty
    :param zero_idxs: a set of indices to lock to zero -- this is used by the
        Waterman-Eggert nonoptimal alignment algorithm to exclude a given
        range in iterations after the first

    """
    def __init__(self, item_sets, score_matrices, gap_one_scores, gap_two_scores,
                 zero_idxs):
        super(NeedlemanWunschAlgorithm, self).__init__(
            item_sets, score_matrices, gap_one_scores, gap_two_scores,
            local=False, zero_idxs=zero_idxs)


class SmithWatermanAlgorithm(AlignmentAlgorithm):
    """Class with an implementation of a standard Smith-Waterman local
    alignment strategy.

    :param items: the items to align, can be either one-dimensional arrays
        containing sequences or two-dimensional arrays containing
        profiles
    :param score_matrix: the score matrix to use during alignment, this
        should be an array containing as many dimensions as supplied items.
    :param gap_series: a one-dimensional array with gap penalties for a gap
        of length l -- if the gap length exceeds the size of this array the
        last element is used to determine the penalty
    :param zero_idxs: a set of indices to lock to zero -- this is used by the
        Waterman-Eggert nonoptimal alignment algorithm to exclude a given
        range in iterations after the first

    """
    def __init__(self, item_sets, score_matrices, gap_one_scores, gap_two_scores,
                 zero_idxs):
        super(SmithWatermanAlgorithm, self).__init__(item_sets,
                                                     score_matrices,
                                                     gap_one_scores,
                                                     gap_two_scores,
                                                     local=True,
                                                     zero_idxs=zero_idxs)

    def get_score(self):
        """Get the alignment score. This corresponds to the score of the
        bottom rightmost cell. Subclasses which have different ways of
        scoring alignments are supposed to override this method.

        :returns: the score of the alignment
        """

        return self.matrix.max()

class SemiGlobalAlignmentAlgorithm(AlignmentAlgorithm):
    """Class implementing a semiglobal alignment strategy.

    :param items: the items to align, can be either one-dimensional arrays
        containing sequences or two-dimensional arrays containing
        profiles
    :param score_matrix: the score matrix to use during alignment, this
        should be an array containing as many dimensions as supplied items.
    :param gap_series: a one-dimensional array with gap penalties for a gap
        of length l -- if the gap length exceeds the size of this array the
        last element is used to determine the penalty
    :param zero_idxs: a set of indices to lock to zero -- this is used by the
        Waterman-Eggert nonoptimal alignment algorithm to exclude a given
        range in iterations after the first

    """
    def __init__(self, item_sets, score_matrices, gap_one_scores, gap_two_scores,
                 zero_idxs):
        spr = super(SemiGlobalAlignmentAlgorithm, self)
        spr.__init__(item_sets, score_matrices, gap_one_scores, gap_two_scores,
                     local=False, zero_idxs=zero_idxs)

    def indices(self):
        """Yields the two-dimensional indices corresponding to
        the cells of the dynamic programming matrix."""
        n, m = self.shape
        for x in range(1, n):
            for y in range(1, m):
                yield (x, y)

    def get_score(self):
        """Get the alignment score. This corresponds to the score of the
        bottom rightmost cell. Subclasses which have different ways of
        scoring alignments are supposed to override this method.

        :returns: the score of the alignment
        """
        n, m = self.shape

        return max(self.matrix[n-1, :].max(), self.matrix[:, m-1].max())


def window(l, size = 2):
    """Slide a window of given size over a list, yielding tuples of the given
    size.

    :param l: the list to slide a window over
    :param size: the window size:
    """
    for n in range(len(l) - size + 1):
        yield tuple(n+m for m in range(size))
