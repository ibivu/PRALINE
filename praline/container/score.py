"""Score matrix container class and default score matrices.

.. moduleauthor:: Maurits Dijkstra <mauritsdijkstra@gmail.com>

"""
from __future__ import division, absolute_import, print_function

import numpy as np
import six
from six.moves import zip

from praline.core import *
from praline.container import ALPHABET_AA


class MatchScoreModel(Container):
    """Container type containing information about the probability that
    symbols are aligned (or matched) in an alignment. This is used by
    the raw pairwise aligner to calculate the match scores in the
    dynamic programming matrix. Generally, a match score model would be
    generated from one or more score matrices together with symbol or
    profile information about the two sequences.

    :param sequence_one: sequence corresponding to the y axis in the array
    :param sequence_two: sequence corresponding to the x axis in the array
    :param scores: two-dimensional array containing the match scores

    """
    tid = "praline.container.MatchScoreModel"

    def __init__(self, sequence_one, sequence_two, scores):
        if len(sequence_one) != scores.shape[0]:
            s = "sequence length {0} does not correspond to array shape {1}"
            raise DataError(s.format(len(sequence_one), scores.shape[0]))
        if len(sequence_two) != scores.shape[1]:
            s = "sequence length {0} does not correspond to array shape {1}"
            raise DataError(s.format(len(sequence_two), scores.shape[1]))


        self.sequence_one = sequence_one
        self.sequence_two = sequence_two
        self.scores = scores


class GapScoreModel(Container):
    """Container type containing information about the probability that
    a gap is aligned against a certain position in a sequence. Two of these
    models are used by the raw pairwise aligner to calculate the gap scores
    in both sequences. The model specifies the gap scores for a gap of length
    n for all positions in the sequence. Generally a gap score model would
    be generated from a constant series of gap penalties, optionally combined
    with a gap profile resulting from (for example) a preprofile alignment.

    :param sequence: sequence for which the gap scores are contained
    :param scores: two-dimensional array containing scores, with the x axis
                   and the y axis respectively denoting the sequence position
                   and the gap length

    """
    tid = "praline.container.GapScoreModel"

    def __init__(self, sequence, scores):
        if len(sequence) != scores.shape[0]:
            s = "sequence length {0} does not correspond to array shape {1}"
            raise DataError(s.format(len(sequence), scores.shape[0]))

        self.sequence = sequence
        self.scores = scores


class ScoreMatrix(Container):
    """The score matrix is the built-in container type that is used by the
    alignment logic to score a given sequence position against another. The
    score matrix container contains, as the name implies, an n-dimensional
    matrix containing score values. Each dimension of the this matrix is
    coupled to a specific alphabet. There is no restriction on the number
    of dimensions, but often it will be an even number, considering when
    aligning two sequences they usually have the same amount of tracks.

    :param scores: a dictionary containing a mapping of n-tuples to a float
        score, where n is the number of dimensions or alphabets
    :param alphabets: a list of alphabets corresponding to the dimensions
    :param matrix: optionally directly provide the raw matrix instead of
        having the object construct it from the scores

    """
    tid = "praline.container.ScoreMatrix"

    def __init__(self, scores, alphabets, matrix = None):
        if len(alphabets) < 2:
            s = "need at least 2 alphabets for a score matrix, got {0}"
            s = s.format(len(alphabets))
            raise DataError(s)

        self.alphabets = alphabets
        size = tuple(alphabet.size for alphabet in self.alphabets)
        self.matrix = np.zeros(size, dtype=np.float32)
        if scores is not None:
            for symbols, score in six.iteritems(scores):
                pairs = list(zip(self.alphabets, symbols))
                indexes = tuple(a.symbol_to_index(s) for a, s in pairs)
                self.matrix[indexes] = score
        else:
            self.matrix = matrix

    @property
    def symbols(self):
        """Return all the symbol pairs contained within this score matrix.
        Generally this is useful when trying to combine, add to or otherwise
        construct a new score matrix from this one. The pairs returned are
        n-tuples, where n is the number of dimensions in this matrix.

        """
        pairs = []
        for indices in np.ndindex(self.matrix.shape):
            pair = []
            for n, index in enumerate(indices):
                pair.append(self.alphabets[n].index_to_symbol(index))
            pairs.append(tuple(pair))

        return pairs

    def score(self, symbols):
        """Score a symbol combination according to this score matrix. This
        method is provided for convenience, but all the algorithms Generally
        directly access the score matrix contained in this object, as it is
        much faster than performing a method call every time a score is
        required.

        :param symbols: a tuple containing the symbol combination to score
        :returns: the score of the provided combination of symbols

        """
        pairs = list(zip(self.alphabets, symbols))
        indexes = tuple(a.symbol_to_index(s) for a, s in pairs)

        return self.matrix[indexes]


def _identity_matrix(alphabet, match_score=1, mismatch_score=0):
    """Helper method to generate an identity matrix for a given alphabet.
    An identity matrix is a 2-dimensional score matrix where both dimensions
    have the same alphabet. There are two possible scores, one score for
    a match between symbols and one score for a mismatch between symbols.

    :param alphabet: the alphabet to use to construct the identity matrix
    :param match_score: the score to use for a match
    :param mismatch_score: the score to use for a mismatch
    :returns: the identity matrix with the requested alphabet, match scores
        and mismatch scores

    """
    scores = {}
    symbols = alphabet.symbols
    for symbol_one in symbols:
        for symbol_two in symbols:
            if symbol_one == symbol_two:
                scores[(symbol_one, symbol_two)] = match_score
            else:
                scores[(symbol_one, symbol_two)] = mismatch_score

    return ScoreMatrix(scores, [alphabet, alphabet])


SCORE_MATRIX_DEFAULT_AA = _identity_matrix(ALPHABET_AA)
SCORE_MATRIX_DEFAULT = SCORE_MATRIX_DEFAULT_AA
