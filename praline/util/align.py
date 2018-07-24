"""Dynamic programming alignment and support methods.

.. moduleauthor:: Maurits Dijkstra <mauritsdijkstra@gmail.com>

"""

from __future__ import division, absolute_import, print_function

import numpy as np
from six.moves import range

from .support import window

# Traceback bit flags for the optimized pairwise aligner.
TRACEBACK_MATCH_MATCH = 1 << 1
TRACEBACK_MATCH_INSERT_UP = 1 << 2
TRACEBACK_MATCH_INSERT_LEFT = 1 << 3
TRACEBACK_INSERT_UP_OPEN = 1 << 4
TRACEBACK_INSERT_UP_EXTEND = 1 << 5
TRACEBACK_INSERT_LEFT_OPEN = 1 << 6
TRACEBACK_INSERT_LEFT_EXTEND = 1 << 7

# Needed in case of missing values in the maximum value determination.
MINUS_INFINITY = -np.inf

# Mode flags. Only used internally.
_MODE_GLOBAL = 0
_MODE_SEMIGLOBAL = 1
_MODE_LOCAL = 2

def align_global(m, g1, g2, o, t, z):
    align(m, g1, g2, o, t, z, _MODE_GLOBAL)

def align_semiglobal(m, g1, g2, o, t, z):
    align(m, g1, g2, o, t, z, _MODE_SEMIGLOBAL)

def align_local(m, g1, g2, o, t, z):
    align(m, g1, g2, o, t, z, _MODE_LOCAL)


def align(m, g1, g2, o, t, z, mode):
    # Fill in the matrices according to the recurrence relations.
    # Loop through them in linear fashion so we know we never have any problems
    # with dependencies between cells.
    for y in range(1, m.shape[0] + 1):
        for x in range(1, m.shape[1] + 1):
            has_up = y > 0
            has_left = x > 0
            has_upleft = has_up and has_left
            masked = z[y, x]

            if (not (has_up or has_left)) or masked:
                continue

            # Calculate all scoring terms for the insert state of sequence one.
            score_insert_up_open = 0.0
            score_insert_up_extend = 0.0
            if has_up:
                score_insert_up_open = o[y - 1, x, 0] + g1[y - 1, 0]
                score_insert_up_extend = o[y - 1, x, 1] + g1[y - 1, 1]

            # Calculate all scoring terms for the insert state of sequence two.
            score_insert_left_open = 0.0
            score_insert_left_extend = 0.0
            if has_left:
                score_insert_left_open = o[y, x - 1, 0] + g2[x - 1, 0]
                score_insert_left_extend = o[y, x - 1, 2] + g2[x - 1, 1]

            # Calculate all scoring terms for the match state.
            score_match_match = 0.0
            score_match_insert_up = 0.0
            score_match_insert_left = 0.0
            if has_upleft:
                match_score = m[y - 1, x - 1]
                score_match_match = o[y - 1, x - 1, 0] + match_score
                score_match_insert_up = o[y - 1, x - 1, 1] + match_score
                score_match_insert_left = o[y - 1, x - 1, 2] + match_score


            # Determine the maximum score for the match state. Add zero if
            # we're doing a local alignment. Write the result to the traceback
            # matrix and to the dp matrix.
            if mode == _MODE_LOCAL:
                score_match_max = 0.0
            else:
                score_match_max = MINUS_INFINITY
            if has_upleft:
                if score_match_match > score_match_max:
                    score_match_max = score_match_match
                if score_match_insert_up > score_match_max:
                    score_match_max = score_match_insert_up
                if score_match_insert_left > score_match_max:
                    score_match_max = score_match_insert_left

                tb_match = 0
                if score_match_match == score_match_max:
                    tb_match |= TRACEBACK_MATCH_MATCH
                if score_match_insert_up == score_match_max:
                    tb_match |= TRACEBACK_MATCH_INSERT_UP
                if score_match_insert_left == score_match_max:
                    tb_match |= TRACEBACK_MATCH_INSERT_LEFT

                t[y, x, 0] = tb_match
                o[y, x, 0] = score_match_max


            # Determine the maximum score for the sequence one insert state.
            # Write the result to the traceback matrix.
            score_insert_up_max = MINUS_INFINITY
            if has_up:
                if score_insert_up_open > score_insert_up_max:
                    score_insert_up_max = score_insert_up_open
                if score_insert_up_extend > score_insert_up_max:
                    score_insert_up_max = score_insert_up_extend

                tb_insert_up = 0
                if score_insert_up_open == score_insert_up_max:
                    tb_insert_up |= TRACEBACK_INSERT_UP_OPEN
                if score_insert_up_extend == score_insert_up_max:
                    tb_insert_up |= TRACEBACK_INSERT_UP_EXTEND

                t[y, x, 1] = tb_insert_up
                o[y, x, 1] = score_insert_up_max

            # Determine the maximum score for the sequence two insert state.
            # Write the result to the traceback matrix.
            score_insert_left_max = MINUS_INFINITY
            if has_left:
                if score_insert_left_open > score_insert_left_max:
                    score_insert_left_max = score_insert_left_open
                if score_insert_left_extend > score_insert_left_max:
                    score_insert_left_max = score_insert_left_extend

                tb_insert_left = 0
                if score_insert_left_open == score_insert_left_max:
                    tb_insert_left |= TRACEBACK_INSERT_LEFT_OPEN
                if score_insert_left_extend == score_insert_left_max:
                    tb_insert_left |= TRACEBACK_INSERT_LEFT_EXTEND

                t[y, x, 2] = tb_insert_left
                o[y, x, 2] = score_insert_left_max


def get_paths(traceback, cell, strip_third_dim=True):
    """Traces the path the alignment takes through the score matrix. Starts
    at the bottom rightmost cell.

    :param traceback: traceback matrix to trace the path in
    :param cell: index of the cell to start tracing at
    :param strip_third_dim: whether to remove the third dimension (state) from
        the paths
    :returns: a list of traced paths

    """
    path = [cell]

    while True:
        n, m, k = cell

        tb_value = traceback[n, m, k]
        if tb_value & TRACEBACK_MATCH_MATCH:
            cell = (n - 1, m - 1, 0)
        elif tb_value & TRACEBACK_MATCH_INSERT_UP:
            cell = (n - 1, m - 1, 1)
        elif tb_value & TRACEBACK_MATCH_INSERT_LEFT:
            cell = (n - 1, m - 1, 2)
        elif tb_value & TRACEBACK_INSERT_UP_OPEN:
            cell = (n - 1, m, 0)
        elif tb_value & TRACEBACK_INSERT_UP_EXTEND:
            cell = (n - 1, m, 1)
        elif tb_value & TRACEBACK_INSERT_LEFT_OPEN:
            cell = (n, m - 1, 0)
        elif tb_value & TRACEBACK_INSERT_LEFT_EXTEND:
            cell = (n, m - 1, 2)
        else:
            break

        path.append(cell)

    path.reverse()

    if strip_third_dim:
        return [[(n, m) for n, m, k in path]]
    else:
        return [path]

def get_frequencies(alignment, trid):
    """Calculate the occurence frequency of each symbol in a given
    alignment.

    :param alignment: the alignment to calculate the frequencies for
    :param trid: the track id of the track to calculate the frequencies for
    :returns: a two-dimensional array containing the absolute frequencies
        for every possible symbol for the given alphabet at every sequence
        position

    """


    path = alignment.path
    tracks = [seq.get_track(trid) for seq in alignment.items]
    alphabet = tracks[0].alphabet
    seqs = [track.values for track in tracks]

    freqs = np.zeros((path.shape[0]-1, alphabet.size), dtype=int)
    for i, i_next in window(list(range(path.shape[0]))):
        inc_cols = (path[i_next, :]-path[i, :]) > 0
        for j, inc_col in enumerate(inc_cols):
            if inc_col:
                seq_idx = path[i_next, j]
                freqs[i, seqs[j][seq_idx-1]] += 1

    return freqs

def compress_path(path, compress_idx):
    """Compresses an alignment path so that any gaps in the master sequence
    are removed. This is useful when building master-slave alignments
    because in such a case it is not important to consider gaps in the
    master sequence.

    :param path: the alignment path to compress
    :param compress_idx: the column index of the master sequence
    :returns: the compressed alignment path
    """
    keep_idxs = [0]

    for i, i_next in window(list(range(path.shape[0]))):
        inc_cols = (path[i_next, :]-path[i, :]) > 0
        if inc_cols[compress_idx]:
            keep_idxs.append(i_next)

    return path[keep_idxs, :]

def extend_path_local(path, extend_length, extend_idx):
    """Extends a local alignment path across the entire matrix. This is
    required when working with local alignments, because the alignment
    routines expect a path that starts at the bottom right and ends at
    the top left of the matrix. The positions that are filled in by this
    method are set to -1 to signify that they are not part of the alignment
    but are unfilled local alignment positions.

    :param path: the alignment path to extend
    :param extend_length: the extension length
    :param extend_idx: the column index of the sequence the incomplete
        second sequence was locally aligned to
    :returns: the extended alignment path

    """

    first_idx = path[0, extend_idx]
    last_idx = path[-1, extend_idx]

    if first_idx > 0:
        extension = np.empty((first_idx, path.shape[1]), dtype=int)
        extension.fill(-1)
        extension[:, extend_idx] = np.arange(first_idx)
        path = np.vstack([extension, path])

    if last_idx < extend_length:
        begin_idx=extend_length-last_idx
        extension = np.empty((begin_idx, path.shape[1]), dtype=int)
        extension.fill(-1)
        extension[:, extend_idx] = np.arange(last_idx+1, extend_length+1)
        path = np.vstack([path, extension])

    return path

def extend_path_semiglobal(path, mat_shape):
    extensions = []
    if path[0, 0] != 0:
        preext = np.zeros((path[0, 0], 2), dtype=int)
        preext[:, 0] = np.arange(path[0, 0], dtype=int)

        extensions.append(preext)
    elif path[0, 1] != 0:
        preext = np.zeros((path[0, 1], 2), dtype=int)
        preext[:, 1] = np.arange(path[0, 1], dtype=int)

        extensions.append(preext)

    extensions.append(path)

    n, m = mat_shape
    if path[-1, 0] != n-1:
        postext = np.empty(((n-1)- path[-1, 0], 2), dtype=int)
        postext[:, 1] = path[-1, 1]
        postext[:, 0] = np.arange(path[-1, 0]+1, n, dtype=int)

        extensions.append(postext)
    elif path[-1, 1] != m-1:
        postext = np.empty(((m-1) - path[-1, 1], 2), dtype=int)
        postext[:, 0] = path[-1, 0]
        postext[:, 1] = np.arange(path[-1, 1]+1, m, dtype=int)

        extensions.append(postext)

    return np.vstack(extensions)

def auto_align_mode(one, two):
    if len(one) > len(two):
        align_mode = "semiglobal_one"
    else:
        align_mode = "semiglobal_two"

    return align_mode
