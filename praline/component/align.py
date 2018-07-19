"""Pairwise aligner and master-slave alignment components.

.. moduleauthor:: Maurits Dijkstra <mauritsdijkstra@gmail.com>

"""
from __future__ import division, absolute_import, print_function

import itertools

import numpy as np
from six.moves import range, zip

from praline.core import *
from praline.container import Sequence, Alignment, ScoreMatrix
from praline.container import SCORE_MATRIX_DEFAULT, PlainTrack, ProfileTrack
from praline.container import MatchScoreModel, GapScoreModel
from praline.util import NeedlemanWunschAlgorithm, SmithWatermanAlgorithm
from praline.util import SemiGlobalAlignmentAlgorithm, get_paths
from praline.util import get_paths_legacy, extend_path_semiglobal
from praline.util import cext_align_global, cext_align_local
from praline.util import cext_build_scores, cext_align_semiglobal_both
from praline.util import cext_align_semiglobal_one, cext_align_semiglobal_two
from praline.util import align_global, align_semiglobal, align_local
from praline.util import TRACEBACK_INSERT_LEFT_EXTEND
from praline.util import TRACEBACK_INSERT_UP_EXTEND

MINUS_INFINITY = -np.inf

_CEXT_ALIGN_FUNCTIONS = {"local": cext_align_local,
                         "global": cext_align_global,
                         "semiglobal_both": cext_align_semiglobal_both,
                         "semiglobal_one": cext_align_semiglobal_one,
                         "semiglobal_two": cext_align_semiglobal_two}

_ALIGN_FUNCTIONS = {"local": align_local,
                    "global": align_global,
                    "semiglobal_both": align_semiglobal}

class LegacyPairwiseAligner(Component):
    """The LegacyPairwiseAligner used to be the main 'pure Python'
    pairwise alignment class, but has since been superseded by the current
    PairwiseAligner component. It is still included in the PRALINE source
    because it is a completely seperate alignment implementation and thus
    may be useful for debugging. In the future we may remove it entirely
    if the API drifts too far away from its current API or if we feel it
    is no longer useful.

    Inputs:
    * mode - string containing the requested alignment mode
             ('local', 'global' or 'semiglobal')
    * sequence_one - the first sequence to align
    * sequence_two - the second sequence to align
    * track_id_sets_one - a list containing lists of strings specifying
                          which tracks should be aligned from sequence one
    * track_id_sets_two - a list containing lists of strings specifying
                          which tracks should be aligned from sequence two
    * zero_idxs - a list of 2-tuples containing indices in the dynamic
                  programming matrix which should be fixed to zero, which is
                  used by multiple alignment techniques such as Waterman-
                  Eggert.
    * score_matrices - use these score matrices for the alignment

    Outputs:
    * alignment - an alignment object containing the resulting alignment
    * score - the score of the alignment, the score is guaranteed to be a
              float, but its scale depends on the alignment method and other
              factors

    Options:
    * gap_series - a list of floats specifying the gap penalties which should
                   be applied for a gap of a certain length (or larger, if
                   the gap length reaches the end of the list)
    * squash_profiles - convert profiles to consensus sequences
    * debug - integer indicating the debug level (0 is silent, 3 is noisiest)

    """
    tid = "praline.component.LegacyPairwiseAligner"
    inputs = {'mode': Port(str),
              'sequence_one': Port(Sequence.tid),
              'sequence_one': Port(Sequence.tid),
              'track_id_sets_one': Port([[str]]),
              'track_id_sets_two': Port([[str]]),
              'score_matrices': Port([ScoreMatrix.tid]),
              'zero_idxs': Port([(int, int)], optional = True)}
    outputs = {'alignment': Port(Alignment.tid), 'score': Port(float)}

    options = {'gap_series': [float], 'squash_profiles': bool,
               'debug': int}
    defaults = {'gap_series': [-11.0, -1.0], 'squash_profiles': False,
                'debug': 0}

    def execute(self, mode, sequence_one, sequence_two,
                track_id_sets_one, track_id_sets_two,
                score_matrices, zero_idxs):
        gap_series = np.array(self.environment['gap_series'],
                              dtype=np.float32)
        debug = self.environment['debug']

        if debug > 0:
            log = LogBundle()
            msg = "Entering component '{0}'".format(self.tid)
            log.message(ROOT_LOG_NAME, msg)

            log.message(ROOT_LOG_NAME, "Alignment mode: '{0}'".format(mode))
            log.message(ROOT_LOG_NAME, "Gap scores: {0}".format(gap_series))

            msg = "Sequence one: '{0}', sequence two: '{1}'"
            msg = msg.format(sequence_one.name, sequence_two.name)
            log.message(ROOT_LOG_NAME, msg)

            log.message(ROOT_LOG_NAME, "Track id sets for sequence one:")
            for track_id_set in track_id_sets_one:
                msg = "\t{0}".format(", ".join(track_id_set))
                log.message(ROOT_LOG_NAME, msg)
            log.message(ROOT_LOG_NAME, "Track id sets for sequence two:")
            for track_id_set in track_id_sets_two:
                msg = "\t{0}".format(", ".join(track_id_set))
                log.message(ROOT_LOG_NAME, msg)

        gap_scores_one = np.empty((len(sequence_one), len(gap_series)),
                                  dtype=np.float32)
        for n, e in enumerate(gap_series):
            gap_scores_one[:, n] = e
        gap_scores_two = np.empty((len(sequence_two), len(gap_series)),
                                  dtype=np.float32)
        for n, e in enumerate(gap_series):
            gap_scores_two[:, n] = e

        if len(track_id_sets_one) != len(track_id_sets_two):
            s = "should have an identical number of track id sets" \
                "for both sequences"
            raise ComponentError(s)

        zipped = list(zip(track_id_sets_one, track_id_sets_two))
        for n, (track_ids_one, track_ids_two) in enumerate(zipped):
            if len(track_ids_one) != len(track_ids_two):
                s = "should have an identical number of tracks to" \
                    "align in both sequences. this is a limitation" \
                    "of the alignment core and likely to go away" \
                    "in the future..."
                raise ComponentError(s)

            total_len = len(track_ids_one)+len(track_ids_two)
            score_matrix = score_matrices[n]
            if score_matrix.matrix.ndim != total_len:
                s = "the score matrix must consist of as many dimensions as" \
                    "there are tracks to be aligned ({0}), but it contains" \
                    "{1}"
                matrix = score_matrix.matrix
                s = s.format(total_len, matrix.ndim)
                raise ComponentError(s)

        input_sets = []
        for n, (track_ids_one, track_ids_two) in enumerate(zipped):
            inputs = []

            score_matrix = score_matrices[n]
            for i in range(len(track_ids_one)):
                track_id_one = track_ids_one[i]
                track_id_two = track_ids_two[i]
                track_one = sequence_one.get_track(track_id_one)
                track_two = sequence_two.get_track(track_id_two)

                track_one_aid = track_one.alphabet.aid
                if score_matrix.alphabets[i*2].aid != track_one_aid:
                    s = "track {0} for sequence one has alphabet '{1}' but " \
                        "the corresponding dimension in the score matrix " \
                        "has alphabet '{2}'"
                    s = s.format(i, track_one_aid,
                                 score_matrix.alphabets[i*2].aid)
                    raise DataError(s)

                track_two_aid = track_two.alphabet.aid
                if score_matrix.alphabets[(i*2)+1].aid != track_two_aid:
                    s = "track {0} for sequence two has alphabet '{1}' but " \
                        "the corresponding dimension in the score matrix " \
                        "has alphabet '{2}'"
                    s = s.format(i, track_two_aid,
                                 score_matrix.alphabets[(i*2)+1].aid)
                    raise DataError(s)

                for track in [track_one, track_two]:
                    if track.tid == PlainTrack.tid:
                        inputs.append(track.values)
                    elif track.tid == ProfileTrack.tid:
                        if self.environment['squash_profiles']:
                            values = np.empty((track.profile.shape[0],),
                                              dtype=np.int32)
                            for i in range(track.profile.shape[0]):
                                values[i] = track.profile[i,:].argmax()
                            inputs.append(values)
                        else:
                            inputs.append(track.profile)
                    else:
                        s = "unknown track type id for this aligner: '{0}'"
                        s = s.format(track.tid)
                        raise DataError(s)
            input_sets.append(inputs)

        raw_score_matrices = [s.matrix for s in score_matrices]

        if mode == 'local':
            swa = SmithWatermanAlgorithm(input_sets, raw_score_matrices,
                                         gap_scores_one, gap_scores_two,
                                         zero_idxs)
            swa.resolve_all()
            cell = np.unravel_index(swa.matrix.argmax(), swa.matrix.shape)
            score = swa.matrix[cell]
            paths = get_paths_legacy(swa.traceback, cell=(cell[0], cell[1]))

            o = swa.matrix
            t = swa.traceback
        elif mode == 'semiglobal_both':
            sgaa = SemiGlobalAlignmentAlgorithm(input_sets, raw_score_matrices,
                                               gap_scores_one, gap_scores_two,
                                               zero_idxs)
            sgaa.resolve_all()
            n, m = sgaa.shape
            last_row = sgaa.matrix[n-1, :]
            last_col = sgaa.matrix[:, m-1]

            if last_row.max() > last_col.max():
                cell = (n-1, last_row.argmax())
            else:
                cell = (last_col.argmax(), m-1)
            score = sgaa.matrix[cell]
            tb = sgaa.traceback
            paths = [np.array(path) for path in get_paths_legacy(tb, cell=cell)]
            paths = [extend_path_semiglobal(path, (n, m)) for path in paths]

            o = sgaa.matrix
            t = sgaa.traceback
        elif mode == 'global':
            nwa = NeedlemanWunschAlgorithm(input_sets, raw_score_matrices,
                                           gap_scores_one, gap_scores_two,
                                           zero_idxs)
            nwa.resolve_all()
            cell = (nwa.shape[0]-1, nwa.shape[1]-1)
            score = nwa.matrix[cell]
            paths = get_paths_legacy(nwa.traceback, cell=cell)

            o = nwa.matrix
            t = nwa.traceback
        else:
            s = "unknown alignment mode: '{0}'".format(mode)
            raise ComponentError(s)

        if debug > 1:
            log.message(ROOT_LOG_NAME, "Dumping DP & traceback matrices...")
            np.savetxt(log.path("dp_matrix.csv"), o, delimiter=",")
            np.savetxt(log.path("tb_matrix.csv"), t, delimiter=",")

        alignment = Alignment([sequence_one, sequence_two], paths[0])
        outputs = {'alignment': alignment, 'score': float(score)}

        if debug > 0:
            log.message(ROOT_LOG_NAME, "Alignment score: {0}".format(score))

            msg = "Done!"
            log.message(ROOT_LOG_NAME, msg)

            archive_path = log.archive()
            log.delete()

            yield LogMessage(path_to_url(archive_path))

        yield CompleteMessage(outputs=outputs)


class PairwiseAligner(Component):
    """This component provides fast implementations of several common pairwise
    alignment scenarios. Currently we implement local, semiglobal and global
    alignments between an unlimited amount of independent sets of single
    profile tracks. This calls through to the raw pairwise aligner to do
    the dynamic programming based alignment. If you want to generate match
    and/or gap scores yourself you should not use this component but instead
    call the raw aligner yourself with suitable match and gap score models.

    Inputs:
    * mode - string containing the requested alignment mode
             ('local', 'global' or 'semiglobal')
    * sequence_one - a sequence object containing the first sequence to align
    * sequence_two - a sequence object containing the second sequence to align
    * track_id_sets_one - a list containing lists of strings specifying
                          which tracks should be aligned from sequence one
    * track_id_sets_two - a list containing lists of strings specifying
                          which tracks should be aligned from sequence two
    * zero_idxs - a list of 2-tuples containing indices in the dynamic
                  programming matrix which should be fixed to zero, which is
                  used by multiple alignment techniques such as Waterman-
                  Eggert.
    * score_matrices - use these score matrices for the alignment


    Outputs:
    * alignment - an alignment object containing the resulting alignment
    * score - the score of the alignment, the score is guaranteed to be a
              float, but its scale depends on the alignment method and other
              factors

    Options:
    * gap_series - a list of floats specifying the gap penalties which should
                   be applied for a gap of a certain length (or larger, if
                   the gap length reaches the end of the list)
    * debug - integer indicating the debug level (0 is silent, 3 is noisiest)

    """
    tid = "praline.component.PairwiseAligner"
    inputs = {'mode': Port(str),
              'sequence_one': Port(Sequence.tid),
              'sequence_two': Port(Sequence.tid),
              'track_id_sets_one': Port([[str]]),
              'track_id_sets_two': Port([[str]]),
              'zero_idxs': Port([(int, int)], optional = True),
              'score_matrices': Port([ScoreMatrix.tid])}
    outputs = {'alignment': Port(Alignment.tid), 'score': Port(float)}

    options = {'gap_series': [float], 'debug': int}
    defaults = {'gap_series': [-11.0, -1.0], 'debug': 0}

    def execute(self, mode, sequence_one, sequence_two, track_id_sets_one,
                track_id_sets_two, zero_idxs, score_matrices):
        gap_series = self.environment['gap_series']
        debug = self.environment['debug']

        if debug > 0:
            log = LogBundle()
            msg = "Entering component '{0}'".format(self.tid)
            log.message(ROOT_LOG_NAME, msg)

            log.message(ROOT_LOG_NAME, "Alignment mode: '{0}'".format(mode))
            log.message(ROOT_LOG_NAME, "Gap scores: {0}".format(gap_series))

            msg = "Sequence one: '{0}', sequence two: '{1}'"
            msg = msg.format(sequence_one.name, sequence_two.name)
            log.message(ROOT_LOG_NAME, msg)

            log.message(ROOT_LOG_NAME, "Track id sets for sequence one:")
            for track_id_set in track_id_sets_one:
                msg = "\t{0}".format(", ".join(track_id_set))
                log.message(ROOT_LOG_NAME, msg)
            log.message(ROOT_LOG_NAME, "Track id sets for sequence two:")
            for track_id_set in track_id_sets_two:
                msg = "\t{0}".format(", ".join(track_id_set))
                log.message(ROOT_LOG_NAME, msg)

        if len(track_id_sets_one) != len(track_id_sets_two):
            s = "should have an identical number of track id sets" \
                "for both sequences"
            raise ComponentError(s)

        input_sets = []
        zipped = list(zip(track_id_sets_one, track_id_sets_two))
        for n, (track_ids_one, track_ids_two) in enumerate(zipped):
            score_matrix = score_matrices[n]

            if len(track_ids_one) != 1 or len(track_ids_two) != 1:
                s = "the fast aligner only supports single-track" \
                    " alignments at the moment"
                raise ComponentError(s)

            total_track_ids_len = len(track_ids_one)+len(track_ids_two)
            if score_matrix.matrix.ndim != total_track_ids_len:
                s = "the score matrix must consist of as many dimensions as" \
                    "there are tracks to be aligned ({0}), but it contains " \
                    "{1}"
                matrix = score_matrix.matrix
                s = s.format(total_track_ids_len, matrix.ndim)
                raise ComponentError(s)

            inputs = []
            for i in range(len(track_ids_one)):
                track_id_one = track_ids_one[i]
                track_id_two = track_ids_two[i]
                track_one = sequence_one.get_track(track_id_one)
                track_two = sequence_two.get_track(track_id_two)

                track_one_aid = track_one.alphabet.aid
                if score_matrix.alphabets[i * 2].aid != track_one_aid:
                    s = "track {0} for sequence one has alphabet '{1}' but " \
                        "the corresponding dimension in the score matrix " \
                        "has alphabet '{2}'"
                    s = s.format(i, track_one_aid,
                                 score_matrix.alphabets[i*2].aid)
                    raise DataError(s)

                track_two_aid = track_two.alphabet.aid
                if score_matrix.alphabets[(i * 2) + 1].aid != track_two_aid:
                    s = "track {0} for sequence two has alphabet '{1}' but " \
                        "the corresponding dimension in the score matrix " \
                        "has alphabet '{2}'"
                    s = s.format(i, track_two_aid,
                                 score_matrix.alphabets[(i*2)+1].aid)
                    raise DataError(s)

                for track in [track_one, track_two]:
                    if track.tid == PlainTrack.tid:
                        # Upconvert the sequence track to a profile.
                        profile = np.zeros((len(track), track.alphabet.size),
                                           dtype=np.float32)
                        for j in range(len(track)):
                            profile[j, track.values[j]] = 1.0
                        inputs.append(profile)
                    elif track.tid == ProfileTrack.tid:
                        inputs.append(track.profile.astype(np.float32))
                    else:
                        s = "unknown track type id for this aligner: '{0}'"
                        s = s.format(track.tid)
                        raise DataError(s)
            input_sets.append(inputs)

        # Our optimized implementation only supports linear and affine gap
        # penalties. Raise an exception if we've gotten a longer gap penalty
        # series than [open, extend].
        if len(gap_series) == 1:
            gap_series = [gap_series[0], gap_series[0]]
        elif len(gap_series) == 2:
            pass
        else:
            s = "the fast aligner only supports linear and affine gap" \
                " penalties at the moment"
            raise ComponentError(s)

        # Setup the arrays which we will pass to the C scoring component.
        # We pre-allocate everything the C code needs in terms of memory here
        # since dealing with python memory management from C is kind of a
        # pain.
        #
        # NOTE: the C side of things performs no type or bounds checking
        # at all. If you pass a smaller array than it is expecting or an
        # array of a different data type, it'll happily read outside of the
        # buffer and crash PRALINE. So you need to be very careful here.
        i1 = [input_set[0] for input_set in input_sets]
        i2 = [input_set[1] for input_set in input_sets]
        i1nz = [build_nonzero_matrix(item) for item in i1]
        i2nz = [build_nonzero_matrix(item) for item in i2]
        shape = (i1[0].shape[0]+1, i2[0].shape[0]+1)
        s = [sm.matrix.astype(np.float32) for sm in score_matrices]
        m = np.zeros((i1[0].shape[0], i2[0].shape[0]), dtype=np.float32)

        # Build the match score model.
        cext_build_scores(i1, i2, i1nz, i2nz, s, m)

        # Build the gap scores from the gap score array.
        g1 = np.empty((i1[0].shape[0], len(gap_series)), dtype=np.float32)
        for n, e in enumerate(gap_series):
            g1[:, n] = e
        g2 = np.empty((i2[0].shape[0], len(gap_series)), dtype=np.float32)
        for n, e in enumerate(gap_series):
            g2[:, n] = e

        match_score_model = MatchScoreModel(sequence_one, sequence_two, m)
        gap_score_model_one = GapScoreModel(sequence_one, g1)
        gap_score_model_two = GapScoreModel(sequence_two, g2)

        # Call on the raw pairwise aligner to do the pairwise alignment
        # with the score models we've just built.
        execution = Execution(self.manager, self.tag)
        task = execution.add_task(RawPairwiseAligner)
        task.environment(self.environment)
        task.inputs(mode=mode, sequence_one=sequence_one,
                    sequence_two=sequence_two,
                    match_score_model=match_score_model,
                    gap_score_model_one=gap_score_model_one,
                    gap_score_model_two=gap_score_model_two,
                    zero_idxs=zero_idxs)

        for msg in execution.run():
            yield msg
        outputs = execution.outputs[0]

        if debug > 0:
            score = outputs["score"]
            log.message(ROOT_LOG_NAME, "Alignment score: {0}".format(score))

            msg = "Done!"
            log.message(ROOT_LOG_NAME, msg)

            archive_path = log.archive()
            log.delete()

            yield LogMessage(path_to_url(archive_path))

        yield CompleteMessage(outputs=outputs)


class RawPairwiseAligner(Component):
    """This is the raw pairwise alignment component which takes a match
    score model and two gap score models and runs a dynamic programming
    pairwise alignment algorithm to generate an alignment. In the past the
    pairwise aligner was a single component that both generated the score
    models and performed the alignment. To increase the flexibility of how
    we perform the scoring the score model generation is now done by another
    component beforehand.

    Inputs:
    * mode - string containing the requested alignment mode
             ('local', 'global' or 'semiglobal')
    * sequence_one - a sequence object containing the first sequence to align
    * sequence_two - a sequence object containing the second sequence to align
    * match_score_model - match score model object containing pairwise scores
    * gap_score_model_one - gap score model object containing gap scores for
                            gaps in sequence two
    * gap_score_model_two - gap score model object containing gap scores for
                            gaps in sequence one
    * zero_idxs - a list of 2-tuples containing indices in the dynamic
                  programming matrix which should be fixed to zero, which is
                  used by multiple alignment techniques such as Waterman-
                  Eggert.


    Outputs:
    * alignment - an alignment object containing the resulting alignment
    * score - the score of the alignment, the score is guaranteed to be a
              float, but its scale depends on the alignment method and other
              factors

    Options:
    * debug - integer indicating the debug level (0 is silent, 3 is noisiest)

    """
    tid = "praline.component.RawPairwiseAligner"
    inputs = {'mode': Port(str),
              'sequence_one': Port(Sequence.tid),
              'sequence_two': Port(Sequence.tid),
              'match_score_model': Port(MatchScoreModel.tid),
              'gap_score_model_one': Port(GapScoreModel.tid),
              'gap_score_model_two': Port(GapScoreModel.tid),
              'zero_idxs': Port([(int, int)], optional = True)}
    outputs = {'alignment': Port(Alignment.tid), 'score': Port(float)}

    options = {'debug': int, 'accelerate': bool}
    defaults = {'debug': 0, 'accelerate': True}

    def execute(self, mode, sequence_one, sequence_two, match_score_model,
                gap_score_model_one, gap_score_model_two, zero_idxs):
        debug = self.environment['debug']
        accelerate = self.environment['accelerate']

        if debug > 0:
            log = LogBundle()
            msg = "Entering component '{0}'".format(self.tid)
            log.message(ROOT_LOG_NAME, msg)

            log.message(ROOT_LOG_NAME, "Alignment mode: '{0}'".format(mode))

            msg = "Sequence one: '{0}', sequence two: '{1}'"
            msg = msg.format(sequence_one.name, sequence_two.name)
            log.message(ROOT_LOG_NAME, msg)

        # We only implement a limited set of alignment modes so far.
        if mode == "local":
            alignment_type = "local"
        elif mode == "semiglobal_both":
            alignment_type = "semiglobal_both"
        elif mode == "semiglobal_one":
            alignment_type = "semiglobal_one"
        elif mode == "semiglobal_two":
            alignment_type = "semiglobal_two"
        elif mode == "global":
            alignment_type = "global"
        else:
            s = "unknown alignment mode: '{0}'".format(mode)
            raise ComponentError(s)

        try:
            if accelerate:
                align_fun = _CEXT_ALIGN_FUNCTIONS[alignment_type]
            else:
                align_fun = _ALIGN_FUNCTIONS[alignment_type]
        except KeyError:
            s = "alignment not implemented for {0}"
            s = s.format(alignment_type)
            raise ComponentError(s)

        # Retrieve score arrays from model objects.
        m = match_score_model.scores
        g1 = gap_score_model_one.scores
        g2 = gap_score_model_two.scores

        # Setup the arrays which we will pass to the C alignment component.
        # We pre-allocate everything the C code needs in terms of memory here
        # since dealing with python memory management from C is kind of a
        # pain.
        #
        # NOTE: the C side of things performs no type or bounds checking
        # at all. If you pass a smaller array than it is expecting or an
        # array of a different data type, it'll happily read outside of the
        # buffer and crash PRALINE. So you need to be very careful here.
        shape = (m.shape[0] + 1, m.shape[1] + 1)
        o = np.zeros((shape[0], shape[1], 3), dtype=np.float32)
        t = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        z = np.zeros(shape, dtype=np.uint8)
        if zero_idxs is not None:
            for idx in zero_idxs:
                z[idx] = 1

        # Initialize the matrices for the dynamic programming function, as well
        # as the traceback matrices.
        o[:, 0, :] = MINUS_INFINITY
        o[0, :, :] = MINUS_INFINITY
        o[0, 0, 0] = 0

        if mode in {"semiglobal_both", "semiglobal_one"}:
            o[:, 0, 1] = 0
        else:
            o[0, 0, 1] = g1[0, 0] - g1[0, 1]
            o[1:, 0, 1] = (np.arange(o.shape[0]-1) * g1[:, 1]) + g1[0, 0]

            t[1:, 0, 1] = TRACEBACK_INSERT_UP_EXTEND

        if mode in {"semiglobal_both", "semiglobal_two"}:
            o[0, :, 2] = 0
        else:
            o[0, 0, 2] = g2[0, 0] - g2[0, 1]
            o[0, 1:, 2] = (np.arange(o.shape[1]-1) * g2[:, 1]) + g2[0, 0]

            t[0, 1:, 2] = TRACEBACK_INSERT_LEFT_EXTEND

        # Initialize the traceback matrices.
        align_fun(m, g1, g2, o, t, z)

        if debug > 1:
            log.message(ROOT_LOG_NAME, "Dumping DP & traceback matrices...")

            np.savetxt(log.path("dp_0_matrix.csv"), o[:, :, 0], delimiter=",")
            np.savetxt(log.path("dp_1_matrix.csv"), o[:, :, 1], delimiter=",")
            np.savetxt(log.path("dp_2_matrix.csv"), o[:, :, 2], delimiter=",")

            np.savetxt(log.path("tb_0_matrix.csv"), t[:, :, 0], delimiter=",")
            np.savetxt(log.path("tb_1_matrix.csv"), t[:, :, 1], delimiter=",")
            np.savetxt(log.path("tb_2_matrix.csv"), t[:, :, 2], delimiter=",")

        if mode == "local":
            cell = np.unravel_index(o.argmax(), o.shape)
            score = o[cell]
            paths = get_paths(t, cell=cell)
        elif mode in {"semiglobal_both", "semiglobal_one", "semiglobal_two"}:
            n, m, _ = o.shape
            last_row = o[n-1, :, :]
            last_col = o[:, m-1, :]
            last_row_max = last_row.max()
            last_col_max = last_col.max()
            trace_from_row = mode in {"semiglobal_both", "semiglobal_two"}

            if last_row_max > last_col_max and trace_from_row:
                for i, j in itertools.product(range(m-1, -1, -1), range(3)):
                    if last_row[i, j] == last_row_max:
                        cell = (n-1, i, j)
                        break
            else:
                for i, j in itertools.product(range(n-1, -1, -1), range(3)):
                    if last_col[i, j] == last_col_max:
                        cell = (i, m-1, j)
                        break

            score = o[cell]
            paths = [np.array(path) for path in get_paths(t, cell=cell)]
            paths = [extend_path_semiglobal(path, (n, m)) for path in paths]
        elif mode == "global":
            y, x = o.shape[0]-1, o.shape[1]-1
            cell = (y, x, np.argmax(o[y, x, :]))
            score = o[cell]
            paths = get_paths(t, cell=cell)

        alignment = Alignment([sequence_one, sequence_two], paths[0])
        outputs = {'alignment': alignment, 'score': float(score)}

        if debug > 0:
            log.message(ROOT_LOG_NAME, "Alignment score: {0}".format(score))

            msg = "Done!"
            log.message(ROOT_LOG_NAME, msg)

            archive_path = log.archive()
            log.delete()

            yield LogMessage(path_to_url(archive_path))

        yield CompleteMessage(outputs=outputs)

def build_nonzero_matrix(i):
    rows, cols = i.shape
    nonzero_mat = np.empty((rows, cols), dtype=np.intp)
    nonzero_mat.fill(-1)

    for n in range(rows):
        row = i[n, :].nonzero()[0]
        nonzero_mat[n, :row.shape[0]] = row

    return nonzero_mat
