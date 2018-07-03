"""Multiple sequence alignment components.

.. moduleauthor:: Maurits Dijkstra <mauritsdijkstra@gmail.com>

"""
from __future__ import division, absolute_import, print_function

import numpy as np
import six
from six.moves import range

from praline import write_alignment_clustal
from praline.core import *
from praline.container import Sequence, ScoreMatrix, TRACK_ID_INPUT
from praline.container import ProfileTrack, PlainTrack
from praline.container import TRACK_ID_PREPROFILE, TRACK_ID_PROFILE
from praline.container import SequenceTree, Alignment
from praline.component import PairwiseAligner
from praline.util import auto_align_mode


MINUS_INFINITY = -(2 ** 32)

class TreeMultipleSequenceAligner(Component):
    """The TreeMultipleSequenceAligner performs a multiple sequence alignment
    (MSA) using a guide tree to determine the order in which the alignments
    should be merged.

    Inputs:
    * sequences - the sequences which should be aligned into a multiple
                  sequence alignment
    * guide_tree - the guide tree which should be used to determine the
                   order of the merges between the subalignments
    * track_id_sets - a list of lists of track ids specifying the tracks in
                      the master and slave sequences to use for the alignment
    * score_matrices - use these score matrices for the alignments

    Outputs:
    * alignment - the resulting multiple sequence alignment

    Options:
    * gap_series - a list of floats specifying the gap penalties which should
                   be applied for a gap of a certain length (or larger, if
                   the gap length reaches the end of the list)
    * score_matrix - use this score matrix for the alignment
    * aligner - a type id specifying which component should be used to
                perform the pairwise alignments
    * aligner_env - an environment object which can be used to pass options
                    to the pairwise alignment component
    * merge_mode - string specifying which alignment mode to use when
                   merging sequence blocks
    * debug - integer indicating the debug level (0 is silent, 3 is noisiest)
    * log_track_ids - log tracks with these ids if debugging is enabled

    """
    tid = "praline.component.TreeMultipleSequenceAligner"

    inputs = {'sequences': Port([Sequence.tid]),
              'guide_tree': Port(SequenceTree.tid),
              'track_id_sets': Port([[str]]),
              'score_matrices': Port([ScoreMatrix.tid])}
    outputs = {'alignment': Port(Alignment.tid)}

    options = {'gap_series': [float],'aligner': str,
               'aligner_env': Environment.tid, 'merge_mode': str,
               'debug': int, 'log_track_ids': [str]}
    defaults = {'gap_series': [-11.0, -1.0], 'aligner': PairwiseAligner.tid,
                'aligner_env': Environment({}), 'merge_mode': 'semiglobal',
                'debug': 0, 'log_track_ids': [TRACK_ID_INPUT]}

    def _initial_clusters(self, sequences, track_id_sets):
        clusters = {}
        for i, sequence in enumerate(sequences):
            name = "Cluster #{0}".format(i)
            new_sequence = Sequence(name, [])
            copied_tracks = set()
            for track_id_set in track_id_sets:
                for track_id in track_id_set:
                    track = sequence.get_track(track_id)
                    if track.tid == PlainTrack.tid:
                        alphabet = track.alphabet
                        counts = np.zeros((len(track), alphabet.size),
                                          dtype=np.int32)
                        for j in range(len(track)):
                            counts[j, track.values[j]] = 1
                        new_track = ProfileTrack(counts, track.alphabet)
                    elif track.tid == ProfileTrack.tid:
                        counts = np.array(track.counts, dtype=np.int32)
                        new_track = ProfileTrack(counts, track.alphabet)

                    if track_id not in copied_tracks:
                        new_sequence.add_track(track_id, new_track)
                        copied_tracks.add(track_id)

            clusters[i] = new_sequence

        return clusters

    def execute(self, sequences, guide_tree, track_id_sets, score_matrices):
        debug = self.environment['debug']
        merge_mode = self.environment['merge_mode']
        log_track_ids = self.environment['log_track_ids']

        if debug > 0:
            log = LogBundle()
            msg = "Entering component '{0}'".format(self.tid)
            log.message(ROOT_LOG_NAME, msg)

        if merge_mode not in {"global", "semiglobal", "semiglobal_auto"}:
            msg = "unknown merge mode '{0}'".format(merge_mode)
            raise ComponentError(msg)

        index = self.manager.index

        seqs = sequences
        tree = guide_tree

        alignments = {i: Alignment([seq], np.arange(len(seq)+1).reshape(
            len(seq)+1, 1)) for i, seq in enumerate(seqs)}
        clusters = self._initial_clusters(seqs, track_id_sets)

        cur_step = 0
        total_steps = len(tree.merge_orders)
        for i, j in tree.merge_orders:
            alignment_one = alignments[i]
            alignment_two = alignments[j]
            cluster_one = clusters[i]
            cluster_two = clusters[j]

            if debug > 0:
                msg = "Step {0} of {1}".format(cur_step + 1, total_steps)
                log.message(ROOT_LOG_NAME, msg)

                msg = "Merging cluster {0} into {1}".format(j, i)
                log.message(ROOT_LOG_NAME, msg)
                header_fmt = "Cluster {0}:"
                name_fmt = "\t{0}"
                log.message(ROOT_LOG_NAME, header_fmt.format(j))
                for item in alignment_two.items:
                    log.message(ROOT_LOG_NAME, name_fmt.format(item.name))
                log.message(ROOT_LOG_NAME, header_fmt.format(i))
                for item in alignment_one.items:
                    log.message(ROOT_LOG_NAME, name_fmt.format(item.name))

            if debug > 1:
                s = 'step{0}_input_{1}_track_{2}.aln'
                for track_id in log_track_ids:
                    input_one_filename = s.format(cur_step + 1, 1, track_id)
                    input_two_filename = s.format(cur_step + 1, 2, track_id)
                    write_alignment_clustal(log.path(input_one_filename),
                                            alignment_one, track_id, None)
                    write_alignment_clustal(log.path(input_two_filename),
                                            alignment_two, track_id, None)

                s = 'step{0}_input_{1}_track_{2}.csv'
                for track_id_set in track_id_sets:
                    for track_id in track_id_set:
                        track_one = cluster_one.get_track(track_id)
                        track_two = cluster_two.get_track(track_id)
                        filename_one = s.format(cur_step + 1, 1, track_id)
                        filename_two = s.format(cur_step + 1, 2, track_id)
                        np.savetxt(log.path(filename_one), track_one.counts,
                                   delimiter=",")
                        np.savetxt(log.path(filename_two), track_two.counts,
                                   delimiter=",")

            if merge_mode == "semiglobal":
                align_mode = "semiglobal_both"
            elif merge_mode == "global":
                align_mode = "global"
            elif merge_mode == "semiglobal_auto":
                align_mode = auto_align_mode(cluster_one, cluster_two)

            sub_env = self.environment['aligner_env']
            sub_component = index.resolve(self.environment['aligner'])
            root_env = self.environment
            execution = Execution(self.manager, self.tag)
            task = execution.add_task(sub_component)
            task.environment(root_env, sub_env)
            task.inputs(mode=align_mode, sequence_one=cluster_one,
                        sequence_two=cluster_two,
                        track_id_sets_one=track_id_sets,
                        track_id_sets_two=track_id_sets,
                        score_matrices=score_matrices)

            if debug > 0:
                msg = "Starting task '{0}' for pairwise alignment"
                msg = msg.format(task.tag)
                log.message(ROOT_LOG_NAME, msg)

            for msg in execution.run():
                yield msg

            outputs = execution.outputs[0]
            if debug > 0:
                msg = "Alignment score = {0}".format(outputs["score"])
                log.message(ROOT_LOG_NAME, msg)

            path_profile = np.array(outputs['alignment'].path)

            replace_tracks = []
            for track_id_set in track_id_sets:
                for track_id in track_id_set:
                    track_one = cluster_one.get_track(track_id)
                    track_two = cluster_two.get_track(track_id)

                    track_new = track_one.merge(track_two, path_profile)

                    replace_tracks.append((track_id, track_new))

                    cluster_one.del_track(track_id)

            for track_id, track_new in replace_tracks:
                cluster_one.add_track(track_id, track_new)

            alignments[i] = alignment_one.merge(alignment_two, path_profile)

            if debug > 1:
                for track_id in log_track_ids:
                    s = 'step{0}_output_track_{1}.aln'
                    output_filename = s.format(cur_step + 1, track_id)
                    write_alignment_clustal(log.path(output_filename),
                                            alignments[i], track_id, None)

                s = 'step{0}_output_track_{1}.csv'
                for track_id_set in track_id_sets:
                    for track_id in track_id_set:
                        track = cluster_one.get_track(track_id)
                        filename = s.format(cur_step + 1, track_id)
                        np.savetxt(log.path(filename), track.counts,
                                   delimiter=",")

            del clusters[j]
            del alignments[j]

            cur_step += 1
            yield ProgressMessage(progress=cur_step / total_steps)

        if debug > 0:
            msg = "Done!"
            log.message(ROOT_LOG_NAME, msg)

            archive_path = log.archive()
            log.delete()

            yield LogMessage(path_to_url(archive_path))

        yield CompleteMessage(outputs={'alignment': list(alignments.values())[0]})

class AdHocMultipleSequenceAligner(Component):
    """The AdHocMultipleSequenceAligner is a component that performs a
    multiple sequence alignment by generating a guide tree 'just in time'.
    Every step it aligns each profile resulting from each subalignments
    against every other profile resulting from every other subalignment. The
    combination with the highest score are then joined. This process is
    repeated until one alignment remains.

    In terms of computational time this is moderately more expensive than a
    tree-guided alignment, but potentially results in a better merge order
    and thus a better alignment. Note that this implementation is optimized
    in the sense that it caches alignments between pairs that have not
    changed between rounds.

    Inputs:
    * sequences - the sequences which should be aligned into a multiple
                  sequence alignment
    * track_id_sets - a list of lists of track ids specifying the tracks in
                      the master and slave sequences to use for the alignment
    * score_matrices - use these score matrices for the alignments

    Outputs:
    * alignment - the resulting multiple sequence alignment

    Options:
    * gap_series - a list of floats specifying the gap penalties which should
                   be applied for a gap of a certain length (or larger, if
                   the gap length reaches the end of the list)
    * score_matrix - use this score matrix for the alignment
    * aligner - a type id specifying which component should be used to
                perform the pairwise alignments
    * aligner_env - an environment object which can be used to pass options
                    to the pairwise alignment component
    * merge_mode - string specifying which alignment mode to use when
                   merging sequence blocks
    * debug - integer indicating the debug level (0 is silent, 3 is noisiest)
    * log_track_ids - log tracks with these ids if debugging is enabled
    """
    tid = "praline.component.AdHocMultipleSequenceAligner"

    inputs = {'sequences': Port([Sequence.tid]),
              'track_id_sets': Port([[str]]),
              'score_matrices': Port([ScoreMatrix.tid])}
    outputs = {'alignment': Port(Alignment.tid)}

    options = {'gap_series': [float], 'aligner': str,
               'aligner_env': Environment.tid, 'merge_mode': str,
               'dist_mode': str, 'debug': int, 'log_track_ids': [str]}
    defaults = {'gap_series': [-11.0, -1.0], 'aligner': PairwiseAligner.tid,
                'aligner_env': Environment({}), 'merge_mode': 'semiglobal',
                'dist_mode': 'global', 'debug': 0,
                'log_track_ids': [TRACK_ID_INPUT]}


    def __init__(self, *args, **kwargs):
        super(AdHocMultipleSequenceAligner, self).__init__(*args, **kwargs)

        self._cached_s = None
        self._dirty_idx = None
        self._cached_index_map = None
        self._cached_r_index_map = None


    def _initial_clusters(self, sequences, track_id_sets):
        clusters = {}
        for i, sequence in enumerate(sequences):
            name = "Cluster #{0}".format(i)
            new_sequence = Sequence(name, [])
            copied_tracks = set()
            for track_id_set in track_id_sets:
                for track_id in track_id_set:
                    track = sequence.get_track(track_id)
                    if track.tid == PlainTrack.tid:
                        alphabet = track.alphabet
                        counts = np.zeros((len(track), alphabet.size),
                                          dtype=np.int32)
                        for j in range(len(track)):
                            counts[j, track.values[j]] = 1
                        new_track = ProfileTrack(counts, track.alphabet)
                    elif track.tid == ProfileTrack.tid:
                        counts = np.array(track.counts, dtype=np.int32)
                        new_track = ProfileTrack(counts, track.alphabet)

                    if track_id not in copied_tracks:
                        new_sequence.add_track(track_id, new_track)
                        copied_tracks.add(track_id)

            clusters[i] = new_sequence

        return clusters


    def execute(self, sequences, track_id_sets, score_matrices):
        debug = self.environment['debug']
        merge_mode = self.environment['merge_mode']
        dist_mode = self.environment['dist_mode']
        log_track_ids = self.environment['log_track_ids']

        if debug > 0:
            log = LogBundle()
            msg = "Entering component '{0}'".format(self.tid)
            log.message(ROOT_LOG_NAME, msg)

            msg = "Distance mode = '{0}', merge mode = '{1}'"
            msg = msg.format(dist_mode, merge_mode)
            log.message(ROOT_LOG_NAME, msg)

        if merge_mode not in {'global', 'semiglobal', 'semiglobal_auto'}:
            msg = "unknown merge mode '{0}'".format(merge_mode)
            raise ComponentError(msg)
        if dist_mode not in {'global', 'semiglobal', 'semiglobal_auto'}:
            msg = "unknown distance mode '{0}'".format(dist_mode)
            raise ComponentError(msg)

        index = self.manager.index

        seqs = sequences

        alignments = {i: Alignment([seq], np.arange(len(seq)+1).reshape(
                      len(seq)+1, 1)) for i, seq in enumerate(seqs)}
        clusters = self._initial_clusters(seqs, track_id_sets)

        cur_step = 0
        total_steps = len(clusters) - 1
        while len(clusters) > 1:
            if debug > 0:
                msg = "Step {0} of {1}".format(cur_step + 1, total_steps)
                log.message(ROOT_LOG_NAME, msg)

            for msg in self._merge_indices(clusters, dist_mode, track_id_sets,
                                           score_matrices):
                yield msg
            i, j = self._merge_i, self._merge_j

            alignment_one = alignments[i]
            alignment_two = alignments[j]

            cluster_one = clusters[i]
            cluster_two = clusters[j]

            if debug > 0:
                msg = "Merging cluster {0} into {1}".format(j, i)
                log.message(ROOT_LOG_NAME, msg)
                header_fmt = "Cluster {0}:"
                name_fmt = "\t{0}"
                log.message(ROOT_LOG_NAME, header_fmt.format(j))
                for item in alignment_two.items:
                    log.message(ROOT_LOG_NAME, name_fmt.format(item.name))
                log.message(ROOT_LOG_NAME, header_fmt.format(i))
                for item in alignment_one.items:
                    log.message(ROOT_LOG_NAME, name_fmt.format(item.name))

            if debug > 1:
                s = 'step{0}_input_{1}_track_{2}.aln'
                for track_id in log_track_ids:
                    input_one_filename = s.format(cur_step + 1, 1, track_id)
                    input_two_filename = s.format(cur_step + 1, 2, track_id)
                    write_alignment_clustal(log.path(input_one_filename),
                                            alignment_one, track_id, None)
                    write_alignment_clustal(log.path(input_two_filename),
                                            alignment_two, track_id, None)


            if merge_mode == "semiglobal":
                align_mode = "semiglobal_both"
            elif merge_mode == "global":
                align_mode = "global"
            elif merge_mode == "semiglobal_auto":
                align_mode = auto_align_mode(cluster_one, cluster_two)

            sub_env = self.environment['aligner_env']
            sub_component = index.resolve(self.environment['aligner'])
            root_env = self.environment
            execution = Execution(self.manager, self.tag)
            task = execution.add_task(sub_component)
            task.environment(root_env, sub_env)
            task.inputs(mode=align_mode, sequence_one=cluster_one,
                        sequence_two=cluster_two,
                        track_id_sets_one=track_id_sets,
                        track_id_sets_two=track_id_sets,
                        score_matrices=score_matrices)

            if debug > 0:
                msg = "Starting task '{0}' for pairwise alignment"
                msg = msg.format(task.tag)
                log.message(ROOT_LOG_NAME, msg)

            for msg in execution.run():
                yield msg
            outputs = execution.outputs[0]

            if debug > 0:
                msg = "Alignment score = {0}".format(outputs["score"])
                log.message(ROOT_LOG_NAME, msg)

            path_profile = np.array(outputs['alignment'].path)

            replace_tracks = []
            for track_id_set in track_id_sets:
                for track_id in track_id_set:
                    track_one = cluster_one.get_track(track_id)
                    track_two = cluster_two.get_track(track_id)

                    track_new = track_one.merge(track_two, path_profile)

                    replace_tracks.append((track_id, track_new))

                    cluster_one.del_track(track_id)

            for track_id, track_new in replace_tracks:
                cluster_one.add_track(track_id, track_new)

            alignments[i] = alignment_one.merge(alignment_two, path_profile)

            if debug > 1:
                for track_id in log_track_ids:
                    s = 'step{0}_output_track_{1}.aln'
                    output_filename = s.format(cur_step + 1, track_id)
                    write_alignment_clustal(log.path(output_filename),
                                            alignments[i], track_id, None)

            del clusters[j]
            del alignments[j]

            cur_step += 1
            yield ProgressMessage(progress=cur_step / total_steps)

        if debug > 0:
            msg = "Done!"
            log.message(ROOT_LOG_NAME, msg)

            archive_path = log.archive()
            log.delete()

            yield LogMessage(path_to_url(archive_path))

        yield CompleteMessage(outputs={'alignment': list(alignments.values())[0]})

    def _merge_indices(self, clusters, dist_mode, track_id_sets,
                       score_matrices):
        index = self.manager.index

        clustlist = list(clusters.values())

        cluster_map = {e: i for i, e in six.iteritems(clusters)}
        index_map = {i: cluster_map[e] for i, e in enumerate(clustlist)}
        r_index_map = {e: i for i, e in six.iteritems(index_map)}

        s = np.empty((len(clustlist), len(clustlist)), dtype=np.float32)
        s.fill(MINUS_INFINITY)

        s_idxs = []
        queued_idxs = set()
        execution = Execution(self.manager, self.tag)
        for i, cluster_one in enumerate(clustlist):
            for j, cluster_two in enumerate(clustlist):
                if i != j:
                    if self._cached_index_map:
                        dirty_idx = self._cached_index_map[self._dirty_idx]
                    else:
                        dirty_idx = None

                    dirty_idxs = {index_map[i], index_map[j]}
                    if dirty_idx is not None and dirty_idx not in dirty_idxs:
                        cached_i = self._cached_r_index_map[index_map[i]]
                        cached_j = self._cached_r_index_map[index_map[j]]
                        s[i, j] = self._cached_s[cached_i, cached_j]
                        s[j, i] = self._cached_s[cached_i, cached_j]

                        queued_idxs.add((i, j))
                    elif (i, j) not in queued_idxs and (j, i) not in queued_idxs:
                        if dist_mode == "semiglobal":
                            align_mode = "semiglobal_both"
                        elif dist_mode == "global":
                            align_mode = "global"
                        elif dist_mode == "semiglobal_auto":
                            align_mode = auto_align_mode(cluster_one, cluster_two)

                        sub_env = self.environment['aligner_env']
                        sub_component = index.resolve(self.environment['aligner'])
                        root_env = self.environment
                        task = execution.add_task(sub_component)
                        task.environment(root_env, sub_env)
                        task.inputs(mode=align_mode, sequence_one=cluster_one,
                                    sequence_two=cluster_two,
                                    track_id_sets_one=track_id_sets,
                                    track_id_sets_two=track_id_sets,
                                    score_matrices=score_matrices)

                        s_idxs.append((i, j))
                        queued_idxs.add((i, j))
                else:
                    s[i, j] = MINUS_INFINITY

        for msg in execution.run():
            yield msg

        for n, outputs in enumerate(execution.outputs):
            i, j = s_idxs[n]
            s[i, j] = outputs['score']
            s[j, i] = outputs['score']

        i, j = np.unravel_index(s.argmax(), s.shape)
        self._cached_s = s
        self._dirty_idx = i
        self._cached_index_map = index_map
        self._cached_r_index_map = r_index_map

        self._merge_i, self._merge_j = index_map[i], index_map[j]
