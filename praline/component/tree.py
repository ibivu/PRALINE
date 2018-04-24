"""Components related to the generation of the guide tree.

.. moduleauthor:: Maurits Dijkstra <mauritsdijkstra@gmail.com>

"""
from __future__ import division

import numpy as np

from praline.core import *
from praline.util import HierarchicalClusteringAlgorithm, auto_align_mode
from praline.container import Sequence, SequenceTree, ScoreMatrix
from praline.container import SCORE_MATRIX_DEFAULT, TRACK_ID_PREPROFILE
from praline.component import PairwiseAligner

MINUS_INFINITY = -(2 ** 32)

class GuideTreeBuilder(Component):
    """The GuideTreeBuilder is a component that constructs the guide tree
    which is used by the multiple sequence alignment component to determine
    the order in which the subalignments should be merged. To generate the
    tree the following steps are performed.

    First pairwise alignments are performed between every possible pair in
    the set of input sequences. The scores of the resulting set of alignments
    are then used to construct a distance matrix. The distance matrix is
    then fed to a hierarchical clustering algorithm which yields the tree.

    Inputs:
    * sequences - the sequences to use to construct the tree
    * track_id_sets - a list of lists of track ids specifying the tracks in
                      the master and slave sequences to use for the alignment
    * score_matrices - use these score matrices for the alignments

    Outputs:
    * guide_tree - the tree generated by clustering the sequences

    Options:
    * gap_series - a list of floats specifying the gap penalties which should
                   be applied for a gap of a certain length (or larger, if
                   the gap length reaches the end of the list)
    * aligner - a type id specifying which component should be used to
                perform the pairwise alignments
    * aligner_env - an environment object which can be used to pass options
                    to the pairwise alignment component
    * linkage_method - which linkage method should be used for the hierarchical
                       clustering
    * dist_mode - string specifying the alignment mode to use when calculating
                  distances
    * debug - integer indicating the debug level (0 is silent, 3 is noisiest)

    """
    tid = "praline.component.GuideTreeBuilder"

    inputs = {'sequences': Port([Sequence.tid]),
              'track_id_sets': Port([[str]]),
              'score_matrices': Port([ScoreMatrix.tid])}
    outputs = {'guide_tree': Port(SequenceTree.tid)}

    options = {'gap_series': [float], 'aligner': str,
               'aligner_env': Environment.tid,
               'linkage_method': str, 'squash_profiles': bool,
               'dist_mode': str, 'debug': int}
    defaults = {'gap_series': [-11.0, -1.0],
                'aligner': PairwiseAligner.tid, 'aligner_env': Environment({}),
                'linkage_method': 'average', 'squash_profiles': False,
                'dist_mode': 'global', 'debug': 0}

    def execute(self, sequences, track_id_sets, score_matrices):
        index = self.manager.index

        seqs = sequences

        gap_series = np.array(self.environment['gap_series'], dtype=np.float32)
        aligner = self.environment['aligner']
        linkage_method = self.environment['linkage_method']
        dist_mode = self.environment['dist_mode']
        debug = self.environment['debug']

        if debug > 0:
            log = LogBundle()
            msg = "Entering component '{0}'".format(self.tid)
            log.message(ROOT_LOG_NAME, msg)

            for n, seq in enumerate(seqs):
                msg = "Sequence #{0}: '{1}'".format(n, seq.name)
                log.message(ROOT_LOG_NAME, msg)

            log.message(ROOT_LOG_NAME, "Generating distance matrix...")

        if not linkage_method in {'single', 'complete', 'average'}:
            s = "unknown linkage method '{0}'".format(linkage_method)
            raise ComponentError(s)

        if not dist_mode in {'semiglobal', 'global', 'semiglobal_auto'}:
            s = "unknown alignment mode '{0}'".format(dist_mode)
            raise ComponentError(s)

        d = np.empty((len(seqs), len(seqs)), dtype=np.float32)
        d.fill(MINUS_INFINITY)

        d_idxs = []
        queued_idxs = set()
        execution = Execution(self.manager, self.tag)
        for i, seq_one in enumerate(seqs):
            for j, seq_two in enumerate(seqs):
                if i != j:
                    if (i, j) not in queued_idxs and (j, i) not in queued_idxs:
                        if dist_mode == "semiglobal":
                            align_mode = "semiglobal_both"
                        elif dist_mode == "global":
                            align_mode = "global"
                        elif dist_mode == "semiglobal_auto":
                            align_mode = auto_align_mode(seq_one, seq_two)

                        sub_env = self.environment['aligner_env']
                        sub_component = index.resolve(aligner)
                        if self.environment['squash_profiles']:
                            sub_env.keys['squash_profiles'] = True
                        root_env = self.environment
                        task = execution.add_task(sub_component)
                        task.environment(root_env, sub_env)
                        task.inputs(mode=align_mode, sequence_one=seq_one,
                                    sequence_two=seq_two,
                                    track_id_sets_one=track_id_sets,
                                    track_id_sets_two=track_id_sets,
                                    score_matrices=score_matrices)
                        d_idxs.append((i, j))
                        queued_idxs.add((i, j))
                else:
                    d[i, j] = 0.0

        current_step = 0
        for msg in execution.run():
            yield msg

            if msg.kind == MESSAGE_KIND_COMPLETE and \
               execution.started_task(msg.tag):
                current_step += 1
                yield ProgressMessage(current_step / len(d_idxs))

        for n, outputs in enumerate(execution.outputs):
            i, j = d_idxs[n]
            d[i, j] = outputs['score']
            d[j, i] = outputs['score']

        dist_matrix = (-d) + d.max()

        if debug > 1:
            log.message(ROOT_LOG_NAME, "Dumping distance matrix to disk...")
            np.savetxt(log.path("dist.csv"), dist_matrix, delimiter=",")

        if debug > 0:
            msg = "Generating guide three through hierarchical clustering "\
                  "with linkage method '{0}'...".format(linkage_method)
            log.message(ROOT_LOG_NAME, msg)

        hc = HierarchicalClusteringAlgorithm(dist_matrix)
        guide_tree = SequenceTree(seqs, list(hc.merge_order(linkage_method)))

        if debug > 0:
            msg = "Done!"
            log.message(ROOT_LOG_NAME, msg)

            archive_path = log.archive()
            log.delete()

            yield LogMessage(path_to_url(archive_path))

        yield CompleteMessage({'guide_tree': guide_tree})
