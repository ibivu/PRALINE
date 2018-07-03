
"""Components which generate master-slave alignments from which the
preprofiles are built.

.. moduleauthor:: Maurits Dijkstra <mauritsdijkstra@gmail.com>

"""
from __future__ import division, absolute_import, print_function

import numpy as np
from six.moves import range

from praline.core import *
from praline.component import PairwiseAligner
from praline.container import Sequence, Alignment, ScoreMatrix
from praline.util import compress_path, extend_path_local
from praline.util import extend_path_semiglobal

class DummyMasterSlaveAligner(Component):
    """Master-slave aligners build an approximate multiple sequence alignment
    by iteratively growing an alignment of a master sequence with a set
    of slave sequences. The alignment is compressed, which means that any
    aligned positions which correspond to a gap in the master sequence are
    removed, leaving only gaps in the slave sequence as a possibility. The
    dummy master-slave aligner doesn't actually perform any alignments, it
    just returns an alignment in which the the master sequence is aligned
    to itself, yielding a profile containing just the master sequence. This
    is used in the commandline tool for its ClustalW-like (preprofile-less)
    strategy.

    Inputs:
    * master_sequence - the master sequence to build the dummy alignment
    * slave_sequences - the list of slave sequences to align against the
                        master sequence (ignored by this component)
    * track_id_sets - a list of lists of track ids specifying the tracks in
                      the master and slave sequences to use for the alignment
    * score_matrices - use these score matrices for the alignment (ignored)


    Outputs:
    * alignment - an alignment object containing the resulting master-slave
                  multiple sequence alignment

    Options:
    """
    tid = "praline.component.DummyMasterSlaveAligner"

    inputs = {'master_sequence': Port(Sequence.tid),
              'slave_sequences': Port([Sequence.tid]),
              'track_id_sets': Port([[str]]),
              'score_matrices': Port([ScoreMatrix.tid], optional=True)}
    outputs = {'alignment': Port(Alignment.tid)}

    options = {}
    defaults = {}

    def execute(self, master_sequence, slave_sequences, track_id_sets,
                score_matrices):
        path = np.array([i for i in range(len(master_sequence) + 1)])
        path = path.reshape(len(master_sequence) + 1, 1)
        alignment = Alignment([master_sequence], path)

        yield CompleteMessage({'alignment': alignment})



class GlobalMasterSlaveAligner(Component):
    """Master-slave aligners build an approximate multiple sequence alignment
    by iteratively growing an alignment of a master sequence with a set
    of slave sequences. The alignment is compressed, which means that any
    aligned positions which correspond to a gap in the master sequence are
    removed, leaving only gaps in the slave sequence as a possibility. The
    global master-slave aligner uses global alignments to build the MSA.

    Inputs:
    * master_sequence - the master sequence to align the slave sequences
                        against
    * slave_sequences - the list of slave sequences to align against the
                        master sequence
    * track_id_sets - a list of lists of track ids specifying the tracks in
                      the master and slave sequences to use for the alignment
    * score_matrices - use these score matrices for the alignment

    Outputs:
    * alignment - an alignment object containing the resulting master-slave
                  multiple sequence alignment

    Options:
    * score_treshold - minimum score before an alignment is included in the
                       master-slave alignment
    * gap_series - a list of floats specifying the gap penalties which should
                   be applied for a gap of a certain length (or larger, if
                   the gap length reaches the end of the list)
    * aligner - a type id specifying which component should be used to
                perform the pairwise alignments
    * aligner_env - an environment object which can be used to pass options
                    to the pairwise alignment component

    """
    tid = "praline.component.GlobalMasterSlaveAligner"

    inputs = {'master_sequence': Port(Sequence.tid),
              'slave_sequences': Port([Sequence.tid]),
              'track_id_sets': Port([[str]]),
              'score_matrices': Port([ScoreMatrix.tid])}
    outputs = {'alignment': Port(Alignment.tid)}

    options = {'gap_series': [float], 'aligner': str,
               'aligner_env': Environment.tid,
               'score_threshold': T(float, nullable=True)}
    defaults = {'gap_series': [-11.0, -1.0], 'aligner': PairwiseAligner.tid,
                'score_threshold': None, 'aligner_env': Environment({})}

    def execute(self, master_sequence, slave_sequences, track_id_sets,
                score_matrices):
        index = self.manager.index

        seq_master = master_sequence
        seq_slaves = slave_sequences

        gap_series = np.array(self.environment['gap_series'], dtype=int)
        score_threshold = self.environment['score_threshold']

        path = np.arange(len(seq_master)+1).reshape(len(seq_master)+1, 1)
        alignment = Alignment([seq_master], path)

        for j, seq_slave in enumerate(seq_slaves):
            sub_env = self.environment['aligner_env']
            sub_component = index.resolve(self.environment['aligner'])
            root_env = self.environment
            execution = Execution(self.manager, self.tag)
            task = execution.add_task(sub_component)
            task.environment(root_env, sub_env)
            task.inputs(mode="global", sequence_one=seq_master,
                        sequence_two=seq_slave,
                        track_id_sets_one=track_id_sets,
                        track_id_sets_two=track_id_sets,
                        score_matrices=score_matrices)

            for msg in execution.run():
                yield msg
            outputs = execution.outputs[0]

            score = outputs['score']
            if score_threshold is None or score >= score_threshold:
                path = outputs['alignment'].path
                path = compress_path(np.array(path), 0)

                merge_range = np.arange(len(seq_slave)+1)
                merge_path = merge_range.reshape(len(seq_slave)+1, 1)
                merge_alignment = Alignment([seq_slave], merge_path)
                alignment = alignment.merge(merge_alignment, path)

            yield ProgressMessage((j+1) / len(seq_slaves))

        yield CompleteMessage({'alignment': alignment})

class LocalMasterSlaveAligner(Component):
    """Master-slave aligners build an approximate multiple sequence alignment
    by iteratively growing an alignment of a master sequence with a set
    of slave sequences. The alignment is compressed, which means that any
    aligned positions which correspond to a gap in the master sequence are
    removed, leaving only gaps in the slave sequence as a possibility. The
    local master-slave aligner uses local alignments to build the MSA. In
    addition, it supports doing non-optimal alignments using the Waterman-
    Eggert algorithm. In this algorithm, after the first alignment the
    dynamic programming matrix cells that are part of the trace of the
    first alignment are fixed to zero and the alignment process is repeated.

    Inputs:
    * master_sequence - the master sequence to align the slave sequences
                        against
    * slave_sequences - the list of slave sequences to align against the
                        master sequence
    * track_id_sets - a list of lists of track ids specifying the tracks in
                      the master and slave sequences to use for the alignment
    * score_matrices - use these score matrices for the alignment

    Outputs:
    * alignment - an alignment object containing the resulting master-slave
                  multiple sequence alignment

    Options:
    * score_treshold - minimum score before an alignment is included in the
                       master-slave alignment
    * gap_series - a list of floats specifying the gap penalties which should
                   be applied for a gap of a certain length (or larger, if
                   the gap length reaches the end of the list)
    * aligner - a type id specifying which component should be used to
                perform the pairwise alignments
    * aligner_env - an environment object which can be used to pass options
                    to the pairwise alignment component
    * waterman_eggert_iterations - an int specifying the number of Waterman-
                                   Eggert iterations to perform

    """
    tid = "praline.component.LocalMasterSlaveAligner"

    inputs = {'master_sequence': Port(Sequence.tid),
              'slave_sequences': Port([Sequence.tid]),
              'track_id_sets': Port([[str]]),
              'score_matrices': Port([ScoreMatrix.tid])}
    outputs = {'alignment': Port(Alignment.tid)}

    options = {'gap_series': [float], 'aligner': str,
               'aligner_env': Environment.tid,
               'waterman_eggert_iterations': int,
               'score_threshold': T(float, nullable=True)}
    defaults = {'gap_series': [-11.0, -1.0], 'aligner': PairwiseAligner.tid,
                'score_threshold': None, 'aligner_env': Environment({}),
                'waterman_eggert_iterations': 2}

    def execute(self, master_sequence, slave_sequences, track_id_sets,
                score_matrices):
        index = self.manager.index
        seq_master = master_sequence
        seq_slaves = slave_sequences

        score_threshold = self.environment['score_threshold']
        gap_series = np.array(self.environment['gap_series'],
                              dtype=np.float32)
        iterations = self.environment['waterman_eggert_iterations']

        path=np.arange(len(seq_master)+1).reshape(len(seq_master)+1, 1)
        alignment = Alignment([seq_master], path)

        for j, seq_slave in enumerate(seq_slaves):
            zero_idxs = []
            for n in range(iterations):
                sub_env = self.environment['aligner_env']
                sub_component = index.resolve(self.environment['aligner'])
                root_env = self.environment
                execution = Execution(self.manager, self.tag)
                task = execution.add_task(sub_component)
                task.environment(root_env, sub_env)
                task.inputs(mode="local", sequence_one=seq_master,
                            sequence_two=seq_slave,
                            track_id_sets_one=track_id_sets,
                            track_id_sets_two=track_id_sets,
                            score_matrices=score_matrices,
                            zero_idxs=zero_idxs)

                for msg in execution.run():
                    yield msg
                outputs = execution.outputs[0]

                path = outputs['alignment'].path
                min_x = min(x for x, y in path)
                max_x = max(x for x, y in path)
                min_y = min(y for x, y in path)
                max_y = max(y for x, y in path)

                for x in range(min_x, max_x+1):
                    for y in range(min_y, max_y+1):
                        zero_idxs.append((x, y))

                score = outputs['score']
                if score_threshold is None or score >= score_threshold:
                    path = compress_path(np.array(path), 0)
                    path = extend_path_local(path, len(seq_master), 0)

                    merge_range = np.arange(len(seq_slave)+1)
                    merge_path = merge_range.reshape(len(seq_slave)+1, 1)
                    merge_alignment = Alignment([seq_slave], merge_path)
                    alignment = alignment.merge(merge_alignment, path)

            yield ProgressMessage((j+1) / len(seq_slaves))

        yield CompleteMessage({'alignment': alignment})
