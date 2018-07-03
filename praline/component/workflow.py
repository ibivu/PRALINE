"""PRALINE MSA workflow component.

.. moduleauthor:: Maurits Dijkstra <mauritsdijkstra@gmail.com>

"""
from __future__ import division, absolute_import, print_function

from praline.core import *
from praline.container import Sequence, Alignment, ScoreMatrix
from praline.component import GuideTreeBuilder, AdHocMultipleSequenceAligner
from praline.component import TreeMultipleSequenceAligner
from praline.component import LocalMasterSlaveAligner, GlobalMasterSlaveAligner
from praline.component import DummyMasterSlaveAligner, ProfileBuilder
from praline.component import PsiBlastPlusSequenceFinder
from praline.container import TRACK_ID_INPUT, TRACK_ID_PREPROFILE

class PralineMultipleSequenceAlignmentWorkflow(Component):
    """This component implements the MSA workflow designed for compatibility
    with the old PRALINE program. The command-line tool is basically a wrapper
    around this component, only performing I/O tasks such as the reading and
    writing of the sequences and alignments.

    Inputs:
    * sequences - two or more input sequences to be aligned
    * score_matrix - the score matrix to score the pairwise alignments

    Outputs:
    * alignment - the resulting multiple sequence alignment

    Options:
    * debug - integer indicating the debug level (0 is silent, 3 is noisiest)
    * preprofile_mode - string indicating which preprofile mode to use
    * msa_mode - string indicating whether to use ad hoc or tree MSA
    * run_psi_blast - run a PSI-BLAST query to find additional sequences
                      for the preprofiles

    """
    tid = "praline.component.PralineMultipleSequenceAlignmentWorkflow"

    inputs = {'sequences': Port([Sequence.tid]),
              'score_matrix': Port(ScoreMatrix.tid)}
    outputs = {'alignment': Port(Alignment.tid)}

    options = {'debug': int, 'preprofile_mode': str, 'msa_mode': str,
               'run_psi_blast': bool}
    defaults = {'debug': 0, 'preprofile_mode': 'global', 'msa_mode': 'adhoc',
                'run_psi_blast': False}

    def __init__(self, *args, **kwargs):
        super(PralineMultipleSequenceAlignmentWorkflow, self).__init__(*args,
                                                                       **kwargs)
        self._master_slave_alignments = None
        self._alignment = None

    def execute(self, sequences, score_matrix):
        debug = self.environment['debug']

        if debug > 0:
            log = LogBundle()
            msg = "Entering component '{0}'".format(self.tid)
            log.message(ROOT_LOG_NAME, msg)

        # Build initial sets of which sequences to align against every master
        # sequence. By default, we want to align every input sequence against
        # every other input sequence.
        master_slave_seqs = []
        all_seqs = list(sequences)
        for master_seq in sequences:
            slave_seqs = []
            for slave_seq in sequences:
                if slave_seq is not master_seq:
                    slave_seqs.append(slave_seq)
            master_slave_seqs.append((master_seq, slave_seqs))

        # If the user has requested this, do a PSI-BLAST lookup on each of the
        # master sequences and add the results to the list of slave sequences.
        if self.environment['run_psi_blast']:
            component = PsiBlastPlusSequenceFinder
            execution = Execution(self.manager, self.tag)
            for master_seq, slave_seqs in master_slave_seqs:
                task = execution.add_task(component)
                task.environment(env)
                task.inputs(sequence=master_seq, track_id=TRACK_ID_INPUT)

            for msg in execution.run():
                yield msg

            for i, output in enumerate(execution.outputs):
                query_sequences = output['sequences']

                master_seq, slave_seqs = master_slave_seqs[i]
                slave_seqs.extend(query_sequences)
                all_seqs.extend(query_sequences)

        # Master-slave alignments
        track_id_sets = [[TRACK_ID_INPUT]]
        for msg in self._do_master_slave_alignments(master_slave_seqs,
                                                    track_id_sets,
                                                    score_matrix):
            yield msg

        # Build preprofiles from master-slave alignments.
        for msg in self._do_preprofiles(sequences):
            yield msg

        # Replace input track ids to align by the preprofiles now.
        msa_track_id_sets = self._replace_input_track_id(track_id_sets)

        # Do multiple sequence alignment from preprofile-annotated sequences.
        for msg in self._do_multiple_sequence_alignment(sequences,
                                                        msa_track_id_sets,
                                                        score_matrix):
            yield msg

        if debug > 0:
            msg = "Done!"
            log.message(ROOT_LOG_NAME, msg)

            archive_path = log.archive()
            log.delete()

            yield LogMessage(path_to_url(archive_path))

        yield CompleteMessage({'alignment': self._alignment})

    def _replace_input_track_id(self, track_id_sets):
        new_track_id_sets = []
        for s in track_id_sets:
            new_s = []
            for tid in s:
                if tid == TRACK_ID_INPUT:
                    new_s.append(TRACK_ID_PREPROFILE)
                else:
                    new_s.append(tid)
            new_track_id_sets.append(new_s)
        return new_track_id_sets


    def _do_master_slave_alignments(self, seqs, track_id_sets, score_matrix):
        execution = Execution(self.manager, self.tag)

        self._master_slave_alignments = [None for seq in seqs]
        for master_seq, slave_seqs in seqs:
            if self.environment['preprofile_mode'] == 'global':
                component = GlobalMasterSlaveAligner
            elif self.environment['preprofile_mode'] == 'local':
                component = LocalMasterSlaveAligner
            elif self.environment['preprofile_mode'] == 'dummy':
                component = DummyMasterSlaveAligner

            task = execution.add_task(component)
            task.environment(self.environment)
            task.inputs(master_sequence=master_seq, slave_sequences=slave_seqs,
                        track_id_sets=track_id_sets,
                        score_matrices=[score_matrix])

        for msg in execution.run():
            yield msg

        for n, output in enumerate(execution.outputs):
            self._master_slave_alignments[n] = output['alignment']


    def _do_multiple_sequence_alignment(self, seqs, track_id_sets,
                                        score_matrix):
        if self.environment['msa_mode'] == 'tree':
            # Dummy preprofiles, so we can safely align by sequence.
            sub_env = Environment(parent=self.environment)

            if not self.environment['preprofile_mode'] in {'local', 'global'}:
                sub_env.keys['squash_profiles'] = True

            # Build guide tree
            component = GuideTreeBuilder
            execution = Execution(self.manager, self.tag)
            task = execution.add_task(component)
            task.environment(sub_env)
            task.inputs(sequences=seqs, track_id_sets=track_id_sets,
                        score_matrices=[score_matrix])

            for msg in execution.run():
                yield msg
            outputs = execution.outputs[0]

            # Build MSA
            component = TreeMultipleSequenceAligner
            execution = Execution(self.manager, self.tag)
            task = execution.add_task(component)
            task.environment(self.environment)
            task.inputs(sequences=seqs, guide_tree=outputs['guide_tree'],
                        track_id_sets=track_id_sets,
                        score_matrices=[score_matrix])

            for msg in execution.run():
                yield msg
            outputs = execution.outputs[0]
        else:
            component = AdHocMultipleSequenceAligner
            execution = Execution(self.manager, self.tag)
            task = execution.add_task(component)
            task.environment(self.environment)
            task.inputs(sequences=seqs, track_id_sets=track_id_sets,
                        score_matrices=[score_matrix])

            for msg in execution.run():
                yield msg
            outputs = execution.outputs[0]

        self._alignment = outputs['alignment']

    def _do_preprofiles(self, seqs):
        for i, alignment in enumerate(self._master_slave_alignments):
            component = ProfileBuilder
            execution = Execution(self.manager, self.tag)
            task = execution.add_task(component)
            task.environment(self.environment)
            task.inputs(alignment=alignment, track_id=TRACK_ID_INPUT)

            for msg in execution.run():
                yield msg
            outputs = execution.outputs[0]

            track = outputs['profile_track']
            seqs[i].add_track(TRACK_ID_PREPROFILE, track)
