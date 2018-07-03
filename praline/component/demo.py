"""Component for the demo I'm giving at the VU.

.. moduleauthor:: Maurits Dijkstra <mauritsdijkstra@gmail.com>

"""
from __future__ import division, absolute_import, print_function

import subprocess
import shutil
import tempfile

import numpy as np
from six.moves import range

from praline import load_alignment_markx3
from praline.core import *
from praline.container import Sequence, Alignment, ScoreMatrix, TRACK_ID_INPUT
from praline.container import SCORE_MATRIX_DEFAULT, PlainTrack

class NeedleAligner(Component):
    """This is a demo component that is supposed to show the advanced
    capabilities of the new PRALINE suite. It implements a pairwise
    aligner by writing out the inputs as fasta files, calling EMBOSS
    needle on it and parsing its output as an alignment container object
    which is then returned. This component, being an example, only supports
    global alignments. Extending it to support local alignments (EMBOSS water)
    is left as an exercise to the reader.

    Inputs:
    * sequence_one - a sequence object containing the first sequence to align
    * sequence_two - a sequence object containing the second sequence to align
    * track_ids_one - a list of strings containing the track ids of the
                      tracks of the first sequence to align sequence to align
    * track_ids_two - a list of strings containing the track ids of the tracks
                      of the second sequence to align
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
    * gap_series - a list of ints specifying the gap penalties which should
                   be applied for a gap of a certain length (or larger, if
                   the gap length reaches the end of the list)
    * local - perform a local alignment instead of a global alignment
    * score_matrix - use this score matrix for the alignment

    """
    tid = "praline.component.PairwiseAligner"
    inputs = {'sequence_one': Port(Sequence.tid),
              'sequence_two': Port(Sequence.tid),
              'track_ids_one': Port([str]),
              'track_ids_two': Port([str]),
              'zero_idxs': Port([(int, int)], optional = True)}
    outputs = {'alignment': Port(Alignment.tid), 'score': Port(float)}

    options = {'gap_series': [int], 'local': bool,
               'score_matrix': ScoreMatrix.tid}
    defaults = {'gap_series': [-11, -1], 'local': False,
                'score_matrix': SCORE_MATRIX_DEFAULT}

    def execute(self, sequence_one, sequence_two, track_ids_one,
                track_ids_two, zero_idxs):
        score_matrix = self.environment['score_matrix']
        local = self.environment['local']
        gap_series = np.array(self.environment['gap_series'],
                              dtype=np.float32)

        if len(track_ids_one) != len(track_ids_two):
            s = "should have an identical number of tracks to" \
                "align in both sequences. this is a limitation" \
                "of the alignment core and likely to go away" \
                "in the future..."
            raise ComponentError(s)

        if len(track_ids_one) != 1:
            s = "this demo component only supports alignments with" \
                "one track"
            raise ComponentError(s)

        if score_matrix.matrix.ndim != len(track_ids_one)+len(track_ids_two):
            s = "the score matrix must consist of as many dimensions as" \
                "there are tracks to be aligned ({0}), but it contains {1}"
            matrix = score_matrix.matrix
            s = s.format(len(track_ids_one)+len(track_ids_two), matrix.ndim)
            raise ComponentError(s)

        inputs = []
        for i in range(len(track_ids_one)):
            track_id_one = track_ids_one[i]
            track_id_two = track_ids_two[i]
            track_one = sequence_one.get_track(track_id_one)
            track_two = sequence_two.get_track(track_id_two)

            if score_matrix.alphabets[i*2].aid != track_one.alphabet.aid:
                s = "track {0} for sequence one has alphabet '{1}' but " \
                    "the corresponding dimension in the score matrix has " \
                    "alphabet '{2}'"
                s = s.format(i, track_one.alphabet.aid,
                             score_matrix.alphabets[i*2].aid)
                raise DataError(s)

            if score_matrix.alphabets[(i*2)+1].aid != track_two.alphabet.aid:
                s = "track {0} for sequence two has alphabet '{1}' but " \
                    "the corresponding dimension in the score matrix has " \
                    "alphabet '{2}'"
                s = s.format(i, track_two.alphabet.aid,
                             score_matrix.alphabets[(i*2)+1].aid)
                raise DataError(s)

            for track in [track_one, track_two]:
                if track.tid == PlainTrack.tid:
                    inputs.append(track.values)
                else:
                    s = "unknown track type id for this aligner: '{0}'"
                    s = s.format(track.tid)
                    raise DataError(s)

        try:
            temp_root = tempfile.mkdtemp()
            output_fasta_path = os.path.join(temp_root, 'output.fasta')
            input_one_fasta_path = os.path.join(temp_root, 'input_one.fasta')
            input_two_fasta_path = os.path.join(temp_root, 'input_two.fasta')

            write_sequence_fasta(input_one_fasta_path, [sequence_one], track_ids_one[0])
            write_sequence_fasta(input_two_fasta_path, [sequence_two], track_ids_two[0])

            args = ["needle", input_one_fasta_path, input_two_fasta_path, output_fasta_path,
                    "-aformat3", "markx3", "-gapopen", "10.0", "-gapextend", "0.5"]
            subprocess.check_call(args)

            alignment = load_alignment_markx3(output_fasta_path, track_one.alphabet)
            # TODO: parse alignment score from markx3 output file
            outputs = {'alignment': alignment, 'score': 0.0}
        finally:
            shutil.rmtree(temp_root)

        yield CompleteMessage(outputs=outputs)
