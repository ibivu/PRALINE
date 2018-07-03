"""Components for integration with the BLAST/BLAST+ suite.

.. moduleauthor:: Maurits Dijkstra <mauritsdijkstra@gmail.com>

"""
from __future__ import division, absolute_import, print_function

import subprocess
import shutil
import tempfile
import os.path
from xml.dom.minidom import parse

from praline import write_sequence_fasta, load_sequence_fasta
from praline.core import *
from praline.container import Sequence, ScoreMatrix, TRACK_ID_INPUT
from praline.container import PlainTrack
from praline.container import ALPHABET_AA, ALPHABET_DNA

class BlastPlusSequenceFinder(Component):
    """ This component integrates with 'plain' protein or nucleotide BLAST
    from the BLAST+ suite. It provides users of the PRALINE suite with an
    easy way to query BLAST for any input sequences. This can dramatically
    increase the quality of the preprofiles a multiple sequence alignment
    has to work with, for example.

    Inputs:
    * sequence - the sequence to run a BLAST query on
    * track_id - the track id of the track to feed to BLAST

    Outputs:
    * sequences - sequences returned by BLAST matching the e-value threshold

    Options:
    * blast_plus_root - a path to the installation root of the BLAST+
                        software
    * db_name - string containing the name of the BLAST database to query
    * num_seqs - number of sequences BLAST should output
    * max_evalue - maximum BLAST e-value for inclusion in the output list
                   of sequences
    """
    tid = "praline.component.BlastPlusSequenceFinder"

    inputs = {'sequence': Port(Sequence.tid),
              'track_id': Port(str)}
    outputs = {'sequences': Port([Sequence.tid])}

    options = {'blast_plus_root': T(str, nullable=True),
               'db_name': str,
               'num_seqs': int,
               'max_evalue': float}
    defaults = {'blast_plus_root': None,
                'db_name': "nr",
                'num_seqs': 10,
                'max_evalue': 1.0}

    def execute(self, sequence, track_id):
        track = sequence.get_track(track_id)
        if track.alphabet.aid == ALPHABET_AA.aid:
            program = "blastp"
        elif track.alphabet.aid == ALPHABET_DNA.aid:
            program = "blastn"
        else:
            s = "BLAST+ will only accept inputs with an AA " \
                "or DNA alphabet"
            raise ComponentError(s)

        if self.environment['blast_plus_root'] is None:
            s = "need to know where you have installed BLAST+, " \
                "supply in shell variable BLAST_PLUS_ROOT or " \
                "PRALINE environment option blast_plus_root"
            raise ComponentError(s)

        blast_root = self.environment['blast_plus_root']
        program_path = os.path.join(blast_root, 'bin', program)
        db_program_path = os.path.join(blast_root, 'bin', 'blastdbcmd')

        temp_root = tempfile.mkdtemp()
        blast_output_path = os.path.join(temp_root, 'blast.out')
        input_fasta_path = os.path.join(temp_root, 'input.fasta')
        output_fasta_path = os.path.join(temp_root, 'output.fasta')
        entries_path = os.path.join(temp_root, 'entries.txt')

        db_name = self.environment['db_name']
        num_seqs = self.environment['num_seqs']
        max_evalue = self.environment['max_evalue']

        try:
            write_sequence_fasta(input_fasta_path, [sequence], track_id)
            with open(blast_output_path, 'w') as fo:
                args = [program_path, "-query", input_fasta_path, "-db",
                        db_name, "-outfmt", "5"]
                subprocess.check_call(args, stdout=fo)


            with open(blast_output_path) as fi, open(entries_path, 'w') as fo:
                for i, (id_, evalue) in enumerate(_parse_xml(fi)):
                    if i >= num_seqs:
                        break
                    if evalue <= max_evalue:
                        fo.write("{0}\n".format(id_))

            with open(output_fasta_path, 'w') as fo:
                args = [db_program_path, "-db", db_name, "-entry_batch",
                        entries_path, "-target_only"]
                subprocess.check_call(args, stdout=fo)

            seqs = load_sequence_fasta(output_fasta_path, track.alphabet)
        finally:
            shutil.rmtree(temp_root)

        yield CompleteMessage(outputs={'sequences': seqs})

class PsiBlastPlusSequenceFinder(Component):
    """This integrates with the PSI-BLAST tool. Mostly the same as the
    component which integrates with BLAST, but it allows you to set a few
    more PSI-BLAST-specific options such as the number of iterations and the
    profile inclusion e-value threshold.

    Inputs:
    * sequence - the sequence to run a BLAST query on
    * track_id - the track id of the track to feed to BLAST

    Outputs:
    * sequences - sequences returned by BLAST matching the e-value threshold

    Options:
    * blast_plus_root - a path to the installation root of the BLAST+
                        software
    * db_name - string containing the name of the BLAST database to query
    * num_seqs - number of sequences BLAST should output
    * max_evalue - maximum BLAST e-value for inclusion in the output list
                   of sequences
    * num_iterations - number of PSI-BLAST iterations to run
    * profile_evalue - e-value threshold for inclusion into the next profile

    """
    tid = "praline.component.PsiBlastPlusSequenceFinder"

    inputs = {'sequence': Port(Sequence.tid),
              'track_id': Port(str)}
    outputs = {'sequences': Port([Sequence.tid])}

    options = {'blast_plus_root': T(str, nullable=True),
               'db_name': str,
               'num_seqs': int,
               'max_evalue': float,
               'profile_evalue': float,
               'num_iterations': int}
    defaults = {'blast_plus_root': None,
                'db_name': "nr",
                'num_seqs': 10,
                'max_evalue': 1.0,
                'profile_evalue': 0.002,
                'num_iterations': 3}

    def execute(self, sequence, track_id):
        track = sequence.get_track(track_id)
        if track.alphabet.aid != ALPHABET_AA.aid:
            s = "PSI-BLAST+ will only accept inputs with an AA " \
                "alphabet"
            raise ComponentError(s)

        if self.environment['blast_plus_root'] is None:
            s = "need to know where you have installed BLAST+, " \
                "supply in shell variable BLAST_PLUS_ROOT or " \
                "PRALINE environment option blast_plus_root"
            raise ComponentError(s)

        blast_root = self.environment['blast_plus_root']
        program_path = os.path.join(blast_root, 'bin', 'psiblast')
        db_program_path = os.path.join(blast_root, 'bin', 'blastdbcmd')

        temp_root = tempfile.mkdtemp()
        blast_output_path = os.path.join(temp_root, 'blast.out')
        input_fasta_path = os.path.join(temp_root, 'input.fasta')
        output_fasta_path = os.path.join(temp_root, 'output.fasta')
        entries_path = os.path.join(temp_root, 'entries.txt')

        db_name = self.environment['db_name']
        num_seqs = self.environment['num_seqs']
        max_evalue = self.environment['max_evalue']
        num_iterations = self.environment['num_iterations']
        profile_evalue = self.environment['profile_evalue']

        try:
            write_sequence_fasta(input_fasta_path, [sequence], track_id)
            with open(blast_output_path, 'w') as fo:
                args = [program_path, "-query", input_fasta_path, "-db",
                        db_name, "-outfmt", "5", "-inclusion_ethresh",
                        str(profile_evalue), "-num_iterations",
                        str(num_iterations)]
                subprocess.check_call(args, stdout=fo)


            with open(blast_output_path) as fi, open(entries_path, 'w') as fo:
                for i, (id_, evalue) in enumerate(_parse_xml(fi)):
                    if i >= num_seqs:
                        break
                    if evalue <= max_evalue:
                        fo.write("{0}\n".format(id_))

            with open(output_fasta_path, 'w') as fo:
                args = [db_program_path, "-db", db_name, "-entry_batch",
                        entries_path, "-target_only"]
                subprocess.check_call(args, stdout=fo)

            seqs = load_sequence_fasta(output_fasta_path, track.alphabet)
        finally:
            shutil.rmtree(temp_root)

        yield CompleteMessage(outputs={'sequences': seqs})


def _parse_xml(f):
    doc = parse(f)

    iterations_elem = doc.getElementsByTagName('BlastOutput_iterations')[0]
    iteration_elems = iterations_elem.getElementsByTagName('Iteration')
    max_num = -1
    last_iteration_elem = None
    for elem in iteration_elems:
        num_elem = elem.getElementsByTagName("Iteration_iter-num")[0]
        num = int(_get_text(num_elem.childNodes))
        if num > max_num:
            max_num = num
            last_iteration_elem = elem

    hits_elem = last_iteration_elem.getElementsByTagName("Iteration_hits")[0]
    hit_elems = hits_elem.getElementsByTagName("Hit")
    for elem in  hit_elems:
        id_elem = elem.getElementsByTagName("Hit_id")[0]
        id_ = _get_text(id_elem.childNodes)

        hsps_elem = elem.getElementsByTagName("Hit_hsps")[0]
        hsp_elem = hsps_elem.getElementsByTagName("Hsp")[0]
        evalue_elem = hsp_elem.getElementsByTagName("Hsp_evalue")[0]
        evalue = float(_get_text(evalue_elem.childNodes))

        yield (id_, evalue)

def _get_text(nodelist):
    rc = []
    for node in nodelist:
        if node.nodeType == node.TEXT_NODE:
            rc.append(node.data)
    return ''.join(rc)
