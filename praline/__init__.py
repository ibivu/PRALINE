"""This is the root package for the PRALINE sequence alignment toolkit.

.. moduleauthor:: Maurits Dijkstra <mauritsdijkstra@gmail.com>

"""

from __future__ import division, absolute_import, print_function

from uuid import uuid4 as uuid
from pkg_resources import resource_stream
import codecs

import numpy as np
from six.moves import range, zip

from praline.container import ScoreMatrix, Alphabet, Sequence, PlainTrack
from praline.container import ProfileTrack, Alignment, TRACK_ID_INPUT
from praline.core import *
from praline.util import window, blocks

def _get_lines(f, encoding):
    """ Read all the lines from a file object or filename string.

    :param f: filename string or file object to read lines from
    :param encoding: encoding of the input file
    :returns: an array of strings containing the lines of the file

    """
    if isinstance(f, str):
        with codecs.open(f, 'r', encoding=encoding) as f:
            lines = f.readlines()
    else:
        lines = f.readlines()

    decoded_lines = []
    for line in lines:
        if isinstance(line, bytes):
            decoded_lines.append(line.decode(encoding))
        else:
            decoded_lines.append(line)

    return decoded_lines

def _strip_comments(line, begin_char = '#'):
    """ Strip comments from a line.
    :param line: the line to strip comments from
    :param begin_char: a character that marks the beginning of a comment
    :returns: the line with possible comments stripped

    """
    i = line.find(begin_char)
    if i >= 0:
        line = line[:i]
    return line

def open_builtin(name):
    """ Open builtin resource for reading.

    :param name: the name of the resource to open
    :returns: a file object reading from resource identified by name

    """

    return resource_stream(__name__, name)

def load_score_matrix(f, alphabet=None, encoding='utf-8'):
    """Read a score matrix into a ScoreMatrix object.

    :param f: filename or file object to read score matrix from
    :param alphabet: alphabet used by score matrix
    :param encoding: encoding to use while reading
    :returns: the score matrix read from f

    """

    lines = [_strip_comments(l).strip() for l in _get_lines(f, encoding)]
    lines = [line.split() for line in lines if len(line) > 0]

    symbol_map = {}
    for i, symbol in enumerate(lines[0]):
        if symbol == "*":
            continue
        symbol_map[i] = symbol

    scores = {}
    for line in lines[1:]:
        symbol_one = line[0]
        for i, value in enumerate(line[1:]):
            try:
                symbol_two = symbol_map[i]
            except KeyError:
                continue
            if "*" in {symbol_one, symbol_two}:
                continue
            scores[symbol_one, symbol_two] = float(value)

    # If an alphabet was not supplied explicitly we generate
    # one based on the symbols defined within the matrix file.
    # This should almost never be used, though...
    if not alphabet:
        mappings = list(zip(list(symbol_map.values()), list(symbol_map.keys())))
        aid = "__anonymous_from_matrix_{0}__".format(uuid().hex)
        alphabet = Alphabet(aid, mappings)

    return ScoreMatrix(scores, [alphabet, alphabet])


def load_sequence_fasta(f, alphabet, encoding='utf-8'):
    """Read a set of sequences in FASTA format into a
    ScoreMatrix object.

    :param f: filename or file object to read sequences from
    :param alphabet: alphabet used by sequences
    :param encoding: encoding to use while reading
    :returns: the sequences read from f

    """

    lines = _get_lines(f, encoding)
    seqs = []
    header = None
    seq = ""
    for line in lines:
        if line.startswith('>'):
            if header and len(seq):
                track = PlainTrack(seq, alphabet)
                seqs.append(Sequence(header, [(TRACK_ID_INPUT, track)]))
            header = line[1:].rstrip()
            seq = ""
            continue
        stripped = line.strip()
        if len(stripped):
            seq += stripped

    if header and len(seq):
        track = PlainTrack(seq, alphabet)
        seqs.append(Sequence(header, [(TRACK_ID_INPUT, track)]))

    return seqs

def load_alignment_fasta(f, alphabet, encoding='utf-8'):
    lines = _get_lines(f, encoding)

    headers = []
    aln_seqs = []

    header = None
    aln_seq = ""
    for line in lines:
        if line.startswith('>'):
            if header and len(seq):
                headers.append(header)
                aln_seqs.append(seq)
            header = line[1:].rstrip()
            seq = ""
            continue

        seq += line.strip()

    if header and len(seq):
        headers.append(header)
        aln_seqs.append(seq)

    consensus_len = None
    for aln_seq in aln_seqs:
        if consensus_len is None:
            consensus_len = len(aln_seq)
        else:
            if len(aln_seq) != consensus_len:
                s = "length {0} does not match consensus length {1}"
                s = s.format(len(aln_seq), consensus_len)

                raise DataError(s)

    if consensus_len is None:
        s = "the input file contains no records"
        raise DataError(s)

    seqs = [[] for aln_seq in aln_seqs]

    path = np.empty((consensus_len + 1, len(aln_seqs)), dtype=int)
    path[0, :] = 0
    for i in range(1, path.shape[0]):
        for j in range(path.shape[1]):
            sym = aln_seqs[j][i - 1]
            if sym == '-':
                path[i, j] = path[i-1, j]
            else:
                path[i, j] = path[i-1, j] + 1
                seqs[j].append(sym)

    seq_objs = []
    for i, seq in enumerate(seqs):
        track = PlainTrack(seq, alphabet)
        header = headers[i]

        seq_obj = Sequence(header, [(TRACK_ID_INPUT, track)])
        seq_objs.append(seq_obj)

    alignment = Alignment(seq_objs, path)

    return alignment

def write_sequence_fasta(f, sequences, trid, encoding='utf-8'):
    """Write a collection of sequence objects in FASTA format.

    :param f: a filename or file object to write the alignment to
    :param alignment: the collection of sequences to write
    :param trid: the track id of the sequence track to write
    :param encoding: encoding to use while writing

    """
    names = [seq.name for seq in sequences]
    tracks = []
    for seq in sequences:
        track = seq.get_track(trid)
        if track.tid != PlainTrack.tid:
            s = "can only write plain tracks to a FASTA formatted " \
                "file"
            raise DataError(s)
        tracks.append(track)

    alphabet = tracks[0].alphabet
    symbolical_sequences = []
    for track in tracks:
        if alphabet.aid != track.alphabet.aid:
            s = "alphabet for track does not match consensus alphabet" \
                "{0}"
            s = s.format(alphabet.aid)
            raise DataError(s)

        symbolical_sequence = []
        for index in track.values:
            symbol = alphabet.index_to_symbol(index)
            symbolical_sequence.append(symbol)
        symbolical_sequences.append(symbolical_sequence)

    lines = []
    for name, symbolical_sequence in zip(names, symbolical_sequences):
        lines.append('>{0}'.format(name))
        for block in blocks(symbolical_sequence):
            lines.append("".join(block))

    should_close = False
    if isinstance(f, str):
        f = codecs.open(f, 'w', encoding)
        should_close = True

    f.write("\n".join(lines))

    if should_close:
        f.close()


def write_alignment_fasta(f, alignment, trid, line_length=72, encoding='utf-8'):
    """Write an alignment object in FASTA format.

    :param f: a filename or file object to write the alignment to
    :param alignment: the alignment to write
    :param trid: the track id of the sequence track to write
    :param line_length: at which length to wrap the alignment
    :param encoding: encoding to use while writing

    """
    alphabet = None
    tracks = []
    for sequence in alignment.items:
        track = sequence.get_track(trid)
        if track.tid != PlainTrack.tid:
            s = "can only write FASTA alignments for plain tracks"
            raise DataError(s)
        alphabet = track.alphabet
        tracks.append(track)

    aligned = [[] for n in range(len(tracks)+1)]
    path = alignment.path
    for i, i_next in window(list(range(path.shape[0]))):
        inc_cols = (path[i_next, :]-path[i, :]) > 0
        symbols = []

        for j, inc_col in enumerate(inc_cols):
            if path[i_next, j] == (-1):
                s = "the FASTA format does not currently support " \
                    "local alignments"
                raise DataError(s)

            if inc_col:
                seq_idx = path[i_next, j]
                aligned[j].append(alphabet.index_to_symbol(tracks[j].values[seq_idx-1]))
            else:
                aligned[j].append('-')


    should_close = False
    if isinstance(f, str):
        f = codecs.open(f, 'w', encoding)
        should_close = True

    names = [sequence.name for sequence in alignment.items]
    for name, contents in zip(names, aligned):
        f.write(">{0}\n".format(name[:line_length - 1]))

        for block in blocks("".join(contents), line_length):
            f.write("{0}\n".format(block))

    if should_close:
        f.close()


def write_alignment_clustal(f, alignment, trid, score_matrix=None, encoding='utf-8'):
    """Write an alignment object in CLUSTAL format.

    :param f: a filename or file object to write the alignment to
    :param alignment: the alignment to write
    :param trid: the track id of the sequence track to write
    :param score_matrix: if supplied, a score matrix used to
                         generate the conservation line
    :param encoding: encoding to use while writing

    """
    alphabet = None
    tracks = []
    for sequence in alignment.items:
        track = sequence.get_track(trid)
        if track.tid != PlainTrack.tid:
            s = "can only write CLUSTAL alignments for plain tracks"
            raise DataError(s)
        alphabet = track.alphabet
        tracks.append(track)

    aligned = [[] for n in range(len(tracks)+1)]
    path = alignment.path
    for i, i_next in window(list(range(path.shape[0]))):
        inc_cols = (path[i_next, :]-path[i, :]) > 0
        symbols = []

        for j, inc_col in enumerate(inc_cols):
            if path[i_next, j] == (-1):
                s = "the CLUSTAL format does not currently support " \
                    "local alignments"
                raise DataError(s)

            if inc_col:
                seq_idx = path[i_next, j]
                symbols.append(alphabet.index_to_symbol(tracks[j].values[seq_idx-1]))
            else:
                symbols.append(None)

        # Calculate maximum score possible for a column for the conservation line
        # if a score matrix has been provided.
        max_self_score = None
        if score_matrix is not None:
            for symbol in symbols:
                if symbol is None:
                    continue
                score = score_matrix.score((symbol, symbol))
                if max_self_score is None or score > max_self_score:
                    max_self_score = score

        col_scores = np.empty((len(symbols), len(symbols)), dtype=np.float32)
        for j, symbol in enumerate(symbols):
            if symbol is not None:
                aligned[j].append(symbol)

                # Calculate column score for conservation line if score matrix
                # has been provided.
                if score_matrix is not None:
                    for k, other_symbol in enumerate(symbols):
                        if other_symbol is None:
                            col_scores[j, k] = 0
                        else:
                            pair = (symbol, other_symbol)
                            col_scores[j, k] = score_matrix.score(pair)
            else:
                aligned[j].append('-')
                for k in range(len(symbols)):
                    col_scores[j, k] = 0

        # Append the conservation symbol if a score matrix has been provided,
        # otherwise just leave it blank.
        if score_matrix is not None:
            col_score = col_scores.sum()
            max_score = max_self_score * len(symbols) * len(symbols)
            if col_score >= max_score * 0.9:
                aligned[-1].append('*')
            elif col_score >= max_score * 0.4:
                aligned[-1].append(':')
            else:
                aligned[-1].append(' ')
        else:
            aligned[-1].append(' ')

    names = [sequence.name for sequence in alignment.items]
    names.append("")

    max_len = max(len(name) for name in names)
    if max_len > 13:
        max_len = 13
    block_len = 78-max_len-5
    if block_len > 60:
        block_len = 60
    aligned_blocks = [blocks("".join(s), block_len) for s in aligned]

    sections = []
    for group in zip(*aligned_blocks):
        processed = []
        for i, line in enumerate(group):
            fmt = u"{0}     {1}"
            name = names[i][:max_len]
            padded_name = name + (" " * (max_len-len(name)))
            processed.append(fmt.format(padded_name, line))
        sections.append("\n".join(processed))

    should_close = False
    if isinstance(f, str):
        f = codecs.open(f, 'w', encoding)
        should_close = True

    f.write("CLUSTAL W format alignment, written by PRALINE\n\n\n")
    f.write("\n\n".join(sections))
    f.write("\n\n")

    if should_close:
        f.close()

def _consensus(row):
    """ Determine the consensus amino acid for a given alignment row.

    :param row: the alignment row to calculate the consensus AA for
    :returns: the index of the consensus AA for the row

    """
    max_e = 0
    max_i = None
    for i, e in enumerate(row):
        if e > max_e or max_i is None:
            max_e = e
            max_i = i

    return max_i

def write_pssm(f, sequence, master_trid, profile_trid, score_matrix, encoding='utf-8'):
    """Construct a position specific scoring matrix (PSSM) from a sequence
    profile track and a base scoring matrix and write it to a file in
    PSI-BLAST tab-delimited PSSM format.

    :param f: filename or file object to write the PSSM to
    :param sequence: sequence to read master and profile tracks from
    :param master_trid: track id of master track
    :param profile_trid: track id of profile track
    :param score_matrix: score matrix to serve as base for PSSM
    :param encoding: encoding to use while writing

    """
    master_track = sequence.get_track(master_trid)
    if master_track.tid != PlainTrack.tid:
        s = "the master track needs to be a plain track for PSSM output"
        raise DataError(s)
    profile_track = sequence.get_track(profile_trid)
    if profile_track.tid != ProfileTrack.tid:
        s = "the profile track needs to be a profile track for PSSM output"
        raise DataError(s)
    if profile_track.alphabet.aid != master_track.alphabet.aid:
        s = "the profile and master tracks need to have the same alphabet"
        raise DataError(s)
    if len(score_matrix.alphabets) != 2:
        s = "this method currently only works with a two dimensional " \
            "score matrix"
        raise DataError(s)
    if score_matrix.alphabets[0].aid != score_matrix.alphabets[1].aid:
        s = "need identical alphabets for dimensions in the score matrix"
        raise DataError(s)
    if score_matrix.alphabets[0].aid != profile_track.alphabet.aid:
        s = "need identical alphabets in scoring matrix and tracks"
        raise DataError(s)

    alphabet = profile_track.alphabet
    counts = profile_track.counts
    scores = score_matrix.matrix

    legend = []
    legend.append("P")
    legend.append("C")
    legend.append("Master")
    for i in range(alphabet.size):
        legend.append(alphabet.index_to_symbol(i))

    rows = [legend]
    for n in range(counts.shape[0]):
        consensus_symbol = alphabet.index_to_symbol(_consensus(counts[n, :]))
        master_symbol = alphabet.index_to_symbol(master_track.values[n])
        profile_row = profile_track.profile[n, :]
        weighted_row = (profile_row * scores).sum(axis=1)

        cols = []
        cols.append(str(n+1))
        cols.append(consensus_symbol)
        cols.append("{0}-{1}".format(n+1, master_symbol))
        for value in weighted_row:
            cols.append("{0:.1f}".format(value))

        rows.append(cols)

    data = "\n".join("\t".join(row) for row in rows)

    should_close = False
    if isinstance(f, str):
        f = codecs.open(f, 'w', encoding)
        should_close = True

    f.write(data)

    if should_close:
        f.close()
