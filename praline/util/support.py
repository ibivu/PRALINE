"""Miscellaneous methods.

.. moduleauthor:: Maurits Dijkstra <mauritsdijkstra@gmail.com>

"""
from __future__ import division, absolute_import, print_function

from six.moves import range

def split_len(seq, length=78):
    """Split a sequence into chunks of at most a given length.

    :param seq: sequence to split
    :param length: maximum length of a chunk
    :returns: a list of lists containing all the chunks for the sequence

    """
    return [seq[i:i+length] for i in range(0, len(seq), length)]

def pad(s, length):
    """Pad a string with spaces at the end until it is of a given length.

    :param s: the string to pad
    :param length: the length to pad the string to
    :returns: the string padded with spaces at the end

    """
    pad_length = length-len(s)

    return s + (" " * pad_length)

def window(l, size = 2):
    """Slide a window of given size over a list, yielding tuples of the given
    size.

    :param l: the list to slide a window over
    :param size: the window size:
    """
    for n in range(len(l) - size + 1):
        yield tuple(n+m for m in range(size))

def blocks(s, size=78):
    """Split a string into blocks of at most a given length.

    :param s: string to split
    :param size: maximum size of a block
    :returns: a list of lists containing all the blocks for the string

    """
    return [s[i:i+size] for i in range(0, len(s), size)]
