"""Alphabet container class and default alphabets.

.. moduleauthor:: Maurits Dijkstra <mauritsdijkstra@gmail.com>

"""
from __future__ import division, absolute_import, print_function

import numpy as np
from six.moves import range
from six.moves import zip

from praline.core import *



class Alphabet(Container):
    """The alphabet container represents a mapping between human readable
    symbols and *indices*, which represent sequences inside the PRALINE
    system. Because PRALINE internally works mostly with indexed arrays,
    it is not possible to use the symbols directly as the key to look
    up items. Also, there often are multiple notations for different symbols,
    for example in different character sets. To keep the PRALINE core free
    of such concerns the alphabet converts every input symbol into an index
    beforehand.

    :param aid: a string containing the alphabet id, which are used for
        equality comparisons between alphabets
    :param mappings: a list of 2-tuples containing the symbols and the
        indices they map to

    """
    tid = "praline.container.Alphabet"

    def __init__(self, aid, mappings):
        self.aid = aid
        self._mappings = {}
        self._inv_mappings = {}
        self._max_index = 0
        for symbol, index in mappings:
            if index > self._max_index:
                self._max_index = index
            self._mappings[symbol] = index
            self._inv_mappings[index] = symbol

    def symbol_to_index(self, symbol):
        """Convert a symbol to an index according to this alphabet.

        :param symbol: a string containing the symbol to convert
        :returns: the index corresponding to the provided symbol

        """
        try:
            return self._mappings[symbol]
        except KeyError:
            s = "symbol '{0}' not found in {1}".format(symbol, self)
            raise AlphabetError(s)

    def index_to_symbol(self, index):
        """Convert an index to a symbol according to this alphabet.

        :param index: an index to convert to the corresponding symbol
        :returns: a string containing the corresponding symbol

        """
        try:
            return self._inv_mappings[index]
        except KeyError:
            s = "index {0} not found in {1}".format(index, self)
            raise AlphabetError(s)

    @property
    def symbols(self):
        """Return all the symbols defined by this alphabet.

        """
        return list(self._mappings.keys())

    @property
    def size(self):
        """Return the size of the alphabet. This is not the number of
        mappings, but the highest index plus one. This allows there to be
        gaps in the alphabet, which may be useful for future additions or
        when a specific alphabet is a subset of another.

        """
        return self._max_index + 1

    def __repr__(self):
        return "<Alphabet aid='{0}'>".format(self.aid)

def _generate_mappings(symbols):
    """Helper method to generate a mapping between symbols and indices.

    :param symbols: list containing the symbols to map
    :returns: a list of 2-tuples containing the generated mapping between
        symbols and indices
    """
    return list(zip(symbols, list(range(len(symbols)))))

_SYMBOLS_AA = [u'A', u'R', u'N', u'D', u'C', u'E', u'Q', u'G', u'H', u'I',
               u'L', u'K', u'M', u'F', u'P', u'S', u'T', u'W', u'Y', u'V',
               u'U', u'O', u'B', u'Z', u'J', u'X']
_MAPPINGS_AA = _generate_mappings(_SYMBOLS_AA)
ALPHABET_AA = Alphabet('praline.alphabet.AAOneLetter', _MAPPINGS_AA)

_SYMBOLS_DNA = ['A', 'T', 'G', 'C', 'S', 'W', 'R', 'Y', 'K', 'M', 'B', 'V',
                'H', 'D', 'N']
_MAPPINGS_DNA = _generate_mappings(_SYMBOLS_DNA)
ALPHABET_DNA = Alphabet('praline.alphabet.DNA', _MAPPINGS_DNA)

_SYMBOLS_RNA = [u'A', u'U', u'C', u'G']
_MAPPINGS_RNA = _generate_mappings(_SYMBOLS_RNA)
ALPHABET_RNA = Alphabet('praline.alphabet.RNA', _MAPPINGS_RNA)
