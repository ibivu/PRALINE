"""Sequence container type, sequence track container types and
default track identifiers.

.. moduleauthor:: Maurits Dijkstra <mauritsdijkstra@gmail.com>

"""
from __future__ import division, absolute_import, print_function

import numpy as np
from six.moves import range

from praline.core import *
from praline.util import window

TRACK_ID_INPUT = "praline.tracks.InputTrack"
TRACK_ID_PREPROFILE = "praline.tracks.Preprofile"
TRACK_ID_PROFILE = "praline.tracks.Profile"

class Sequence(Container):
    """The Sequence container is the root of the PRALINE container
    hierarchy for sequence data. This container does not store the actual
    sequence data itself, but acts a collection object for an arbitrary
    number of sequence *tracks*. Within a sequence object, all tracks are
    required to be of the same length and should provide a property to
    access the alphabet of the sequence data they contain.

    :param name: a string containing an identifier or name for this sequence
    :param tracks: a list of 2-tuples containing track ids and tracks to
        initially construct this sequence with

    """
    tid = "praline.container.Sequence"

    def __init__(self, name, tracks):
        self.name = name
        self._length = None
        self._tracks = {}
        for trid, track in tracks:
            self.add_track(trid, track)

    def __len__(self):
        return self._length

    def add_track(self, trid, track):
        """Add a track with a given track id to this sequence.

        :param trid: a string containing the track id
        :param track: the track to add to this sequence

        """
        if trid in self._tracks:
            s = "track with id {0} already present in this sequence"
            s = s.format(trid)
            raise DataError(s)

        if self._length is None:
            self._length = len(track)

        if len(track) != len(self):
            s = "track length {0} does not match sequence length {1}"
            s = s.format(len(track), len(self))
            raise DataError(s)

        self._tracks[trid] = track

    def del_track(self, trid):
        """Delete a track with a given track id from this sequence.

        :param trid: the track id of the track to delete

        """
        if trid not in self._tracks:
            s = "track with id {0} not found"
            s = s.format(trid)
            raise DataError(s)

        del self._tracks[trid]

        if len(self._tracks) == 0:
            self._length = None

    def replace_track(self, trid, track):
        """Replace the track with the given track id with another.

        :param trid: the track id of the track to replace
        :param track: the track to replace the given track with

        """
        self.del_track(trid)
        self.add_track(trid, track)

    def get_track(self, trid):
        """Get the track with the given track id.

        :param trid: the track id of the track to return
        :returns: the track identified by the given track id

        """
        if trid not in self._tracks:
            s = "track with id {0} not found"
            s = s.format(trid)
            raise DataError(s)
        return self._tracks[trid]

    @property
    def tracks(self):
        """Return a list of 2-tuples containing all track ids and the tracks
        they identify.

        """
        return list(self._tracks.items())

    def __repr__(self):
        f = "<Sequence name='{0}' length={1}>"

        return f.format(self.name, len(self))


class Track(Container):
    """A Track container is an abstract interface for container types
    containing the actual sequence data. Other than a requirement that all
    subclasses provide a way to query the track for an alphabet and its
    length there are no required methods or properties to implement. This
    informal protocol allows track objects to represent sequence data of
    a large number of types, including (but not limited to) sequences,
    profiles and HMM priors.

    This object should be subclassed whenever you want to add a new sequence
    container to PRALINE but should never be instantiated directly.

    """
    tid = "praline.container.Track"

    def __len__(self):
        s = "please implement __len__ in your Track subclass"
        raise NotImplementedError(s)


class PlainTrack(Track):
    """The PlainTrack is the simplest type of track, containing plain
    sequence data, or simply a signle index identifying a specific symbol
    per sequence position.

    :param values: a list of symbols corresponding to the symbols at each
        sequence position
    :param alphabet: the alphabet of this track
    :param raw_indices: numpy array containing the raw indices for the track,
        used for the copy constructor. if provided, values will not be used.
        does not check whether indices map to symbols in an alphabet, so
        tread carefully!

    """
    tid = "praline.container.PlainTrack"

    def __init__(self, values, alphabet, raw_indices=None):
        if raw_indices is None:
            indices = [alphabet.symbol_to_index(value) for value in values]
        else:
            indices = raw_indices
        self.values = np.array(indices, dtype=np.int32)
        self.alphabet = alphabet

    def __len__(self):
        return self.values.shape[0]


class ProfileTrack(Track):
    """A ProfileTrack stores sequence information about the probability of
    finding a specific symbol type at a given sequence position. Internally
    the profile track stores the raw counts in a 2-dimensional array. When
    probabilities are required they are generated on the fly and cached
    for quick lookup in the future. This allows access to a probability
    profile while still keeping the raw count data accessible in case it is
    required.

    :param counts: a two-dimensional array containing the counts for every
        sequence position
    :param alphabet: the alphabet for this track

    """
    tid = "praline.container.ProfileTrack"

    def __init__(self, counts, alphabet):
        self.counts = np.array(counts, dtype=int)
        self.alphabet = alphabet
        self._profile = None

    def __len__(self):
        return self.counts.shape[0]

    @property
    def profile(self):
        """Generates the probability profile if required and returns it.
        The probabilities are generated by summing the total symbol count
        for every sequence position and then dividing the entire array by
        them.

        """
        if self._profile is None:
            totals = np.array(self.counts.sum(axis=1), dtype=np.float32)
            self._profile = np.array(self.counts/totals[:, np.newaxis],
                                     dtype=np.float32)
        return self._profile

    def merge(self, track, path):
        """Merge this profile track with another one. This is never done
        in-place. Instead, a new profile track is created representing the
        merge between this profile and the provided profile.

        :param track: the track to merge with
        :param path: a path through the dynamic programming matrix to guide
            the merge between this track and the provided track
        :returns: a new profile track representing the merge between this
            profile and the provided profile

        """
        if track.tid != self.tid:
            s = "can not merge with non-profile track {0}"
            s = s.format(track.tid)
            raise DataError(s)

        if self.alphabet.aid != track.alphabet.aid:
            s = "our alphabet {0} does not match track alphabet {1}"
            s = s.format(self.alphabet.aid, profile.alphabet.aid)
            raise DataError(s)

        size = (path.shape[0]-1, self.counts.shape[1])
        merged_counts = np.zeros(size, dtype=np.float32)
        for i, i_next in window(list(range(path.shape[0]))):
            inc_cols = (path[i_next, :]-path[i, :]) > 0
            for j, inc_col in enumerate(inc_cols):
                if inc_col:
                    prof_idx = path[i_next, j]
                    if j == 0:
                        merged_counts[i, :] += self.counts[prof_idx-1, :]
                    else:
                        merged_counts[i, :] += track.counts[prof_idx-1, :]

        return ProfileTrack(merged_counts, self.alphabet)
