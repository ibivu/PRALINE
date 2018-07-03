"""Profile-related components.

.. moduleauthor:: Maurits Dijkstra <mauritsdijkstra@gmail.com>

"""
from __future__ import division, absolute_import, print_function

import numpy as np

from praline.core import *
from praline.util import get_frequencies
from praline.container import Alignment, ProfileTrack, Alphabet


class ProfileBuilder(Component):
    """This is a very simple component used in the construction of a
    ProfileTrack from an alignment. It first gets the frequencies of each
    symbol type for every sequence position and then feeds these to the
    profile track to build a profile.

    Inputs:
    * alignment - the alignment to use for the profile track construction
    * track_id - the id of the track to use for the profile track construction

    Outputs:
    * profile_track - the resulting profile track

    Options:
    * debug - integer indicating the debug level (0 is silent, 3 is noisiest)

    """
    tid = "praline.component.ProfileBuilder"

    inputs = {'alignment': Port(Alignment.tid),
              'track_id': Port(str)}
    outputs = {'profile_track': Port(ProfileTrack.tid)}

    options = {'debug': int}
    defaults = {'debug': 0}

    def execute(self, alignment, track_id):
        debug = self.environment["debug"]

        if debug > 0:
            log = LogBundle()
            msg = "Entering component '{0}'".format(self.tid)
            log.message(ROOT_LOG_NAME, msg)

            msg = "{0} sequences:".format(len(alignment.items))
            log.message(ROOT_LOG_NAME, msg)

            for item in alignment.items:
                msg = "\t{0}".format(item.name)
                log.message(ROOT_LOG_NAME, msg)

        freqs = get_frequencies(alignment, track_id)
        if debug > 1:
            log.message(ROOT_LOG_NAME, "Dumping frequency table...")
            np.savetxt(log.path("freqs.csv"), freqs, delimiter=",")

        track = alignment.items[0].get_track(track_id)
        profile_track = ProfileTrack(freqs, track.alphabet)

        if debug > 0:
            msg = "Done!"
            log.message(ROOT_LOG_NAME, msg)

            archive_path = log.archive()
            log.delete()

            yield LogMessage(path_to_url(archive_path))


        yield CompleteMessage(outputs={'profile_track': profile_track})
