"""
Module to implement the propagation of the
electrons towards the anode.
"""

from math import fabs, exp, sqrt
from numba import cuda
from . import consts
from . import indeces as i
from .consts import module_borders

import logging
logging.basicConfig()
logger = logging.getLogger('drifting')
logger.setLevel(logging.WARNING)
logger.info("DRIFTING MODULE PARAMETERS")

@cuda.jit
def drift(tracks):
    """
    This function takes as input an array of track segments and calculates
    the properties of the segments at the anode:

      * z coordinate at the anode
      * number of electrons taking into account electron lifetime
      * longitudinal diffusion
      * transverse diffusion
      * time of arrival at the anode

    Args:
        tracks (:obj:`numpy.ndarray`): array containing the tracks segment information
    """
    itrk = cuda.grid(1)

    if itrk < tracks.shape[0]:
        pixel_plane = -1

        track = tracks[itrk]

        for ip, plane in enumerate(module_borders):
            if plane[0][0] < track['x'] < plane[0][1] and \
               plane[1][0] < track['y'] < plane[1][1] and \
               plane[2][0] < track['z'] < plane[2][1]:
                pixel_plane = ip
                break

        track['pixel_plane'] = pixel_plane
        z_anode = module_borders[pixel_plane][2][0]
        drift_distance = fabs(track['z'] - z_anode)
        drift_start = fabs(min(track['z_start'],track['z_end']) - z_anode)
        drift_end = fabs(max(track['z_start'],track['z_end']) - z_anode)

        drift_time = drift_distance / consts.vdrift
        track['z'] = z_anode

        lifetime_red = exp(-drift_time / consts.lifetime)
        track['n_electrons'] *= lifetime_red

        track['long_diff'] = sqrt(drift_time * 2 * consts.long_diff)
        track['tran_diff'] = sqrt(drift_time * 2 * consts.tran_diff)
        track['t'] += drift_time
        track['t_start'] += drift_start / consts.vdrift
        track['t_end'] += drift_end / consts.vdrift
