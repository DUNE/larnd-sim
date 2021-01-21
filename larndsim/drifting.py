"""
Module to implement the propagation of the
electrons towards the anode.
"""

from math import fabs, exp, sqrt
from numba import cuda
from .consts import long_diff, tran_diff, vdrift, tpc_borders, lifetime, module_borders
from . import indeces as i

import logging
logging.basicConfig()
logger = logging.getLogger('drifting')
logger.setLevel(logging.WARNING)
logger.info("DRIFTING MODULE PARAMETERS")
logger.info("""Drift velocity: {vdrift} us/cm
Longitudinal diffusion coefficient: {long_diff} cm^2 / us,
Transverse diffusion coefficient: {tran_diff} cm
Electron lifetime: {lifetime} us
TPC borders:
({tpc_borders[0][0]} cm, {tpc_borders[0][1]} cm) x,
({tpc_borders[1][0]} cm, {tpc_borders[1][1]} cm) y,
({tpc_borders[2][0]} cm, {tpc_borders[2][1]} cm) z
""")

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
        pixel_plane = 0

        track = tracks[itrk]

        for ip,plane in enumerate(module_borders):
            if plane[0][0] < track[i.x] < plane[0][1] and plane[1][0] < track[i.y] < plane[1][1]:
                pixel_plane = ip
                break

        track[i.pixel_plane] = pixel_plane
        z_anode = module_borders[pixel_plane][2][0]

        drift_distance = fabs(track[i.z] - z_anode)
        drift_start = fabs(min(track[i.z_start],track[i.z_end]) - z_anode)
        drift_end = fabs(max(track[i.z_start],track[i.z_end]) - z_anode)

        drift_time = drift_distance / vdrift
        track[i.z] = z_anode

        lifetime_red = exp(-drift_time / lifetime)
        track[i.n_electrons] *= lifetime_red

        track[i.long_diff] = sqrt(drift_time*2*long_diff)
        track[i.tran_diff] = sqrt(drift_time*2*tran_diff)
        track[i.t] += drift_time
        track[i.t_start] += drift_start / vdrift
        track[i.t_end] += drift_end / vdrift
