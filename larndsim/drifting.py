"""
Module to implement the propagation of the
electrons towards the anode.
"""

from math import fabs, exp, sqrt
from numba import cuda
from .consts import long_diff, tran_diff, vdrift, tpc_borders, lifetime
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
    z_anode = tpc_borders[2][0] 

    itrk = cuda.grid(1)
    if itrk < tracks.shape[0]:
        track = tracks[itrk]

        drift_distance = fabs(track[i.z] - z_anode)
        drift_start = fabs(track[i.z_start] - z_anode)
        drift_end = fabs(track[i.z_end] - z_anode)

        drift_time = drift_distance / vdrift
        track[i.z] = z_anode

        lifetime_red = exp(-drift_time / lifetime)
        track[i.n_electrons] *= lifetime_red

        track[i.long_diff] = sqrt(drift_time) * long_diff
        track[i.tran_diff] = sqrt(drift_time) * tran_diff
        track[i.t] += drift_time + track[i.tran_diff] / vdrift
        track[i.t_start] += (drift_start + track[i.tran_diff]) / vdrift 
        track[i.t_end] += (drift_end + track[i.tran_diff]) / vdrift 
