"""
Module to implement the propagation of the
electrons towards the anode.
"""

from math import exp, sqrt
from numba import cuda
from . import consts
from .consts import tpc_borders

import logging
logging.basicConfig()
logger = logging.getLogger("drifting")
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

        track = tracks[itrk]

        pixel_plane = -1

        for ip, plane in enumerate(tpc_borders):
            if plane[0][0]-2e-2 <= track["x"] <= plane[0][1]+2e-2 and \
               plane[1][0]-2e-2 <= track["y"] <= plane[1][1]+2e-2 and \
               min(plane[2][1]-2e-2,plane[2][0]-2e-2) <= track["z"] <= max(plane[2][1]+2e-2,plane[2][0]+2e-2):
                pixel_plane = ip
                break 

        track["pixel_plane"] = pixel_plane

        if pixel_plane >= 0:
            z_anode = tpc_borders[pixel_plane][2][0]
            drift_distance = abs(track["z"] - z_anode) - 0.5
            drift_start = abs(min(track["z_start"],track["z_end"]) - z_anode) - 0.5
            drift_end = abs(max(track["z_start"],track["z_end"]) - z_anode) - 0.5
            drift_time = drift_distance / consts.vdrift

            lifetime_red = exp(-drift_time / consts.lifetime)
            track["z"] = z_anode
            track["n_electrons"] *= lifetime_red

            track["long_diff"] = sqrt((drift_time + 0.5 / consts.vdrift) * 2 * consts.long_diff)
            track["tran_diff"] = sqrt((drift_time + 0.5 / consts.vdrift) * 2 * consts.tran_diff)
            track["t"] += drift_time
            track["t_start"] += min(drift_start, drift_end) / consts.vdrift
            track["t_end"] += max(drift_start, drift_end) / consts.vdrift
        else:
            print(track['z'])
