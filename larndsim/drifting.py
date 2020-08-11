"""
Module to implement the propagation of the
electrons towards the anode.
"""

from math import fabs, exp, sqrt
import numba as nb
from .consts import *


@nb.njit(parallel=True, fastmath=True)
def Drift(tracks, cols):
    """
    This function takes as input an array of track segments and calculates
    the properties of the segments at the anode:
        - z coordinate at the anode
        - number of electrons taking into account electron lifetime
        - longitudinal diffusion
        - transverse diffusion
        - time of arrival at the anode

    Args:
        tracks (:obj:`numpy.array`): array containing the tracks segment information
        cols (:obj:`numba.typed.Dict`): Numba dictionary containing columns names for the track array
    """
    zAnode = tpc_borders[2][0]

    for index in nb.prange(tracks.shape[0]):
        driftDistance = fabs(tracks[index, cols["z"]] - zAnode)
        driftStart = fabs(tracks[index, cols["z_start"]] - zAnode)
        driftEnd = fabs(tracks[index, cols["z_end"]] - zAnode)

        driftTime = driftDistance / vdrift
        tracks[index, cols["z"]] = zAnode

        lifetime_red = exp(-driftTime / lifetime)
        tracks[index, cols["NElectrons"]] *= lifetime_red

        tracks[index, cols["longDiff"]] = sqrt(driftTime) * longDiff
        tracks[index, cols["tranDiff"]] = sqrt(driftTime) * tranDiff
        tracks[index, cols["t"]] += driftTime + tracks[index, cols["tranDiff"]] / vdrift
        tracks[index, cols["t_start"]] += (driftStart + tracks[index, cols["tranDiff"]]) / vdrift
        tracks[index, cols["t_end"]] += (driftEnd + tracks[index, cols["tranDiff"]]) / vdrift
