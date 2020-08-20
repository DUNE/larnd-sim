"""
Module to implement the propagation of the
electrons towards the anode.
"""

from math import fabs, exp, sqrt
import numba as nb
from numba import cuda
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

@cuda.jit
def GPU_Drift(z_start, z_end, z_mid,
          t_start, t_end, t_mid,
          nElectrons, longDiff, tranDiff):

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
    zAnode = tpc_zStart

    index = cuda.grid(1)
    if index < z_start.shape[0]:
        driftDistance = fabs(z_mid[index] - zAnode)
        driftStart = fabs(z_start[index] - zAnode)
        driftEnd = fabs(z_end[index] - zAnode)

        driftTime = driftDistance / vdrift
        z_mid[index] = zAnode

        lifetime_red = exp(-driftTime / lifetime)
        nElectrons[index] *= lifetime_red

        longDiff[index] = sqrt(driftTime) * long_diff
        tranDiff[index] = sqrt(driftTime) * tran_diff
        t_mid[index] += driftTime + tranDiff[index] / vdrift
        t_start[index] += (driftStart + tranDiff[index]) / vdrift
        t_end[index] += (driftEnd + tranDiff[index]) / vdrift
