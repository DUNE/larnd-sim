"""
Module to implement the propagation of the
electrons towards the anode.
"""

from math import fabs, exp, sqrt
import numba as nb
from numba import cuda
from .consts import *
from . import indeces as i

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

        tracks[index, cols["longDiff"]] = sqrt(driftTime) * long_diff
        tracks[index, cols["tranDiff"]] = sqrt(driftTime) * tran_diff
        tracks[index, cols["t"]] += driftTime + tracks[index, cols["tranDiff"]] / vdrift
        tracks[index, cols["t_start"]] += (driftStart + tracks[index, cols["tranDiff"]]) / vdrift
        tracks[index, cols["t_end"]] += (driftEnd + tracks[index, cols["tranDiff"]]) / vdrift

@cuda.jit
def GPU_Drift(tracks):

    """
    CUDA version of `Drift`

    Args:
        tracks (:obj:`numpy.array`): array containing the tracks segment information
    """
    zAnode = tpc_borders[2][0]

    itrk = cuda.grid(1)
    if itrk < tracks.shape[0]:
        track = tracks[itrk]
        
        driftDistance = fabs(track[i.z] - zAnode)
        driftStart = fabs(track[i.z_start] - zAnode)
        driftEnd = fabs(track[i.z_end] - zAnode)

        driftTime = driftDistance / vdrift
        track[i.z] = zAnode

        lifetime_red = exp(-driftTime / lifetime)
        track[i.n_electrons] *= lifetime_red

        track[i.long_diff] = sqrt(driftTime) * long_diff
        track[i.tran_diff] = sqrt(driftTime) * tran_diff
        track[i.t] += driftTime + track[i.tran_diff] / vdrift
        track[i.t_start] += (driftStart + track[i.tran_diff]) / vdrift
        track[i.t_end] += (driftEnd + track[i.tran_diff]) / vdrift
