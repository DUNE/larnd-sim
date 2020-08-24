"""
Module to implement the propagation of the
electrons towards the anode.
"""

from math import fabs, exp, sqrt
import numba as nb
from numba import cuda
from .consts import long_diff, tran_diff, vdrift, tpc_borders, lifetime
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
        cols (:obj:`numba.typed.Dict`): Numba dictionary containing columns names for
        the track array
    """
    z_anode = tpc_borders[2][0]

    for index in nb.prange(tracks.shape[0]):
        drift_distance = fabs(tracks[index, cols["z"]] - z_anode)
        drift_start = fabs(tracks[index, cols["z_start"]] - z_anode)
        drift_end = fabs(tracks[index, cols["z_end"]] - z_anode)

        drift_time = drift_distance / vdrift
        tracks[index, cols["z"]] = z_anode

        lifetime_red = exp(-drift_time / lifetime)
        tracks[index, cols["NElectrons"]] *= lifetime_red

        tracks[index, cols["longDiff"]] = sqrt(drift_time) * long_diff
        tracks[index, cols["tranDiff"]] = sqrt(drift_time) * tran_diff
        tracks[index, cols["t"]] += drift_time + tracks[index, cols["tranDiff"]] / vdrift
        tracks[index, cols["t_start"]] += (drift_start + tracks[index, cols["tranDiff"]]) / vdrift
        tracks[index, cols["t_end"]] += (drift_end + tracks[index, cols["tranDiff"]]) / vdrift

@cuda.jit
def GPU_Drift(tracks):

    """
    CUDA version of `Drift`

    Args:
        tracks (:obj:`numpy.array`): array containing the tracks segment information
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
