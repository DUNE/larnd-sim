"""
Module to implement the propagation of the
electrons towards the anode.
"""

from math import fabs, exp, sqrt
import numba as nb
from .consts import *
from tqdm import tqdm_notebook as progress_bar

@nb.njit(parallel=True, fastmath=True)
def Drift(tracks, col):
    """
    CPU Drift function
    """
    zAnode = tpcBorders[2][0]

    for index in nb.prange(tracks.shape[0]):
        driftDistance = fabs(tracks[index, col["z"]] - zAnode)
        driftStart = fabs(tracks[index, col["z_start"]] - zAnode)
        driftEnd = fabs(tracks[index, col["z_end"]] - zAnode)

        driftTime = driftDistance / vdrift
        tracks[index, col["z"]] = zAnode

        lifetime_red = exp(-driftTime / lifetime)
        tracks[index, col["NElectrons"]] *= lifetime_red

        tracks[index, col["longDiff"]] = sqrt(driftTime) * longDiff
        tracks[index, col["tranDiff"]] = sqrt(driftTime) * tranDiff
        tracks[index, col["t"]] += driftTime + tracks[index, col["tranDiff"]] / vdrift
        tracks[index, col["t_start"]] += (driftStart + tracks[index, col["tranDiff"]]) / vdrift
        tracks[index, col["t_end"]] += (driftEnd + tracks[index, col["tranDiff"]]) / vdrift
