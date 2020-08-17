"""
Module to implement the propagation of the
electrons towards the anode.
"""

from math import fabs, exp, sqrt
from numba import cuda
from .consts import *


@cuda.jit
def Drift(z_start, z_end, z_mid,
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

        longDiff[index] = sqrt(driftTime) * LONGDIFF
        tranDiff[index] = sqrt(driftTime) * TRANDIFF
        t_mid[index] += driftTime + tranDiff[index] / vdrift
        t_start[index] += (driftStart + tranDiff[index]) / vdrift
        t_end[index] += (driftEnd + tranDiff[index]) / vdrift
