import numpy as np
import numba as nb
from math import fabs, exp, sqrt
from . import consts

'''
Module to implement the propagation of the
electrons towards the anode.
'''

@nb.njit(parallel=True)
def Drift(tracks, col):
    '''
    CPU Drift function
    '''
    zAnode = consts.tpcZStart

    for index in nb.prange(tracks.shape[0]):
        driftDistance        = fabs(tracks[index , col['z']] - zAnode)
        driftStart           = fabs(tracks[index ,col['z_start']] - zAnode)
        driftEnd             = fabs(tracks[index,col['z_end']] - zAnode)

        driftTime            = driftDistance / consts.vdrift
        tracks[index,col['z']]   = zAnode

        lifetime                     =  exp(-driftTime / consts.lifetime)
        tracks[index,col['NElectrons']] *= lifetime

        tracks[index , col['longDiff']] = sqrt(driftTime) * consts.longDiff
        tracks[index , col['tranDiff']] = sqrt(driftTime) * consts.tranDiff
        tracks[index , col['t']]       += driftTime    + tracks[index , col['tranDiff']]/ consts.vdrift
        tracks[index , col['t_start']]  += (driftStart + tracks[index , col['tranDiff']])/ consts.vdrift
        tracks[index , col['t_end']]    += (driftEnd   + tracks[index , col['tranDiff']]) / consts.vdrift

