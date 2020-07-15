import numpy as np
import numba as nb

import consts

'''
Module to implement the propagation of the
electrons towards the anode.
'''

@nb.njit
def Drift(NElectrons, z, zStart, zEnd, t, tStart, tEnd, longDiff, tranDiff):
    '''    
    CPU Drift function
     '''       
    zAnode = consts.tpcZStart

    for index in range(NElectrons.shape[0]): 
        driftDistance = np.abs(z[index] - zAnode)
        driftStart    = np.abs(zStart[index] - zAnode)
        driftEnd      = np.abs(zEnd[index] - zAnode)
        
        driftTime     = driftDistance / consts.vdrift
        z[index]      = zAnode
        
        lifetime          = np.exp(-driftTime / consts.lifetime)
        NElectrons[index] = NElectrons[index] * lifetime
        
        longDiff[index] = np.sqrt(driftTime) * consts.longDiff
        tranDiff[index] = np.sqrt(driftTime) * consts.tranDiff
        t[index]       += driftTime + tranDiff[index]/ consts.vdrift
        tStart[index]  += (driftStart + tranDiff[index])/ consts.vdrift
        tEnd[index]    += (driftEnd + tranDiff[index]) / consts.vdrift

        
