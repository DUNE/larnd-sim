import numpy as np
import numba as nb

import consts


@nb.njit
def Quench(dEdx, dE, NElectrons):
    '''    
    CPU Quenching Kernel function
    '''       
    for index in range(dE.shape[0]):    
        recomb = np.log(consts.alpha + consts.beta * dEdx[index] \
                        / (consts.beta * dEdx[index]))
        if(recomb <= 0 or np.isnan(recomb)):
            recomb = 0
            
        NElectrons[index]  = recomb * dE[index] * consts.MeVToElectrons



