"""
Module to implement the quenching of the ionized electrons
through the detector
"""

from math import log, isnan
import numba as nb
from . import consts


@nb.njit(parallel=True, fastmath=True)
def Quench(tracks, col, mode="box"):
    """
    CPU Quenching Kernel function
    """
    for index in nb.prange(tracks.shape[0]):
        dedx = tracks[index, col["dEdx"]]

        recomb = 0

        if mode == "box":
            # Baller, 2013 JINST 8 P08005
            csi = consts.beta * dedx / (consts.eField * consts.lArDensity)
            recomb = max(0, log(consts.alpha + csi)/csi)
        elif mode == "birks":
            # Amoruso, et al NIM A 523 (2004) 275
            recomb = consts.Ab / (1 + consts.kb * dedx / (consts.eField * consts.lArDensity))
        else:
            raise ValueError("Invalid recombination mode: must be 'box' or 'birks'")

        if isnan(recomb):
            raise RuntimeError("Invalid recombination value")

        tracks[index, col["NElectrons"]] = recomb * tracks[index, col["dE"]] * consts.MeVToElectrons
