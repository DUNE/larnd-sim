"""
Module to implement the quenching of the ionized electrons
through the detector
"""

from math import log, isnan
from numba import cuda
import numba as nb
from . import consts

@nb.njit(fastmath=True)
def Quench(tracks, cols, mode="box"):
    """
    This function takes as input an array of track segments and calculates
    the number of electrons that reach the anode plane after recombination.
    It is possible to pick among two models: Box (Baller, 2013 JINST 8 P08005) or
    Birks (Amoruso, et al NIM A 523 (2004) 275).
    Args:
        tracks (:obj:`numpy.array`): array containing the tracks segment information
        cols (:obj:`numba.typed.Dict`): Numba dictionary containing columns names for the track array
        mode (string, optional): recombination model. Default is "box"
    """
    for index in nb.prange(tracks.shape[0]):
        dedx = tracks[index, cols["dEdx"]]

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

        tracks[index, cols["NElectrons"]] = recomb * tracks[index, cols["dE"]] * consts.MeVToElectrons


@cuda.jit
def GPU_Quench(dEdx, dE, nElectrons):
    """
    This function takes as input an array of track segments and calculates
    the number of electrons that reach the anode plane after recombination.
    It is possible to pick among two models: Box (Baller, 2013 JINST 8 P08005) or
    Birks (Amoruso, et al NIM A 523 (2004) 275).

    Args:
        tracks (:obj:`numpy.array`): array containing the tracks segment information
        cols (:obj:`numba.typed.Dict`): Numba dictionary containing columns names for the track array
        mode (string, optional): recombination model. Default is "box"
    """
    index = cuda.grid(1)
    if index < dEdx.shape[0]:

        mode = "box"
        recomb = 0
        if mode == "box":
            # Baller, 2013 JINST 8 P08005
            csi = consts.beta * dEdx[index] / (consts.eField * consts.lArDensity)
            recomb = max(0, log(consts.alpha + csi)/csi)
        elif mode == "birks":
            # Amoruso, et al NIM A 523 (2004) 275
            recomb = consts.Ab / (1 + consts.kb * dEdx[index] / (consts.eField * consts.lArDensity))
        else:
            raise ValueError("Invalid recombination mode: must be 'box' or 'birks'")

        if isnan(recomb):
            raise RuntimeError("Invalid recombination value")

        nElectrons[index] = recomb * dE[index] * consts.MeVToElectrons
