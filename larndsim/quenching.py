"""
Module to implement the quenching of the ionized electrons
through the detector
"""

from math import log, isnan
from numba import cuda

from .consts import detector, physics, light

@cuda.jit
def quench(tracks, mode):
    """
    This CUDA kernel takes as input an array of track segments and calculates
    the number of electrons and photons that reach the anode plane after recombination.
    It is possible to pick among two models: Box (Baller, 2013 JINST 8 P08005) or
    Birks (Amoruso, et al NIM A 523 (2004) 275).

    Args:
        tracks (:obj:`numpy.ndarray`): array containing the tracks segment information
        mode (int): recombination model (physics.BOX or physics.BIRKS).
    """
    itrk = cuda.grid(1)

    if itrk < tracks.shape[0]:
        dEdx = tracks[itrk]["dEdx"]
        dE = tracks[itrk]["dE"]

        dEdx_nonIonizing = tracks[itrk]["dEdx_secondary"]
        dE_nonIonizing = tracks[itrk]["dE_secondary"]

        dEdx_ionizing = dEdx - dEdx_nonIonizing
        dE_ionizing = dE - dE_nonIonizing

        recomb = 0
        if mode == physics.BOX:
            # Baller, 2013 JINST 8 P08005
            csi = physics.BOX_BETA * dEdx_ionizing / (detector.E_FIELD * detector.LAR_DENSITY)
            recomb = max(0, log(physics.BOX_ALPHA + csi)/csi)
        elif mode == physics.BIRKS:
            # Amoruso, et al NIM A 523 (2004) 275
            recomb = physics.BIRKS_Ab / (1 + physics.BIRKS_kb * dEdx_ionizing / (detector.E_FIELD * detector.LAR_DENSITY))
        else:
            raise ValueError("Invalid recombination mode: must be 'physics.BOX' or 'physics.BIRKS'")

        if isnan(recomb):
            raise RuntimeError("Invalid recombination value")

        tracks[itrk]["n_electrons"] = recomb * dE_ionizing / physics.W_ION
        tracks[itrk]["n_photons"] = (dE/light.W_PH - tracks[itrk]["n_electrons"]) * light.SCINT_PRESCALE
