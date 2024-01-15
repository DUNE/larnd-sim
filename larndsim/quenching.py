"""
Module to implement the quenching of the ionized electrons
through the detector
"""

from math import log, isnan
from numba import cuda
#import random

from .consts import detector, physics, light

@cuda.jit
def random_value_from_distribution(x_vals, normalized_distribution):
    return np.random.choice(x_vals, p=normalized_distribution/np.sum(normalized_distribution))

@cuda.jit
def MIP_dEdx(mc_array):
    bin_centers = mc_array[:,0]
    fitted_data = mc_array[:,1]
    random_sample = random_value_from_distribution(bin_centers, fitted_data)
    return random_sample

@cuda.jit
def quench(tracks, mode, recomb_file):
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

        recomb = 0
        if mode == physics.BOX:
            # Baller, 2013 JINST 8 P08005
            csi = physics.BOX_BETA * dEdx / (detector.E_FIELD * detector.LAR_DENSITY)
            recomb = max(0, log(physics.BOX_ALPHA + csi)/csi)
        elif mode == physics.BIRKS:
            # Amoruso, et al NIM A 523 (2004) 275
            moyald = MIP_dEdx(recomb_file)
            recomb = physics.BIRKS_Ab / (1 + physics.BIRKS_kb * moyald / (detector.E_FIELD * detector.LAR_DENSITY))
            #recomb = physics.BIRKS_Ab / (1 + physics.BIRKS_kb * dEdx / (detector.E_FIELD * detector.LAR_DENSITY))
            #recomb = physics.BIRKS_Ab / (1 + physics.BIRKS_kb * 1.85 / (detector.E_FIELD * detector.LAR_DENSITY))
        else:
            raise ValueError("Invalid recombination mode: must be 'physics.BOX' or 'physics.BIRKS'")

        if isnan(recomb):
            raise RuntimeError("Invalid recombination value")

        tracks[itrk]["n_electrons"] = recomb * dE / physics.W_ION
        tracks[itrk]["n_photons"] = (dE/light.W_PH - tracks[itrk]["n_electrons"]) * light.SCINT_PRESCALE
