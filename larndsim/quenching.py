"""
Module to implement the quenching of the ionized electrons
through the detector
"""

from math import log, isnan, sqrt
from numba import cuda

from .consts import detector, physics, light

@cuda.jit
def BOX(dEdx):
    """
    Box recombination model. Baller, 2013 JINST 8 P08005
    
    Args:
        dEdx (float): Segment dE/dx in MeV/cm
    """
    csi = physics.BOX_BETA * dEdx / (detector.E_FIELD * detector.LAR_DENSITY)
    return max(0, log(physics.BOX_ALPHA + csi)/csi)

@cuda.jit
def BIRKS(dEdx):
    """
    Birks recombination model. Amoruso, et al NIM A 523 (2004) 275
    
    Args:
        dEdx (float): Segment dE/dx in MeV/cm
    """
    return physics.BIRKS_Ab / (1 + physics.BIRKS_kb * dEdx / (detector.E_FIELD * detector.LAR_DENSITY))

@cuda.jit
def NEST_ER(E, er_energies, er_recomb_factors):
    """
    LArNEST electron recoil (ER) recombination model used for low-energy electrons. https://github.com/NESTCollaboration/larnestpy
    
    Args:
        E (float): Starting energy in MeV of the trajectory corresponding to the current segment.
        er_energies (:obj:`numpy.ndarray`): ER energies from LArNEST.
        er_recomb_factors (:obj:`numpy.ndarray`): ER recombination factors from LArNEST.
    """
    recomb = linear_interpolation(E, er_energies, er_recomb_factors, 0.0)
    return recomb

@cuda.jit
def NEST_ALPHA():
    """
    LArNEST alpha recombination model. https://github.com/NESTCollaboration/larnestpy
    
    """
    ALPHA_R_FACTOR = 0.01848
    return ALPHA_R_FACTOR

@cuda.jit
def linear_interpolation(x, x_data, y_data, default_value):
    """
    Linearly interpolation function.
    
    Args:
        x (float): value to evaluate interpolation at.
        x_data (:obj:`numpy.ndarray`): x data 
        y_data (:obj:`numpy.ndarray`): y data
        default_value (float): value to return if x is out of range.
    """
    for i in range(x_data.shape[0] - 1):
        if x_data[i] <= x <= x_data[i + 1]:
            t = (x - x_data[i]) / (x_data[i + 1] - x_data[i])
            return (1 - t) * y_data[i] + t * y_data[i + 1]
    return default_value  
    
@cuda.jit
def quench(tracks, mode, recomb_energies=None, Ne_yields=None, Nph_yields=None):
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
        # TODO: ensure 'dE' is visible energy (may require subtracting off the SecondaryEnergyDeposit
        #       calculated by edepsim that accounts for non-ionizing energy loss)
        
        # initialize to zero
        tracks[itrk]["n_electrons"] = 0
        tracks[itrk]["n_photons"]   = 0

        # if this isn't a gamma or neutron, calculate charge/light
        pdg = tracks[itrk]["pdg_id"]
        use_nest = False
        if dEdx > 0 and pdg != 22 and pdg != 2112:
            recomb = 0
            if mode == physics.BOX:
                # Baller, 2013 JINST 8 P08005
                csi = physics.BOX_BETA * dEdx / (detector.E_FIELD * detector.LAR_DENSITY)
                recomb = max(0, log(physics.BOX_ALPHA + csi)/csi)
            elif mode == physics.BIRKS:
                # Amoruso, et al NIM A 523 (2004) 275
                recomb = physics.BIRKS_Ab / (1 + physics.BIRKS_kb * dEdx / (detector.E_FIELD * detector.LAR_DENSITY))
            elif mode == physics.NEST_ER or pdg == 11:
                # Starting energy needed for NEST  electron recoil recombination calculation
                p_start = tracks[itrk]['p_mag_traj_start']
                energy = sqrt(0.511*0.511 + p_start*p_start) - 0.511
                Ne, Nph = NEST_ER(energy, recomb_energies, Ne_yields), NEST_ER(energy, recomb_energies, Nph_yields)
                tracks[itrk]["n_electrons"] = Ne * dE / energy
                tracks[itrk]["n_photons"] = Nph * dE / energy
                use_nest = True
            elif mode == physics.NEST_ALPHA or pdg == 1000020040: # alpha particle PDG code
                recomb = NEST_ALPHA()
            else:
                raise ValueError("Invalid recombination mode: must be 'physics.BOX', 'physics.BIRKS', 'physics.NEST_ER', or 'physics.NEST_ALPHA'")
                recomb = NEST_ALPHA()
            if isnan(recomb):
                raise RuntimeError("Invalid recombination value")
                
            if not use_nest:
                tracks[itrk]["n_electrons"] = recomb * dE / physics.W_ION
                tracks[itrk]["n_photons"]   = (dE/light.W_PH - tracks[itrk]["n_electrons"]) * light.SCINT_PRESCALE
