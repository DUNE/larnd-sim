"""
Module to implement the quenching of the ionized electrons
through the detector
"""

from math import log, isnan, sqrt
from numba import cuda

from .consts import detector, physics, light, pdg

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
    recomb = linear_interpolation(E, er_energies, er_recomb_factors, physics.ER_ASYMPTOTE_AVG)
    return recomb

@cuda.jit
def NEST_NR(E, nr_energies, nr_recomb_factors):
    """
    LArNEST nuclear recoil (NR) recombination model. https://github.com/NESTCollaboration/larnestpy
    
    Args:
        E (float): Segment dE in MeV.
    """
    recomb = linear_interpolation(E, nr_energies, nr_recomb_factors, physics.NR_ASYMPTOTE_AVG)
    return recomb

@cuda.jit
def NEST_ALPHA(E):
    """
    LArNEST alpha recombination model. https://github.com/NESTCollaboration/larnestpy
    
    Args:
        E (float): Segment dE in MeV.
    """
    return physics.ALPHA_R_FACTOR


@cuda.jit
def DEFAULT_MODEL(default_model, dEdx):
    """
    Function for calling the default model and by pass the if statements. Note only two default models
    
    Args:
        default_model (int): integer corresponding to the default recombination model
    """
    recomb = 0
    if default_model == physics.BOX:
        recomb = BOX(dEdx)
    elif default_model == physics.BIRKS:
        recomb = BIRKS(dEdx)
    else:
        raise ValueError("Only `physics.BOX` and `physics.BIRKS` supported as default recombination models.")
    return recomb

@cuda.jit
def pick_model(model, E, dEdx, er_energy_threshold, default_model, use_default_model,\
               E_ER, E_NR, R_ER, R_NR):
    """
    Function to pick a recombination model for a particular segment. 
    
    Args:
        model (int): recombination model number, as defined in consts.physics.
        E (float): energy in MeV, either corresponding to particle starting energy or segment dE
        dEdx (float): segment dE/dx in MeV/cm
        er_energy_threshold (float): threshold energy in MeV for using NEST ER model 
            (NEST ER is used if particle energy is less than this threshold). Only relevant for electrons. but must still be specified.
        default_model (int): number corresponding to the model that should be used 
            if the segment pdgID is not in the pdg->model dictionary in the simulation properties file.
        use_default_model (bool): if True, bypasses the if statements and always uses the default model.
    
    """
    recomb = 0
    if use_default_model:
        recomb = DEFAULT_MODEL(default_model, dEdx)
    elif model == physics.BOX:
        recomb = BOX(dEdx)
    elif model == physics.BIRKS:
        recomb = BIRKS(dEdx)
    elif model == physics.NEST_ER: 
        recomb = NEST_ER(E, E_ER, R_ER)
    elif model == physics.NEST_ALPHA:
        recomb = NEST_ALPHA(E)
    elif model == physics.NEST_NR:
        recomb = NEST_NR(E, E_NR, R_NR)
    else:
        recomb = DEFAULT_MODEL(default_model, dEdx)
        
    return recomb

@cuda.jit
def find_index(array_to_search, value):
    """
    Find index of value in array.
    
    Args:
        array_to_search (:obj:`numpy.ndarray`): array in which to search for the index of `value`.
        value (int): value to search for in `array_to_search`.
    """
    for i, element in enumerate(array_to_search):
        if element == value:
            return i
    return -1

@cuda.jit
def linear_interpolation(x, x_data, y_data, default_value):
    """
    Linearly interpolation of plotting data, x and y.
    
    Args:
        x (float): value to evaluate interpolation at.
        x_data (:obj:`numpy.ndarray`): x data of graph.
        y_data (:obj:`numpy.ndarray`): y data of graph.
        default_value (float): value to return if x is out of range.
    """
    for i in range(x_data.shape[0] - 1):
        if x_data[i] <= x <= x_data[i + 1]:
            # Compute the interpolation
            t = (x - x_data[i]) / (x_data[i + 1] - x_data[i])
            return (1 - t) * y_data[i] + t * y_data[i + 1]
    return default_value  

@cuda.jit
def quench(tracks, d_pdg_codes, d_model_codes, er_energy_threshold, default_model,\
          E_ER, E_NR, R_ER, R_NR):
    """
    This CUDA kernel takes as input an array of track segments and calculates
    the number of electrons and photons that reach the anode plane after recombination.
    It is possible to pick among two models: Box (Baller, 2013 JINST 8 P08005) or
    Birks (Amoruso, et al NIM A 523 (2004) 275).

    Args:
        tracks (:obj:`numpy.ndarray`): array containing the tracks segment information
        d_pdg_codes (:obj:`numpy.ndarray`): array containing pdg codes from the `pdg_to_recombination_model` dictionary 
            in simulation properties file.
        d_model_codes (:obj:`numpy.ndarray`): array containing pdg codes from the `pdg_to_recombination_model` dictionary 
            in simulation properties file.
        er_energy_threshold (float): energy threshold for using the NEST ER model.
        default_model (int): default recombination model.
    """
    itrk = cuda.grid(1)
    
    if itrk < tracks.shape[0]:
        dEdx = tracks[itrk]["dEdx"]
        dE = tracks[itrk]["dE"]
        pdgID = tracks[itrk]['pdg_id']

        recomb, energy = 0, 0
        USE_DEFAULT_MODEL=False
        if physics.USE_DEFAULT_MODEL:
            USE_DEFAULT_MODEL=True
        if physics.USE_DEFAULT_MODEL:
            model = -1
        else:
            # get energy value if relevant for this segment
            if abs(pdgID) == pdg.electron:
                p_start = tracks[itrk]['p_mag_traj_start']
                energy = sqrt(0.511*0.511 + p_start*p_start) - 0.511
                if energy > er_energy_threshold:
                    model = default_model
                    USE_DEFAULT_MODEL=True
            elif pdgID == pdg.alpha:
                energy = dE

            # look up the model to use for this segment
            model = -1
            index_of_model = find_index(d_pdg_codes, abs(pdgID))
            if index_of_model != -1:
                model = d_model_codes[index_of_model]
            if pdgID > 1000000000 and pdgID != pdg.alpha and pdgID != pdg.deuteron:
                model = 5 # NR
                energy = dE
        recomb = pick_model(model, energy, dEdx, er_energy_threshold, default_model, USE_DEFAULT_MODEL,\
                           E_ER, E_NR, R_ER, R_NR)
        
        if isnan(recomb):
            raise RuntimeError("Invalid recombination value")
        #print('Using model: ', model, ", ", 'Recombination value = ', recomb)
        tracks[itrk]["n_electrons"] = recomb * dE / physics.W_ION
        tracks[itrk]["n_photons"] = (dE/light.W_PH - tracks[itrk]["n_electrons"]) * light.SCINT_PRESCALE
