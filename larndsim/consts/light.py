"""
Sets ligth-related constants
"""
import yaml
import numpy as np

LUT_VOX_DIV = np.zeros(0)
N_OP_CHANNEL = 0
LIGHT_SIMULATED = True
OP_CHANNEL_EFFICIENCY = np.zeros(0)
OP_CHANNEL_TO_TPC = np.zeros(0)
#: Prescale factor analogous to ScintPreScale in LArSoft FIXME
SCINT_PRESCALE = 1
#: Ion + excitation work function in `MeV`
W_PH = 19.5e-6 # MeV

#: Step size for light simulation [microseconds]
LIGHT_TICK_SIZE = 0.005 # us
#: Pre- and post-window for light simulation [microseconds]
LIGHT_WINDOW = (0.1,10) # us

#: Fraction of total light emitted from singlet state
SINGLET_FRACTION = 0.3
#: Singlet decay time [microseconds]
TAU_S = 0.001 # us
#: Triplet decay time [microseconds]
TAU_T = 1.530

#: Conversion from PE to ADC
LIGHT_GAIN = 80 # ADC/PE
#: Response RC time [microseconds]
LIGHT_RESPONSE_TIME = 0.055
#: Reponse oscillation period [microseconds]
LIGHT_OSCILLATION_PERIOD = 0.095
LIGHT_DET_NOISE_SAMPLE_SPACING = 0.01 # us


def set_light_properties(detprop_file):
    """
    The function loads the detector properties YAML file
    and stores the light-related constants as global variables

    Args:
        detprop_file (str): detector properties YAML filename

    """
    global LUT_VOX_DIV
    global N_OP_CHANNEL
    global LIGHT_SIMULATED
    global OP_CHANNEL_EFFICIENCY
    global OP_CHANNEL_TO_TPC

    global LIGHT_TICK_SIZE
    global LIGHT_WINDOW
    
    global SINGLET_FRACTION
    global TAU_S
    global TAU_T
    
    global LIGHT_GAIN
    global LIGHT_DET_NOISE_SAMPLE_SPACING

    with open(detprop_file) as df:
        detprop = yaml.load(df, Loader=yaml.FullLoader)

    try:
        LUT_VOX_DIV = np.array(detprop['lut_vox_div'])
        N_OP_CHANNEL = detprop['n_op_channel']
        OP_CHANNEL_EFFICIENCY = np.array(detprop['op_channel_efficiency'])
        
        tpc_to_op_channel = detprop['tpc_to_op_channel']
        OP_CHANNEL_TO_TPC = np.zeros((N_OP_CHANNEL,), int)
        for itpc in range(len(tpc_to_op_channel)):
            for idet in range(N_OP_CHANNEL):
                if idet in tpc_to_op_channel[itpc]:
                    OP_CHANNEL_TO_TPC[idet] = itpc

        LIGHT_TICK_SIZE = detprop.get('light_tick_size', LIGHT_TICK_SIZE)
        LIGHT_WINDOW = detprop.get('light_window', LIGHT_WINDOW)
        
        SINGLET_FRACTION = detprop.get('singlet_fraction', SINGLET_FRACTION)
        TAU_S = detprop.get('tau_s', TAU_S)
        TAU_T = detprop.get('tau_t', TAU_T)
        
        LIGHT_GAIN = np.array(detprop.get('light_gain', np.full(OP_CHANNEL_EFFICIENCY.shape, LIGHT_GAIN)))
        LIGHT_DET_NOISE_SAMPLE_SPACING = detprop.get('light_det_noise_sample_spacing', LIGHT_DET_NOISE_SAMPLE_SPACING)
    except KeyError:
        LIGHT_SIMULATED = False
