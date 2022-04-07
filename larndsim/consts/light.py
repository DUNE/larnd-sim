"""
Sets ligth-related constants
"""
import yaml
import numpy as np

LUT_VOX_DIV = np.zeros(0)
N_OP_CHANNEL = 0
LIGHT_SIMULATED = True
OP_CHANNEL_EFFICIENCY = np.zeros(0)
#: Prescale factor analogous to ScintPreScale in LArSoft FIXME
SCINT_PRESCALE = 1
#: Ion + excitation work function in `MeV`
W_PH = 19.5e-6 # MeV

#: Step size for light simulation [microseconds]
LIGHT_TICK_SIZE = 0.01 # us
#: Pre- and post-window for light simulation [microseconds]
LIGHT_WINDOW = (0,10) # us

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

    global LIGHT_TICK_SIZE
    global LIGHT_WINDOW

    with open(detprop_file) as df:
        detprop = yaml.load(df, Loader=yaml.FullLoader)

    try:
        LUT_VOX_DIV = np.array(detprop['lut_vox_div'])
        N_OP_CHANNEL = detprop['n_op_channel']
        OP_CHANNEL_EFFICIENCY = np.array(detprop['op_channel_efficiency'])

        LIGHT_TICK_SIZE = detprop.get('light_tick_size', LIGHT_TICK_SIZE)
        LIGHT_WINDOW = detprop.get('light_window', LIGHT_WINDOW)
    except KeyError:
        LIGHT_SIMULATED = False
