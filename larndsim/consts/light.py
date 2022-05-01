"""
Sets ligth-related constants
"""
import yaml
import numpy as np
import os

#: Number of true segments to track for each time tick (`MAX_MC_TRUTH_IDS=0` to disable complete truth tracking)
MAX_MC_TRUTH_IDS = 0 #256
#: Threshold for propogating truth information on a given SiPM
MC_TRUTH_THRESHOLD = 0.1 # pe/us

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
LIGHT_WINDOW = (1, 10) # us

#: Fraction of total light emitted from singlet state
SINGLET_FRACTION = 0.3
#: Singlet decay time [microseconds]
TAU_S = 0.001 # us
#: Triplet decay time [microseconds]
TAU_T = 1.530 # us

#: Conversion from PE/microsecond to ADC
LIGHT_GAIN = -2.30 # ADC * us/PE
#: Set response model type (0=RLC response, 1=arbitrary input)
SIPM_RESPONSE_MODEL = 0
#: Response RC time [microseconds]
LIGHT_RESPONSE_TIME = 0.055
#: Reponse oscillation period [microseconds]
LIGHT_OSCILLATION_PERIOD = 0.095
#: Sample rate for input noise spectrum [microseconds]
LIGHT_DET_NOISE_SAMPLE_SPACING = 0.01 # us
#: Arbitrary input model (normalized to sum of 1)
IMPULSE_MODEL = np.array([1,0])
#: Arbitrary input model tick size [microseconds]
IMPULSE_TICK_SIZE = 0.001

#: Total detector light threshold [ADC]
LIGHT_TRIG_THRESHOLD = -10000
#: Light digitization window [microseconds]
LIGHT_TRIG_WINDOW = (0.9, 1.66) # us
#: Light waveform sample rate [microseconds]
LIGHT_DIGIT_SAMPLE_SPACING = 0.01 # us
#: Light digitizer bits
LIGHT_NBIT = 10

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
    global SIPM_RESPONSE_MODEL
    global LIGHT_RESPONSE_TIME
    global LIGHT_OSCILLATION_PERIOD
    global LIGHT_DET_NOISE_SAMPLE_SPACING
    global IMPULSE_MODEL
    global IMPULSE_TICK_SIZE

    global LIGHT_TRIG_THRESHOLD
    global LIGHT_TRIG_WINDOW
    global LIGHT_DIGIT_SAMPLE_SPACING
    global LIGHT_NBIT

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

        LIGHT_TICK_SIZE = float(detprop.get('light_tick_size', LIGHT_TICK_SIZE))
        LIGHT_WINDOW = tuple(detprop.get('light_window', LIGHT_WINDOW))
        assert len(LIGHT_WINDOW) == 2

        SINGLET_FRACTION = float(detprop.get('singlet_fraction', SINGLET_FRACTION))
        TAU_S = float(detprop.get('tau_s', TAU_S))
        TAU_T = float(detprop.get('tau_t', TAU_T))

        LIGHT_GAIN = np.array(detprop.get('light_gain', np.full(OP_CHANNEL_EFFICIENCY.shape, LIGHT_GAIN)))
        if LIGHT_GAIN.size == 1:
            LIGHT_GAIN = np.full(OP_CHANNEL_EFFICIENCY.shape, LIGHT_GAIN)
        assert LIGHT_GAIN.shape == OP_CHANNEL_EFFICIENCY.shape
        SIPM_RESPONSE_MODEL = int(detprop.get('sipm_response_model', SIPM_RESPONSE_MODEL))
        assert SIPM_RESPONSE_MODEL in (0,1)
        LIGHT_DET_NOISE_SAMPLE_SPACING = float(detprop.get('light_det_noise_sample_spacing', LIGHT_DET_NOISE_SAMPLE_SPACING))
        LIGHT_RESPONSE_TIME = float(detprop.get('light_response_time', LIGHT_RESPONSE_TIME))
        LIGHT_OSCILLATION_PERIOD = float(detprop.get('light_oscillation_period', LIGHT_OSCILLATION_PERIOD))
        impulse_model_filename = str(detprop.get('impulse_model', ''))
        if impulse_model_filename and SIPM_RESPONSE_MODEL == 1:
            print('Light impulse model:', impulse_model_filename)
            try:
                # first try to load from current directory
                IMPULSE_MODEL = np.load(impulse_model_filename)
            except FileNotFoundError:
                # then try from larnd-sim base directory
                try:
                    IMPULSE_MODEL = np.load(os.path.join(os.path.dirname(__file__), '../../') + impulse_model_filename)
                except FileNotFoundError:
                    print("Impulse model file not found:", impulse_model_filename)
        IMPULSE_TICK_SIZE = float(detprop.get('impulse_tick_size', IMPULSE_TICK_SIZE))

        LIGHT_TRIG_THRESHOLD = float(detprop.get('light_trig_threshold', LIGHT_TRIG_THRESHOLD))
        LIGHT_TRIG_WINDOW = tuple(detprop.get('light_trig_window', LIGHT_TRIG_WINDOW))
        assert len(LIGHT_TRIG_WINDOW) == 2
        LIGHT_DIGIT_SAMPLE_SPACING = float(detprop.get('light_digit_sample_spacing', LIGHT_DIGIT_SAMPLE_SPACING))
        LIGHT_NBIT = int(detprop.get('light_nbit', LIGHT_NBIT))

    except KeyError:
        LIGHT_SIMULATED = False
