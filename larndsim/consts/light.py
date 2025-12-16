"""
Sets ligth-related constants
"""
import yaml
import numpy as np
import os
import numbers

from . import detector

LIGHT_SIMULATED = True

ENABLE_LUT_SMEARING = False

N_OP_CHANNEL = 0
OP_CHANNEL_EFFICIENCY = np.ones(0)
OP_CHANNEL_TO_TPC = np.zeros(0)
TPC_TO_OP_CHANNEL = np.zeros((0,0))

#: PDE correction factors (data/MC efficiency ratios) applied to LUT visibility
#: Shape: (n_tpc, n_detectors_per_tpc) - one correction per detector unit
#: Each correction is applied uniformly to all channels in that detector
#: (e.g., all 6 channels in an ACL, or both channels in an LCM)
#: Internally expanded to per-channel array for kernel use
#: Default value of 1.0 means no correction
PDE_CORRECTION_2D = np.ones((0,0))  # 2D array: (n_tpc, n_detectors_per_tpc)
OP_CHANNEL_PDE_CORRECTION = np.ones(0)  # 1D expanded version for kernel (n_op_channel,)
#: Detector to channel mapping: list of (tpc_id, detector_id, [channel_ids]) for each detector unit
#: Used to expand 2D PDE corrections to per-channel array
#: Format: [(tpc, det, [ch0, ch1, ...]), (tpc, det, [ch0, ..., ch5]), ...]
DETECTOR_CHANNEL_MAP = []
#: Enable PDE correction feature
ENABLE_PDE_CORRECTION = False

#: Prescale factor analogous to ScintPreScale in LArSoft FIXME
SCINT_PRESCALE = 1
#: Ion + excitation work function in `MeV`
W_PH = 19.5e-6 # MeV

#: Step size for light simulation [microseconds]
LIGHT_TICK_SIZE = 0.001 # us
#: Pre- and post-window for light simulation [microseconds]
LIGHT_WINDOW = (1, 10) # us

#: Use triple exponential scintillation model (default: False for double exponential)
USE_TRIPLE_EXPONENTIAL = False

#: Double exponential model parameters
#: Fraction of total light emitted from singlet state
SINGLET_FRACTION = 0.3
#: Singlet decay time [microseconds]
TAU_S = 0.001 # us
#: Triplet decay time [microseconds]
TAU_T = 1.530

#: Triple exponential model parameters (used when USE_TRIPLE_EXPONENTIAL = True)
#: Fast component fraction
FAST_FRACTION = 0.3
#: Fast component decay time [microseconds]
TAU_FAST = 0.001 # us
#: Intermediate component fraction
INTERMEDIATE_FRACTION = 0.0
#: Intermediate component decay time [microseconds]
TAU_INTERMEDIATE = 0.1 # us (placeholder, needs tuning)
#: Slow component fraction (calculated as 1 - FAST_FRACTION - INTERMEDIATE_FRACTION)
#: Slow component decay time [microseconds]
TAU_SLOW = 1.530 # us

#: Conversion from PE/microsecond to ADC
DEFAULT_LIGHT_GAIN = -2.30 # ADC * us/PE
LIGHT_GAIN = np.zeros((0,))
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
IMPULSE_TICK_SIZE = 0.01 # 10 ns a tick for the response measurement

#: Number of SiPMs per detector (used by trigger)
OP_CHANNEL_PER_TRIG = 6
#: Light trigger mode (0: threshold each module, 1: beam and threshold)
LIGHT_TRIG_MODE = 0
#: Total detector light threshold [ADC] (one value for every OP_CHANNEL_PER_TRIG detector sum)
LIGHT_TRIG_THRESHOLD = np.zeros((0,))
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
    global LIGHT_SIMULATED

    global N_OP_CHANNEL
    global OP_CHANNEL_EFFICIENCY
    global OP_CHANNEL_TO_TPC
    global TPC_TO_OP_CHANNEL
    global PDE_CORRECTION_2D
    global OP_CHANNEL_PDE_CORRECTION
    global DETECTOR_CHANNEL_MAP
    global ENABLE_PDE_CORRECTION

    global ENABLE_LUT_SMEARING
    global LIGHT_TICK_SIZE
    global LIGHT_WINDOW

    global USE_TRIPLE_EXPONENTIAL
    global SINGLET_FRACTION
    global TAU_S
    global TAU_T
    global FAST_FRACTION
    global TAU_FAST
    global INTERMEDIATE_FRACTION
    global TAU_INTERMEDIATE
    global TAU_SLOW

    global LIGHT_GAIN
    global SIPM_RESPONSE_MODEL
    global LIGHT_RESPONSE_TIME
    global LIGHT_OSCILLATION_PERIOD
    global LIGHT_DET_NOISE_SAMPLE_SPACING
    global IMPULSE_MODEL
    global IMPULSE_TICK_SIZE

    global OP_CHANNEL_PER_TRIG
    global LIGHT_TRIG_MODE
    global LIGHT_TRIG_THRESHOLD
    global LIGHT_TRIG_WINDOW
    global LIGHT_DIGIT_SAMPLE_SPACING
    global LIGHT_NBIT

    with open(detprop_file) as df:
        detprop = yaml.load(df, Loader=yaml.FullLoader)

    try:
        LIGHT_SIMULATED = bool(detprop.get('light_simulated', LIGHT_SIMULATED))

        mod_ids = detector.get_n_modules(detprop_file)
        n_tpc = len(mod_ids)*2
        N_OP_CHANNEL = detprop['n_op_channel']
        if N_OP_CHANNEL % n_tpc != 0:
            raise ValueError("N_OP_CHANNEL should be a multiple of n_tpc.")
        if N_OP_CHANNEL % OP_CHANNEL_PER_TRIG != 0:
            raise ValueError("N_OP_CHANNEL should be a multiple of number of SiPM per light unit (The default is 6).")
        OP_CHANNEL_EFFICIENCY = np.array(detprop.get('op_channel_efficiency', OP_CHANNEL_EFFICIENCY))
        if OP_CHANNEL_EFFICIENCY.size == 1:
            OP_CHANNEL_EFFICIENCY = np.full(N_OP_CHANNEL, OP_CHANNEL_EFFICIENCY)

        # Load PDE correction factors (data/MC efficiency ratios)
        # Expected shape: (n_tpc, n_detectors_per_tpc) where detectors may have varying channel counts (2 or 6)
        ENABLE_PDE_CORRECTION = bool(detprop.get('enable_pde_correction', ENABLE_PDE_CORRECTION))
        pde_correction_file = str(detprop.get('pde_correction_file', ''))

        # Build or load detector-to-channel mapping
        # This handles mixed detector types (e.g., 2-channel LCMs and 6-channel ACLs)
        detector_map_config = detprop.get('detector_channel_map', None)

        if detector_map_config:
            # Load explicit mapping from YAML: [(tpc, det, [channels]), ...]
            DETECTOR_CHANNEL_MAP = [(entry['tpc'], entry['detector'], entry['channels'])
                                     for entry in detector_map_config]
        else:
            # Infer mapping assuming uniform OP_CHANNEL_PER_TRIG channels per detector
            # This is a fallback - explicit mapping is preferred for mixed detector types
            DETECTOR_CHANNEL_MAP = []
            for itpc in range(n_tpc):
                tpc_channels = TPC_TO_OP_CHANNEL[itpc]
                n_detectors_this_tpc = len(tpc_channels) // OP_CHANNEL_PER_TRIG
                for idet in range(n_detectors_this_tpc):
                    det_channels = tpc_channels[idet*OP_CHANNEL_PER_TRIG:(idet+1)*OP_CHANNEL_PER_TRIG]
                    DETECTOR_CHANNEL_MAP.append((itpc, idet, list(det_channels)))

        # Determine 2D shape from detector map
        n_detectors_per_tpc = max([det for tpc, det, _ in DETECTOR_CHANNEL_MAP]) + 1 if DETECTOR_CHANNEL_MAP else 0

        if ENABLE_PDE_CORRECTION and pde_correction_file:
            # Load from file (numpy array with shape (n_tpc, n_detectors_per_tpc))
            try:
                # First try to load from current directory
                PDE_CORRECTION_2D = np.load(pde_correction_file)
            except FileNotFoundError:
                # Then try from larnd-sim base directory
                try:
                    PDE_CORRECTION_2D = np.load(os.path.join(os.path.dirname(__file__), '../../') + pde_correction_file)
                except FileNotFoundError:
                    print("PDE correction file not found:", pde_correction_file, ", using default correction of 1.0")
                    PDE_CORRECTION_2D = np.ones((n_tpc, n_detectors_per_tpc))
        else:
            # Load from YAML or use default
            pde_corr_yaml = detprop.get('pde_correction_2d', None)
            if pde_corr_yaml is None:
                PDE_CORRECTION_2D = np.ones((n_tpc, n_detectors_per_tpc))
            else:
                pde_corr_yaml = np.array(pde_corr_yaml)
                if pde_corr_yaml.size == 1:
                    # Single value: apply to all detectors
                    PDE_CORRECTION_2D = np.full((n_tpc, n_detectors_per_tpc), pde_corr_yaml)
                else:
                    PDE_CORRECTION_2D = pde_corr_yaml.reshape(n_tpc, n_detectors_per_tpc)

        # Validate 2D shape
        if PDE_CORRECTION_2D.shape[0] != n_tpc:
            raise ValueError(f"PDE correction array has {PDE_CORRECTION_2D.shape[0]} TPCs, expected {n_tpc}")

        # Expand 2D array to 1D per-channel array using detector-channel mapping
        # This correctly handles mixed detector types (2-ch and 6-ch)
        OP_CHANNEL_PDE_CORRECTION = np.ones(N_OP_CHANNEL)
        for tpc_id, det_id, channels in DETECTOR_CHANNEL_MAP:
            correction = PDE_CORRECTION_2D[tpc_id, det_id]
            for ch in channels:
                OP_CHANNEL_PDE_CORRECTION[ch] = correction

        try:
            tpc_to_op_channel = detprop['tpc_to_op_channel']
            OP_CHANNEL_TO_TPC = np.zeros((N_OP_CHANNEL,), int)
            TPC_TO_OP_CHANNEL = np.zeros((len(tpc_to_op_channel), len(tpc_to_op_channel[0])), int)
            for itpc in range(len(tpc_to_op_channel)):
                TPC_TO_OP_CHANNEL[itpc] = np.array(tpc_to_op_channel[itpc])
                for idet in tpc_to_op_channel[itpc]:
                    OP_CHANNEL_TO_TPC[idet] = itpc
        except:
            n_op_per_tpc = int(N_OP_CHANNEL/n_tpc)
            OP_CHANNEL_TO_TPC = np.zeros((N_OP_CHANNEL,), int)
            TPC_TO_OP_CHANNEL = np.zeros((n_tpc, n_op_per_tpc), int)
            for itpc in range(n_tpc):
                TPC_TO_OP_CHANNEL[itpc] = np.arange(itpc*n_op_per_tpc, (itpc+1)*n_op_per_tpc)
                for idet in TPC_TO_OP_CHANNEL[itpc]:
                    OP_CHANNEL_TO_TPC[idet] = itpc

        ENABLE_LUT_SMEARING = bool(detprop.get('enable_lut_smearing', ENABLE_LUT_SMEARING))
        LIGHT_TICK_SIZE = float(detprop.get('light_tick_size', LIGHT_TICK_SIZE))
        LIGHT_WINDOW = tuple(detprop.get('light_window', LIGHT_WINDOW))
        assert len(LIGHT_WINDOW) == 2

        USE_TRIPLE_EXPONENTIAL = bool(detprop.get('use_triple_exponential', USE_TRIPLE_EXPONENTIAL))

        # Double exponential model parameters
        SINGLET_FRACTION = float(detprop.get('singlet_fraction', SINGLET_FRACTION))
        TAU_S = float(detprop.get('tau_s', TAU_S))
        TAU_T = float(detprop.get('tau_t', TAU_T))

        # Triple exponential model parameters
        FAST_FRACTION = float(detprop.get('fast_fraction', FAST_FRACTION))
        TAU_FAST = float(detprop.get('tau_fast', TAU_FAST))
        INTERMEDIATE_FRACTION = float(detprop.get('intermediate_fraction', INTERMEDIATE_FRACTION))
        TAU_INTERMEDIATE = float(detprop.get('tau_intermediate', TAU_INTERMEDIATE))
        TAU_SLOW = float(detprop.get('tau_slow', TAU_SLOW))

        LIGHT_GAIN = np.array(detprop.get('light_gain', [DEFAULT_LIGHT_GAIN]))
        if LIGHT_GAIN.size == 1:
            LIGHT_GAIN = np.full(N_OP_CHANNEL, LIGHT_GAIN)
        assert LIGHT_GAIN.shape == OP_CHANNEL_EFFICIENCY.shape
        SIPM_RESPONSE_MODEL = int(detprop.get('sipm_response_model', SIPM_RESPONSE_MODEL))
        assert SIPM_RESPONSE_MODEL in (0,1)
        LIGHT_DET_NOISE_SAMPLE_SPACING = float(detprop.get('light_det_noise_sample_spacing', LIGHT_DET_NOISE_SAMPLE_SPACING))
        LIGHT_RESPONSE_TIME = float(detprop.get('light_response_time', LIGHT_RESPONSE_TIME))
        LIGHT_OSCILLATION_PERIOD = float(detprop.get('light_oscillation_period', LIGHT_OSCILLATION_PERIOD))
        impulse_model_filename = str(detprop.get('impulse_model', ''))
        if impulse_model_filename and SIPM_RESPONSE_MODEL == 1:
            try:
                # first try to load from current directory
                IMPULSE_MODEL = np.load(impulse_model_filename)
            except FileNotFoundError:
                # then try from larnd-sim base directory
                try:
                    IMPULSE_MODEL = np.load(os.path.join(os.path.dirname(__file__), '../../') + impulse_model_filename)
                except FileNotFoundError:
                    SIPM_RESPONSE_MODEL = 0
                    print("Impulse model file not found:", impulse_model_filename, ", and setting SIPM_RESPONSE_MODEL to 0 (RLC model).")
        IMPULSE_TICK_SIZE = float(detprop.get('impulse_tick_size', IMPULSE_TICK_SIZE))

        OP_CHANNEL_PER_TRIG = int(detprop.get('op_channel_per_det', OP_CHANNEL_PER_TRIG))
        LIGHT_TRIG_MODE = int(detprop.get('light_trig_mode', LIGHT_TRIG_MODE))
        assert LIGHT_TRIG_MODE in (0,1)
        # One threshold for an ArCLight unit or three LCM units (6 SiPMs each)
        # A single threshold for all channels
        if isinstance(detprop['light_trig_threshold'], (float, int)):
            LIGHT_TRIG_THRESHOLD = np.full(N_OP_CHANNEL // OP_CHANNEL_PER_TRIG, float(detprop['light_trig_threshold']))
        # Assuming the same threshold is applied for all ACL and another one for all LCM
        elif isinstance(detprop['light_trig_threshold'], list) and len(detprop['light_trig_threshold']) == 2:
            LIGHT_TRIG_THRESHOLD = np.tile(np.array(detprop['light_trig_threshold'], dtype=float), N_OP_CHANNEL // OP_CHANNEL_PER_TRIG)
        else:
            LIGHT_TRIG_THRESHOLD = np.array(detprop['light_trig_threshold'], dtype=float)
            if len(LIGHT_TRIG_THRESHOLD) != (N_OP_CHANNEL // OP_CHANNEL_PER_TRIG):
                raise ValueError("The light_trig_threshold is provided as a list but with a length not matched with n_op_channel.")
        LIGHT_TRIG_WINDOW = tuple(detprop.get('light_trig_window', LIGHT_TRIG_WINDOW))
        assert len(LIGHT_TRIG_WINDOW) == 2
        LIGHT_DIGIT_SAMPLE_SPACING = float(detprop.get('light_digit_sample_spacing', LIGHT_DIGIT_SAMPLE_SPACING))
        LIGHT_NBIT = int(detprop.get('light_nbit', LIGHT_NBIT))



    except KeyError:
        LIGHT_SIMULATED = False
        LIGHT_TRIG_MODE = int(detprop.get('light_trig_mode', LIGHT_TRIG_MODE))
        assert LIGHT_TRIG_MODE in (0,1)
