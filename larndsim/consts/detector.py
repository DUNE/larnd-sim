"""
Set detector constants
"""
import warnings

import numpy as np
import cupy as cp
import yaml

from collections import defaultdict

from .units import mm, cm, mV, V, kV, e

###################
# LArTPC drift
###################
#: Detector temperature in K
TEMPERATURE = 87.17
#: Liquid argon density in :math:`g/cm^3`
LAR_DENSITY = 1.38 # g/cm^3
#: Electric field magnitude in :math:`kV/cm`
E_FIELD = 0.50 # kV/cm
#: Drift velocity in :math:`cm/\mu s`
V_DRIFT = 0.1648 # cm / us,
#: Electron mobility constants
ELECTRON_MOBILITY_PARAMS = 551.6, 7158.3, 4440.43, 4.29, 43.63, 0.2053
#: Electron lifetime in :math:`\mu s`
ELECTRON_LIFETIME = 2.2e3 # us,
#: Longitudinal diffusion coefficient in :math:`cm^2/\mu s`
LONG_DIFF = 4.0e-6 # cm * cm / us
#: Transverse diffusion coefficient in :math:`cm^2/\mu s`
TRAN_DIFF = 8.8e-6 # cm * cm / us

###################
# TPC geometry
###################
#: TPC drift length in :math:`cm`
DRIFT_LENGTH = 0
#: Time for the full drift calculated from the nominal drift velocity
DRIFT_MAX_TIME = 0
#: Borders of each TPC volume in :math:`cm`
TPC_BORDERS = np.zeros((0, 3, 2))
#: TPC offsets wrt the origin in :math:`cm`
TPC_OFFSETS = np.zeros((0, 3, 2))
#: Pixel tile borders in :math:`cm`
TILE_BORDERS = np.zeros((2,2))

###################
# LArPix
###################
#: Time sampling in :math:`\mu s`
TIME_SAMPLING = 0.1 # us
#: Time sampling in the pixel response file in :math:`\mu s`
RESPONSE_SAMPLING = 0.05
#: Spatial sampling in the pixel reponse file in :math:`cm`
RESPONSE_BIN_SIZE = 0.04434
#: The longest cathode charge response in time :math:`\mu s`
RESPONSE_MAX_TIME = 0.0
#: The maximum radius to consider the neighbouring charge response
MAX_RADIUS = 4
#: The step size to chop up segments :math:`cm`
#: MIN_STEP_SIZE should be comparable to the smallest bin size in x,y,t of the response file
#: The bin size in x, y is ~0.04 cm (1/10 of a pixel size),
#: and the bin size in t is 50 ns, which is roughly 0.007 cm
MIN_STEP_SIZE = 0.001 # allow per response bin to have a "reasonable" diffusion distribution
#: Default value for pixel_plane, to indicate out-of-bounds edep
DEFAULT_PLANE_INDEX = 0x0000BEEF
#: Total number of pixels
N_PIXELS = 0, 0
#: Number of pixels in each tile
N_PIXELS_PER_TILE = 0, 0
#: Dictionary between pixel ID and its position in the pixel array
PIXEL_CONNECTION_DICT = {}
#: Pixel pitch in :math:`cm`
PIXEL_PITCH = 0.4434
#: Tile position wrt the center of the anode in :math:`cm`
TILE_POSITIONS = {}
#: Tile orientations in each anode
TILE_ORIENTATIONS = {}
#: Map of tiles in each anode
TILE_MAP = ()
#: Association between chips and io channels
TILE_CHIP_TO_IO = {}
#: Association between modules and io groups
MODULE_TO_IO_GROUPS = {}
#: Association between modules and tpcs
MODULE_TO_TPCS = {}
TPC_TO_MODULE = {}

###################
# LArPix FEE
###################
#: Discrimination threshold in e-
DISCRIMINATION_THRESHOLD = 5e3 # e-
#: ADC hold delay in clock cycles
ADC_HOLD_DELAY = 15
#: ADC busy delay in clock cycles
ADC_BUSY_DELAY = 9
#: Reset time in clock cycles
RESET_CYCLES = 1
#: Clock cycle time in :math:`\mu s`
CLOCK_CYCLE = 0.1
#: Clock rollover / reset time in larpix clock ticks (32-digit clock)
ROLLOVER_CYCLES =  2**31
#: PPS reset time
PPS_CYCLES = 10**6 / CLOCK_CYCLE
#: True if using PPS reset / false for clock rollover
USE_PPS_ROLLOVER = True # leaving True as default
#: Clock reset, either ROLLOVER_CYCLES or PPS_CYCLES
if USE_PPS_ROLLOVER:
    CLOCK_RESET_PERIOD = int(PPS_CYCLES)
else:
    CLOCK_RESET_PERIOD = int(ROLLOVER_CYCLES)
#: Front-end gain in :math:`mV/e-`
GAIN = 4 / 1e3 # mV/e
#: Buffer risetime in :math:`\mu s` (set >0 to include buffer response simulation)
BUFFER_RISETIME = 0.100
#: Common-mode voltage in :math:`mV`
V_CM = 478.1 # mV
#: Reference voltage in :math:`mV`
V_REF = 1568 # mV
#: Pedestal voltage in :math:`mV`
V_PEDESTAL = 580 # mV
#: Number of ADC counts
ADC_COUNTS = 2**8
#: Reset noise in e-
RESET_NOISE_CHARGE = 900 # e
#: Uncorrelated noise in e-
UNCORRELATED_NOISE_CHARGE = 500 # e
#: Discriminator noise in e-
DISCRIMINATOR_NOISE = 650 # e
#: Average time between events in microseconds
EVENT_RATE = 100000 # 10Hz
#: Offset of the non-beam event time in microseconds
NON_BEAM_EVENT_GAP = 0 # us
#: Pad the signal range to allow N sigmas of diffusion
DIFF_N_SIGMAS = 5
#: Distances of the neighboring pixels to the center pixel
NEIGHBORING_PIX_DIST = []

def electron_mobility(efield, temperature):
    """
    Calculation of the electron mobility w.r.t temperature and electric
    field.
    References:
     - https://lar.bnl.gov/properties/trans.html (summary)
     - https://doi.org/10.1016/j.nima.2016.01.073 (parameterization)
     
    Args:
        efield (float): electric field in kV/cm
        temperature (float): temperature
        
    Returns:
        float: electron mobility in cm^2/kV/us

    """
    a0, a1, a2, a3, a4, a5 = ELECTRON_MOBILITY_PARAMS

    num = a0 + a1 * efield + a2 * pow(efield, 1.5) + a3 * pow(efield, 2.5)
    denom = 1 + (a1 / a0) * efield + a4 * pow(efield, 2) + a5 * pow(efield, 3)
    temp_corr = pow(temperature / 89, -1.5)

    mu = num / denom * temp_corr * V / kV

    return mu

def load_detector_properties(config_keyword):
    from ..config import get_config
    cfg = get_config(config_keyword)
    set_detector_properties(cfg['DET_PROPERTIES'],cfg['PIXEL_LAYOUT'])

def get_n_modules(detprop_file):
    """
    The function loads the global detector properties (not subject to the module variations)
    stores the constants as global variables

    Args:
        detprop_file (str): detector properties YAML
            filename
    """
    with open(detprop_file) as df:
        detprop = yaml.load(df, Loader=yaml.FullLoader)
                
    return list(detprop['module_to_tpcs'].keys())

def set_multi_properties(bucket, n_mod, i_module, message=""):
    if isinstance(bucket, list) and (len(bucket) != n_mod and len(bucket) != 1):
        raise KeyError(f'The length of provided {message} configuration file is unexpected. Please check again.')
    if not isinstance(bucket, list):
        prop = float(bucket)
    elif i_module < 0:
        prop = float(bucket[0])
        if len(bucket) > 1:
            warnings.warn(f'Module variation seems to be not activated, but the {message} is provided as a list. Taking the first given value.')
    elif i_module > len(bucket):
        prop = float(bucket[0])
        warnings.warn(f'Module variation seems to be activated, but the {message} is not specified per module. Taking the first given value.')
    else:
        prop = float(bucket[i_module-1])
    return prop

def set_detector_properties(detprop_file, pixel_file, response_file=None, i_module=-1, geo_only=False):
    """
    The function loads the detector properties and
    the pixel geometry YAML files and stores the constants
    as global variables

    Args:
        detprop_file (str): detector properties YAML filename
                            It is acceptable to provide a single value or a list with one element
                            for electric field and electron lifetime.
        pixel_file (str): pixel layout YAML filename
        i_module (int): module id, default value i_module = -1.
                        i_module < 0 means all module share the same detector configuration.
    """

    global PIXEL_PITCH
    global TPC_BORDERS
    global PIXEL_CONNECTION_DICT
    global N_PIXELS
    global N_PIXELS_PER_TILE
    global V_DRIFT
    global E_FIELD
    global TEMPERATURE
    global ELECTRON_LIFETIME
    global LONG_DIFF
    global TRAN_DIFF
    global TILE_POSITIONS
    global TILE_ORIENTATIONS
    global TILE_MAP
    global TILE_CHIP_TO_IO
    global DRIFT_LENGTH
    global DRIFT_MAX_TIME
    global MODULE_TO_IO_GROUPS
    global MODULE_TO_TPCS
    global TPC_TO_MODULE
    global RESPONSE_SAMPLING
    global RESPONSE_BIN_SIZE
    global MAX_RADIUS
    global MIN_STEP_SIZE
    global TPC_OFFSETS
    global MOD_IDS
    global DISCRIMINATION_THRESHOLD
    global ADC_HOLD_DELAY
    global ADC_BUSY_DELAY
    global RESET_CYCLES
    global CLOCK_CYCLE
    global ROLLOVER_CYCLES
    global PPS_CYCLES
    global USE_PPS_ROLLOVER
    global CLOCK_RESET_PERIOD
    global GAIN
    global BUFFER_RISETIME
    global V_CM
    global V_REF
    global V_PEDESTAL
    global ADC_COUNTS
    global RESET_NOISE_CHARGE
    global UNCORRELATED_NOISE_CHARGE
    global DISCRIMINATOR_NOISE
    global EVENT_RATE
    global NON_BEAM_EVENT_GAP
    global DIFF_N_SIGMAS
    global NEIGHBORING_PIX_DIST

    with open(detprop_file) as df:
        detprop = yaml.load(df, Loader=yaml.FullLoader)

    MOD_IDS = get_n_modules(detprop_file)

    n_mod = len(MOD_IDS)

    TPC_OFFSETS = np.array(detprop['tpc_offsets'])
    # Inverting x and z axes
    TPC_OFFSETS[:, [2, 0]] = TPC_OFFSETS[:, [0, 2]]

    # if module variation for pixel layout file exist, "pixel_file" is a list of pixel layout file with the length of module number
    if isinstance(pixel_file, list):
        if i_module < 0:
            pixel_file = pixel_file[0]
        else:
            pixel_file = pixel_file[i_module-1]
    with open(pixel_file, 'r') as pf:
        tile_layout = yaml.load(pf, Loader=yaml.FullLoader)

    PIXEL_PITCH = tile_layout['pixel_pitch'] * mm / cm
    chip_channel_to_position = tile_layout['chip_channel_to_position']
    PIXEL_CONNECTION_DICT = {tuple(pix): (chip_channel//1000,chip_channel%1000) for chip_channel, pix in chip_channel_to_position.items()}
    TILE_CHIP_TO_IO = tile_layout['tile_chip_to_io']

    xs = np.array(list(chip_channel_to_position.values()))[:,0] * PIXEL_PITCH
    ys = np.array(list(chip_channel_to_position.values()))[:,1] * PIXEL_PITCH
    TILE_BORDERS[0] = [-(max(xs) + PIXEL_PITCH)/2, (max(xs) + PIXEL_PITCH)/2]
    TILE_BORDERS[1] = [-(max(ys) + PIXEL_PITCH)/2, (max(ys) + PIXEL_PITCH)/2]

    tile_indeces = tile_layout['tile_indeces']
    TILE_ORIENTATIONS = tile_layout['tile_orientations']
    TILE_POSITIONS = tile_layout['tile_positions']
    tpc_ids = np.unique(np.array(list(tile_indeces.values()))[:,0], axis=0)

    anodes = defaultdict(list)
    cathode_directions = dict() # What direction the cathode is relative to anode

    for tpc_id in tpc_ids:
        tile_cathode_directions = []
        for tile in tile_indeces:
            if tile_indeces[tile][0] == tpc_id:
                anodes[tpc_id].append(TILE_POSITIONS[tile])
                tile_cathode_directions.append(TILE_ORIENTATIONS[tile][0])

        if len(set(tile_cathode_directions)) != 1:
            raise ValueError("Tiles in same anode plane have different drift directions.")

        if tile_cathode_directions[0] not in [1, -1]:
            raise ValueError("Cathode direction should be either 1 or -1.")

        cathode_directions[tpc_id] = tile_cathode_directions[0]

    try:
        DRIFT_LENGTH = tile_layout['drift_length'] * mm / cm
    except:
        mod_anodes = np.array(list(TILE_POSITIONS.values()))[:, 0]
        DRIFT_LENGTH = 0.5 * (max(mod_anodes) - min(mod_anodes)) * mm / cm

    DRIFT_MAX_TIME = DRIFT_LENGTH / V_DRIFT

    TPC_OFFSETS = np.array(detprop['tpc_offsets'])
    TPC_OFFSETS[:, [2, 0]] = TPC_OFFSETS[:, [0, 2]]

    TPC_BORDERS = np.empty((TPC_OFFSETS.shape[0] * tpc_ids.shape[0], 3, 2))

    for it, tpc_offset in enumerate(TPC_OFFSETS):
        for ia, anode in enumerate(anodes):
            tiles = np.vstack(anodes[anode])
            tiles *= mm / cm
            cathode_direction = cathode_directions[anode]
            x_border = min(tiles[:,2]) + TILE_BORDERS[0][0] + tpc_offset[0], \
                       max(tiles[:,2]) + TILE_BORDERS[0][1] + tpc_offset[0]
            y_border = min(tiles[:,1]) + TILE_BORDERS[1][0] + tpc_offset[1], \
                       max(tiles[:,1]) + TILE_BORDERS[1][1] + tpc_offset[1]
            z_border = min(tiles[:,0]) + tpc_offset[2], \
                       max(tiles[:,0]) + DRIFT_LENGTH * cathode_direction + tpc_offset[2]
            TPC_BORDERS[it*2+ia] = (x_border, y_border, z_border)

    TILE_MAP = detprop['tile_map']

    ntiles_x = len(TILE_MAP[0])
    ntiles_y = len(TILE_MAP[0][0])

    N_PIXELS = len(np.unique(xs))*ntiles_x, len(np.unique(ys))*ntiles_y
    N_PIXELS_PER_TILE = len(np.unique(xs)), len(np.unique(ys))
    MODULE_TO_IO_GROUPS = detprop['module_to_io_groups']
    MODULE_TO_TPCS = detprop['module_to_tpcs']
    TPC_TO_MODULE = dict([(tpc, mod) for mod,tpcs in MODULE_TO_TPCS.items() for tpc in tpcs])

    DIFF_N_SIGMAS = detprop.get('signal_overlap_cut', DIFF_N_SIGMAS)

    if not geo_only:
         # Get response sampling and bin size
        if response_file is not None:
            # if module variation for response file exist, "response_file" is a list of response file with the length of module number
            if isinstance(response_file, list):
                if i_module < 0:
                    response_file = response_file[0]
                else:
                    response_file = response_file[i_module-1]
            response = cp.load(response_file)
            RESPONSE_SAMPLING = float(response['time_tick'])
            RESPONSE_BIN_SIZE = float(response['bin_size'])
            MAX_RADIUS = int(response['response'].shape[0] * RESPONSE_BIN_SIZE // PIXEL_PITCH) # assuming y,z in the response file has the symmetry
        else:
            warnings.warn(f'Charge response not set!')

        dis_threshold_bucket = detprop.get('discrimination_threshold', DISCRIMINATION_THRESHOLD)
        DISCRIMINATION_THRESHOLD = set_multi_properties(dis_threshold_bucket, n_mod, i_module, message="larpix discrimination threshold")
        ADC_HOLD_DELAY = detprop.get('adc_hold_delay', ADC_HOLD_DELAY)
        ADC_BUSY_DELAY = detprop.get('adc_busy_delay', ADC_BUSY_DELAY)
        RESET_CYCLES = detprop.get('reset_cycles', RESET_CYCLES)
        CLOCK_CYCLE = detprop.get('clock_cycle', CLOCK_CYCLE)
        ROLLOVER_CYCLES = detprop.get('rollover_cycles', ROLLOVER_CYCLES)
        PPS_CYCLES = detprop.get('pps_cycles', PPS_CYCLES)
        USE_PPS_ROLLOVER = detprop.get('use_pps_rollover', USE_PPS_ROLLOVER)
        CLOCK_RESET_PERIOD = detprop.get('clock_reset_period', CLOCK_RESET_PERIOD)
        GAIN = detprop.get('larpix_gain', GAIN)
        BUFFER_RISETIME = detprop.get('buffer_risetime', BUFFER_RISETIME)
        V_CM = detprop.get('v_cm', V_CM)
        V_REF = detprop.get('v_ref', V_REF)
        V_PEDESTAL = detprop.get('v_pedestal', V_PEDESTAL)
        ADC_COUNTS = detprop.get('adc_counts', ADC_COUNTS)
        RESET_NOISE_CHARGE = detprop.get('reset_noise_charge', RESET_NOISE_CHARGE)
        UNCORRELATED_NOISE_CHARGE = detprop.get('uncorrelated_noise_charge', UNCORRELATED_NOISE_CHARGE)
        DISCRIMINATOR_NOISE = detprop.get('discriminator_noise', DISCRIMINATOR_NOISE)
        EVENT_RATE = detprop.get('event_rate', EVENT_RATE)
        NON_BEAM_EVENT_GAP = detprop.get('non_beam_event_gap', NON_BEAM_EVENT_GAP)

        TEMPERATURE = detprop.get('temperature', TEMPERATURE)

        e_field_bucket = detprop.get('e_field', E_FIELD)
        E_FIELD = set_multi_properties(e_field_bucket, n_mod, i_module, message="electric field")
        V_DRIFT = E_FIELD * electron_mobility(E_FIELD, TEMPERATURE)

        lifetime_bucket = detprop.get('lifetime', ELECTRON_LIFETIME)
        ELECTRON_LIFETIME = set_multi_properties(lifetime_bucket, n_mod, i_module, message="electron lifetime")

        LONG_DIFF = float(detprop.get('long_diff', LONG_DIFF))
        TRAN_DIFF = float(detprop.get('tran_diff', TRAN_DIFF))

        MAX_RADIUS = int(detprop.get('max_radius', MAX_RADIUS))

        # Prepare neighbouring pixel distance for backtracking
        # Currently backtracking range is used to convert from segment base to pixel base
        NEIGHBORING_PIX_DIST = []
        for i in range(MAX_RADIUS + 1):
            for j in range(MAX_RADIUS + 1):
                NEIGHBORING_PIX_DIST.append(np.sqrt(i*i + j*j))
        NEIGHBORING_PIX_DIST = np.unique(np.array(NEIGHBORING_PIX_DIST, dtype=np.float32))

def load_response(response_file):
    global RESPONSE_MAX_TIME

    # load the charge response for the full drift length
    # shape it to be consistent with detector drift length
    # by either pad or chop the values at the beginning (the side further away from the cathode)
    f_res = cp.load(response_file)
    response = f_res['response']
    res_drift_length = float(f_res['drift_length'])
    drift_ticks_diff = round((res_drift_length - DRIFT_LENGTH) / V_DRIFT / RESPONSE_SAMPLING) # time difference of the response file vs. the TPC in terms of response time ticks

    if drift_ticks_diff > 0: # response is too long, chop
        response = response[drift_ticks_diff:]
        warnings.warn(f'The TPC drift_length is {DRIFT_LENGTH} cm and the charge response simulation uses drift_length of {res_drift_length} cm; The response drfit_length is too long, chop {drift_ticks_diff} values at the beginning (close to the readout).')
    elif drift_ticks_diff < 0: # response is too short, pad
        response = cp.pad(response, abs(drift_ticks_diff))
        warnings.warn(f'The TPC drift_length is {DRIFT_LENGTH} cm and the charge response simulation uses drift_length of {res_drift_length} cm; The response drfit_length is too short, pad {drift_ticks_diff} 0s at the beginning (close to the readout).')

    RESPONSE_MAX_TIME = float(cp.max(cp.nonzero(response)[2]) * RESPONSE_SAMPLING) # axis 0,1 are pixel plane bins, and axis 2 is the time axis

    return response
