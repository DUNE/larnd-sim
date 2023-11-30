"""
Set detector constants
"""
import warnings

import numpy as np
import yaml

from collections import defaultdict

from .units import mm, cm, V, kV

#: Detector temperature in K
TEMPERATURE = 87.17
#: Liquid argon density in :math:`g/cm^3`
LAR_DENSITY = 1.38 # g/cm^3
#: Electric field magnitude in :math:`kV/cm`
E_FIELD = 0.50 # kV/cm
#: Drift velocity in :math:`cm/\mu s`
V_DRIFT = 0.1648 # cm / us,
#: Electron lifetime in :math:`\mu s`
ELECTRON_LIFETIME = 2.2e3 # us,
#: Time sampling in :math:`\mu s`
TIME_SAMPLING = 0.1 # us
#: Drift time window in :math:`\mu s`
TIME_INTERVAL = (0, 200.) # us
#: Signal time window padding in :math:`\mu s`
TIME_PADDING = 10
#: Number of sampled points for each segment slice
SAMPLED_POINTS = 40
#: Longitudinal diffusion coefficient in :math:`cm^2/\mu s`
LONG_DIFF = 4.0e-6 # cm * cm / us
#: Transverse diffusion coefficient in :math:`cm^2/\mu s`
TRAN_DIFF = 8.8e-6 # cm * cm / us
#: Numpy array containing all the time ticks in the drift time window
TIME_TICKS = np.linspace(TIME_INTERVAL[0],
                         TIME_INTERVAL[1],
                         int(round(TIME_INTERVAL[1]-TIME_INTERVAL[0])/TIME_SAMPLING)+1)
#: Time window of current response in :math:`\mu s`
TIME_WINDOW = 8.9 # us
#: TPC drift length in :math:`cm`
DRIFT_LENGTH = 0
#: Time sampling in the pixel response file in :math:`\mu s`
RESPONSE_SAMPLING = 0.1
#: Spatial sampling in the pixel reponse file in :math:`cm`
RESPONSE_BIN_SIZE = 0.04434
#: Borders of each TPC volume in :math:`cm`
TPC_BORDERS = np.zeros((0, 3, 2))
#: TPC offsets wrt the origin in :math:`cm`
TPC_OFFSETS = np.zeros((0, 3, 2))
#: Pixel tile borders in :math:`cm`
TILE_BORDERS = np.zeros((2,2))
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

ELECTRON_MOBILITY_PARAMS = 551.6, 7158.3, 4440.43, 4.29, 43.63, 0.2053


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

def set_detector_properties(detprop_file, pixel_file, i_module=-1):
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
    global TIME_INTERVAL
    global TIME_TICKS
    global TIME_PADDING
    global TIME_WINDOW
    global LONG_DIFF
    global TRAN_DIFF
    global TILE_POSITIONS
    global TILE_ORIENTATIONS
    global TILE_MAP
    global TILE_CHIP_TO_IO
    global DRIFT_LENGTH
    global MODULE_TO_IO_GROUPS
    global MODULE_TO_TPCS
    global TPC_TO_MODULE
    global RESPONSE_SAMPLING
    global RESPONSE_BIN_SIZE
    global TPC_OFFSETS

    with open(detprop_file) as df:
        detprop = yaml.load(df, Loader=yaml.FullLoader)

    DRIFT_LENGTH = detprop['drift_length']

    TPC_OFFSETS = np.array(detprop['tpc_offsets'])
    # Inverting x and z axes
    TPC_OFFSETS[:, [2, 0]] = TPC_OFFSETS[:, [0, 2]]

    TIME_INTERVAL = np.array(detprop['time_interval'])
    TIME_TICKS = np.linspace(TIME_INTERVAL[0],
                             TIME_INTERVAL[1],
                             int(round(TIME_INTERVAL[1]-TIME_INTERVAL[0])/TIME_SAMPLING)+1)

    TIME_PADDING = detprop.get('time_padding', TIME_PADDING)
    TIME_WINDOW = detprop.get('time_window', TIME_WINDOW)
    TEMPERATURE = detprop.get('temperature', TEMPERATURE)

    e_field_bucket = detprop.get('e_field', E_FIELD)
    if hasattr(e_field_bucket, "__len__") and (len(e_field_bucket) != len(get_n_modules(detprop_file)) and len(e_field_bucket) != 1):
        raise KeyError("The length of provided E_field in the detector configuration file is unexpected. Please check again.")
    if not hasattr(e_field_bucket, "__len__"):
        E_FIELD = e_field_bucket
    elif i_module < 0:
        E_FIELD = e_field_bucket[0]
        if len(e_field_bucket) > 1:
            warnings.warn('It seems module variation is not activated, but electric field is provided as a list. Taking the first given value.')
    elif i_module > len(e_field_bucket):
        E_FIELD = e_field_bucket[0]
        warnings.warn('Simulation with module variation seems to be activated, but the electric field is not specified per module.Taking the first given value.')
    else:
        E_FIELD = e_field_bucket[i_module-1]
    V_DRIFT = E_FIELD * electron_mobility(E_FIELD, TEMPERATURE)

    lifetime_bucket = detprop.get('lifetime', ELECTRON_LIFETIME)
    if hasattr(lifetime_bucket, "__len__") and (len(lifetime_bucket) != len(get_n_modules(detprop_file)) and len(lifetime_bucket) != 1):
        raise KeyError("The length of provided lifetime in the detector configuration file is unexpected. Please check again.")
    if not hasattr(lifetime_bucket, "__len__"):
        ELECTRON_LIFETIME = lifetime_bucket
    elif i_module < 0:
        ELECTRON_LIFETIME = lifetime_bucket[0]
        if len(lifetime_bucket) > 1:
            warnings.warn('It seems module variation is not activated, but electron lifetime is provided as a list. Taking the first given value.')
    elif i_module > len(lifetime_bucket):
        ELECTRON_LIFETIME = lifetime_bucket[0]
        warnings.warn('Simulation with module variation seems to be activated, but the electron lifetime is not specified per module. Taking the first given value.')
    else:
        ELECTRON_LIFETIME = lifetime_bucket[i_module-1]

    LONG_DIFF = detprop.get('long_diff', LONG_DIFF)
    TRAN_DIFF = detprop.get('tran_diff', TRAN_DIFF)
    RESPONSE_SAMPLING = detprop.get('response_sampling', RESPONSE_SAMPLING)

    response_bin_size_bucket = detprop.get('response_bin_size', RESPONSE_BIN_SIZE)
    print("len(response_bin_size_bucket): ", len(response_bin_size_bucket))
    print("get_n_modules(detprop_file): ", get_n_modules(detprop_file))
    if hasattr(response_bin_size_bucket, "__len__") and (len(response_bin_size_bucket) != len(get_n_modules(detprop_file)) and len(response_bin_size_bucket) != 1):
        raise KeyError("The length of provided induction response bin size in the detector configuration file is unexpected. Please check again.")
    if not hasattr(response_bin_size_bucket, "__len__"):
        RESPONSE_BIN_SIZE = response_bin_size_bucket
    elif i_module < 0:
        RESPONSE_BIN_SIZE = response_bin_size_bucket[0]
        if len(response_bin_size_bucket) > 1:
            warnings.warn('It seems module variation is not activated, but the induction response bin size is provided as a list. Taking the first given value.')
    elif i_module > len(response_bin_size_bucket):
        RESPONSE_BIN_SIZE = response_bin_size_bucket[0]
        warnings.warn('Simulation with module variation seems to be activated, but the induction response bin size is not specified per module. Taking the first given value.')
    else:
        RESPONSE_BIN_SIZE = response_bin_size_bucket[i_module-1]

    # if module variation for pixel layout file exist, "pixel_file" is a list of pixel layout file with the length of module number
    if isinstance(pixel_file, list):
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
    for tpc_id in tpc_ids:
        for tile in tile_indeces:
            if tile_indeces[tile][0] == tpc_id:
                anodes[tpc_id].append(TILE_POSITIONS[tile])

    DRIFT_LENGTH = detprop['drift_length']

    TPC_OFFSETS = np.array(detprop['tpc_offsets'])
    TPC_OFFSETS[:, [2, 0]] = TPC_OFFSETS[:, [0, 2]]

    TPC_BORDERS = np.empty((TPC_OFFSETS.shape[0] * tpc_ids.shape[0], 3, 2))

    for it, tpc_offset in enumerate(TPC_OFFSETS):
        for ia, anode in enumerate(anodes):
            tiles = np.vstack(anodes[anode])
            tiles *= mm / cm
            drift_direction = 1 if anode == 1 else -1
            x_border = min(tiles[:,2]) + TILE_BORDERS[0][0] + tpc_offset[0], \
                       max(tiles[:,2]) + TILE_BORDERS[0][1] + tpc_offset[0]
            y_border = min(tiles[:,1]) + TILE_BORDERS[1][0] + tpc_offset[1], \
                       max(tiles[:,1]) + TILE_BORDERS[1][1] + tpc_offset[1]
            z_border = min(tiles[:,0]) + tpc_offset[2], \
                       max(tiles[:,0]) + DRIFT_LENGTH * drift_direction + tpc_offset[2]
            TPC_BORDERS[it*2+ia] = (x_border, y_border, z_border)

    TILE_MAP = detprop['tile_map']

    ntiles_x = len(TILE_MAP[0])
    ntiles_y = len(TILE_MAP[0][0])

    N_PIXELS = len(np.unique(xs))*ntiles_x, len(np.unique(ys))*ntiles_y
    N_PIXELS_PER_TILE = len(np.unique(xs)), len(np.unique(ys))
    MODULE_TO_IO_GROUPS = detprop['module_to_io_groups']
    MODULE_TO_TPCS = detprop['module_to_tpcs']
    TPC_TO_MODULE = dict([(tpc, mod) for mod,tpcs in MODULE_TO_TPCS.items() for tpc in tpcs])
