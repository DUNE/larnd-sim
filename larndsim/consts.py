"""
Module containing constants needed by the simulation
"""

import numpy as np
import yaml

## Detector constants
#: Liquid argon density in :math:`g/cm^3`
lArDensity = 1.38 # g/cm^3
#: Electric field magnitude in :math:`kV/cm`
eField = 0.50 # kV/cm

## Unit Conversions
MeVToElectrons = 4.237e+04

## Physical params
#: Recombination :math:`\alpha` constant for the Box model
alpha = 0.93
#: Recombination :math:`\beta` value for the Box model in :math:`(kV/cm)(g/cm^2)/MeV`
beta = 0.207 #0.3 (MeV/cm)^-1 * 1.383 (g/cm^3)* 0.5 (kV/cm), R. Acciarri et al JINST 8 (2013) P08005
#: Recombination :math:`A_b` value for the Birks Model
Ab = 0.800
#: Recombination :math:`k_b` value for the Birks Model in :math:`(kV/cm)(g/cm^2)/MeV`
kb = 0.0486 # g/cm2/MeV Amoruso, et al NIM A 523 (2004) 275
#: Electron charge in Coulomb
e_charge = 1.602e-19

## TPC params
#: Drift velocity in :math:`cm/\mu s`
vdrift = 0.1648 # cm / us,
#: Electron lifetime in :math:`\mu s`
lifetime = 2.2e3 # us,
#: Time sampling in :math:`\mu s`
t_sampling = 0.1 # us
#: Drift time window in :math:`\mu s`
time_interval = (0, 200.) # us
#: Signal time window padding in :math:`\mu s`
time_padding = 5
#: Number of sampled points for each segment slice
sampled_points = 15
#: Longitudinal diffusion coefficient in :math:`cm^2/\mu s`
long_diff = 4.0e-6 # cm * cm / us
#: Transverse diffusion coefficient in :math:`cm^2/\mu s`
tran_diff = 8.8e-6 # cm * cm / us
#: Numpy array containing all the time ticks in the drift time window
time_ticks = np.linspace(time_interval[0],
                         time_interval[1],
                         int(round(time_interval[1]-time_interval[0])/t_sampling)+1)
## Quenching parameters
box = 1
birks = 2

mm2cm = 0.1
cm2mm = 10

tpc_borders = np.zeros((0, 3, 2))
tile_borders = np.zeros((2,2))
tile_size = np.zeros(3)
n_pixels = 0, 0
n_pixels_per_tile = 0, 0
pixel_connection_dict = {}
pixel_pitch = 0
tile_positions = {}
tile_orientations = {}
tile_map = ()
tile_chip_to_io = {}

variable_types = {
    "eventID": "u4",
    "z_end": "f4",
    "trackID": "u4",
    "tran_diff": "f4",
    "z_start": "f4",
    "x_end": "f4",
    "y_end": "f4",
    "n_electrons": "u4",
    "pdgId": "i4",
    "x_start": "f4",
    "y_start": "f4",
    "t_start": "f4",
    "dx": "f4",
    "long_diff": "f4",
    "pixel_plane": "u4",
    "t_end": "f4",
    "dEdx": "f4",
    "dE": "f4",
    "t": "f4",
    "y": "f4",
    "x": "f4",
    "z": "f4"
}

anode_layout = (2,4)
xs = 0
ys = 0

def load_detector_properties(detprop_file, pixel_file):
    """
    The function loads the detector properties and
    the pixel geometry YAML files and stores the constants
    as global variables

    Args:
        detprop_file (str): detector properties YAML
            filename
        pixel_file (str): pixel layout YAML filename
    """
    global xs
    global ys
    global pixel_pitch
    global tpc_borders
    global pixel_connection_dict
    global n_pixels
    global n_pixels_per_tile
    global vdrift
    global lifetime
    global time_interval
    global long_diff
    global tran_diff
    global tile_positions
    global tile_orientations
    global tile_map
    global tile_chip_to_io

    with open(detprop_file) as df:
        detprop = yaml.load(df, Loader=yaml.FullLoader)

    tpc_centers = np.array(detprop['tpc_centers'])
    tpc_centers[:, [2, 0]] = tpc_centers[:, [0, 2]]

    time_interval = np.array(detprop['time_interval'])

    vdrift = detprop['vdrift']
    lifetime = detprop['lifetime']
    long_diff = detprop['long_diff']
    tran_diff = detprop['tran_diff']

    with open(pixel_file, 'r') as pf:
        tile_layout = yaml.load(pf, Loader=yaml.FullLoader)

    pixel_pitch = tile_layout['pixel_pitch'] * mm2cm
    chip_channel_to_position = tile_layout['chip_channel_to_position']
    pixel_connection_dict = {tuple(pix): (chip_channel//1000,chip_channel%1000) for chip_channel, pix in chip_channel_to_position.items()}
    tile_chip_to_io = tile_layout['tile_chip_to_io']

    xs = np.array(list(chip_channel_to_position.values()))[:,0] * pixel_pitch
    ys = np.array(list(chip_channel_to_position.values()))[:,1] * pixel_pitch
    tile_borders[0] = [-(max(xs)+pixel_pitch)/2, (max(xs)+pixel_pitch)/2]
    tile_borders[1] = [-(max(ys)+pixel_pitch)/2, (max(ys)+pixel_pitch)/2]

    tile_positions = np.array(list(tile_layout['tile_positions'].values())) * mm2cm
    tile_orientations = np.array(list(tile_layout['tile_orientations'].values()))
    tpcs = np.unique(tile_positions[:,0])
    tpc_borders = np.zeros((len(tpcs), 3, 2))

    for itpc,tpc_id in enumerate(tpcs):
        this_tpc_tile = tile_positions[tile_positions[:,0] == tpc_id]
        this_orientation = tile_orientations[tile_positions[:,0] == tpc_id]

        x_border = min(this_tpc_tile[:,2])+tile_borders[0][0]+tpc_centers[itpc][0], \
                   max(this_tpc_tile[:,2])+tile_borders[0][1]+tpc_centers[itpc][0]
        y_border = min(this_tpc_tile[:,1])+tile_borders[1][0]+tpc_centers[itpc][1], \
                   max(this_tpc_tile[:,1])+tile_borders[1][1]+tpc_centers[itpc][1]
        z_border = min(this_tpc_tile[:,0])+tpc_centers[itpc][2], \
                   max(this_tpc_tile[:,0])+detprop['drift_length']*this_orientation[:,0][0]+tpc_centers[itpc][2]

        tpc_borders[itpc] = (x_border, y_border, z_border)

    #: Number of pixels per axis
    n_pixels = len(np.unique(xs))*2, len(np.unique(ys))*4
    n_pixels_per_tile = len(np.unique(xs)), len(np.unique(ys))

    tile_map = ((7,5,3,1),(8,6,4,2)),((16,14,12,10),(15,13,11,9))
