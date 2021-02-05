"""
Module containing constants needed by the simulation
"""

import numpy as np
import larpixgeometry.pixelplane
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
vdrift = 0.153812 # cm / us,
#: Electron lifetime in :math:`\mu s`
lifetime = 10e3 # us,
#: Time sampling in :math:`\mu s`
t_sampling = 0.05 # us
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

tpc_centers = np.array([
        [-487.949, -218.236, -335.],
        [-182.051, -218.236, -335.],
        [182.051, -218.236, -335.],
        [487.949, -218.236, -335.],
        [-487.949, -218.236, 335.],
        [-182.051, -218.236, 335.],
        [182.051, -218.236, 335.],
        [487.949, -218.236, 335.],
])

# Swap z->x
tpc_centers[:, [2, 0]] = tpc_centers[:, [0, 2]]
tpc_centers *= mm2cm

module_borders = np.zeros((tpc_centers.shape[0], 3, 2))
tpc_borders = np.zeros((3,2))
tpc_size = np.zeros(3)
n_pixels = 0, 0
pixel_connection_dict = {}
pixel_size = np.zeros(2)

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
    global pixel_size
    global tpc_borders
    global tpc_size
    global pixel_connection_dict
    global module_borders
    global n_pixels
    global tpc_centers
    global vdrift
    global lifetime
    global time_interval
    global long_diff
    global tran_diff

    with open(detprop_file) as df:
        detprop = yaml.load(df, Loader=yaml.FullLoader)

    tpc_centers = np.array(detprop['tpc_centers'])
    time_interval = np.array(detprop['time_interval'])
    
    vdrift = detprop['vdrift']
    lifetime = detprop['lifetime']
    long_diff = detprop['long_diff']
    tran_diff = detprop['tran_diff']

    with open(pixel_file, 'r') as pf:
        board = larpixgeometry.pixelplane.PixelPlane.fromDict(yaml.load(pf, Loader=yaml.FullLoader))

    xs = np.array([board.pixels[ip].x/10 for ip in board.pixels])
    ys = np.array([board.pixels[ip].y/10 for ip in board.pixels])

    #: Number of pixels per axis
    n_pixels = len(np.unique(xs)), len(np.unique(ys))

    #: Size of pixels per axis in :math:`cm`
    pixel_size[0] = (max(xs)-min(xs)) / (n_pixels[0] - 1)
    pixel_size[1] = (max(ys)-min(ys)) / (n_pixels[1] - 1)

    chipids = list(board.chips.keys())

    for chip in chipids:
        for channel, pixel in enumerate(board.chips[chip].channel_connections):
            if pixel.x !=0 and pixel.y != 0:
                pixel_connection_dict[(round(pixel.x/pixel_size[0]),
                                       round(pixel.y/pixel_size[1]))] = channel, chip

    #: TPC borders coordinates in :math:`cm`
    tpc_borders[0] = [min(xs)-pixel_size[0]/2, max(xs)+pixel_size[0]/2]
    tpc_borders[1] = [min(ys)-pixel_size[1]/2, max(ys)+pixel_size[1]/2]
    tpc_borders[2] = [-15, 15]

    tpc_size = tpc_borders[:,1] - tpc_borders[:,0]

    module_borders = np.zeros((tpc_centers.shape[0], 3, 2))
    for iplane, tpc_center in enumerate(tpc_centers):
        module_borders[iplane] = (tpc_borders.T+tpc_center).T
