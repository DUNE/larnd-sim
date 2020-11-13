"""
Module containing constants needed by the simulation
"""
import numpy as np
import larpixgeometry.pixelplane
import yaml
import os, sys

## Detector constants
#: Liquid argon density in :math:`g/cm^3`
lArDensity = 1.38 # g/cm^3
#: Electric field magnitude in :math:`kV/cm`
eField = 0.50 # kV/cm

## Unit Conversions
MeVToElectrons = 4.237e+04

## Physical params
#: Recombination :math:`\alpha` constant for the Box model
alpha = 0.847
#: Recombination :math:`\beta` value for the Box model in :math:`cm/MeV`
beta = 0.2061
#: Recombination :math:`A_b` value for the Birks Model
Ab = 0.800
#: Recombination :math:`k_b` value for the Birks Model in :math:`g/cm^2/MeV`
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
time_interval = (0, 40.) # us
#: Signal time window padding in :math:`\mu s`
time_padding = 1
#: Number of sampled points for each segment slice
sampled_points = 50
#: Longitudinal diffusion coefficient in :math:`cm^2/\mu s`
long_diff = 4.0e-6 # cm * cm / us,
#: Transverse diffusion coefficient in :math:`cm`
tran_diff = 8.8e-6 # cm
#: Numpy array containing all the time ticks in the drift time window
time_ticks = np.linspace(time_interval[0],
                         time_interval[1],
                         int(round(time_interval[1]-time_interval[0])/t_sampling))

## Pixel params
with open(os.path.join(sys.path[0], "examples/pixel_geometry.yaml"), 'r') as f:
    board = larpixgeometry.pixelplane.PixelPlane.fromDict(yaml.load(f,Loader=yaml.FullLoader))
    
xs = np.array([board.pixels[ip].x/10 for ip in board.pixels])
ys = np.array([board.pixels[ip].y/10 for ip in board.pixels])
    
def load_pixel_geometry(filename):
    global xs
    global ys
    with open(filename, 'r') as f:
        board = larpixgeometry.pixelplane.PixelPlane.fromDict(yaml.load(f,Loader=yaml.FullLoader))
        xs = np.array([board.pixels[ip].x/10 for ip in board.pixels])
        ys = np.array([board.pixels[ip].y/10 for ip in board.pixels])

#: Number of pixels per axis
n_pixels = len(np.unique(xs)), len(np.unique(ys))
x_pixel_size = (max(xs)-min(xs)) / (n_pixels[0] - 1) 
y_pixel_size = (max(ys)-min(xs)) / (n_pixels[1] - 1) 
#: Size of pixels per axis in :math:`cm`
pixel_size = np.array([x_pixel_size, y_pixel_size])

pixel_connection_dict = {}
chipids = list(board.chips.keys())
for chip in chipids:
    for channel, pixel in enumerate(board.chips[chip].channel_connections):
        if pixel.x !=0 and pixel.y != 0:
            pixel_connection_dict[(round(pixel.x/pixel_size[0]), round(pixel.y/pixel_size[1]))] = channel, chip

#: TPC borders coordinates in :math:`cm`
tpc_borders = np.array([(min(xs)-x_pixel_size/2, max(xs)+x_pixel_size/2), 
                        (min(ys)-y_pixel_size/2, max(ys)+y_pixel_size/2), 
                        (0, 3)])

## Quenching parameters
box = 1
birks = 2

def get_pixel_coordinates(pixel_id):
    return pixel_id[0]*pixel_size[0]+tpc_borders[0][0]+pixel_size[0]/2, pixel_id[1]*pixel_size[1]+tpc_borders[1][0]+pixel_size[1]/2