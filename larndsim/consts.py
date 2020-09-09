"""
Module containing constants needed by the simulation
"""
import numpy as np

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

## TPC params
#: Drift velocity in :math:`cm/\mu s`
vdrift = 0.153812 # cm / us,
#: Electron lifetime in :math:`\mu s`
lifetime = 10e3 # us,
#: TPC borders coordinates in :math:`cm`
tpc_borders = np.array([(0, 100), (-150, 150), (0, 100)]) # cm,
#: Time sampling in :math:`\mu s`
t_sampling = 0.1 # us
#: Drift time window in :math:`\mu s`
time_interval = (0., 700.) # us
#: Signal time window padding in :math:`\mu s`
time_padding = 50
#: Number of sampled points for each segment slice
sampled_points = 50
#: Longitudinal diffusion coefficient in :math:`cm^2/\mu s`
long_diff = 4.0e-3 # cm * cm / us,
#: Transverse diffusion coefficient in :math:`cm`
tran_diff = 8.8e-3 # cm
#: Numpy array containing all the time ticks in the drift time window
time_ticks = np.linspace(time_interval[0],
                         time_interval[1],
                         int(round(time_interval[1]-time_interval[0])/t_sampling))

## Pixel params
#: Number of pixels per axis
n_pixels = 250, 750
x_pixel_size = (tpc_borders[0][1] - tpc_borders[0][0]) / n_pixels[0]
y_pixel_size = (tpc_borders[1][1] - tpc_borders[1][0]) / n_pixels[1]
#: Size of pixels per axis in :math:`cm`
pixel_size = np.array([x_pixel_size, y_pixel_size])

## Quenching parameters
box = 1
birks = 2
