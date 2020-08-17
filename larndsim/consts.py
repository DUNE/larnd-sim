import numpy as np


"""
Detector constants
"""
lArDensity = 1.38 # g/cm^3
eField = 0.50 # kV/cm

"""
Unit Conversions
"""
GeVToMeV = 1e3 # MeV
MeVToElectrons = 4.237e+04
msTous = 10e3 # us

"""
PHYSICAL_PARAMS
"""
MeVToElectrons = 4.237e+04
alpha = 0.847
beta = 0.2061
Ab = 0.800
kb = 0.0486 # g/cm2/MeV Amoruso, et al NIM A 523 (2004) 275

"""
TPC_PARAMS
"""
vdrift = 0.153812 # cm / us,
lifetime = 10e3 # us,
tpc_borders = np.array([(0, 100), (-50, 50), (-50, 50)]) # cm,
tpc_xStart = 0
tpc_yStart = -50
timeInterval = (0, 3000) # us
longDiff = 4.0e-6 # cm * cm / us,
tranDiff = 8.8e-6 # cm


"""
PIXEL CONFIG
"""

n_pixels = 333
x_pixel_size = (tpc_borders[0][1] - tpc_borders[0][0]) / n_pixels
y_pixel_size = (tpc_borders[1][1] - tpc_borders[1][0]) / n_pixels
