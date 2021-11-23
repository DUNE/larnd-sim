"""
Module that simulates the scattering of photons throughout the detector from the
location of the edep to the location of each photodetector
"""

import sys, time, math
import argparse
import numpy as np
from math import log, isnan
from . import consts    

def get_voxel(pos, itpc):
    """
    Indexes the ID of the voxel in which the edep occurs in.
    Args:
        pos (:obj:`numpy.ndarray`): list of x, y, z coordinates within a generic TPC volume
        itpc (int): index of the tpc corresponding to this position (calculated in drift)
    Returns:
        :obj:`numpy.float64`: index of the voxel containing the input position
    """

    tpc_borders = consts.tpc_borders[itpc]

    # If we are in an "odd" TPC, we need to rotate x 
    # this is to preserve the "left/right"-ness of the optical channels
    # with respect to the anode plane
    is_odd = tpc_borders[2][1] > tpc_borders[2][0]

    # Assigns tpc borders to variables 
    xMin = tpc_borders[0][0] - 2e-2
    xMax = tpc_borders[0][1] + 2e-2
    yMin = tpc_borders[1][0] - 2e-2
    yMax = tpc_borders[1][1] + 2e-2
    zMin = tpc_borders[2][0] - 2e-2
    zMax = tpc_borders[2][1] + 2e-2

    if is_odd:
        i = int((pos[0] - xMin)/(xMax - xMin)*consts.lut_vox_div[0])
    else:
        i = int((xMax - pos[0])/(xMax - xMin)*consts.lut_vox_div[0])
    j = int((pos[1] - yMin)/(yMax - yMin)*consts.lut_vox_div[1])
    k = int((pos[2] - zMin)/(zMax - zMin)*consts.lut_vox_div[2])

    return i,j,k

def calculate_light_incidence(tracks, lut_path, light_dep, light_incidence):
    """
    Simulates the number of photons read by each optical channel depending on 
        where the edep occurs as well as the time it takes for a photon to reach the 
        nearest photomultiplier tube (the "fastest" photon)
    Args:
        tracks (:obj:`numpy.ndarray`): track array containing edep segments, positions are used for lookup
        lut_path (str): filename of numpy array (.npy) containing light calculation
        light_dep (:obj:`numpy.ndarray`): 1-Dimensional array containing number of photons produced
            in each edep segment.
        light_incidence (:obj:`numpy.ndarray`): to contain the result of light incidence calculation.
            this array has dimension (n_tracks, n_optical_channels) and each entry
            is a structure of type (n_photons_det (float32), t0_det (float32))
            these correspond to the number detected in each channel (n_photons_edep*visibility), 
            and the time of earliest arrival at that channel.
    """
    
    # Loads in LUT file
    np_lut = np.load(lut_path)
    
    # Defines variables of global position. Currently using the average between the start and end positions of the edep
    x = tracks['x']
    y = tracks['y']
    z = tracks['z']

    nEdepSegments = tracks.shape[0]
    
    # Loop edep positions
    for edepInd in range(nEdepSegments):

        # Global position
        pos = (np.array((x[edepInd],y[edepInd],z[edepInd])))

        # Defining number of produced photons from quencing.py
        n_photons = light_dep['n_photons_edep'][edepInd]

        # tpc
        itpc = tracks["pixel_plane"][edepInd]
        
        # voxel containing LUT position
        voxel = get_voxel(pos, itpc)
        
        # Calls data on voxel
        lut_vox = np_lut[voxel[0], voxel[1], voxel[2],:,:]

        # the indices corresponding the the channels in a given tpc
        output_channels = np.arange(consts.n_op_channel) + int(itpc*consts.n_op_channel)

        # Gets visibility data for the voxel
        vis_dat = lut_vox[:,0]

        # Gets T1 data for the voxel
        T1_dat = lut_vox[:,1]

        for outputInd, eff, vis, t1 in zip(output_channels, consts.op_channel_efficiency, vis_dat, T1_dat):
            light_incidence[edepInd, outputInd] = (eff*vis*n_photons, t1)
