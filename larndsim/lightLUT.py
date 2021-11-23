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

    j = int((pos[1] - tpc_borders[1][0])/(tpc_borders[1][1] - tpc_borders[1][0])*consts.lut_vox_div[1]) 

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
    
    # Data containers
    time = np.full((tracks['dE'].size,consts.n_op_channel*2),20.)
    tphotons = np.zeros((tracks['dE'].size,consts.n_op_channel*2))
    
    # Defines variables of global position. Currently using the average between the start and end positions of the edep
    x = tracks['x']
    y = tracks['y']
    z = tracks['z']

    # Defining number of produced photons from quencing.py
    n_photons = light_dep['n_photons_edep']

    nEdepSegments = tracks.shape[0]
    
    # Loop edep positions
    for edepInd in range(nEdepSegments):

        # Global position
        pos = (np.array((x[edepInd],y[edepInd],z[edepInd])))

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

        # arclight and LCM modules have a different efficiency
        # (this should go in the detector properties yaml)
        OMefficiency = sum([6*[1.],
                            6*[consts.norm_lcm_acl],
                            6*[1.],
                            6*[consts.norm_lcm_acl],
                            6*[1.],
                            6*[consts.norm_lcm_acl],
                            6*[1.],
                            6*[consts.norm_lcm_acl]],
                           start = [])
        
        for outputInd, eff, vis, t1 in zip(output_channels, OMefficiency, vis_dat, T1_dat):
            light_incidence[edepInd, outputInd] = (eff*vis*n_photons[edepInd], t1)
