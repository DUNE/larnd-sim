"""
Module that simulates the scattering of photons throughout the detector from the
location of the edep to the location of each photodetector
"""

import sys, time, math
import argparse
import numpy as np
from math import log, isnan
from . import consts    

def get_voxel(pos):
    """
    Indexes the ID of the voxel in which the edep occurs in.
    Args:
        pos (:obj:`numpy.ndarray`): list of x, y, z coordinates within a generic TPC volume

    Returns:
        :obj:`numpy.float64`: index of the voxel containing the input position
    """

    # i = np.floor(pos[2]/consts.lut_vox_div[2])
    # j = np.floor(pos[1]/consts.lut_vox_div[1])
    # k = np.floor(pos[0]/consts.lut_vox_div[0])
    i = int((pos[0] - consts.lut_xrange[0])/(consts.lut_xrange[1] - consts.lut_xrange[0])*consts.lut_vox_div[2]) 
    j = int((pos[1] - consts.lut_yrange[0])/(consts.lut_yrange[1] - consts.lut_yrange[0])*consts.lut_vox_div[1]) 
    k = int((pos[2] - consts.lut_zrange[0])/(consts.lut_zrange[1] - consts.lut_zrange[0])*consts.lut_vox_div[0]) 

    print('\n\n\n\n')
    print(pos)
    print(i,j,k)
    print('\n\n\n\n')

    return i,j,k


def get_tpc(pos):
    """
    Determines in which TPC the edep takes place.
    This should actually call the detector properties yaml.
    Args:
        pos (:obj:`numpy.ndarray`): list of x, y, z coordinates within a detector geometry
    Returns:
        int: index of the TPC containing the input posision
    """
    
    tpc_x = math.floor(pos[2]/(consts.ModuleDimension[2]/2.))+consts.n_mod[2]/2*consts.n_tpc[2]

    return int(tpc_x)%2

def larnd_to_lut_coord(pos, itpc):
    """
    Converts the LArND-sim coord to its respective location in the LUT coordinate system.
    LUT should be updated to LArND-sim system and this function removed.
    Args:
        pos (:obj:`numpy.ndarray`): list of x, y, z coordinates within a generic TPC
        itpc (int): index of the tpc corresponding to the input position
    Returns:
        :obj:`numpy.ndarray`: list of x, y, z coordinates translated to the LUT system
    """
    # shifts the larnd coord to the LUT coord system

    if pos[2] < 0:
        lut_pos -= np.array([0,0,consts.lut_zrange[0]])
    else:
        lut_pos += np.array([0,0,consts.lut_zrange[0]])

    return (lut_pos)

def calculate_light_incidence(t_data, lut_path, light_dep, light_incidence):
    """
    Simulates the number of photons read by each optical channel depending on 
        where the edep occurs as well as the time it takes for a photon to reach the 
        nearest photomultiplier tube (the "fastest" photon)
    Args:
        t_data (:obj:`numpy.ndarray`): track array containing edep segments, positions are used for lookup
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
    time = np.full((t_data['dE'].size,consts.n_op_channel*2),20.)
    tphotons = np.zeros((t_data['dE'].size,consts.n_op_channel*2))
    
    # Defines variables of global position. Currently using the average between the start and end positions of the edep
    x = t_data['x']
    y = t_data['y']
    z = t_data['z']

    # Defining number of produced photons from quencing.py
    n_photons = light_dep['n_photons_edep']

    nEdepSegments = t_data.shape[0]
    
    # Loop edep positions
    for edepInd in range(nEdepSegments):

        # Global position
        pos = (np.array((x[edepInd],y[edepInd],z[edepInd])))*consts.cm2mm

        # tpc
        itpc = get_half_det_copy(pos)
        
        # LUT position
        lut_pos = larnd_to_lut_coord(pos, itpc) 

        # voxel containing LUT position
        voxel = get_voxel(lut_pos)
        
        # Calls data on voxel
        # lut_vox = np_lut[np_lut['Voxel'] == voxel]
        lut_vox = np_lut[voxel[0], voxel[1], voxel[2],:,:]
        
        # Makes a list of op channel indecies
        op_dat = np.arange(len(lut_vox))

        # Gets visibility data for the voxel
        vis_dat = lut_vox[:,0]

        # Gets T1 data for the voxel
        T1_dat = lut_vox[:,1]
        
        # Loop through each op channel
        for entry in range(len(lut_vox)):
            # Defines which op channel data is collected from
            op_channel = op_dat[entry]
            
            # Flips op channels if edep occurs in tpc 2
            if (itpc==1):
                op_channel = (op_channel+consts.n_op_channel/2)%consts.n_op_channel

            # Calculates the number of photons reaching each optical channel
            n_photons_read = vis_dat[entry]*n_photons[edepInd]
            
            # Determines the travel time of the "fastest" photon
            if (T1_dat[entry] < time[edepInd,int(op_channel)]):  
                time[edepInd, int(op_channel)] = T1_dat[entry]
            
            # Accounts for higher light collection on the LCM detectors
            if (op_channel % 12) > 5: 
                n_photons_read *= consts.norm_lcm_acl
                
            # Index for storing light data the optical channels in the second TPC
            if itpc == 1:
                op_channel += consts.n_op_channel
            
            # Assigns the number of photons read for each optical channel to an array
            tphotons[edepInd, int(op_channel)] += n_photons_read
    
    # Assigns data to the h5 file
    light_incidence['n_photons_det'] += tphotons
    light_incidence['t0_det'] += time
