"""
Module that simulates the scattering of photons throughout the detector from the
location of the edep to the location of each photodetector
"""

import sys, time, math
import argparse
import numpy as np
from math import log, isnan
from . import consts    

def get_lut_geometry(lut_path):
    """
    Finds the maximum and minimum values along the x,y and z axis as well as how
    many divisions (voxels) are along each axis.
    Args:
        lut_path (str): filename of numpy array (.npy) containing light calculation
    Returns:
        :obj:`numpy.ndarray`: 3x3 array of voxelization information (minimum, maximum, 
            number of divisions) in each dimension
    """
    
    f = np.load(lut_path)
    lut_min = np.array([f['Min'][0],f['Min'][1],f['Min'][2]])
    lut_max = np.array([f['Max'][0],f['Max'][1],f['Max'][2]])
    lut_ndiv = np.array([f['NDivisions'][0],f['NDivisions'][1],f['NDivisions'][2]])

    return np.array([lut_min,lut_max,lut_ndiv])

def get_voxel(pos,lut_geometry):
    """
    Indexes the ID of the voxel in which the edep occurs in.
    Args:
        pos (:obj:`numpy.ndarray`): list of x, y, z coordinates within a generic TPC volume
        lut_geometry (obj:`numpy.ndarray`): 3x3 array of voxelization information 
            (minimum, maximum, number of divisions) in each dimension
    Returns:
        :obj:`numpy.float64`: index of the voxel containing the input position
    """
    
    (lut_min,lut_max,lut_ndiv) = lut_geometry

    vox_xyz = np.floor(pos/(lut_max-lut_min)*lut_ndiv).astype(int)+lut_ndiv/2

    voxel = vox_xyz[2]*lut_ndiv[0]*lut_ndiv[1]+vox_xyz[1]*lut_ndiv[0]+vox_xyz[0]

    return voxel


def get_half_det_copy(pos):
    """
    Determines in which TPC the edep takes place.
    This should actually call the detector properties yaml.
    Args:
        pos (:obj:`numpy.ndarray`): list of x, y, z coordinates within a detector geometry
    Returns:
        int: index of the TPC containing the input posision
    """
    
    tpc_x = math.floor(pos[0]/(consts.ModuleDimension[0]/2.))+consts.n_mod[0]/2*consts.n_tpc[0]

    return int(tpc_x)%2

def larnd_to_lut_coord(pos, lut_geometry, itpc):
    """
    Converts the LArND-sim coord to its respective location in the LUT coordinate system.
    LUT should be updated to LArND-sim system and this function removed.
    Args:
        pos (:obj:`numpy.ndarray`): list of x, y, z coordinates within a generic TPC
        lut_geometry (obj:`numpy.ndarray`): 3x3 array of voxelization information 
            (minimum, maximum, number of divisions) in each dimension
        itpc (int): index of the tpc corresponding to the input position
    Returns:
        :obj:`numpy.ndarray`: list of x, y, z coordinates translated to the LUT system
    """
    
    # access LUT geometry
    lut_min,lut_max,lut_ndiv = lut_geometry
    
    # shifts the larnd coord to the LUT coord system
    lut_pos = pos - consts.tpc_offsets[itpc]*consts.cm2mm
    
    return (lut_pos)

def calculate_light_incidence(t_data,lut_path,light_dat):
    """
    Simulates the number of photons read by each optical channel depending on 
        where the edep occurs as well as the time it takes for a photon to reach the 
        nearest photomultiplier tube (the "fastest" photon)
    Args:
        t_data (:obj:`numpy.ndarray`): track array containing edep segments, positions are used for lookup
        lut_path (str): filename of numpy array (.npy) containing light calculation
        light_dat (:obj:`numpy.ndarray`): to contain the result of light incidence calculation.
            this array has dimension (n_tracks, n_optical_channels) and each entry
            is a structure of type (n_photons_edep (float32), n_photons_det (float32), t0_det (float32))
            these correspond the number of photons produced by a given edep (stored only in the 0th channel),
            the number detected in each channel (n_photons_edep*visibility), and the time of earliest
            arrival at that channel.
    """
    
    # Loads in LUT file
    np_lut = np.load(lut_path)
    
    # Obtains lut geometry
    lut_geometry = get_lut_geometry(lut_path)  
    
    # Data containers
    time = np.full((t_data['dE'].size,consts.n_op_channel*2),20.)
    tphotons = np.zeros((t_data['dE'].size,consts.n_op_channel*2))
    
    # Defines variables of global position. Currently using the average between the start and end positions of the edep
    x = t_data['x']
    y = t_data['y']
    z = t_data['z']

    # Defining number of produced photons from quencing.py
    n_photons = light_dat['n_photons_edep'][:,0]

    nEdepSegments = t_data.shape[0]
    
    # Loop edep positions
    for edepInd in range(nEdepSegments):

        # Global position
        pos = (np.array((z[edepInd],y[edepInd],x[edepInd])))*consts.cm2mm

        # tpc
        itpc = get_half_det_copy(pos)
        
        # LUT position
        lut_pos = larnd_to_lut_coord(pos, lut_geometry, itpc) 

        # voxel containing LUT position
        voxel = get_voxel(lut_pos,lut_geometry)
        
        # Calls voxel data
        lut_vox = np_lut[np_lut['Voxel'] == voxel]
        
        op_dat = lut_vox['OpChannel']
        vis_dat = lut_vox['Visibility']
        T1_dat = lut_vox['T1']
        
        # loop voxel entry-list
        for entry in range(len(lut_vox)):
            # Defines which op channel data is collected from
            op_channel = op_dat[entry]
            
            # Calculates the number of photons reaching each optical channel
            n_photons_read = vis_dat[entry]*n_photons[edepInd]
            
            # Flips op channels if edep occurs in tpc 2
            if (itpc==1):
                op_channel = (op_channel+consts.n_op_channel/2)%consts.n_op_channel
            
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
    light_dat['n_photons_det'] += tphotons
    light_dat['t0_det'] += time
