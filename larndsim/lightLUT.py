"""
Module that simulates the scattering of photons throughout the detector from the
location of the edep to the location of each photodetector
"""

import numba as nb

from numba import cuda

from .consts import light
from .consts.light import LUT_VOX_DIV, OP_CHANNEL_EFFICIENCY
from .consts.detector import TPC_BORDERS
from .consts import units as units

@nb.njit
def get_voxel(pos, itpc):
    """
    Finds and returns the indices of the voxel in which the edep occurs.

    Args:
        pos (tuple): x, y, z coordinates within a generic TPC volume
        itpc (int): index of the tpc corresponding to this position (calculated in drift)
    Returns:
        tuple: indices (in x, y, z dimensions) of the voxel containing the input position
    """

    this_tpc_borders = TPC_BORDERS[itpc]

    # If we are in an "odd" TPC, that is, if the index of
    # the tpc is an odd number, we need to rotate x
    # this is to preserve the "left/right"-ness of the optical channels
    # with respect to the anode plane
    is_even = this_tpc_borders[2][1] > this_tpc_borders[2][0]

    # Assigns tpc borders to variables
    # +- 2e-2 mimics the logic used in drifting.py to prevent event
    # voxel indicies from being located outside the LUT
    x_min = this_tpc_borders[0][0] - 2e-2
    x_max = this_tpc_borders[0][1] + 2e-2
    y_min = this_tpc_borders[1][0] - 2e-2
    y_max = this_tpc_borders[1][1] + 2e-2
    z_min = this_tpc_borders[2][0] - 2e-2
    z_max = this_tpc_borders[2][1] + 2e-2

    # Determines which voxel the event takes place in
    # based on the fractional dstance the event takes place in the volume
    # for the x, y, and z dimensions
    if is_even:
        i = int((pos[0] - x_min)/(x_max - x_min) * LUT_VOX_DIV[0])
    else:
        # if is_even, is false we measure i from the xMax side
        # rather than the xMin side as means of rotating the x component
        i = int((x_max - pos[0])/(x_max - x_min) * LUT_VOX_DIV[0])

    j = int((pos[1] - y_min)/(y_max - y_min) * LUT_VOX_DIV[1])
    k = int((pos[2] - z_min)/(z_max - z_min) * LUT_VOX_DIV[2])

    return i, j, k

@cuda.jit
def calculate_light_incidence(tracks, lut, light_incidence):
    """
    Simulates the number of photons read by each optical channel depending on
        where the edep occurs as well as the time it takes for a photon to reach the
        nearest photomultiplier tube (the "fastest" photon)

    Args:
        tracks (:obj:`numpy.ndarray`): track array containing edep segments, positions are used for lookup.
        lut (:obj:`numpy.ndarray`): Numpy array (.npy) containing light lookup table.
        light_incidence (:obj:`numpy.ndarray`): to contain the result of light incidence calculation.
            This array has dimension (n_tracks, n_optical_channels) and each entry
            is a structure of type (n_photons_det (float32), t0_det (float32)).
            These correspond to the number detected in each channel (n_photons_edep*visibility),
            and the time of earliest arrival at that channel.
    """
    itrk = cuda.grid(1)

    if itrk < tracks.shape[0]:

        # Global position
        pos = (tracks['x'][itrk],tracks['y'][itrk],tracks['z'][itrk])

        # Defining number of produced photons from quencing.py
        n_photons = tracks['n_photons'][itrk]

        # Identifies which tpc event takes place in
        itpc = tracks["pixel_plane"][itrk]

        # Voxel containing LUT position
        voxel = get_voxel(pos, itpc)

        # Calls data from voxel
        lut_vox = lut[voxel[0], voxel[1], voxel[2]]

        # Calls visibility data for the voxel
        vis_dat = lut_vox['vis']

        # Calls T1 data for the voxel
        T1_dat = lut_vox['t0']

        # Assigns the LUT data to the light_incidence array
        for output_i in range(light.N_OP_CHANNEL):
            op_channel_index = output_i + int(itpc*light.N_OP_CHANNEL)
            eff = OP_CHANNEL_EFFICIENCY[output_i]
            vis = vis_dat[output_i]
            t1 = (T1_dat[output_i] * units.ns + tracks['t0'][itrk] * units.mus) / units.mus

            light_incidence['n_photons_det'][itrk,op_channel_index] = eff*vis*n_photons
            light_incidence['t0_det'][itrk,op_channel_index] = t1
