"""
Module that simulates the scattering of photons throughout the detector from the
location of the edep to the location of each photodetector
"""

import numba as nb

from numba import cuda

#from .consts import light
#from .consts.light import OP_CHANNEL_EFFICIENCY, OP_CHANNEL_TO_TPC
#from .consts.detector import TPC_BORDERS
from .consts import units, detector, light

@nb.njit
def get_voxel(pos, itpc, lut_vox_div):
    """
    Finds and returns the indices of the voxel in which the edep occurs.

    Args:
        pos (tuple): x, y, z coordinates within a generic TPC volume
        itpc (int): index of the tpc corresponding to this position (calculated in drift)
        lut_vox_div (tuple): number of lut voxels in x,y,z direction
    Returns:
        tuple: indices (in x, y, z dimensions) of the voxel containing the input position
    """

    this_tpc_borders = detector.TPC_BORDERS[itpc]

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
        i = int((pos[0] - x_min)/(x_max - x_min) * lut_vox_div[0])
    else:
        # if is_even, is false we measure i from the xMax side
        # rather than the xMin side as means of rotating the x component
        i = int((x_max - pos[0])/(x_max - x_min) * lut_vox_div[0])

    j = int((y_max - pos[1])/(y_max - y_min) * lut_vox_div[1])
    k = int((pos[2] - z_min)/(z_max - z_min) * lut_vox_div[2])

    i = min(lut_vox_div[0] - 1, max(0, i))
    j = min(lut_vox_div[1] - 1, max(0, j))
    k = min(lut_vox_div[2] - 1, max(0, k))

    return i, j, k

@cuda.jit
def calculate_light_incidence(tracks, lut, light_incidence, voxel):
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
        voxel (:obj:`numpy.ndarray`): to contain the voxel for each track, dimension (n_tracks, 3)
    """
    itrk = cuda.grid(1)

    if itrk < tracks.shape[0]:

        # Global position
        pos = (tracks['x'][itrk],tracks['y'][itrk],tracks['z'][itrk])

        # Defining number of produced photons from quencing.py
        n_photons = tracks['n_photons'][itrk]

        # Identifies which tpc and module event takes place in
        itpc = tracks["pixel_plane"][itrk]
        imod = itpc // 2

        # ignore any edeps with the default itpc value,
        # they are outside any tpc
        if itpc != detector.DEFAULT_PLANE_INDEX:
            # Voxel containing LUT position
            lut_vox_div = lut.shape[:-1]
            i_voxel = get_voxel(pos, itpc,lut_vox_div)
            voxel[itrk,0] = i_voxel[0]
            voxel[itrk,1] = i_voxel[1]
            voxel[itrk,2] = i_voxel[2]

            # Calls data from voxel
            lut_vox = lut[i_voxel[0], i_voxel[1], i_voxel[2]]

            # Calls visibility data for the voxel
            vis_dat = lut_vox['vis']

            if light.LIGHT_TRIG_MODE == 0:
                # Calls T1 data for the voxel
                T1_dat = lut_vox['t0']

            # When mod2mod variation is enabled, we simulate one module at a
            # time. In that case, use channel_offset to go from "relative" to
            # "absolute" channels when doing lookups in e.g.
            # OP_CHANNEL_EFFICIENCY.
            if light_incidence.shape[1] < light.N_OP_CHANNEL:
                channel_offset = light_incidence.shape[1] * imod
            else:
                channel_offset = 0

            # Assigns the LUT data to the light_incidence array
            for output_i in range(light_incidence.shape[1]):
                op_channel_index = output_i + channel_offset
                lut_index = output_i % vis_dat.shape[0]

                eff = light.OP_CHANNEL_EFFICIENCY[op_channel_index]
                vis = vis_dat[lut_index] * (light.OP_CHANNEL_TO_TPC[op_channel_index] == itpc)
                light_incidence['n_photons_det'][itrk, output_i] = eff * vis * n_photons

                if light.LIGHT_TRIG_MODE == 0:
                    t1 = (T1_dat[lut_index] * units.ns + tracks['t0'][itrk] * units.mus) / units.mus
                    light_incidence['t0_det'][itrk, output_i] = t1
