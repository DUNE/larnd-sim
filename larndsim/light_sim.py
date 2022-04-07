"""
Module that simulates smearing effects of the light incident on each
photodetector
"""

import numba as nb

from numba import cuda

import numpy as np

from .consts import light
from .consts.light import LIGHT_TICK_SIZE, LIGHT_WINDOW
from .consts.detector import TPC_BORDERS
from .consts import units as units

def get_nticks(light_incidence):
    """
    Calculates the number of time ticks needed to simulate light signals of the
    event (plus the desired pre- and post-intervals)

    Args:
        light_incidence(array): shape `(ntracks, ndet)`, containing first hit time and number of photons on each detector

    Returns:
        tuple: number of time ticks (`int`), time of first tick (`float`) [in microseconds]
    """
    mask = light_incidence['n_photons_det'] > 0
    start_time = np.min(light_incidence['t0_det'][mask]) - LIGHT_WINDOW[0]
    end_time = np.max(light_incidence['t0_det'][mask]) + LIGHT_WINDOW[1]
    return int(np.ceil((end_time - start_time)/LIGHT_TICK_SIZE)), start_time

@cuda.jit
def sum_light_signals(segments, segment_voxel, light_inc, lut, start_time, light_sample_inc):
    """
    Sums the number of photons observed by each light detector at each time tick

    Args:
        segments(array): shape `(ntracks,)`, edep-sim tracks to simulate
        segment_voxel(array): shape `(ntracks, 3)`, LUT voxel for eack edep-sim track
        light_inc(array): shape `(ntracks, ndet)`, number of photons incident on each detector and voxel id
        lut(array): shape `(nx,ny,nz)`, light look up table
        start_time(float): start time of light simulation in microseconds
        light_sample_inc(array): output array, shape `(ndet, nticks)`, number of photons incident on each detector at each time tick (propogation delay only)
    """
    idet,itick = cuda.grid(2)

    if idet < light_sample_inc.shape[0]:
        if itick < light_sample_inc.shape[1]:

            start_tick_time = itick * LIGHT_TICK_SIZE + start_time
            end_tick_time = start_tick_time + LIGHT_TICK_SIZE

            # find tracks that contribute light to this time tick
            for itrk in range(segments.shape[0]):
                if light_inc[itrk,idet]['n_photons_det'] > 0:
                    voxel = segment_voxel[itrk]
                    time_profile = lut[voxel[0],voxel[1],voxel[2],idet]['time_dist']
                    track_time = segments[itrk]['t0']
                    track_end_time = track_time + time_profile.shape[0] * units.ns / units.mus # FIXME: assumes light LUT time profile bins are 1ns (might not be true in general)

                    if track_end_time < start_tick_time or track_time > end_tick_time:
                        continue

                    # normalize propogation delay time profile
                    norm = 0
                    for iprof in range(time_profile.shape[0]):
                        norm += time_profile[iprof]

                    # add photons to time tick
                    for iprof in range(time_profile.shape[0]):
                        profile_time = track_time + iprof * units.ns / units.mus # FIXME: assumes light LUT time profile bins are 1ns (might not be true in general)
                        if profile_time < end_tick_time and profile_time > start_tick_time:
                            light_sample_inc[idet,itick] += light_inc['n_photons_det'][itrk,idet] * time_profile[iprof] / norm

