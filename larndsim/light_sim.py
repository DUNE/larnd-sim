"""
Module that simulates smearing effects of the light incident on each
photodetector
"""

import numba as nb

from numba import cuda

import numpy as np
from math import ceil, exp, sqrt, sin

from .consts import light
from .consts.light import LIGHT_TICK_SIZE, LIGHT_WINDOW, SINGLET_FRACTION, TAU_S, TAU_T, LIGHT_GAIN, LIGHT_OSCILLATION_PERIOD, LIGHT_RESPONSE_TIME, LIGHT_DET_NOISE_SAMPLE_SPACING
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


@nb.njit
def scintillation_model(time_tick):
    """
    Calculates the fraction of scintillation photons emitted 
    during time interval `time_tick` to `time_tick + 1`
    
    Args:
        time_tick(int): time tick relative to t0
    
    Returns:
        float: fraction of scintillation photons
    """
    p1 = SINGLET_FRACTION * exp(-time_tick * LIGHT_TICK_SIZE / TAU_S) * (1 - exp(-LIGHT_TICK_SIZE / TAU_S))
    p3 = (1 - SINGLET_FRACTION) * exp(-time_tick * LIGHT_TICK_SIZE / TAU_T) * (1 - exp(-LIGHT_TICK_SIZE / TAU_T))
    return (p1 + p3) * (time_tick >= 0)


@cuda.jit
def calc_scintillation_effect(light_sample_inc, light_sample_inc_scint):
    """
    Applies a smearing effect due to the liquid argon scintillation time profile using
    a two decay component scintillation model.
    
    Args:
        light_sample_inc(array): shape `(ndet, ntick)`, light incident on each detector
        light_sample_inc_scint(array): output array, shape `(ndet, ntick)`, light incident on each detector after accounting for scintillation time
    """
    idet,itick = cuda.grid(2)

    if idet < light_sample_inc.shape[0]:
        if itick < light_sample_inc.shape[1]:
            conv_ticks = ceil((LIGHT_WINDOW[1] - LIGHT_WINDOW[0])/LIGHT_TICK_SIZE)
            
            for jtick in range(max(itick - conv_ticks, 0), itick+1):
                light_sample_inc_scint[idet,itick] += scintillation_model(itick-jtick) * light_sample_inc[idet,jtick]


@nb.njit
def xoroshiro128p_poisson_int32(mean, states, index):
    """
    Return poisson distributed int32 and advance `states[index]`
    
    Args:
        mean(float): mean of poisson distribution
        states(array): array of RNG states
        index(int): offset in states to update
    """
    if mean < 100: # poisson statistics are important
        u = cuda.random.xoroshiro128p_uniform_float32(states, index)
        x = 0
        p = exp(-mean)
        s = p
        while u > s:
            x += 1
            p = p * mean / x
            s = s + p
        return x
    return max(int(cuda.random.xoroshiro128p_normal_float32(states, index) * sqrt(mean) + mean),0)
    
                
@cuda.jit
def calc_stat_fluctuations(light_sample_inc, light_sample_inc_disc, rng_states):
    """
    Simulates Poisson fluctuations in the number of PE per time tick.
    
    Args:
        light_sample_inc(array): shape `(ndet, ntick)`, light incident on each detector
        light_sample_inc_disc(array): output array, shape `(ndet, ntick)`, number PE in each time interval
        rng_states(array): shape `(>ndet*ntick,)`, random number states
    """
    idet,itick = cuda.grid(2)

    if idet < light_sample_inc.shape[0]:
        if itick < light_sample_inc.shape[1]:
            if light_sample_inc[idet,itick] > 0:
                light_sample_inc_disc[idet,itick] = 1. * xoroshiro128p_poisson_int32(
                    light_sample_inc[idet,itick], rng_states, idet*light_sample_inc.shape[1] + itick)
            else:
                light_sample_inc_disc[idet,itick] = 0.
                

@nb.njit
def sipm_response_model(idet, time_tick):
    """
    Calculates the SiPM response from a PE at `time_tick` relative to the PE time
    
    Args:
        idet(int): SiPM index
        time_tick(int): time tick relative to t0
    
    Returns:
        float: response
    """
    t = time_tick * LIGHT_TICK_SIZE
    impulse = (t>=0) * exp(-t/LIGHT_RESPONSE_TIME) * sin(t/LIGHT_OSCILLATION_PERIOD)
    # normalize to 1
    impulse /= LIGHT_OSCILLATION_PERIOD * LIGHT_RESPONSE_TIME**2
    impulse *= LIGHT_OSCILLATION_PERIOD**2 + LIGHT_RESPONSE_TIME**2
    return impulse * LIGHT_TICK_SIZE
            

@cuda.jit
def calc_light_detector_response(light_sample_inc, light_response):
    """
    Simulates the SiPM reponse and digitization
    
    Args:
        light_sample_inc(array): shape `(ndet, ntick)`, PE produced on each SiPM at each time tick
        light_response(array): shape `(ndet, ntick)`, ADC value at each time tick
    """
    idet,itick = cuda.grid(2)

    if idet < light_sample_inc.shape[0]:
        if itick < light_sample_inc.shape[1]:
            conv_ticks = ceil((LIGHT_WINDOW[1] - LIGHT_WINDOW[0])/LIGHT_TICK_SIZE)
            
            for jtick in range(max(itick - conv_ticks, 0), itick+1):
                light_response[idet,itick] += LIGHT_GAIN[idet] * sipm_response_model(idet, itick-jtick) * light_sample_inc[idet,jtick]
                

def gen_light_detector_noise(shape, light_det_noise):
    noise_freq = np.fft.rfftfreq((light_det_noise.shape[-1]-1)*2, d=LIGHT_DET_NOISE_SAMPLE_SPACING)
    desired_freq = np.fft.rfftfreq(shape[-1], d=LIGHT_TICK_SIZE)
    
    bin_size = np.diff(desired_freq).mean()
    noise_spectrum = np.zeros((shape[0], desired_freq.shape[0]))
    for idet in range(shape[0]):
        noise_spectrum[idet] = np.interp(desired_freq, noise_freq, light_det_noise[idet] * np.diff(noise_freq).mean(), left=0, right=0) / (bin_size)
    
    noise = np.fft.irfft(noise_spectrum * np.exp(1j * np.random.uniform(size=noise_spectrum.shape) * 2* np.pi), axis=-1)
    if noise.shape != shape:
        noise = np.concatenate([noise,np.zeros((noise.shape[0],shape[1]-noise.shape[1]))],axis=-1)
    return noise