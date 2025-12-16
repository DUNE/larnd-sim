"""
Module that simulates smearing effects of the light incident on each
photodetector
"""

import numba as nb

from numba import cuda

import numpy as np
import cupy as cp
from math import ceil, floor, exp, sqrt, sin

import h5py

#from .consts.light import LIGHT_TICK_SIZE, LIGHT_WINDOW, SINGLET_FRACTION, TAU_S, TAU_T, LIGHT_GAIN, LIGHT_OSCILLATION_PERIOD, LIGHT_RESPONSE_TIME, LIGHT_DET_NOISE_SAMPLE_SPACING, LIGHT_TRIG_MODE, LIGHT_TRIG_THRESHOLD, LIGHT_TRIG_WINDOW, LIGHT_DIGIT_SAMPLE_SPACING, LIGHT_NBIT, OP_CHANNEL_TO_TPC, OP_CHANNEL_PER_TRIG, TPC_TO_OP_CHANNEL, SIPM_RESPONSE_MODEL, IMPULSE_TICK_SIZE, IMPULSE_MODEL, MC_TRUTH_THRESHOLD, ENABLE_LUT_SMEARING
                                    if photons > sim.MC_TRUTH_THRESHOLD:
@nb.njit
            for itrk in sorted_indices[idet]:
    """
    Calculates the fraction of scintillation photons emitted
    during time interval `time_tick` to `time_tick + 1`.
    If triple_exponential is True, includes an intermediate component.

    Args:
        time_tick(int): time tick relative to t0
        triple_exponential (bool): If True, use triple exponential model
        intermediate_fraction (float or None): Fraction for intermediate component (if None, uses light.INTERMEDIATE_FRACTION)
        tau_i (float or None): Intermediate decay time (if None, uses light.TAU_I)

    Returns:
        float: fraction of scintillation photons
    """
    singlet = light.SINGLET_FRACTION
    triplet = 1.0 - singlet
    interm = 0.0
    tau_s = light.TAU_S
    tau_t = light.TAU_T
    tau_i_val = light.TAU_I if tau_i is None else tau_i
    interm_frac = light.INTERMEDIATE_FRACTION if intermediate_fraction is None else intermediate_fraction
    if triple_exponential:
        # Repartition fractions
        triplet = 1.0 - singlet - interm_frac
        interm = interm_frac
    t = time_tick * light.LIGHT_TICK_SIZE
    dt = light.LIGHT_TICK_SIZE
    p1 = singlet * exp(-t / tau_s) * (1 - exp(-dt / tau_s))
    p2 = interm * exp(-t / tau_i_val) * (1 - exp(-dt / tau_i_val)) if triple_exponential else 0.0
    p3 = triplet * exp(-t / tau_t) * (1 - exp(-dt / tau_t))
    return (p1 + p2 + p3) * (t >= 0)
                n_photons = light_inc[itrk,op_channel[idet]]['n_photons_det']
                if n_photons > 0:
                    voxel = segment_voxel[itrk]
                    track_time = segments[itrk]['t0']
                    # For each segment, build its time profile (LUT or average)
                    if light.ENABLE_LUT_SMEARING:
                        time_profile = lut[voxel[0],voxel[1],voxel[2],idet_lut]['time_dist']
                        n_profile = time_profile.shape[0]
                        for iprof in range(n_profile):
                            profile_time = track_time + iprof * units.ns / units.mus
                            for iscin in range(scint_model.shape[0]):
                                scint_time = profile_time + iscin * light.LIGHT_TICK_SIZE
                                scint_weight = scint_model[iscin]
                                tick_time = itick * light.LIGHT_TICK_SIZE + start_time
                                if abs(tick_time - scint_time) < 0.5 * light.LIGHT_TICK_SIZE:
                                    photons = n_photons * time_profile[iprof] * scint_weight / light.LIGHT_TICK_SIZE
                                    light_sample_inc[idet,itick] += photons
                                    if photons > sim.MC_TRUTH_THRESHOLD:
                                        for itrue in range(light_sample_inc_true_track_id.shape[-1]):
                                            if light_sample_inc_true_track_id[idet,itick,itrue] == -1 or light_sample_inc_true_track_id[idet,itick,itrue] == segment_track_id[itrk]:
                                                light_sample_inc_true_track_id[idet,itick,itrue] = segment_track_id[itrk]
                                                light_sample_inc_true_photons[idet,itick,itrue] += photons
                                                break
                    else:
                        t0_avg = lut[voxel[0],voxel[1],voxel[2],idet_lut]['t0_avg'] * units.ns / units.mus
                        for iscin in range(scint_model.shape[0]):
                            scint_time = track_time + t0_avg + iscin * light.LIGHT_TICK_SIZE
                            scint_weight = scint_model[iscin]
                            tick_time = itick * light.LIGHT_TICK_SIZE + start_time
                            if abs(tick_time - scint_time) < 0.5 * light.LIGHT_TICK_SIZE:
                                photons = n_photons * scint_weight / light.LIGHT_TICK_SIZE
                                light_sample_inc[idet,itick] += photons
                                if photons > sim.MC_TRUTH_THRESHOLD:
                                    for itrue in range(light_sample_inc_true_track_id.shape[-1]):
                                        if light_sample_inc_true_track_id[idet,itick,itrue] == -1 or light_sample_inc_true_track_id[idet,itick,itrue] == segment_track_id[itrk]:
                                            light_sample_inc_true_track_id[idet,itick,itrue] = segment_track_id[itrk]
                                            light_sample_inc_true_photons[idet,itick,itrue] += photons
                                            break
                                        for itrue in range(light_sample_inc_true_track_id.shape[-1]):
                                            if light_sample_inc_true_track_id[idet,itick,itrue] == -1 or light_sample_inc_true_track_id[idet,itick,itrue] == segment_track_id[itrk]:
                                                light_sample_inc_true_track_id[idet,itick,itrue] = segment_track_id[itrk]
                                                light_sample_inc_true_photons[idet,itick,itrue] += photons
                                                break
            end_tick_time = start_tick_time + light.LIGHT_TICK_SIZE

            # find tracks that contribute light to this time tick
            idet_lut = op_channel[idet] % lut.shape[3]
            for itrk in sorted_indices[idet]:
                if light_inc[itrk,op_channel[idet]]['n_photons_det'] > 0:
                    voxel = segment_voxel[itrk]
                    track_time = segments[itrk]['t0']
                    track_end_time = track_time + t0_profile_length * units.ns / units.mus # FIXME: assumes light LUT time profile bins are 1ns (might not be true in general)

                    if track_end_time < start_tick_time or track_time > end_tick_time:
                        continue

                    # use LUT time smearing
                    if light.ENABLE_LUT_SMEARING:
                        time_profile = lut[voxel[0],voxel[1],voxel[2],idet_lut]['time_dist'] # normalised

                        # add photons to time tick
                        for iprof in range(time_profile.shape[0]):
                            profile_time = track_time + iprof * units.ns / units.mus # FIXME: assumes light LUT time profile bins are 1ns (might not be true in general)
                            if profile_time < end_tick_time and profile_time > start_tick_time:
                                photons = light_inc['n_photons_det'][itrk,op_channel[idet]] * time_profile[iprof] / light.LIGHT_TICK_SIZE
                                light_sample_inc[idet,itick] += photons

                                if photons > sim.MC_TRUTH_THRESHOLD:
                                    # get truth information for time tick
                                    for itrue in range(light_sample_inc_true_track_id.shape[-1]):
                                        if light_sample_inc_true_track_id[idet,itick,itrue] == -1 or light_sample_inc_true_track_id[idet,itick,itrue] == segment_track_id[itrk]:
                                            light_sample_inc_true_track_id[idet,itick,itrue] = segment_track_id[itrk]
                                            light_sample_inc_true_photons[idet,itick,itrue] += photons
                                            break
                    # use average time only
                    else:
                        # calculate average delay time
                        t0_avg = lut[voxel[0],voxel[1],voxel[2],idet_lut]['t0_avg'] * units.ns / units.mus # normalised averagein us

                        # add photons to time tick
                        profile_time = track_time + t0_avg
                        if profile_time < end_tick_time and profile_time > start_tick_time:
                            photons = light_inc['n_photons_det'][itrk,op_channel[idet]] / light.LIGHT_TICK_SIZE
                            light_sample_inc[idet,itick] += photons
                            if photons > sim.MC_TRUTH_THRESHOLD:
                                # get truth information for time tick
                                for itrue in range(light_sample_inc_true_track_id.shape[-1]):
                                    if light_sample_inc_true_track_id[idet,itick,itrue] == -1 or light_sample_inc_true_track_id[idet,itick,itrue] == segment_track_id[itrk]:
                                        light_sample_inc_true_track_id[idet,itick,itrue] = segment_track_id[itrk]
                                        light_sample_inc_true_photons[idet,itick,itrue] += photons
                                        break

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
    p1 = light.SINGLET_FRACTION * exp(-time_tick * light.LIGHT_TICK_SIZE / light.TAU_S) * (1 - exp(-light.LIGHT_TICK_SIZE / light.TAU_S))
    p3 = (1 - light.SINGLET_FRACTION) * exp(-time_tick * light.LIGHT_TICK_SIZE / light.TAU_T) * (1 - exp(-light.LIGHT_TICK_SIZE / light.TAU_T))
    return (p1 + p3) * (time_tick >= 0)

@nb.njit
def scintillation_array(scint_model):
    """
    Calculates the fraction of scintillation photons emitted
    during time interval `time_tick` to `time_tick + 1` for
    the entire input array.

    Args:
        scint_model: array to store result
    """
    for time_tick in range(scint_model.shape[0]):
        p1 = light.SINGLET_FRACTION * exp(-time_tick * light.LIGHT_TICK_SIZE / light.TAU_S) * (1 - exp(-light.LIGHT_TICK_SIZE / light.TAU_S))
        p3 = (1 - light.SINGLET_FRACTION) * exp(-time_tick * light.LIGHT_TICK_SIZE / light.TAU_T) * (1 - exp(-light.LIGHT_TICK_SIZE / light.TAU_T))
        scint_model[time_tick] = p1 + p3

def calc_scintillation_effect(light_sample_inc, light_sample_inc_true_track_id, light_sample_inc_true_photons, light_sample_inc_scint, light_sample_inc_scint_true_track_id, light_sample_inc_scint_true_photons, scint_model):
@nb.njit
            conv_ticks = ceil((light.LIGHT_WINDOW[1] - light.LIGHT_WINDOW[0])/light.LIGHT_TICK_SIZE)
    """
    Calculates the fraction of scintillation photons emitted
    during time interval `time_tick` to `time_tick + 1` for
    the entire input array.
    If triple_exponential is True, includes an intermediate component.

    Args:
        scint_model: array to store result
        triple_exponential (bool): If True, use triple exponential model
        intermediate_fraction (float or None): Fraction for intermediate component
        tau_i (float or None): Intermediate decay time
    """
    for time_tick in range(scint_model.shape[0]):
        scint_model[time_tick] = scintillation_model(time_tick, triple_exponential, intermediate_fraction, tau_i)

            for jtick in range(max(itick - conv_ticks, 0), itick+1):
                if light_sample_inc[idet,jtick] == 0:
                    continue
                tick_weight = scint_model[itick-jtick]
                light_sample_inc_scint[idet,itick] += tick_weight * light_sample_inc[idet,jtick]

                # loop over convolution tick truth
                for itrue in range(light_sample_inc_true_track_id.shape[-1]):
                    if light_sample_inc_true_track_id[idet,jtick,itrue] == -1:
                        break

                    if tick_weight * light_sample_inc_true_photons[idet,jtick,itrue] < sim.MC_TRUTH_THRESHOLD:
                        continue

                    # loop over current tick truth
                    for jtrue in range(light_sample_inc_scint_true_track_id.shape[-1]):
                        if light_sample_inc_scint_true_track_id[idet,itick,jtrue] == light_sample_inc_true_track_id[idet,jtick,itrue] or light_sample_inc_scint_true_track_id[idet,itick,jtrue] == -1:
                            light_sample_inc_scint_true_track_id[idet,itick,jtrue] = light_sample_inc_true_track_id[idet,jtick,itrue]
                            light_sample_inc_scint_true_photons[idet,itick,jtrue] += tick_weight * light_sample_inc_true_photons[idet,jtick,itrue]
                            break


@nb.njit
def xoroshiro128p_poisson_int32(mean, states, index):
    """
    Return poisson distributed int32 and advance `states[index]`. For efficiency,
    if `mean > 30`, returns a gaussian distributed int32 with `mean == mean`
    and `std = sqrt(mean)` truncated at 0 (approximately equivalent to a poisson-
    distributed number)

    [DOI:10.1007/978-1-4613-8643-8_10]

    Args:
        mean(float): mean of poisson distribution
        states(array): array of RNG states
        index(int): offset in states to update
    """
    if mean <= 0:
        return 0
    if mean < 30: # poisson statistics are important
        u = cuda.random.xoroshiro128p_uniform_float32(states, index)
        x = 0
        p = exp(-mean)
        s = p
        prev_s = s
        while u > s:
            x += 1
            p = p * mean / x
            prev_s = s
            s = s + p
            if s == prev_s: # break if machine precision reached
                break
        return x
    return max(int(cuda.random.xoroshiro128p_normal_float32(states, index) * sqrt(mean) + mean),0)


@cuda.jit
def calc_stat_fluctuations(light_sample_inc, light_sample_inc_disc, rng_states):
    """
    Simulates Poisson fluctuations in the number of PE per time tick.

    Args:
        light_sample_inc(array): shape `(ndet, ntick)`, effective photocurrent on each detector
        light_sample_inc_disc(array): output array, shape `(ndet, ntick)`, effective photocurrent on each detector (with stocastic fluctuations)
        rng_states(array): shape `(>ndet*ntick,)`, random number states
    """
    idet,itick = cuda.grid(2)

    if idet < light_sample_inc.shape[0]:
        if itick < light_sample_inc.shape[1]:
            if light_sample_inc[idet,itick] > 0:
                light_sample_inc_disc[idet,itick] = 1. / light.LIGHT_TICK_SIZE * xoroshiro128p_poisson_int32(
                    light_sample_inc[idet,itick] * light.LIGHT_TICK_SIZE, rng_states, idet*light_sample_inc.shape[1] + itick)
            else:
                light_sample_inc_disc[idet,itick] = 0.


@nb.njit
def interp(idx, arr, low, high):
    """
    Performs a simple linear interpolation of an array at a given floating point
    index

    Args:
        idx(float): index into array to interpolate
        arr(array): 1D array of values to interpolate
        low(float): value to return if index is less than 0
        high(float): value to return if index is above `len(arr)-1`

    Returns:
        float: interpolated array value
    """
    i0 = int(floor(idx))

    if i0 < 0:
        return low
        return high
    @cuda.jit
    if i0 == idx:
        return arr[i0]
    if i0 > len(arr)-2:
        return high

    i1 = i0 + 1
        Args: (same as before, but no sipm_response)
            scint_model(array): LAr scintillation time profile (default, for fixed singlet fraction)
            variable_singlet_fraction (bool): If True, use per-segment singlet fraction (based on PDG code or other info)
    return v0 + (v1 - v0) * (idx - i0)

@nb.njit()
def sipm_response_model(time_tick):
    """
    Calculates the SiPM response from a PE at `time_tick` relative to the PE time

    Args:
        idet(int): SiPM index
        time_tick(int): time tick relative to t0

                        # Optionally get per-segment singlet fraction (e.g., by PDG code)
                        singlet_fraction = light.SINGLET_FRACTION
                        if variable_singlet_fraction:
                            pdg = 0
                            if 'pdg' in segments.dtype.fields:
                                pdg = segments[itrk]['pdg']
                            # Set singlet fraction for neutrons or any nuclei (PDG > 1e9 covers all nuclei, including alphas)
                            # Neutron: PDG 2112, nuclei: |PDG| > 1000000000
                            if pdg == 2112 or abs(pdg) > 1000000000:
                                singlet_fraction = 0.7
                            else:
                                singlet_fraction = 0.3
    Returns:
        float: response
    """
    # use RLC response model
    if light.SIPM_RESPONSE_MODEL == 0:
        t = time_tick * light.LIGHT_TICK_SIZE
                                for iscin in range(scint_model.shape[0]):
                                    scint_time = profile_time + iscin * light.LIGHT_TICK_SIZE
                                    # Compute per-segment scintillation weight if variable_singlet_fraction
                                    if variable_singlet_fraction:
                                        tau_s = light.TAU_S
                                        tau_t = light.TAU_T
                                        t = iscin * light.LIGHT_TICK_SIZE
                                        p1 = singlet_fraction * exp(-t / tau_s) * (1 - exp(-light.LIGHT_TICK_SIZE / tau_s))
                                        p3 = (1 - singlet_fraction) * exp(-t / tau_t) * (1 - exp(-light.LIGHT_TICK_SIZE / tau_t))
                                        scint_weight = (p1 + p3) * (t >= 0)
                                    else:
                                        scint_weight = scint_model[iscin]
                                    tick_time = itick * light.LIGHT_TICK_SIZE + start_time
                                    if abs(tick_time - scint_time) < 0.5 * light.LIGHT_TICK_SIZE:
                                        photons = n_photons * time_profile[iprof] * scint_weight / light.LIGHT_TICK_SIZE
                                        light_sample_inc[idet,itick] += photons
                                        if photons > sim.MC_TRUTH_THRESHOLD:
                                            for itrue in range(light_sample_inc_true_track_id.shape[-1]):
                                                if light_sample_inc_true_track_id[idet,itick,itrue] == -1 or light_sample_inc_true_track_id[idet,itick,itrue] == segment_track_id[itrk]:
                                                    light_sample_inc_true_track_id[idet,itick,itrue] = segment_track_id[itrk]
                                                    light_sample_inc_true_photons[idet,itick,itrue] += photons
                                                    break
@nb.njit()
def sipm_response_array(sipm_response):
                            for iscin in range(scint_model.shape[0]):
                                scint_time = track_time + t0_avg + iscin * light.LIGHT_TICK_SIZE
                                if variable_singlet_fraction:
                                    tau_s = light.TAU_S
                                    tau_t = light.TAU_T
                                    t = iscin * light.LIGHT_TICK_SIZE
                                    p1 = singlet_fraction * exp(-t / tau_s) * (1 - exp(-light.LIGHT_TICK_SIZE / tau_s))
                                    p3 = (1 - singlet_fraction) * exp(-t / tau_t) * (1 - exp(-light.LIGHT_TICK_SIZE / tau_t))
                                    scint_weight = (p1 + p3) * (t >= 0)
                                else:
                                    scint_weight = scint_model[iscin]
                                tick_time = itick * light.LIGHT_TICK_SIZE + start_time
                                if abs(tick_time - scint_time) < 0.5 * light.LIGHT_TICK_SIZE:
                                    photons = n_photons * scint_weight / light.LIGHT_TICK_SIZE
                                    light_sample_inc[idet,itick] += photons
                                    if photons > sim.MC_TRUTH_THRESHOLD:
                                        for itrue in range(light_sample_inc_true_track_id.shape[-1]):
                                            if light_sample_inc_true_track_id[idet,itick,itrue] == -1 or light_sample_inc_true_track_id[idet,itick,itrue] == segment_track_id[itrk]:
                                                light_sample_inc_true_track_id[idet,itick,itrue] = segment_track_id[itrk]
                                                light_sample_inc_true_photons[idet,itick,itrue] += photons
                                                break

    if light.SIPM_RESPONSE_MODEL == 1:
        for time_tick in range(sipm_response.shape[0]):
            sipm_response[time_tick] = interp(time_tick * light.LIGHT_TICK_SIZE / light.IMPULSE_TICK_SIZE, light.IMPULSE_MODEL, 0, 0)
        sipm_response /= light.IMPULSE_TICK_SIZE/light.LIGHT_TICK_SIZE

@cuda.jit
def calc_light_detector_response(light_sample_inc, light_sample_inc_true_track_id, light_sample_inc_true_photons, light_response, light_response_true_track_id, light_response_true_photons, light_gain, sipm_response):
    """
    Simulates the SiPM reponse and digit

    Args:
        light_sample_inc(array): shape `(ndet, ntick)`, PE produced on each SiPM at each time tick
        light_response(array): shape `(ndet, ntick)`, ADC value at each time tick
    """
    idet,itick = cuda.grid(2)

    if idet < light_sample_inc.shape[0]:
        if itick < light_sample_inc.shape[1]:
            conv_ticks = ceil((light.LIGHT_WINDOW[1] - light.LIGHT_WINDOW[0])/light.LIGHT_TICK_SIZE)

            for jtick in range(max(itick - conv_ticks, 0), itick+1):
                tick_weight = sipm_response[itick-jtick]
                light_response[idet,itick] += light_gain[idet] * tick_weight * light_sample_inc[idet,jtick]

                # loop over convolution tick truth
                for itrue in range(light_sample_inc_true_track_id.shape[-1]):
                    if light_sample_inc_true_track_id[idet,jtick,itrue] == -1:
                        break

                    if abs(tick_weight * light_sample_inc_true_photons[idet,jtick,itrue]) < sim.MC_TRUTH_THRESHOLD:
                        continue

                    # loop over current tick truth
                    for jtrue in range(light_response_true_track_id.shape[-1]):
                        # apply convolution if convolution tick matches or if available truth slot
                        if light_sample_inc_true_track_id[idet,itick,jtrue] == light_sample_inc_true_track_id[idet,itick,itrue] or light_sample_inc_true_track_id[idet,itick,jtrue] == -1:
                            light_response_true_track_id[idet,itick,jtrue] = light_sample_inc_true_track_id[idet,itick,itrue]
                            light_response_true_photons[idet,itick,jtrue] += tick_weight * light_sample_inc_true_photons[idet,jtick,itrue]
                            break


def gen_light_detector_noise(shape, light_det_noise):
    """
    Generates uncorrelated noise with a defined frequency spectrum

    Args:
        shape(tuple): desired shape of output noise, `shape[0]` must equal `light_det_noise.shape[0]`
        light_det_noise(array): FFT of noise, `light_det_noise.ndim == 2`

    Returns:
        array: shape `(shape[0], shape[1])`, randomly generated sample noise
    """
    if not shape[0]:
        return cp.empty(shape)

    noise_freq = cp.fft.rfftfreq((light_det_noise.shape[-1]-1)*2, d=light.LIGHT_DET_NOISE_SAMPLE_SPACING)
    desired_freq = cp.fft.rfftfreq(shape[-1], d=light.LIGHT_TICK_SIZE)

    bin_size = cp.diff(desired_freq).mean()
    noise_spectrum = cp.zeros((shape[0], desired_freq.shape[0]))
    for idet in range(shape[0]):
        noise_spectrum[idet] = cp.interp(desired_freq, noise_freq, light_det_noise[idet], left=0, right=0)
    # rescale noise spectrum to have constant noise power with digitizer sample spacing
    noise_spectrum *= cp.sqrt(cp.diff(noise_freq, axis=-1).mean()/bin_size) * light.LIGHT_DIGIT_SAMPLE_SPACING / light.LIGHT_TICK_SIZE


    # generate an FFT with the same frequency power, but with random phase
    noise = noise_spectrum * cp.exp(2j * cp.pi * cp.random.uniform(size=noise_spectrum.shape))
    if shape[1] < 2:
        # special case where inverse FFT does not exist - just generate one sample
        noise = cp.round(cp.real(noise)) * 2**(16-light.LIGHT_NBIT)
    else:
        # invert FFT to create a noise waveform
        noise = cp.round(cp.fft.irfft(noise, axis=-1)) * 2**(16-light.LIGHT_NBIT)

    if noise.shape[1] < shape[1]:
        # FFT must have even samples, so append 0 if an odd number of samples is requested
        noise = cp.concatenate([noise, cp.zeros((noise.shape[0],shape[1]-noise.shape[1]))],axis=-1)

    return noise[:,:shape[1]]


def get_triggers(signal, group_threshold, op_channel_idx, i_subbatch):
    """
    Identifies each simulated ticks that would initiate a trigger taking into account the ADC digitization window

    Args:
        signal(array): shape `(ndet, nticks)`, simulated signal on each channel
        group_threshold(array): shape `(ngrp,)`, threshold on group sum (requires `ndet/ngrp == OP_CHANNEL_PER_TRIG`)
        op_channel_idx(array): shape `(ndet,)`, optical channel index for each signal
        i_subbatch(int): index of the sub_batch numbering ("itrk in the batch for loop")

    Returns:
        tuple: array of tick indices at each trigger (shape `(ntrigs,)`) and array of op channel index (shape `(ntrigs, ndet_module)`)
    """

    shape = signal.shape
    # sum over all signals on a single detector (shape: (ndet, nticks) -> (ngrp, ndetpergrp, nticks) -> (ngrp, 1, nticks))
    signal_sum = signal.reshape(shape[0]//light.OP_CHANNEL_PER_TRIG, light.OP_CHANNEL_PER_TRIG, shape[-1]).sum(axis=1, keepdims=True)
    # reduce to approximate ADC sample rate (padded with zeros)
    sample_factor = round(light.LIGHT_DIGIT_SAMPLE_SPACING / light.LIGHT_TICK_SIZE)
    padding = sample_factor - shape[-1] % sample_factor
    if padding > 0:
        signal_sum = cp.concatenate((signal_sum, cp.zeros((shape[0]//light.OP_CHANNEL_PER_TRIG, 1, padding))), axis=-1)
    signal_sum = signal_sum.reshape(-1, 1, signal_sum.shape[-1]//sample_factor, sample_factor).mean(axis=-1, keepdims=True)
    signal_sum = cp.broadcast_to(signal_sum, signal_sum.shape[:3] + (sample_factor,))
    signal_sum = signal_sum.reshape(-1, 1, shape[-1] + padding)[...,:(-padding if padding > 0 else shape[-1])]

    # apply trigger threshold
    sample_above_thresh = cp.broadcast_to(signal_sum < group_threshold[:,cp.newaxis,cp.newaxis], (shape[0]//light.OP_CHANNEL_PER_TRIG, light.OP_CHANNEL_PER_TRIG, shape[-1]))
    # cast back into the original signal array
    sample_above_thresh = sample_above_thresh.reshape(signal.shape)

    # calculate the minimum number of ticks between two triggers on the same module
    digit_ticks = ceil((light.LIGHT_TRIG_WINDOW[1] + light.LIGHT_TRIG_WINDOW[0])/light.LIGHT_TICK_SIZE)

    tpc_ids = np.unique(light.OP_CHANNEL_TO_TPC[op_channel_idx.get()])
    mod_ids = np.unique([detector.TPC_TO_MODULE[tpc_id] for tpc_id in tpc_ids])

    trigger_idx_list = []
    op_channel_idx_list = []
    trigger_type_list = [] # 0 --> threshold, # 1 --> beam
    if light.LIGHT_TRIG_MODE == 0:
        # treat each module independently
        for mod_id, tpc_ids in [(mod_id, detector.MODULE_TO_TPCS[mod_id]) for mod_id in mod_ids]:
            # get active channels for the module
            op_channels = light.TPC_TO_OP_CHANNEL[tpc_ids].ravel()
            op_channel_mask = np.isin(op_channel_idx.get(), op_channels)
            #module_above_thresh = cp.any(sample_above_thresh[op_channels], axis=0)
            module_above_thresh = np.any(sample_above_thresh[op_channel_mask], axis=0)

            last_trigger = 0
            while cp.any(module_above_thresh):
                # find next time signal goes above threshold
                next_idx = cp.sort(cp.nonzero(module_above_thresh)[0])[0] + (last_trigger if last_trigger != 0 else 0)
                next_trig_type = cp.asarray(0)
                # keep track of trigger time
                trigger_idx_list.append(next_idx)
                trigger_type_list.append(next_trig_type)
                op_channel_idx_list.append(op_channels)
                # ignore samples during digitization window
                module_above_thresh = module_above_thresh[next_idx+digit_ticks:]
                last_trigger = next_idx + digit_ticks

    # if i_subbatch > 0, means this batch/event should already have a trigger
    # then return an empty list
    elif light.LIGHT_TRIG_MODE == 1 and i_subbatch == 0:
        # always add a trigger for a simulated spill/event
        # this function is called in the batch script
        # which means it is executed per event
        # keep track of trigger time (initial comment)
        trigger_idx_list.append(cp.asarray(0)) # the first trigger in the event
        op_channel_idx_list.append(op_channel_idx)
        trigger_type_list.append(cp.asarray(1)) # beam

        ## would we ever get these secondary triggers? -- Not at the moment
        ## 1. currently the internal light simulation window is the same as the light readout window,
        ##    and 16us is large enough (for NuMI at least)
        ## 2. potentially an off-beam event if ever simulated together with the beam, will be considered as a separate event
        ##    therefore, likely will not be in the same batch
        ##    if we ever overlay beam and off-beam, we should consider if we should compare the two signals and overlay the two light signals
        ##    wether they would be in the same readout window; how to deal with the dead time between the two triggers etc...
        #module_above_thresh = np.any(sample_above_thresh, axis=0)
        #module_above_thresh = module_above_thresh[digit_ticks:]
        #last_trigger = digit_ticks
        #while cp.any(module_above_thresh):
        #    # find next time signal goes above threshold
        #    next_idx = cp.sort(cp.nonzero(module_above_thresh)[0])[0] + (last_trigger if last_trigger != 0 else 0)
        #    next_trig_type = cp.asarray(0)
        #    # keep track of trigger time
        #    trigger_idx_list.append(next_idx)
        #    trigger_type_list.append(next_trig_type)
        #    op_channel_idx_list.append(op_channel_idx)
        #    # ignore samples during digitization window
        #    module_above_thresh = module_above_thresh[next_idx+digit_ticks:]
        #    last_trigger = next_idx + digit_ticks

    if len(trigger_idx_list):
        return cp.array(trigger_idx_list), cp.array(op_channel_idx_list), cp.array(trigger_type_list)
    return cp.empty((0,), dtype=int), cp.empty((0, len(op_channel_idx)), dtype=int), cp.empty((0,), dtype=int)


@cuda.jit
def digitize_signal(signal, signal_op_channel_idx, trigger_idx, trigger_op_channel_idx, signal_true_track_id, signal_true_photons, digit_signal, digit_signal_true_track_id, digit_signal_true_photons):
    """
    Interpolate signal to the appropriate sampling frequency

    Args:
        signal(array): shape `(ndet, nticks)`, simulated signal on each channel
        signal_op_channel_idx(array): shape `(ndet,)`, optical channel index for each simulated signal
        trigger_idx(array): shape `(ntrigs,)`, tick index for each trigger
        trigger_op_channel_idx(array): shape `(ntrigs, ndet_module)`, optical channel index for each trigger
        digit_signal(array): output array, shape `(ntrigs, ndet_module, nsamples)`, digitized signal
    """
    itrig,idet_module,isample = cuda.grid(3)

    if itrig < digit_signal.shape[0]:
        if idet_module < digit_signal.shape[1]:
            if isample < digit_signal.shape[2]:
                #sample_tick = isample * light.LIGHT_DIGIT_SAMPLE_SPACING / light.LIGHT_TICK_SIZE - light.LIGHT_TRIG_WINDOW[0] / light.LIGHT_TICK_SIZE + trigger_idx[itrig]
                sample_tick = isample * light.LIGHT_DIGIT_SAMPLE_SPACING / light.LIGHT_TICK_SIZE
                idet = trigger_op_channel_idx[itrig, idet_module]
                idet_signal = 0
                for idet_signal in range(signal.shape[0]):
                    if idet == signal_op_channel_idx[idet_signal]:
                        break
                if idet_signal == signal.shape[0]:
                    return
                digit_signal[itrig,idet_module,isample] = interp(sample_tick, signal[idet_signal], 0, 0)

                itick0 = int(floor(sample_tick))
                itick1 = int(ceil(sample_tick))

                itrue = 0
                # loop over previous tick truth
                for jtrue in range(signal_true_track_id.shape[-1]):
                    if itrue >= digit_signal_true_track_id.shape[-1]:
                        break
                    if signal_true_track_id[idet_signal,itick0,jtrue] == -1:
                        break


                    @cuda.jit
        signal_true_track_id = cp.concatenate([signal_true_track_id, cp.full(pad_shape + signal_true_track_id.shape[-1:], -1, dtype=signal_true_track_id.dtype)], axis=-2)
                        """
                        Sums the number of photons observed by each light detector at each time tick,
                        but first convolves (smears) each segment's LUT output with the SiPM pulse shape
                        before summing. This enables per-segment pulse shape variation in the future.

                        Args: (same as before, plus sipm_response)
                            sipm_response(array): SiPM response function (impulse response)
                        """
                        idet,itick = cuda.grid(2)

                        if idet < light_sample_inc.shape[0]:
                            if itick < light_sample_inc.shape[1]:
                                idet_lut = op_channel[idet] % lut.shape[3]
                                for itrk in sorted_indices[idet]:
                                    n_photons = light_inc[itrk,op_channel[idet]]['n_photons_det']
                                    if n_photons > 0:
                                        voxel = segment_voxel[itrk]
                                        track_time = segments[itrk]['t0']
                                        # For each segment, build its time profile (LUT or average)
                                        if light.ENABLE_LUT_SMEARING:
                                            time_profile = lut[voxel[0],voxel[1],voxel[2],idet_lut]['time_dist']
                                            n_profile = time_profile.shape[0]
                                            # Convolve segment's time profile with SiPM response
                                            for iprof in range(n_profile):
                                                # Time of this profile bin
                                                profile_time = track_time + iprof * units.ns / units.mus
                                                # For each tick in output, apply pulse shape
                                                for iresp in range(sipm_response.shape[0]):
                                                    # Output tick time
                                                    tick_time = itick * light.LIGHT_TICK_SIZE + start_time
                                                    # Time difference between tick and this photon emission
                                                    dt = tick_time - profile_time
                                                    # Find corresponding response bin
                                                    resp_bin = int(dt / light.LIGHT_TICK_SIZE)
                                                    if 0 <= resp_bin < sipm_response.shape[0]:
                                                        photons = n_photons * time_profile[iprof] * sipm_response[resp_bin] / light.LIGHT_TICK_SIZE
                                                        light_sample_inc[idet,itick] += photons
                                                        if photons > sim.MC_TRUTH_THRESHOLD:
                                                            for itrue in range(light_sample_inc_true_track_id.shape[-1]):
                                                                if light_sample_inc_true_track_id[idet,itick,itrue] == -1 or light_sample_inc_true_track_id[idet,itick,itrue] == segment_track_id[itrk]:
                                                                    light_sample_inc_true_track_id[idet,itick,itrue] = segment_track_id[itrk]
                                                                    light_sample_inc_true_photons[idet,itick,itrue] += photons
                                                                    break
                                        else:
                                            t0_avg = lut[voxel[0],voxel[1],voxel[2],idet_lut]['t0_avg'] * units.ns / units.mus
                                            for iresp in range(sipm_response.shape[0]):
                                                tick_time = itick * light.LIGHT_TICK_SIZE + start_time
                                                dt = tick_time - (track_time + t0_avg)
                                                resp_bin = int(dt / light.LIGHT_TICK_SIZE)
                                                if 0 <= resp_bin < sipm_response.shape[0]:
                                                    photons = n_photons * sipm_response[resp_bin] / light.LIGHT_TICK_SIZE
                                                    light_sample_inc[idet,itick] += photons
                                                    if photons > sim.MC_TRUTH_THRESHOLD:
                                                        for itrue in range(light_sample_inc_true_track_id.shape[-1]):
                                                            if light_sample_inc_true_track_id[idet,itick,itrue] == -1 or light_sample_inc_true_track_id[idet,itick,itrue] == segment_track_id[itrk]:
                                                                light_sample_inc_true_track_id[idet,itick,itrue] = segment_track_id[itrk]
                                                                light_sample_inc_true_photons[idet,itick,itrue] += photons
                                                                break
        signal_true_photons = cp.concatenate([signal_true_photons, cp.zeros(pad_shape + signal_true_photons.shape[-1:], dtype=signal_true_photons.dtype)], axis=-2)

    # add noise to padded (in readout time) signal
    signal += cp.array(gen_light_detector_noise(signal.shape, light_det_noise[signal_op_channel_idx]))

    # add noise for any channels that had no signal
    if cp.any(~cp.isin(op_channel_idx, signal_op_channel_idx)):
        missing = cp.unique(op_channel_idx[~cp.isin(op_channel_idx, signal_op_channel_idx)])
        pad_shape = (missing.shape[0], signal.shape[-1])
        signal = cp.concatenate([signal, gen_light_detector_noise(pad_shape, light_det_noise[missing])], axis=0)
        signal_op_channel_idx = cp.concatenate([signal_op_channel_idx, missing], axis=0)
        signal_true_track_id = cp.concatenate([signal_true_track_id, cp.full(pad_shape + signal_true_track_id.shape[-1:], -1, dtype=signal_true_track_id.dtype)], axis=0)
        signal_true_photons = cp.concatenate([signal_true_photons, cp.zeros(pad_shape + signal_true_photons.shape[-1:], dtype=signal_true_photons.dtype)], axis=0)

        # sort to keep consistency across events
        order = cp.argsort(signal_op_channel_idx, axis=-1)
        signal = cp.take_along_axis(signal, order[...,np.newaxis], axis=0)
        signal_op_channel_idx = cp.take_along_axis(signal_op_channel_idx, order, axis=0)
        signal_true_track_id = cp.take_along_axis(signal_true_track_id, order[...,np.newaxis,np.newaxis], axis=0)
        signal_true_photons = cp.take_along_axis(signal_true_photons, order[...,np.newaxis,np.newaxis], axis=0)

    digitize_signal[bpg,tpb](signal, signal_op_channel_idx, padded_trigger_idx, op_channel_idx, signal_true_track_id, signal_true_photons,
        digit_signal, digit_signal_true_track_id, digit_signal_true_photons)

    # truncate to correct number of bits
    digit_signal = cp.round(digit_signal / 2**(16-light.LIGHT_NBIT)) * 2**(16-light.LIGHT_NBIT)

    return digit_signal, digit_signal_true_track_id, digit_signal_true_photons

def zero_suppress_waveform_truth(waveforms_true_track_id, waveforms_true_photons, i_evt, i_trig, i_mod=-1):
    """
    Filter empty light waveform backtracking which is filled with the default value '-1', and flatten the backtracking info to 1D array.

    Args:
        i_evt(int): event id
        waveforms_true_track_id(array): shape `(ntrigs, ndet, nsamples)`, segment ids contributing to each sample
        waveforms_true_photons(array): shape `(ntrigs, ndet, nsamples)`, true photocurrent at each sample
        i_evt(int): true event id
        i_trig(int): light trigger or light event id
        i_mod(int): module id. The default value is -1 which indicates that there is no modular variation activated.

    Returns:
        1D array which logs ['trigger_id', 'op_channel_id', 'tick', 'event_id', 'segment_id', 'pe_current']. The first three locate a unique data point on a recorded light waveform, and the last three provide the truth information associated to it. `event_id` can be infered from `segment_id` but the information helps searching the corresponding reco information from a true event.
    """

    op_channel = light.TPC_TO_OP_CHANNEL[(i_mod-1)*2:i_mod*2].ravel() if i_mod > 0 else light.TPC_TO_OP_CHANNEL[:].ravel()

    # Get total number of non-default entries
    mask = waveforms_true_track_id != -1
    num_idx = np.prod(waveforms_true_track_id[mask].shape)
    # Get indices of those valid entires and destructure the tuple
    idx0, idx1, idx2, idx3 = np.nonzero(waveforms_true_track_id != -1)

    truth_dtype = np.dtype([('trigger_id', 'i4'), ('op_channel_id','i4'), ('tick','i4'), ('event_id','i4'), ('segment_id','i8'), ('pe_current','f8')])
    truth_data = np.empty(num_idx, dtype=truth_dtype)

    # Careful with all the indices
    for x, (i, j, k, l) in enumerate(zip(idx0, idx1, idx2, idx3)):
        this_trig = i #idx0
        i_trig = i_trig + i #FIXME currently idx0 is always 0. probably further change is needed for multiple light triggers in one trueevent
        i_op_channel = j #idx1
        i_sample = k  #idx2
        i_content = l #idx3
        truth_data[x] = (i_trig, op_channel[i_op_channel], i_sample, i_evt,
                         waveforms_true_track_id[this_trig][i_op_channel][i_sample][i_content],
                         waveforms_true_photons[this_trig][i_op_channel][i_sample][i_content])

    return truth_data

def export_light_wvfm_to_hdf5(event_id, waveforms, output_filename, waveforms_true_track_id, waveforms_true_photons, i_trig, i_mod=-1, compression=None):
    """
    Saves waveforms to output file

    Args:
        event_id(array): shape `(ntrigs,)`, event id for each trigger
        waveforms(array): shape `(ntrigs, ndet_module, nsamples)`, simulated waveforms to save
        output_filename(str): output hdf5 file path
        waveforms_true_track_id(array): shape `(ntrigs, ndet, nsamples)`, segment ids contributing to each sample
        waveforms_true_photons(array): shape `(ntrigs, ndet, nsamples)`, true photocurrent at each sample
        i_mod(int): module id. The default value is -1 which indicates that there is no modular variation activated.

    """
    if event_id.shape[0] == 0:
        return

    with h5py.File(output_filename, 'a') as f:

        # the final dataset will be (n_triggers, all op channels in the detector, waveform samples)
        # it would take too much memory if we hold the information until all the modules been simulated
        # therefore, let's store the intermediate data with (n_triggers, op channels in a module, waveform samples) per module
        # and we will cast it into the final shape
        # FIXME currently this does not support threshold triggering
        if sim.MOD2MOD_VARIATION and light.LIGHT_TRIG_MODE == 1:
            if i_mod > 0:
                if f'light_wvfm/light_wvfm_mod{i_mod-1}' not in f:
                    f.create_dataset(f'light_wvfm/light_wvfm_mod{i_mod-1}', data=waveforms, maxshape=(None,None,None), compression=compression)
                else:
                    f[f'light_wvfm/light_wvfm_mod{i_mod-1}'].resize(f[f'light_wvfm/light_wvfm_mod{i_mod-1}'].shape[0] + waveforms.shape[0], axis=0)
                    f[f'light_wvfm/light_wvfm_mod{i_mod-1}'][-waveforms.shape[0]:] = waveforms
            else:
                raise ValueError("Mod2mod variation is activated, but the module id is not provided correctly.")

        else:
            if 'light_wvfm' not in f:
                f.create_dataset('light_wvfm', data=waveforms, maxshape=(None,None,None), compression=compression)
            else:
                f['light_wvfm'].resize(f['light_wvfm'].shape[0] + waveforms.shape[0], axis=0)
                f['light_wvfm'][-waveforms.shape[0]:] = waveforms

        # Store the light truth backtracking, in the same way for module variation turned on and off
        # skip creating the truth dataset if there is no truth information to store
        truth_data=None
        if sim.MAX_MC_TRUTH_IDS > 0:
            truth_data = zero_suppress_waveform_truth(waveforms_true_track_id, waveforms_true_photons, event_id[0], i_trig, i_mod)
            if truth_data.shape[0] > 0:
                if f'light_wvfm_mc_assn' not in f:
                    f.create_dataset(f'light_wvfm_mc_assn', data=truth_data, maxshape=(None,), compression=compression)
                else:
                    f[f'light_wvfm_mc_assn'].resize(f[f'light_wvfm_mc_assn'].shape[0] + truth_data.shape[0], axis=0)
                    f[f'light_wvfm_mc_assn'][-truth_data.shape[0]:] = truth_data

def export_light_trig_to_hdf5(event_id, start_times, trigger_idx, op_channel_idx, output_filename, event_times, compression=None):
    """
    Saves light trigger to output file

    Args:
        event_id(array): shape `(ntrigs,)`, event id for each trigger
        start_times(array): shape `(ntrigs,)`, simulation time offset for each trigger [microseconds]
        trigger_idx(array): shape `(ntrigs,)`, simulation time tick of each trigger
        op_channel_idx(array): shape `(ntrigs, ndet_module)`, optical channel index for each trigger
        output_filename(str): output hdf5 file path
        event_times(array): shape `(nevents,)`, global event t0 for each unique event [microseconds]

    """
    if event_id.shape[0] == 0:
        return

    unique_events, unique_events_inv = np.unique(event_id, return_inverse=True)
    event_start_times = event_times[unique_events_inv]
    event_sync_times = (event_times[unique_events_inv] / detector.CLOCK_CYCLE).astype(int) % detector.CLOCK_RESET_PERIOD

    with h5py.File(output_filename, 'a') as f:
        trig_data = np.empty(trigger_idx.shape[0], dtype=np.dtype([('op_channel','i4',(op_channel_idx.shape[-1])), ('ts_s','f8'), ('ts_sync','u8')]))
        trig_data['op_channel'] = op_channel_idx
        trig_data['ts_s'] = ((start_times + trigger_idx * light.LIGHT_TICK_SIZE + event_start_times) * units.mus / units.s)
        trig_data['ts_sync'] = (((start_times + trigger_idx * light.LIGHT_TICK_SIZE)/detector.CLOCK_CYCLE + event_sync_times).astype(int) % detector.CLOCK_RESET_PERIOD)

        if 'light_trig' not in f:
            f.create_dataset('light_trig', data=trig_data, maxshape=(None,), compression=compression)
        else:
            f['light_trig'].resize(f['light_trig'].shape[0] + trigger_idx.shape[0], axis=0)
            f['light_trig'][-trigger_idx.shape[0]:] = trig_data

def export_to_hdf5(event_id, start_times, trigger_idx, op_channel_idx, waveforms, output_filename, event_times, waveforms_true_track_id, waveforms_true_photons, i_trig, i_mod, compression=None):
    """
    Saves waveforms to output file

    Args:
        event_id(array): shape `(ntrigs,)`, event id for each trigger
        start_times(array): shape `(ntrigs,)`, simulation time offset for each trigger [microseconds]
        trigger_idx(array): shape `(ntrigs,)`, simulation time tick of each trigger
        op_channel_idx(array): shape `(ntrigs, ndet_module)`, optical channel index for each trigger
        waveforms(array): shape `(ntrigs, ndet_module, nsamples)`, simulated waveforms to save
        output_filename(str): output hdf5 file path
        event_times(array): shape `(nevents,)`, global event t0 for each unique event [microseconds]
        waveforms_true_track_id(array): shape `(ntrigs, ndet, nsamples)`, segment ids contributing to each sample
        waveforms_true_photons(array): shape `(ntrigs, ndet, nsamples)`, true photocurrent at each sample

    """
    export_light_trig_to_hdf5(event_id, start_times, trigger_idx, op_channel_idx, output_filename, event_times, compression)
    export_light_wvfm_to_hdf5(event_id, waveforms, output_filename, waveforms_true_track_id, waveforms_true_photons, i_trig, i_mod, compression)

def merge_module_light_wvfm_same_trigger(output_filename, compression=None):
    """
    Merge light waveforms for each module if the modular variation is activated
    """
    with h5py.File(output_filename, 'a') as f:
        for i_, i_mod in enumerate(detector.MOD_IDS):
            if i_ == 0:
                merged_wvfm = f[f'light_wvfm/light_wvfm_mod{i_mod-1}']
            else:
                mod_wvfm = f[f'light_wvfm/light_wvfm_mod{i_mod-1}']
                if mod_wvfm.shape[0] != merged_wvfm.shape[0]:
                    raise ValueError("The number of triggers should be the same in each module with light trigger mode 1 (light waveform).")
                merged_wvfm = np.append(merged_wvfm, mod_wvfm, axis=1)
        del f['light_wvfm']
        f.create_dataset(f'light_wvfm', data=merged_wvfm, maxshape=(None,None,None), compression=compression)
