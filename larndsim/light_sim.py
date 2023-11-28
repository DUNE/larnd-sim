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

from .consts import light
from .consts.light import LIGHT_TICK_SIZE, LIGHT_WINDOW, SINGLET_FRACTION, TAU_S, TAU_T, LIGHT_GAIN, LIGHT_OSCILLATION_PERIOD, LIGHT_RESPONSE_TIME, LIGHT_DET_NOISE_SAMPLE_SPACING, LIGHT_TRIG_MODE, LIGHT_TRIG_THRESHOLD, LIGHT_TRIG_WINDOW, LIGHT_DIGIT_SAMPLE_SPACING, LIGHT_NBIT, OP_CHANNEL_TO_TPC, OP_CHANNEL_PER_TRIG, TPC_TO_OP_CHANNEL, OP_CHANNEL_PER_TRIG, SIPM_RESPONSE_MODEL, IMPULSE_TICK_SIZE, IMPULSE_MODEL, MC_TRUTH_THRESHOLD, ENABLE_LUT_SMEARING

from .consts.detector import TPC_BORDERS, MODULE_TO_TPCS, TPC_TO_MODULE
from .consts import units as units

from .fee import CLOCK_CYCLE, ROLLOVER_CYCLES


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
    if np.any(mask) and not LIGHT_TRIG_MODE == 1:
        start_time = np.min(light_incidence['t0_det'][mask]) - LIGHT_WINDOW[0]
        end_time = np.max(light_incidence['t0_det'][mask]) + LIGHT_WINDOW[1]
        return int(np.ceil((end_time - start_time)/LIGHT_TICK_SIZE)), start_time
    return int((LIGHT_WINDOW[1] + LIGHT_WINDOW[0])/LIGHT_TICK_SIZE), 0

def get_active_op_channel(light_incidence):
    """
    Returns an array of optical channels that need to be simulated

    Args:
        light_incidence(array): shape `(ntracks, ndet)`, containing first hit time and number of photons on each detector

    Returns:
        array: shape `(ndet_active,)` op detector index of each active channel (`int`)
    """
    mask = light_incidence['n_photons_det'] > 0
    if np.any(mask):
        return cp.array(np.where(np.any(mask, axis=0))[0], dtype='i4')
    return cp.empty((0,), dtype='i4')
    
@cuda.jit
def sum_light_signals(segments, segment_voxel, segment_track_id, light_inc, op_channel, lut, start_time, light_sample_inc, light_sample_inc_true_track_id, light_sample_inc_true_photons):
    """
    Sums the number of photons observed by each light detector at each time tick

    Args:
        segments(array): shape `(ntracks,)`, edep-sim tracks to simulate
        segment_voxel(array): shape `(ntracks, 3)`, LUT voxel for eack edep-sim track
        segment_track_id(array): shape `(ntracks,)`, unique id for each track segment (for MC truth backtracking)
        light_inc(array): shape `(ntracks, ndet)`, number of photons incident on each detector and voxel id
        op_channel(array): shape `(ntracks, ndet_active)`, optical channel index, will use lut[:,:,:,op_channel%lut.shape[3]] to look up timing information
        lut(array): shape `(nx,ny,nz,ndet_tpc)`, light look up table
        start_time(float): start time of light simulation in microseconds
        light_sample_inc(array): output array, shape `(ndet, nticks)`, number of photons incident on each detector at each time tick (propogation delay only)
        light_sample_inc_true_track_id(array): output array, shape `(ndet, nticks, maxtracks)`, true track ids on each detector at each time tick (propogation delay only)
        light_sample_inc_true_photons(array): output array, shape `(ndet, nticks, maxtracks)`, number of photons incident on each detector at each time tick from each track
    """
    idet,itick = cuda.grid(2)

    if idet < light_sample_inc.shape[0]:
        if itick < light_sample_inc.shape[1]:

            start_tick_time = itick * LIGHT_TICK_SIZE + start_time
            end_tick_time = start_tick_time + LIGHT_TICK_SIZE

            # find tracks that contribute light to this time tick
            idet_lut = op_channel[idet] % lut.shape[3]
            for itrk in range(segments.shape[0]):
                if light_inc[itrk,op_channel[idet]]['n_photons_det'] > 0:
                    voxel = segment_voxel[itrk]
                    time_profile = lut[voxel[0],voxel[1],voxel[2],idet_lut]['time_dist']
                    track_time = segments[itrk]['t0']
                    track_end_time = track_time + time_profile.shape[0] * units.ns / units.mus # FIXME: assumes light LUT time profile bins are 1ns (might not be true in general)

                    if track_end_time < start_tick_time or track_time > end_tick_time:
                        continue

                    # use LUT time smearing
                    if ENABLE_LUT_SMEARING:
                        # normalize propogation delay time profile
                        norm = 0
                        for iprof in range(time_profile.shape[0]):
                            norm += time_profile[iprof]

                        # add photons to time tick
                        for iprof in range(time_profile.shape[0]):
                            profile_time = track_time + iprof * units.ns / units.mus # FIXME: assumes light LUT time profile bins are 1ns (might not be true in general)
                            if profile_time < end_tick_time and profile_time > start_tick_time:
                                photons = light_inc['n_photons_det'][itrk,op_channel[idet]] * time_profile[iprof] / norm / LIGHT_TICK_SIZE
                                light_sample_inc[idet,itick] += photons

                                if photons > MC_TRUTH_THRESHOLD:
                                    # get truth information for time tick
                                    for itrue in range(light_sample_inc_true_track_id.shape[-1]):
                                        if light_sample_inc_true_track_id[idet,itick,itrue] == -1 or light_sample_inc_true_track_id[idet,itick,itrue] == segment_track_id[itrk]:
                                            light_sample_inc_true_track_id[idet,itick,itrue] = segment_track_id[itrk]
                                            light_sample_inc_true_photons[idet,itick,itrue] += photons
                                            break
                    # use average time only
                    else:
                        # calculate average delay time
                        avg = 0
                        norm = 0
                        for iprof in range(time_profile.shape[0]):
                            avg += iprof * units.ns / units.mus * time_profile[iprof]
                            norm += time_profile[iprof]
                        avg = avg / norm

                        # add photons to time tick
                        profile_time = track_time + avg
                        if profile_time < end_tick_time and profile_time > start_tick_time:
                            photons = light_inc['n_photons_det'][itrk,op_channel[idet]] / LIGHT_TICK_SIZE
                            light_sample_inc[idet,itick] += photons
                            if photons > MC_TRUTH_THRESHOLD:
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
    p1 = SINGLET_FRACTION * exp(-time_tick * LIGHT_TICK_SIZE / TAU_S) * (1 - exp(-LIGHT_TICK_SIZE / TAU_S))
    p3 = (1 - SINGLET_FRACTION) * exp(-time_tick * LIGHT_TICK_SIZE / TAU_T) * (1 - exp(-LIGHT_TICK_SIZE / TAU_T))
    return (p1 + p3) * (time_tick >= 0)


@cuda.jit
def calc_scintillation_effect(light_sample_inc, light_sample_inc_true_track_id, light_sample_inc_true_photons, light_sample_inc_scint, light_sample_inc_scint_true_track_id, light_sample_inc_scint_true_photons):
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
                if light_sample_inc[idet,jtick] == 0:
                    continue
                tick_weight = scintillation_model(itick-jtick)
                light_sample_inc_scint[idet,itick] += tick_weight * light_sample_inc[idet,jtick]

                # loop over convolution tick truth
                for itrue in range(light_sample_inc_true_track_id.shape[-1]):
                    if light_sample_inc_true_track_id[idet,jtick,itrue] == -1:
                        break
                        
                    if tick_weight * light_sample_inc_true_photons[idet,jtick,itrue] < MC_TRUTH_THRESHOLD:
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
                light_sample_inc_disc[idet,itick] = 1. / LIGHT_TICK_SIZE * xoroshiro128p_poisson_int32(
                    light_sample_inc[idet,itick] * LIGHT_TICK_SIZE, rng_states, idet*light_sample_inc.shape[1] + itick)
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
    if i0 > len(arr)-1:
        return high
    if i0 == idx:
        return arr[i0]
    if i0 > len(arr)-2:
        return high

    i1 = i0 + 1
    v0 = arr[i0]
    v1 = arr[i1]

    return v0 + (v1 - v0) * (idx - i0)
                

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
    # use RLC response model
    if SIPM_RESPONSE_MODEL == 0:
        t = time_tick * LIGHT_TICK_SIZE
        impulse = (t>=0) * exp(-t/LIGHT_RESPONSE_TIME) * sin(t/LIGHT_OSCILLATION_PERIOD)
        # normalize to 1
        impulse /= LIGHT_OSCILLATION_PERIOD * LIGHT_RESPONSE_TIME**2
        impulse *= LIGHT_OSCILLATION_PERIOD**2 + LIGHT_RESPONSE_TIME**2
        return impulse * LIGHT_TICK_SIZE

    # use measured response model
    if SIPM_RESPONSE_MODEL == 1:
        impulse = interp(time_tick * LIGHT_TICK_SIZE / IMPULSE_TICK_SIZE, IMPULSE_MODEL, 0, 0)
        # normalize to 1
        impulse /=  IMPULSE_TICK_SIZE/LIGHT_TICK_SIZE
        return impulse
            

@cuda.jit
def calc_light_detector_response(light_sample_inc, light_sample_inc_true_track_id, light_sample_inc_true_photons, light_response, light_response_true_track_id, light_response_true_photons):
    """
    Simulates the SiPM reponse and digit
    
    Args:
        light_sample_inc(array): shape `(ndet, ntick)`, PE produced on each SiPM at each time tick
        light_response(array): shape `(ndet, ntick)`, ADC value at each time tick
    """
    idet,itick = cuda.grid(2)

    if idet < light_sample_inc.shape[0]:
        if itick < light_sample_inc.shape[1]:
            conv_ticks = ceil((LIGHT_WINDOW[1] - LIGHT_WINDOW[0])/LIGHT_TICK_SIZE)
            
            for jtick in range(max(itick - conv_ticks, 0), itick+1):
                tick_weight = sipm_response_model(idet, itick-jtick)
                light_response[idet,itick] += LIGHT_GAIN[idet] * tick_weight * light_sample_inc[idet,jtick]
                    
                # loop over convolution tick truth
                for itrue in range(light_sample_inc_true_track_id.shape[-1]):
                    if light_sample_inc_true_track_id[idet,jtick,itrue] == -1:
                        break
                        
                    if abs(tick_weight * light_sample_inc_true_photons[idet,jtick,itrue]) < MC_TRUTH_THRESHOLD:
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
    
    noise_freq = cp.fft.rfftfreq((light_det_noise.shape[-1]-1)*2, d=LIGHT_DET_NOISE_SAMPLE_SPACING)
    desired_freq = cp.fft.rfftfreq(shape[-1], d=LIGHT_TICK_SIZE)
    
    bin_size = cp.diff(desired_freq).mean()
    noise_spectrum = cp.zeros((shape[0], desired_freq.shape[0]))
    for idet in range(shape[0]):
        noise_spectrum[idet] = cp.interp(desired_freq, noise_freq, light_det_noise[idet], left=0, right=0)
    # rescale noise spectrum to have constant noise power with digitizer sample spacing
    noise_spectrum *= cp.sqrt(cp.diff(noise_freq, axis=-1).mean()/bin_size) * LIGHT_DIGIT_SAMPLE_SPACING / LIGHT_TICK_SIZE
    

    # generate an FFT with the same frequency power, but with random phase
    noise = noise_spectrum * cp.exp(2j * cp.pi * cp.random.uniform(size=noise_spectrum.shape))
    if shape[1] < 2:
        # special case where inverse FFT does not exist - just generate one sample
        noise = cp.round(cp.real(noise)) * 2**(16-LIGHT_NBIT)
    else:
        # invert FFT to create a noise waveform
        noise = cp.round(cp.fft.irfft(noise, axis=-1)) * 2**(16-LIGHT_NBIT)

    if noise.shape[1] < shape[1]:
        # FFT must have even samples, so append 0 if an odd number of samples is requested
        noise = cp.concatenate([noise, cp.zeros((noise.shape[0],shape[1]-noise.shape[1]))],axis=-1)

    return noise[:,:shape[1]]


def get_triggers(signal, group_threshold, op_channel_idx):
    """
    Identifies each simulated ticks that would initiate a trigger taking into account the ADC digitization window
    
    Args:
        signal(array): shape `(ndet, nticks)`, simulated signal on each channel
        group_threshold(array): shape `(ngrp,)`, threshold on group sum (requires `ndet/ngrp == OP_CHANNEL_PER_TRIG`)
        op_channel_idx(array): shape `(ndet,)`, optical channel index for each signal
        
    Returns:
        tuple: array of tick indices at each trigger (shape `(ntrigs,)`) and array of op channel index (shape `(ntrigs, ndet_module)`)
    """
    
    shape = signal.shape
    # sum over all signals on a single detector (shape: (ndet, nticks) -> (ngrp, ndetpergrp, nticks) -> (ngrp, 1, nticks))
    signal_sum = signal.reshape(shape[0]//OP_CHANNEL_PER_TRIG, OP_CHANNEL_PER_TRIG, shape[-1]).sum(axis=1, keepdims=True)
    # reduce to approximate ADC sample rate (padded with zeros)
    sample_factor = round(LIGHT_DIGIT_SAMPLE_SPACING / LIGHT_TICK_SIZE)
    padding = sample_factor - shape[-1] % sample_factor
    if padding > 0:
        signal_sum = cp.concatenate((signal_sum, cp.zeros((shape[0]//OP_CHANNEL_PER_TRIG, 1, padding))), axis=-1)
    signal_sum = signal_sum.reshape(-1, 1, signal_sum.shape[-1]//sample_factor, sample_factor).mean(axis=-1, keepdims=True)
    signal_sum = cp.broadcast_to(signal_sum, signal_sum.shape[:3] + (sample_factor,))
    signal_sum = signal_sum.reshape(-1, 1, shape[-1] + padding)[...,:(-padding if padding > 0 else shape[-1])]
    
    # apply trigger threshold
    sample_above_thresh = cp.broadcast_to(signal_sum < group_threshold[:,cp.newaxis,cp.newaxis], (shape[0]//OP_CHANNEL_PER_TRIG, OP_CHANNEL_PER_TRIG, shape[-1]))
    # cast back into the original signal array
    sample_above_thresh = sample_above_thresh.reshape(signal.shape)

    # calculate the minimum number of ticks between two triggers on the same module
    digit_ticks = ceil((LIGHT_TRIG_WINDOW[1] + LIGHT_TRIG_WINDOW[0])/LIGHT_TICK_SIZE)

    tpc_ids = np.unique(OP_CHANNEL_TO_TPC[op_channel_idx.get()])
    mod_ids = np.unique([TPC_TO_MODULE[tpc_id] for tpc_id in tpc_ids])
    
    trigger_idx_list = []
    op_channel_idx_list = []
    trigger_type_list = [] # 0 --> threshold, # 1 --> beam
    # treat each module independently
    for mod_id, tpc_ids in [(mod_id, MODULE_TO_TPCS[mod_id]) for mod_id in mod_ids]:
        # get active channels for the module
        op_channels = TPC_TO_OP_CHANNEL[tpc_ids].ravel()
        op_channel_mask = np.isin(op_channel_idx.get(), op_channels)
        #module_above_thresh = cp.any(sample_above_thresh[op_channels], axis=0)
        module_above_thresh = np.any(sample_above_thresh[op_channel_mask], axis=0)        

        last_trigger = 0
        while cp.any(module_above_thresh):
            # find next time signal goes above threshold
            if last_trigger == 0 and LIGHT_TRIG_MODE == 1:
                next_idx = cp.asarray(0)
                next_trig_type = cp.asarray(1)
            else:
                next_idx = cp.sort(cp.nonzero(module_above_thresh)[0])[0] + (last_trigger if last_trigger != 0 else 0)
                next_trig_type = cp.asarray(0)
                print("seondary trigger")
            # keep track of trigger time
            trigger_idx_list.append(next_idx)
            trigger_type_list.append(next_trig_type)
            op_channel_idx_list.append(op_channels)
            # ignore samples during digitization window
            module_above_thresh = module_above_thresh[next_idx+digit_ticks:]
            last_trigger = next_idx + digit_ticks

    if len(trigger_idx_list):
        return cp.array(trigger_idx_list), cp.array(op_channel_idx_list), cp.array(trigger_type_list)
    return cp.empty((0,), dtype=int), cp.empty((0,len(TPC_TO_OP_CHANNEL[list(MODULE_TO_TPCS.values())[0]].ravel())), dtype=int), cp.empty((0,), dtype=int)


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
                sample_tick = isample * LIGHT_DIGIT_SAMPLE_SPACING / LIGHT_TICK_SIZE - LIGHT_TRIG_WINDOW[0] / LIGHT_TICK_SIZE + trigger_idx[itrig]
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
                            
                    photons0, photons1 = 0, 0

                    # if matches the current sample track or we have empty truth slot, add truth info
                    if signal_true_track_id[idet_signal,itick0,jtrue] == digit_signal_true_track_id[itrig,idet_module,isample,itrue] or digit_signal_true_track_id[itrig,idet_module,isample,itrue] == -1:
                        digit_signal_true_track_id[itrig,idet_module,isample,itrue] = signal_true_track_id[idet_signal,itick0,jtrue]
                        itrue += 1
                        # interpolate true photons
                        photons0 = signal_true_photons[idet,itick0,jtrue]
                        
                        if abs(photons0) < MC_TRUTH_THRESHOLD:
                            continue

                        # loop over next tick
                        # first try same position (for speed-up)
                        if signal_true_track_id[idet_signal,itick0,jtrue] == signal_true_track_id[idet_signal,itick1,jtrue]:
                            photons1 = signal_true_photons[idet_signal,itick1,jtrue]
                        else:
                            for ktrue in range(signal_true_track_id.shape[-1]):
                                if signal_true_track_id[idet_signal,itick0,jtrue] == signal_true_track_id[idet_signal,itick1,ktrue]:
                                    photons1 = signal_true_photons[idet_signal,itick1,ktrue]
                                    break

                    # if a valid truth entry was found, do interpolation
                    if digit_signal_true_track_id[itrig,idet_module,isample,itrue-1] != -1:
                        digit_signal_true_photons[itrig,idet_module,isample,itrue-1] = interp(sample_tick-itick0, (photons0,photons1), 0, 0)


def sim_triggers(bpg, tpb, signal, signal_op_channel_idx, signal_true_track_id, signal_true_photons, trigger_idx, op_channel_idx, digit_samples, light_det_noise):
    """
    Generates digitized waveforms at specified simulation tick indices
    
    Args:
        bpg(tuple): blocks per grid used to generate digitized waveforms, `len(bpg) == 3`, `prod(bpg) * prod(tpb) >= digit_samples.size`
        tpb(tuple): threads per grid used to generate digitized waveforms, `len(bpg) == 3`, `bpg[i] * tpb[i] >= digit_samples.shape[i]`
        signal(array): shape `(ndet, nticks)`, simulated signal on each channel
        signal_op_channel_idx(array): shape `(ndet,)`, optical channel index for each simulated signal
        signal_true_track_id(array): shape `(ndet, nticks, ntruth)`, true segments associated with each tick
        signal_true_photons(array): shape `(ndet, nticks, ntruth)`, true photons associated with each tick from each track
        trigger_idx(array): shape `(ntrigs,)`, tick index for each trigger to digitize
        op_channel_idx(array): shape `(ntrigs, ndet_module)`, optical channel indices for each trigger
        digit_samples(int): number of digitizations per waveform
        light_det_noise(array): shape `(ndet, nnoise_bins)`, noise spectrum for each channel (only used if waveforms extend past simulated signal)
        
    Returns:
        array: shape `(ntrigs, ndet_module, digit_samples)`, digitized waveform on each channel for each trigger
    """
    digit_signal = cp.zeros((trigger_idx.shape[0], op_channel_idx.shape[-1], digit_samples), dtype='f8')
    digit_signal_true_track_id = cp.full((trigger_idx.shape[0], op_channel_idx.shape[-1], digit_samples, signal_true_track_id.shape[-1]), -1, dtype=signal_true_track_id.dtype)
    digit_signal_true_photons = cp.zeros((trigger_idx.shape[0], op_channel_idx.shape[-1], digit_samples, signal_true_photons.shape[-1]), dtype=signal_true_photons.dtype)
    # exit if no triggers
    if digit_signal.shape[0] == 0:
        return digit_signal, digit_signal_true_track_id, digit_signal_true_photons
    
    padded_trigger_idx = trigger_idx.copy()

    # pad front of simulation with noise, if trigger close to start of simulation window
    pre_digit_ticks = int(ceil(LIGHT_TRIG_WINDOW[0]/LIGHT_TICK_SIZE))
    if trigger_idx.min() - pre_digit_ticks < 0:
        pad_shape = (signal.shape[0], int(pre_digit_ticks - trigger_idx.min()))
        signal = cp.concatenate([gen_light_detector_noise(pad_shape, light_det_noise[signal_op_channel_idx]), signal], axis=-1)
        signal_true_track_id = cp.concatenate([cp.full(pad_shape + signal_true_track_id.shape[-1:], -1, dtype=signal_true_track_id.dtype), signal_true_track_id], axis=-2)
        signal_true_photons = cp.concatenate([cp.zeros(pad_shape + signal_true_photons.shape[-1:], signal_true_photons.dtype), signal_true_photons], axis=-2)
        padded_trigger_idx += pad_shape[1]
    
    # pad end of simulation with noise, if trigger close to end of simulation window
    post_digit_ticks = int(ceil(LIGHT_TRIG_WINDOW[1]/LIGHT_TICK_SIZE))
    if post_digit_ticks + padded_trigger_idx.max() > signal.shape[1]:
        pad_shape = (signal.shape[0], int(post_digit_ticks + padded_trigger_idx.max() - signal.shape[1]))

        signal = cp.concatenate([signal, gen_light_detector_noise(pad_shape, light_det_noise[signal_op_channel_idx])], axis=-1)
        signal_true_track_id = cp.concatenate([signal_true_track_id, cp.full(pad_shape + signal_true_track_id.shape[-1:], -1, dtype=signal_true_track_id.dtype)], axis=-2)
        signal_true_photons = cp.concatenate([signal_true_photons, cp.zeros(pad_shape + signal_true_photons.shape[-1:], dtype=signal_true_photons.dtype)], axis=-2)

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
    digit_signal = cp.round(digit_signal / 2**(16-LIGHT_NBIT)) * 2**(16-LIGHT_NBIT)
    
    return digit_signal, digit_signal_true_track_id, digit_signal_true_photons


def export_to_hdf5(event_id, start_times, trigger_idx, op_channel_idx, waveforms, output_filename, event_times, waveforms_true_track_id, waveforms_true_photons):
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
        waveforms_true_track_id(array): shape `(ntrigs, ndet, nsamples, ntruth)`, segment ids contributing to each sample
        waveforms_true_photons(array): shape `(ntrigs, ndet, nsamples, ntruth)`, true photocurrent at each sample
    
    """
    if event_id.shape[0] == 0:
        return
    
    unique_events, unique_events_inv = np.unique(event_id, return_inverse=True)
    event_start_times = event_times[unique_events_inv]
    event_sync_times = (event_times[unique_events_inv] / CLOCK_CYCLE).astype(int) % ROLLOVER_CYCLES
    
    with h5py.File(output_filename, 'a') as f:
        trig_data = np.empty(trigger_idx.shape[0], dtype=np.dtype([('op_channel','i4',(op_channel_idx.shape[-1])), ('ts_s','f8'), ('ts_sync','u8')]))
        trig_data['op_channel'] = op_channel_idx
        trig_data['ts_s'] = ((start_times + trigger_idx * LIGHT_TICK_SIZE + event_start_times) * units.mus / units.s)
        trig_data['ts_sync'] = (((start_times + trigger_idx * LIGHT_TICK_SIZE)/CLOCK_CYCLE + event_sync_times).astype(int) % ROLLOVER_CYCLES)

        # skip creating the truth dataset if there is no truth information to store
        if waveforms_true_track_id.size > 0:
            truth_dtype = np.dtype([('track_ids', 'i8', (waveforms_true_track_id.shape[-1],)), ('pe_current', 'f8', (waveforms_true_photons.shape[-1],))])
            truth_data = np.empty(waveforms_true_track_id.shape[:-1], dtype=truth_dtype)
            truth_data['track_ids'] = waveforms_true_track_id
            truth_data['pe_current'] = waveforms_true_photons
        
        if 'light_wvfm' not in f:
            f.create_dataset('light_wvfm', data=waveforms, maxshape=(None,None,None))
            #f.create_dataset('light_from_seg', data=from_seg, maxshape=(None,None,None))
            f.create_dataset('light_trig', data=trig_data, maxshape=(None,))
            if waveforms_true_track_id.size > 0:
                f.create_dataset('light_wvfm_mc_assn', data=truth_data, maxshape=(None,None,None))
        else:
            f['light_wvfm'].resize(f['light_wvfm'].shape[0] + waveforms.shape[0], axis=0)
            f['light_wvfm'][-waveforms.shape[0]:] = waveforms

            #f['light_from_seg'].resize(f['light_from_seg'].shape[0] + from_seg.shape[0], axis=0)
            #f['light_from_seg'][-from_seg.shape[0]:] = from_seg
            
            f['light_trig'].resize(f['light_trig'].shape[0] + trigger_idx.shape[0], axis=0)
            f['light_trig'][-trigger_idx.shape[0]:] = trig_data

            if waveforms_true_track_id.size > 0:
                f['light_wvfm_mc_assn'].resize(f['light_wvfm_mc_assn'].shape[0] + truth_data.shape[0], axis=0)
                f['light_wvfm_mc_assn'][-truth_data.shape[0]:] = truth_data

    return waveforms_true_track_id, waveforms_true_photons
