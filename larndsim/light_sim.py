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
from .consts.light import LIGHT_TICK_SIZE, LIGHT_WINDOW, SINGLET_FRACTION, TAU_S, TAU_T, LIGHT_GAIN, LIGHT_OSCILLATION_PERIOD, LIGHT_RESPONSE_TIME, LIGHT_DET_NOISE_SAMPLE_SPACING, LIGHT_TRIG_THRESHOLD, LIGHT_TRIG_WINDOW, LIGHT_DIGIT_SAMPLE_SPACING, LIGHT_NBIT, OP_CHANNEL_TO_TPC, OP_CHANNEL_PER_DET, TPC_TO_OP_CHANNELS, OP_CHANNEL_PER_TRIG
from .consts.detector import TPC_BORDERS, MODULE_TO_TPCS
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
    if np.any(mask):
        start_time = np.min(light_incidence['t0_det'][mask]) - LIGHT_WINDOW[0]
        end_time = np.max(light_incidence['t0_det'][mask]) + LIGHT_WINDOW[1]
        return int(np.ceil((end_time - start_time)/LIGHT_TICK_SIZE)), start_time
    return int((LIGHT_WINDOW[1] + LIGHT_WINDOW[0])/LIGHT_TICK_SIZE), 0
    


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
                            light_sample_inc[idet,itick] += light_inc['n_photons_det'][itrk,idet] * time_profile[iprof] / norm / LIGHT_TICK_SIZE


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
    Return poisson distributed int32 and advance `states[index]`. For efficiency, 
    if `mean > 100`, returns a gaussian distributed int32 with `mean == mean`
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
    if mean < 100: # poisson statistics are important
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
                light_response[idet,itick] += LIGHT_GAIN[idet] * sipm_response_model(idet, itick-jtick) * light_sample_inc[idet,jtick]
                

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
    if shape[0] < 2:
        # special case where inverse FFT does not exist - just generate one sample
        noise = cp.real(noise)
    else:
        # invert FFT to create a noise waveform
        noise = cp.fft.irfft(noise, axis=-1)

    if noise.shape != shape:
        # FFT must have even samples, so append 0 if an odd number of samples is requested
        noise = cp.concatenate([noise, cp.zeros((noise.shape[0],shape[1]-noise.shape[1]))],axis=-1)

    return noise


def get_triggers(signal):
    """
    Identifies each simulated ticks that would initiate a trigger taking into account the ADC digitization window
    
    Args:
        signal(array): shape `(ndet, nticks)`, simulated signal on each channel
        
    Returns:
        tuple: array of tick indices at each trigger (shape `(ntrigs,)`) and array of op channel index (shape `(ntrigs, ndet_module)`)
    """
    # sum over all signals on a single detector
    signal_sum = signal.reshape(signal.shape[0]//OP_CHANNEL_PER_DET, OP_CHANNEL_PER_DET, signal.shape[-1]).sum(axis=1, keepdims=True)
    # apply trigger threshold
    sample_above_thresh = cp.broadcast_to(signal_sum < LIGHT_TRIG_THRESHOLD, (signal.shape[0]//OP_CHANNEL_PER_DET, OP_CHANNEL_PER_DET, signal.shape[-1]))
    # cast back into the original signal array
    sample_above_thresh = sample_above_thresh.reshape(signal.shape)

    # calculate the minimum number of ticks between two triggers on the same module
    digit_ticks = ceil((LIGHT_TRIG_WINDOW[1] + LIGHT_TRIG_WINDOW[0])/LIGHT_TICK_SIZE)

    trigger_idx = []
    op_channel_idx = []
    # treat each module independently
    for mod_id, tpc_ids in MODULE_TO_TPCS.items():
        # get active channels for the module
        op_channels = TPC_TO_OP_CHANNELS[tpc_ids].ravel()
        module_above_thresh = np.any(sample_above_thresh[op_channels], axis=0)

        last_trigger = 0
        while cp.any(module_above_thresh):
            # find next time signal goes above threshold
            next_idx = cp.sort(cp.nonzero(module_above_thresh)[0])[0] + (last_trigger + digit_ticks if last_trigger == 0 else 0)
            # keep track of trigger time
            trigger_idx.append(next_idx)
            op_channel_idx.append(op_channels)
            # ignore samples during digitization window
            module_above_thresh = module_above_thresh[next_idx+digit_ticks:]
        
    return cp.array(trigger_idx), cp.array(op_channel_idx)


@cuda.jit
def digitize_signal(signal, trigger_idx, op_channel_idx, digit_signal):
    """
    Interpolate signal to the appropriate sampling frequency
    
    Args:
        signal(array): shape `(ndet, nticks)`, simulated signal on each channel
        trigger_idx(array): shape `(ntrigs,)`, tick index for each trigger
        op_channel_idx(array): shape `(ntrigs, ndet_module)`, optical channel index for each trigger
        digit_signal(array): output array, shape `(ntrigs, ndet_module, nsamples)`, digitized signal
    """
    itrig,idet_module,isample = cuda.grid(3)
    
    if itrig < digit_signal.shape[0]:
        if idet_module < digit_signal.shape[1]:
            if isample < digit_signal.shape[2]:
                sample_tick = isample * LIGHT_DIGIT_SAMPLE_SPACING / LIGHT_TICK_SIZE - LIGHT_TRIG_WINDOW[0] / LIGHT_TICK_SIZE + trigger_idx[itrig]
                
                tick0 = int(floor(sample_tick))
                tick1 = int(ceil(sample_tick))
                
                signal0 = signal[op_channel_idx[itrig, idet_module], tick0]
                
                if tick0 == tick1:
                    digit_signal[itrig,idet_module,isample] = signal0
                else:
                    signal1 = signal[idet_module, tick1]
                    
                    digit_signal[itrig,idet_module,isample] = signal0 + (signal1 - signal0) * (sample_tick - tick0)


def sim_triggers(bpg, tpb, signal, trigger_idx, op_channel_idx, digit_samples, light_det_noise):
    """
    Generates digitized waveforms at specified simulation tick indices
    
    Args:
        bpg(tuple): blocks per grid used to generate digitized waveforms, `len(bpg) == 3`, `prod(bpg) * prod(tpb) >= digit_samples.size`
        tpb(tuple): threads per grid used to generate digitized waveforms, `len(bpg) == 3`, `bpg[i] * tpb[i] >= digit_samples.shape[i]`
        signal(array): shape `(ndet, nticks)`, simulated signal on each channel
        trigger_idx(array): shape `(ntrigs,)`, tick index for each trigger to digitize
        op_channel_idx(array): shape `(ntrigs, ndet_module)`, optical channel indices for each trigger
        digit_samples(int): number of digitizations per waveform
        light_det_noise(array): shape `(ndet, nnoise_bins)`, noise spectrum for each channel (only used if waveforms extend past simulated signal)
        
    Returns:
        array: shape `(ntrigs, ndet_module, digit_samples)`, digitized waveform on each channel for each trigger
    """
    # exit if no triggers
    digit_signal = cp.zeros((trigger_idx.shape[0], op_channel_idx.shape[-1], digit_samples), dtype='f8')
    if digit_signal.shape[0] == 0:
        return digit_signal
    
    padded_trigger_idx = trigger_idx.copy()

    # pad front of simulation with noise, if trigger close to start of simulation window
    pre_digit_ticks = int(ceil(LIGHT_TRIG_WINDOW[0]/LIGHT_TICK_SIZE))
    if trigger_idx.min() - pre_digit_ticks < 0:
        pad_shape = (signal.shape[0], int(pre_digit_ticks - trigger_idx.min()))
        signal = cp.concatenate([gen_light_detector_noise(pad_shape, light_det_noise), signal], axis=-1)
        padded_trigger_idx += pad_shape[1]
    
    # pad end of simulation with noise, if trigger close to end of simulation window
    post_digit_ticks = int(ceil(LIGHT_TRIG_WINDOW[1]/LIGHT_TICK_SIZE))
    if post_digit_ticks + trigger_idx.max() > signal.shape[-1]:
        pad_shape = (signal.shape[0], int(signal.shape[1] - (post_digit_ticks + trigger_idx.max())))
        signal = cp.concatenate([signal, gen_light_detector_noise(pad_shape, light_det_noise)], axis=-1)
        
    digitize_signal[bpg,tpb](signal, padded_trigger_idx, op_channel_idx, digit_signal)

    # truncate to correct number of bits
    digit_signal = cp.round(digit_signal / 2**(16-LIGHT_NBIT)) * 2**(16-LIGHT_NBIT)
    
    return digit_signal


def export_to_hdf5(event_id, start_times, trigger_idx, waveforms, output_filename, event_times):
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
    
    """
    if event_id.shape[0] == 0:
        return
    
    unique_events, unique_events_inv = np.unique(event_id, return_inverse=True)
    event_start_times = event_times[unique_events_inv]
    event_sync_times = (event_times[unique_events_inv] / CLOCK_CYCLE).astype(int) % ROLLOVER_CYCLES
    
    with h5py.File(output_filename, 'a') as f:
        trig_data = np.empty(trigger_idx.shape[0], dtype=np.dtype([('op_channel','i4',(op_channel_idx.shape[-1])), ('ts_s','f8'), ('ts_sync','u8')]))
        trig_data['op_channel'] = op_channel_idx
        trig_data['ts_s'] = ((start_times + trigger_idx * LIGHT_TICK_SIZE + event_start_times) * units.mus / units.s).get()
        trig_data['ts_sync'] = (((start_times + trigger_idx * LIGHT_TICK_SIZE)/CLOCK_CYCLE + event_sync_times).astype(int) % ROLLOVER_CYCLES).get()
        
        if 'light_wvfm' not in f:
            f.create_dataset('light_wvfm', data=waveforms.get(), maxshape=(None,None,None))
            f.create_dataset('light_trig', data=trig_data, maxshape=(None,))
        else:
            f['light_wvfm'].resize(f['light_wvfm'].shape[0] + waveforms.shape[0], axis=0)
            f['light_wvfm'][-waveforms.shape[0]:] = waveforms.get()

            f['light_trig'].resize(f['light_trig'].shape[0] + trigger_idx.shape[0], axis=0)
            f['light_trig'][-trigger_idx.shape[0]:] = trig_data

            
