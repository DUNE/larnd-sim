from math import exp, floor, ceil
import numpy as np
import cupy as cp

from numba import cuda
from numba.cuda.random import xoroshiro128p_normal_float32, xoroshiro128p_uniform_float32, create_xoroshiro128p_states

# Load in frozen inputs and outputs.
inputs = np.load('get_adc_values_inputs_03.npz')
outputs = np.load('get_adc_values_outputs_03.npz')

# Load in constants.
PERIODIC_RESET_CYCLES = inputs['PERIODIC_RESET_CYCLES'].item()
RESET_NOISE_CHARGE = inputs['RESET_NOISE_CHARGE'].item()
BUFFER_RISETIME = inputs['BUFFER_RISETIME'].item()
TIME_SAMPLING = inputs['TIME_SAMPLING'].item()
UNCORRELATED_NOISE_CHARGE = inputs['UNCORRELATED_NOISE_CHARGE'].item()
DISCRIMINATOR_NOISE = inputs['DISCRIMINATOR_NOISE'].item()
CLOCK_CYCLE = inputs['CLOCK_CYCLE'].item()
ADC_HOLD_DELAY = inputs['ADC_HOLD_DELAY'].item()
RESET_CYCLES = inputs['RESET_CYCLES'].item()
ADC_BUSY_DELAY = inputs['ADC_BUSY_DELAY'].item()
MAX_ADC_VALUES = inputs['MAX_ADC_VALUES'].item()
MAX_TRACKS_PER_PIXEL = inputs["current_fractions"].shape[2]
e = inputs['e'].item()

# Extract inputs and convert to CuPy arrays where necessary for GPU processing
pixels_signals = cp.asarray(inputs['pixels_signals'])
pixels_tracks_signals = cp.asarray(inputs['pixels_tracks_signals'])
num_backtrack = cp.asarray(inputs['num_backtrack'])
offset_backtrack = cp.asarray(inputs['offset_backtrack'])
time_ticks = cp.asarray(inputs['time_ticks'])

integral_list = cp.asarray(inputs['integral_list']) # adc_list
adc_ticks_list = cp.asarray(inputs['adc_ticks_list'])
time_padding_val = inputs['time_padding'].item() # Ensure scalar
rng_states = inputs['rng_states']
cuda.to_device(rng_states)
current_fractions = cp.asarray(inputs['current_fractions'])
pixel_thresholds = cp.asarray(inputs['pixel_thresholds'])

TPB = 4 #[1, 4, 8, 16, 32, 64, 128, 256]
BPG = ceil(pixels_signals.shape[0] / TPB)

# --- Phase 2: RNG Pullout ---

# Use pixels_signals shape to initialize noise (random number) arrays.
n_pixels = pixels_signals.shape[0]
n_ticks  = pixels_signals.shape[1]
# RNG usage breaks down into 3 patterns (see kernel for patterns)
# noise_uncorr.shape = (n_pixels, n_ticks)
# noise_disc.shape   = (n_pixels, n_ticks)
# noise_reset.shape  = (n_pixels, n_ticks)
# periodic_reset_phase.shape = (n_pixels,)

# RNG Kernel
@cuda.jit
def generate_noise(
    rng_states,
    noise_uncorr,
    noise_disc,
    noise_reset,
    periodic_reset_phase
):
    ip = cuda.grid(1)
    if ip >= noise_uncorr.shape[0]:
        return

    # 1) periodic reset phase (exactly once)
    periodic_reset_phase[ip] = int(
        xoroshiro128p_uniform_float32(rng_states, ip)
        * (PERIODIC_RESET_CYCLES + 1)
    )

    # 2) initialize noise
    noise_reset[ip, 0] = (
        xoroshiro128p_normal_float32(rng_states, ip)
        * RESET_NOISE_CHARGE * e
    )

    noise_uncorr[ip, 0] = (
        xoroshiro128p_normal_float32(rng_states, ip)
        * UNCORRELATED_NOISE_CHARGE * e
    )

    noise_disc[ip, 0] = (
        xoroshiro128p_normal_float32(rng_states, ip)
        * DISCRIMINATOR_NOISE * e
    )


    # 3) per-tick noise
    for t in range(1, noise_uncorr.shape[1]):
        noise_uncorr[ip, t] = (
            xoroshiro128p_normal_float32(rng_states, ip)
            * UNCORRELATED_NOISE_CHARGE * e
        )

        noise_disc[ip, t] = (
            xoroshiro128p_normal_float32(rng_states, ip)
            * DISCRIMINATOR_NOISE * e
        )

        noise_reset[ip, t] = (
            xoroshiro128p_normal_float32(rng_states, ip)
            * RESET_NOISE_CHARGE * e
        )

# Initialize the noise arrays to the GPU.
noise_uncorr = cuda.device_array(
    (n_pixels, n_ticks), dtype=np.float32
)
noise_disc = cuda.device_array(
    (n_pixels, n_ticks), dtype=np.float32
)
noise_reset = cuda.device_array(
    (n_pixels, n_ticks), dtype=np.float32
)
periodic_reset_phase = cuda.device_array(
    (n_pixels,), dtype=np.int32
)

# Populate the arrays with random numbers
generate_noise[BPG, TPB](rng_states, noise_uncorr, noise_disc, noise_reset, periodic_reset_phase)

# --- Phase 3a: Signal Etraction ---

@cuda.jit
def integrate_signal(
    pixels_signals,
    signal_charge
):
    ip, ic = cuda.grid(2)

    if ip >= pixels_signals.shape[0]:
        return
    if ic >= pixels_signals.shape[1]:
        return

    curre = pixels_signals[ip]

    q = 0.0

    # Match original logic
    if BUFFER_RISETIME > 0:
        conv_start = max(
            0,
            int(ic - 10 * BUFFER_RISETIME / TIME_SAMPLING)
        )
        for jc in range(conv_start, ic + 1):
            if jc >= curre.shape[0]:
                break

            w = exp((jc - ic) * TIME_SAMPLING / BUFFER_RISETIME) * \
                (1.0 - exp(-TIME_SAMPLING / BUFFER_RISETIME))

            q += curre[jc] * TIME_SAMPLING * w

            # (optional: track-resolved version later)
    else:
        q = curre[ic] * TIME_SAMPLING

    signal_charge[ip, ic] = q

# Initialize signal charge array
signal_charge = cuda.device_array(
    (n_pixels, n_ticks),
    dtype=np.float32
)

TPB_2D = (16, 16)
BPG_2D = (
    ceil(pixels_signals.shape[0] / TPB_2D[0]),
    ceil(pixels_signals.shape[1] / TPB_2D[1]),
)

print("BPG_2D, TPB_2D:", BPG_2D, TPB_2D)

integrate_signal[BPG_2D, TPB_2D](
    pixels_signals,
    signal_charge
)

# --- Phase 3b/4: Tracks Accumulation ---

@cuda.jit
def integrate_signal_tracks(
    pixels_signals_tracks,
    num_backtrack,
    offset_backtrack,
    signal_charge_track  # (n_pixels, n_ticks, MAX_TRACKS_PER_PIXEL)
):
    ip, ic = cuda.grid(2)

    if ip >= num_backtrack.shape[0]:
        return
    if ic >= signal_charge_track.shape[1]:
        return

    ntrks = min(num_backtrack[ip], signal_charge_track.shape[2])
    off   = offset_backtrack[ip]

    # equivalent to num_backtrack.sum()
    # TODO: Check if this is true
    total_backtracks = offset_backtrack[-1] + num_backtrack[-1]
    #total_backtracks = num_backtrack.sum() <- numba no likey

    for itrk in range(ntrks):
        q = 0.0

        if BUFFER_RISETIME > 0:
            conv_start = max(
                0,
                int(ic - 10 * BUFFER_RISETIME / TIME_SAMPLING)
            )

            for jc in range(conv_start, ic + 1):
                idx = total_backtracks * jc + off + itrk

                w = exp((jc - ic) * TIME_SAMPLING / BUFFER_RISETIME) * \
                    (1.0 - exp(-TIME_SAMPLING / BUFFER_RISETIME))

                q += pixels_signals_tracks[idx] * TIME_SAMPLING * w

        else:
            idx = total_backtracks * ic + off + itrk
            q = pixels_signals_tracks[idx] * TIME_SAMPLING

        signal_charge_track[ip, ic, itrk] = q

signal_charge_track = cuda.device_array(
    (n_pixels, n_ticks, MAX_TRACKS_PER_PIXEL),
    dtype=np.float32
)

integrate_signal_tracks[BPG_2D, TPB_2D](
    pixels_tracks_signals,
    num_backtrack,
    offset_backtrack,
    signal_charge_track
)

# --- Phase 5: ADC Window Discovery ---

@cuda.jit(max_registers=96)
def discover_adc_windows(
    signal_charge,
    noise_uncorr,
    noise_disc,
    noise_reset,
    periodic_reset_phase,
    pixel_thresholds,
    adc_start,
    adc_end,
    adc_ticks_idx,
    adc_counts
):
    ip = cuda.grid(1)

    if ip >= signal_charge.shape[0]:
        return

    n_ticks = signal_charge.shape[1]

    ic = 0
    iadc = 0
    adc_busy = 0
    true_q = 0.0
    q_sum = noise_reset[ip, 0]

    apply_periodic_reset = PERIODIC_RESET_CYCLES > 0
    reset_phase = periodic_reset_phase[ip] if apply_periodic_reset else -1

    while ic < n_ticks or adc_busy > 0:

        if iadc >= MAX_ADC_VALUES:
            break

        q = signal_charge[ip, ic]

        if apply_periodic_reset and ic % PERIODIC_RESET_CYCLES == reset_phase:
            q_sum = noise_reset[ip, ic]
            true_q = 0.0
            ic += 1
            continue

        q_sum += q
        true_q += q

        if adc_busy > 0:
            adc_busy -= 1

        q_noise = noise_uncorr[ip, ic]
        disc_noise = noise_disc[ip, ic]

        if q_sum + q_noise >= pixel_thresholds[ip] + disc_noise and adc_busy == 0:

            interval = round(
                (3 * CLOCK_CYCLE + ADC_HOLD_DELAY * CLOCK_CYCLE) / TIME_SAMPLING
            )
            start = ic
            end = min(ic + interval, n_ticks - 1)

            adc_start[ip, iadc] = start
            adc_end[ip, iadc]   = end
            adc_ticks_idx[ip, iadc] = ic

            iadc += 1

            ic += round(RESET_CYCLES * CLOCK_CYCLE / TIME_SAMPLING)
            adc_busy = round(ADC_BUSY_DELAY * CLOCK_CYCLE / TIME_SAMPLING)

            q_sum = noise_reset[ip, ic]
            true_q = 0.0
            continue

        ic += 1

    adc_counts[ip] = iadc

# Init ADC state arrays.
adc_start = cuda.device_array(
    (n_pixels, MAX_ADC_VALUES),
    dtype=np.int32
)

adc_end = cuda.device_array(
    (n_pixels, MAX_ADC_VALUES),
    dtype=np.int32
)

adc_ticks_idx = cuda.device_array(
    (n_pixels, MAX_ADC_VALUES),
    dtype=np.int32
)

adc_counts = cuda.device_array(
    (n_pixels),
    dtype=np.int32
)

threads = 128
blocks = ceil(n_pixels / threads)

discover_adc_windows[blocks, threads](signal_charge,
    noise_uncorr,
    noise_disc,
    noise_reset,
    periodic_reset_phase,
    pixel_thresholds,
    adc_start,
    adc_end,
    adc_ticks_idx,
    adc_counts)

# --- Phase 5b: Integrate Windows ---

@cuda.jit
def integrate_windows(
    signal_charge,
    signal_charge_track,
    adc_start,
    adc_end,
    adc_counts,
    adc_ticks_idx,
    adc_list,
    adc_ticks_list,
    current_fractions,
    time_ticks,
    time_padding
):
    ip, iadc = cuda.grid(2)
    if ip >= signal_charge.shape[0]:
        return
    if iadc >= adc_counts[ip]:
        return

    start = adc_start[ip, iadc]
    end   = adc_end[ip, iadc]

    q_sum = 0.0
    true_q = 0.0

    ntrks = min(
        current_fractions.shape[2],
        signal_charge_track.shape[2]
    )

    for ic in range(start, end + 1):
        q = signal_charge[ip, ic]
        q_sum += q
        true_q += q

        for itrk in range(ntrks):
            current_fractions[ip, iadc, itrk] += signal_charge_track[ip, ic, itrk]

    if true_q > 0:
        for itrk in range(ntrks):
            current_fractions[ip, iadc, itrk] /= true_q

    adc_list[ip, iadc] = q_sum

    tick = adc_ticks_idx[ip, iadc]
    adc_ticks_list[ip, iadc] = (
        time_ticks[min(tick, len(time_ticks)-1)]
        + time_padding
    )

threads = (16, 8)
blocks = (
    ceil(n_pixels / threads[0]),
    ceil(MAX_ADC_VALUES / threads[1])
)

integrate_windows[blocks, threads](signal_charge,
    signal_charge_track,
    adc_start,
    adc_end,
    adc_counts,
    adc_ticks_idx,
    integral_list,
    adc_ticks_list,
    current_fractions,
    time_ticks,
    time_padding_val)

# Quick and dirty array simularity check of new flow compared to frozen outputs (old get_adc_value kernel).
print((np.sum(integral_list), np.sum(outputs["integral_list"])), (np.sum(adc_ticks_list), np.sum(outputs["adc_ticks_list"])), (np.sum(current_fractions), np.sum(outputs["current_fractions"])))