"""
Module that simulates the front-end electronics (triggering, ADC)
"""

import numpy as np
import cupy as cp
import h5py
import yaml

from numba import cuda
from numba.cuda.random import xoroshiro128p_normal_float32
from math import exp, floor

from larpix.packet import Packet_v2, TimestampPacket, TriggerPacket, SyncPacket, PacketCollection
from larpix.key import Key
from larpix.format import hdf5format
from .consts import detector

from .pixels_from_track import id2pixel

from .consts.units import mV, e
from .consts import units

from .consts.light import LIGHT_TRIG_MODE

#: Number of back-tracked segments to be recorded
ASSOCIATION_COUNT_TO_STORE = 10
#: Maximum number of ADC values stored per pixel
MAX_ADC_VALUES = 20
#: Discrimination threshold in e-
DISCRIMINATION_THRESHOLD = 7e3 * e
#: ADC hold delay in clock cycles
ADC_HOLD_DELAY = 15
#: ADC busy delay in clock cycles
ADC_BUSY_DELAY = 9
#: Reset time in clock cycles
RESET_CYCLES = 1
#: Clock cycle time in :math:`\mu s`
CLOCK_CYCLE = 0.1
#: Clock rollover / reset time in larpix clock ticks
ROLLOVER_CYCLES = 2**31
#: Front-end gain in :math:`mV/e-`
GAIN = 4 * mV / (1e3 * e)
#: Buffer risetime in :math:`\mu s` (set >0 to include buffer response simulation)
BUFFER_RISETIME = 0.100
#: Common-mode voltage in :math:`mV`
V_CM = 288 * mV
#: Reference voltage in :math:`mV`
V_REF = 1300 * mV
#: Pedestal voltage in :math:`mV`
V_PEDESTAL = 580 * mV
#: Number of ADC counts
ADC_COUNTS = 2**8
#: Reset noise in e-
RESET_NOISE_CHARGE = 900 * e
#: Uncorrelated noise in e-
UNCORRELATED_NOISE_CHARGE = 500 * e 
#: Discriminator noise in e-
DISCRIMINATOR_NOISE = 650 * e 
#: Average time between events in microseconds
EVENT_RATE = 100000 # 10Hz
#: Beam trig io group
BEAM_TRIG_IO = 4
#: Threshold/LRS trig io group
THRES_TRIG_IO = 5

import logging
logging.basicConfig()
logger = logging.getLogger('fee')
logger.setLevel(logging.WARNING)
logger.info("ELECTRONICS SIMULATION")

def rotate_tile(pixel_id, tile_id):
    """
    Returns the pixel ID of the rotated tile.

    Args:
        pixel_id(int): pixel ID
        tile_id(int): tile ID

    Returns:
        tuple: pixel indeces
    """
    axes = detector.TILE_ORIENTATIONS[tile_id]
    x_axis = axes[2]
    y_axis = axes[1]

    pix_x = pixel_id[0]
    if x_axis < 0:
        pix_x = detector.N_PIXELS_PER_TILE[0]-pixel_id[0]-1

    pix_y = pixel_id[1]
    if y_axis < 0:
        pix_y = detector.N_PIXELS_PER_TILE[1]-pixel_id[1]-1

    return pix_x, pix_y


def gen_event_times(nevents, t0):
    """
    Generate sequential event times assuming events are uncorrelated

    Args:
        nevents(int): number of event times to generate
        t0(int): offset to apply [microseconds]

    Returns:
        array: shape `(nevents,)`, sequential event times [microseconds]
    """
    event_start_time = cp.random.exponential(scale=EVENT_RATE, size=int(nevents))
    event_start_time = cp.cumsum(event_start_time)
    event_start_time += t0

    return event_start_time


def export_to_hdf5(event_id_list,
                   adc_list,
                   adc_ticks_list,
                   unique_pix,
                   current_fractions,
                   track_ids,
                   filename,
                   event_start_times,
                   is_first_batch,
                   light_trigger_times=None,
                   light_trigger_event_id=None,
                   light_trigger_modules=None,
                   bad_channels=None):
    """
    Saves the ADC counts in the LArPix HDF5 format.
    Args:
        event_id_list (:obj:`numpy.ndarray`): list of event ids for each ADC value for each pixel
        adc_list (:obj:`numpy.ndarray`): list of ADC values for each pixel
        adc_ticks_list (:obj:`numpy.ndarray`): list of time ticks for each pixel
        unique_pix (:obj:`numpy.ndarray`): list of pixel IDs
        current_fractions (:obj:`numpy.ndarray`): array containing the fractional current
            induced by each track on each pixel
        track_ids (:obj:`numpy.ndarray`): 2D array containing the track IDs associated
            to each pixel
        filename (str): filename of HDF5 output file
        event_times (:obj:`numpy.ndarray`): list of timestamps for start each unique event [in microseconds]
        is_first_batch (bool): `True` if this is the first batch to save to the file
        light_trigger_times (array): 1D array of light trigger timestamps (relative to event t0) [in microseconds]
        light_trigger_event_id (array): 1D array of event id for each light trigger
        light_trigger_modules (array): 1D array of module id for each light trigger
        bad_channels (dict): dictionary containing as value a list of bad channels and as
            the chip key
    Returns:
        tuple: a tuple containing the list of LArPix packets and the list of entries for the `mc_packets_assn` dataset
    """

    io_groups = np.unique(np.array(list(detector.MODULE_TO_IO_GROUPS.values())))
    packets = []
    packets_mc = []
    packets_frac = []

    if is_first_batch:
        for io_group in io_groups:
            packets.append(TimestampPacket(timestamp=0))
            packets[-1].chip_key = Key(io_group,0,0)
            packets_mc.append([-1] * track_ids.shape[1])
            packets_frac.append([0] * current_fractions.shape[2])

            packets.append(SyncPacket(sync_type=b'S', timestamp=0, io_group=io_group))
            packets_mc.append([-1] * track_ids.shape[1])
            packets_frac.append([0] * current_fractions.shape[2])

    packets_mc_ds = []
    last_event = -1

    if bad_channels:
        with open(bad_channels, 'r') as bad_channels_file:
            bad_channels_list = yaml.load(bad_channels_file, Loader=yaml.FullLoader)

    unique_events, unique_events_inv = np.unique(event_id_list[...,0], return_inverse=True)
    event_start_time_list = (event_start_times[unique_events_inv] / CLOCK_CYCLE).astype(int)

    light_trigger_times = np.empty((0,)) if light_trigger_times is None else light_trigger_times
    light_trigger_event_id = np.empty((0,), dtype=int) if light_trigger_event_id is None else light_trigger_event_id

    rollover_count = 0
    for itick, adcs in enumerate(adc_list):
        ts = adc_ticks_list[itick]
        pixel_id = unique_pix[itick]

        pix_x, pix_y, plane_id = id2pixel(pixel_id)
        module_id = plane_id//2+1

        if module_id not in detector.MODULE_TO_IO_GROUPS.keys():
            logger.warning("Pixel ID not valid %i" % module_id)
            continue

        tile_x = int(pix_x//detector.N_PIXELS_PER_TILE[0])
        tile_y = int(pix_y//detector.N_PIXELS_PER_TILE[1])
        anode_id = 0 if plane_id % 2 == 0 else 1
        tile_id = detector.TILE_MAP[anode_id][tile_x][tile_y]

        for iadc, adc in enumerate(adcs):
            t = ts[iadc]

            if adc > digitize(0):
                while True:
                    event = event_id_list[itick,iadc]
                    event_t0 = event_start_time_list[itick]
                    time_tick = int(np.floor(t / CLOCK_CYCLE + event_t0))

                    if event_t0 > ROLLOVER_CYCLES-1 or time_tick > ROLLOVER_CYCLES-1:
                        # 31-bit rollover
                        rollover_count += 1
                        for io_group in io_groups:
                            packets.append(SyncPacket(sync_type=b'S',
                                                      timestamp=ROLLOVER_CYCLES-1, io_group=io_group))
                            packets_mc.append([-1] * track_ids.shape[1])
                            packets_frac.append([0] * current_fractions.shape[2])
                        event_start_time_list[itick:] -= ROLLOVER_CYCLES
                    else:
                        break

                event_t0 = event_t0 % ROLLOVER_CYCLES
                time_tick = time_tick % ROLLOVER_CYCLES

                # new event, insert light triggers and timestamp flag
                if event != last_event:
                    for io_group in io_groups:
                        packets.append(TimestampPacket(timestamp=event_start_times[unique_events_inv[itick]] * units.mus / units.s))
                        packets[-1].chip_key = Key(io_group,0,0)
                        packets_mc.append([-1] * track_ids.shape[1])
                        packets_frac.append([0] * current_fractions.shape[2])

                    trig_mask = light_trigger_event_id == event
                    if any(trig_mask):
                        for t_trig, module_trig in zip(light_trigger_times[trig_mask], light_trigger_modules[trig_mask]):
                            t_trig = int(np.floor(t_trig / CLOCK_CYCLE + event_t0)) % ROLLOVER_CYCLES
                            if LIGHT_TRIG_MODE == 0:
                                for io_group in detector.MODULE_TO_IO_GROUPS[int(module_trig)]:
                                    packets.append(TriggerPacket(io_group=io_group, trigger_type=b'\x02', timestamp=t_trig))
                                    packets_mc.append([-1] * track_ids.shape[1])
                                    packets_frac.append([0] * current_fractions.shape[2])
                            elif LIGHT_TRIG_MODE == 1:
                                if module_trig == 0: #beam trigger
                                    io_group = BEAM_TRIG_IO
                                elif module_trig == 1: #threshold trigger
                                    io_group = THRES_TRIG_IO
                                packets.append(TriggerPacket(io_group=io_group, trigger_type=b'\x02', timestamp=t_trig))
                                packets_mc.append([-1] * track_ids.shape[1])
                                packets_frac.append([0] * current_fractions.shape[2])
                    last_event = event

                p = Packet_v2()

                try:
                    chip, channel = detector.PIXEL_CONNECTION_DICT[rotate_tile((pix_x % detector.N_PIXELS_PER_TILE[0],
                                                                                pix_y % detector.N_PIXELS_PER_TILE[1]),
                                                                   tile_id)]
                except KeyError:
                    logger.warning(f"Pixel ID not valid {pixel_id}")
                    continue

                p.dataword = int(adc)
                p.timestamp = time_tick

                try:
                    io_group_io_channel = detector.TILE_CHIP_TO_IO[tile_id][chip]
                except KeyError:
                    logger.warning(f"Chip {chip} on tile {tile_id} not found")
                    continue

                io_group, io_channel = io_group_io_channel // 1000, io_group_io_channel % 1000
                io_group = detector.MODULE_TO_IO_GROUPS[module_id][io_group-1]
                chip_key = "%i-%i-%i" % (io_group, io_channel, chip)

                if bad_channels:
                    if chip_key in bad_channels_list:
                        if channel in bad_channels_list[chip_key]:
                            logger.info(f"Channel {channel} on chip {chip_key} disabled")
                            continue

                p.chip_key = chip_key
                p.channel_id = channel
                p.receipt_timestamp = time_tick
                p.packet_type = 0
                p.first_packet = 1
                p.assign_parity()

                packets_mc.append(track_ids[itick])
                packets_frac.append(current_fractions[itick][iadc])
                packets.append(p)
            else:
                break

    if packets:
        packet_list = PacketCollection(packets, read_id=0, message='')
        hdf5format.to_file(filename, packet_list, workers=1)

        dtype = np.dtype([('track_ids',f'({ASSOCIATION_COUNT_TO_STORE},)i8'),
                          ('fraction', f'({ASSOCIATION_COUNT_TO_STORE},)f8')])
        packets_mc_ds = np.empty(len(packets), dtype=dtype)

        # First, sort the back-tracking information by the magnitude of the fraction
        packets_frac = np.array(packets_frac)
        packets_mc   = np.array(packets_mc)
        
        frac_order = np.flip(np.argsort(np.abs(packets_frac),axis=1),axis=1)
        ass_track_ids = np.take_along_axis(packets_mc,   frac_order, axis=1)
        ass_fractions = np.take_along_axis(packets_frac, frac_order, axis=1)

        # Second, only store the relevant portion.
        if ass_track_ids.shape[1] >= ASSOCIATION_COUNT_TO_STORE:
            packets_mc_ds['track_ids'] = ass_track_ids[:,:ASSOCIATION_COUNT_TO_STORE]
            packets_mc_ds['fraction' ] = ass_fractions[:,:ASSOCIATION_COUNT_TO_STORE]
        else:
            num_to_pad = ASSOCIATION_COUNT_TO_STORE - ass_track_ids.shape[1]
            packets_mc_ds['track_ids'] = np.pad(ass_track_ids,
                pad_width=((0,0),(0,num_to_pad)),
                mode='constant',
                constant_values=-1)
            packets_mc_ds['fraction' ] = np.pad(ass_fractions,
                pad_width=((0,0),(0,num_to_pad)),
                mode='constant',
                constant_values=0.)

        with h5py.File(filename, 'a') as f:
            if is_first_batch:
                f.create_dataset("mc_packets_assn", data=packets_mc_ds, maxshape=(None,))
            else:
                f['mc_packets_assn'].resize((f['mc_packets_assn'].shape[0] + packets_mc_ds.shape[0]), axis=0)
                f['mc_packets_assn'][-packets_mc_ds.shape[0]:] = packets_mc_ds

            f['configs'].attrs['vdrift'] = detector.V_DRIFT
            f['configs'].attrs['long_diff'] = detector.LONG_DIFF
            f['configs'].attrs['tran_diff'] = detector.TRAN_DIFF
            f['configs'].attrs['lifetime'] = detector.ELECTRON_LIFETIME
            f['configs'].attrs['drift_length'] = detector.DRIFT_LENGTH

    return packets, packets_mc_ds


def digitize(integral_list, gain=GAIN):
    """
    The function takes as input the integrated charge and returns the digitized
    ADC counts.

    Args:
        integral_list(:obj:`numpy.ndarray`): list of charge collected by each pixel
        gain(:obj:`numpy.ndarray`): list of gain values (or float) for each pixel

    Returns:
        :obj:`numpy.ndarray`: list of ADC values for each pixel
    """
    xp = cp.get_array_module(integral_list)
    adcs = xp.minimum(xp.around(xp.maximum((integral_list * gain + V_PEDESTAL - V_CM), 0)
                                * ADC_COUNTS / (V_REF - V_CM)), ADC_COUNTS-1)

    return adcs

@cuda.jit
def get_adc_values(pixels_signals,
                   pixels_signals_tracks,
                   time_ticks,
                   adc_list,
                   adc_ticks_list,
                   time_padding,
                   rng_states,
                   current_fractions,
                   pixel_thresholds):
    """
    Implementation of self-trigger logic

    Args:
        pixels_signals (:obj:`numpy.ndarray`): list of induced currents for
            each pixel
        pixels_signals_tracks (:obj:`numpy.ndarray`): list of induced currents
            for each track that induces current on each pixel
        time_ticks (:obj:`numpy.ndarray`): list of time ticks for each pixel
        adc_list (:obj:`numpy.ndarray`): list of integrated charges for each
            pixel
        adc_ticks_list (:obj:`numpy.ndarray`): list of the time ticks that
            correspond to each integrated charge
        time_padding (float): time interval to add to each time tick.
        rng_states (:obj:`numpy.ndarray`): array of random states for noise
            generation
        current_fractions (:obj:`numpy.ndarray`): 2D array that will contain
            the fraction of current induced on the pixel by each track
        pixel_thresholds(: obj: `numpy.ndarray`): list of discriminator
            thresholds for each pixel
    """
    ip = cuda.grid(1)

    if ip < pixels_signals.shape[0]:
        curre = pixels_signals[ip]
        ic = 0
        iadc = 0
        adc_busy = 0
        last_reset = 0
        true_q = 0
        q_sum = xoroshiro128p_normal_float32(rng_states, ip) * RESET_NOISE_CHARGE

        while ic < curre.shape[0] or adc_busy > 0:

            if iadc >= MAX_ADC_VALUES:
                print("More ADC values than possible,", MAX_ADC_VALUES)
                break

            q = 0
            if BUFFER_RISETIME > 0:
                conv_start = max(last_reset, floor(ic - 10*BUFFER_RISETIME/detector.TIME_SAMPLING))
                for jc in range(conv_start, min(ic+1, curre.shape[0])):
                    w = exp((jc - ic) * detector.TIME_SAMPLING / BUFFER_RISETIME) * (1 - exp(-detector.TIME_SAMPLING/BUFFER_RISETIME))
                    q += curre[jc] * detector.TIME_SAMPLING * w

                    for itrk in range(current_fractions.shape[2]):
                        current_fractions[ip][iadc][itrk] += pixels_signals_tracks[ip][jc][itrk] * detector.TIME_SAMPLING * w

            elif ic < curre.shape[0]:
                q += curre[ic] * detector.TIME_SAMPLING
                for itrk in range(current_fractions.shape[2]):
                    current_fractions[ip][iadc][itrk] += pixels_signals_tracks[ip][ic][itrk] * detector.TIME_SAMPLING

            q_sum += q
            true_q += q

            q_noise = xoroshiro128p_normal_float32(rng_states, ip) * UNCORRELATED_NOISE_CHARGE
            disc_noise = xoroshiro128p_normal_float32(rng_states, ip) * DISCRIMINATOR_NOISE

            if adc_busy > 0:
                adc_busy -= 1

            if q_sum + q_noise >= pixel_thresholds[ip] + disc_noise and adc_busy == 0:
                interval = round((3 * CLOCK_CYCLE + ADC_HOLD_DELAY * CLOCK_CYCLE) / detector.TIME_SAMPLING)
                integrate_end = ic+interval

                ic+=1

                while ic <= integrate_end:
                    q = 0

                    if BUFFER_RISETIME > 0:
                        conv_start = max(last_reset, floor(ic - 10*BUFFER_RISETIME/detector.TIME_SAMPLING))
                        for jc in range(conv_start, min(ic+1, curre.shape[0])):
                            w = exp((jc - ic) * detector.TIME_SAMPLING / BUFFER_RISETIME) * (1 - exp(-detector.TIME_SAMPLING/BUFFER_RISETIME))
                            q += curre[jc] * detector.TIME_SAMPLING * w

                            for itrk in range(current_fractions.shape[2]):
                                current_fractions[ip][iadc][itrk] += pixels_signals_tracks[ip][jc][itrk] * detector.TIME_SAMPLING * w

                    elif ic < curre.shape[0]:
                        q += curre[ic] * detector.TIME_SAMPLING
                        for itrk in range(current_fractions.shape[2]):
                            current_fractions[ip][iadc][itrk] += pixels_signals_tracks[ip][ic][itrk] * detector.TIME_SAMPLING

                    q_sum += q
                    true_q += q
                    ic+=1

                adc = q_sum + xoroshiro128p_normal_float32(rng_states, ip) * UNCORRELATED_NOISE_CHARGE
                disc_noise = xoroshiro128p_normal_float32(rng_states, ip) * DISCRIMINATOR_NOISE

                if adc < pixel_thresholds[ip] + disc_noise:
                    ic += round(RESET_CYCLES * CLOCK_CYCLE / detector.TIME_SAMPLING)
                    q_sum = xoroshiro128p_normal_float32(rng_states, ip) * RESET_NOISE_CHARGE
                    true_q = 0

                    for itrk in range(current_fractions.shape[2]):
                        current_fractions[ip][iadc][itrk] = 0
                    last_reset = ic
                    continue

                #tot_backtracked = 0
                #for itrk in range(current_fractions.shape[2]):
                #    tot_backtracked += current_fractions[ip][iadc][itrk]

                if true_q > 0:
                    for itrk in range(current_fractions.shape[2]):
                        current_fractions[ip][iadc][itrk] /= true_q

                adc_list[ip][iadc] = adc

                crossing_time_tick = min((ic, len(time_ticks)-1))
                # handle case when tick extends past end of current array
                post_adc_ticks = max((ic - crossing_time_tick, 0))
                #+2-tick delay from when the PACMAN receives the trigger and when it registers it.
                adc_ticks_list[ip][iadc] = time_ticks[crossing_time_tick]+time_padding-2+post_adc_ticks

                ic += round(RESET_CYCLES * CLOCK_CYCLE / detector.TIME_SAMPLING)
                last_reset = ic
                adc_busy = round(ADC_BUSY_DELAY * CLOCK_CYCLE / detector.TIME_SAMPLING)

                q_sum = xoroshiro128p_normal_float32(rng_states, ip) * RESET_NOISE_CHARGE
                true_q = 0

                iadc += 1
                continue

            ic += 1
