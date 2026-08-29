"""
Module that simulates the front-end electronics (triggering, ADC)
"""

import numpy as np
import cupy as cp
import h5py
import yaml
import warnings

from numba import cuda, float32, int32
from numba.cuda.random import xoroshiro128p_normal_float32, xoroshiro128p_uniform_float32
from math import exp, floor, ceil

from larpix.packet import Packet_v2, Packet_v3, TimestampPacket, TriggerPacket, SyncPacket, PacketCollection
from larpix.key import Key
from larpix.format import hdf5format

from .pixels_from_track import id2pixel

from .consts.units import mV, e
from .consts import units, detector, light, sim

import logging
logging.basicConfig()
logger = logging.getLogger('fee')
logger.setLevel(logging.WARNING)
logger.info("ELECTRONICS SIMULATION")

def get_trig_io():
    """
    Returns the io_group if the trigger is only forwarded to one pacman
    """
    if light.LIGHT_TRIG_MODE == 0: #: Threshold/LRS trigger
        trig_io = 2
    elif light.LIGHT_TRIG_MODE == 1: #: Beam trigger
        trig_io = 1
    return trig_io

def rotate_tile(pixel_id, tile_id):
    """
    Returns the pixel ID of the rotated tile.

    Args:
        pixel_id(int): pixel ID
        tile_id(int): tile ID

    Returns:
        tuple: pixel indices
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


def gen_event_times(nevents, t0=detector.NON_BEAM_EVENT_GAP):
    """
    Generate sequential event times assuming events are uncorrelated

    Args:
        nevents (int): number of event times to generate
        t0 (int): offset to apply [microseconds]

    Returns:
        :obj:`numpy.ndarray`: shape `(nevents,)`, sequential event times [microseconds]
    """
    event_start_time = cp.random.exponential(scale=detector.EVENT_RATE, size=int(nevents))
    event_start_time = cp.cumsum(event_start_time)
    event_start_time += t0

    return event_start_time


def export_to_hdf5(event_id_list,
                   adc_list,
                   adc_ticks_list,
                   unique_pix,
                   current_fractions,
                   track_ids,
                   traj_ids,
                   filename,
                   event_start_times,
                   light_trigger_times=None,
                   light_trigger_event_id=None,
                   light_trigger_modules=None,
                   bad_channels=None,
                   i_mod=-1,
                   compression=None):
    """
    Saves the ADC counts in the LArPix HDF5 format.

    Args:
        event_id_list (:obj:`numpy.ndarray`): event ids for each tick;
                shape (nticks, max_adcs); dtype uint32
        adc_list (:obj:`numpy.ndarray`): ADC values for each tick;
                shape (nticks, max_adcs); dtype float64
        adc_ticks_list (:obj:`numpy.ndarray`): timestamps for each tick;
                shape (nticks, max_adcs); dtype float64
        unique_pix (:obj:`numpy.ndarray`): pixel IDs for each tick;
                shape (nticks,); dtype int32
        current_fractions (:obj:`numpy.ndarray`): fractional current induced by
                each track on each pixel;
                shape (nticks, max_adcs, max_backtracks); dtype float64
        track_ids (:obj:`numpy.ndarray`): track IDs associated to each pixel;
                shape (nticks, max_backtracks); dtype int64
        filename (str): filename of HDF5 output file
        event_start_times (:obj:`numpy.ndarray`): timestamps of start of each
                unique event [in microseconds];
                shape (nevents,); dtype float64
        light_trigger_times (:obj:`numpy.ndarray`): light trigger timestamps
                (relative to event t0) [in microseconds];
                shape (ntrigs,); dtype float64
        light_trigger_event_id (:obj:`numpy.ndarray`): event id for each light trigger;
                shape (ntrigs,); dtype uint32
        light_trigger_modules (:obj:`numpy.ndarray`): module id for each light trigger;
                shape (ntrigs,); dtype int64
        bad_channels (dict): dictionary mapping a chip key to a list of bad channels
        i_mod (int): module index for saving the result in each module
                individually if needed.
        compression (:obj:`str`, optional): enable file compression of the output HDF5 datasets. Defaults to None,
                supported options are 'lzf' and 'gzip'

    Returns:
        tuple: a tuple containing the list of LArPix packets and the list of
                entries for the `mc_packets_assn` dataset
    """

    io_groups = np.unique(np.array(list(detector.MODULE_TO_IO_GROUPS.values())))
    io_groups = io_groups if i_mod < 0 else io_groups[(i_mod-1)*2: i_mod*2]
    packets = []
    packets_mc_evt = []
    packets_mc_trk = []
    packets_mc_trj = []
    packets_frac = []

    packets_mc_ds = []
    last_event = -1

    if bad_channels:
        with open(bad_channels, 'r') as bad_channels_file:
            bad_channels_list = yaml.load(bad_channels_file, Loader=yaml.FullLoader)

    unique_events, unique_events_inv = np.unique(event_id_list[...,0], return_inverse=True)
    event_start_time_list = (event_start_times[unique_events_inv] / detector.CLOCK_CYCLE).astype(int)
    light_trigger_times = np.empty((0,)) if light_trigger_times is None else light_trigger_times
    light_trigger_event_id = np.empty((0,), dtype=int) if light_trigger_event_id is None else light_trigger_event_id

    rollover_count = 0
    last_time_tick = -1
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

            if adc > 0:
                while True:
                    event = event_id_list[itick,iadc]
                    event_t0 = event_start_time_list[itick]
                    time_tick = int(np.floor(t / detector.CLOCK_CYCLE + event_t0))

                    if event_t0 > detector.CLOCK_RESET_PERIOD-1 or time_tick > detector.CLOCK_RESET_PERIOD-1:
                        # rollover (reset) at either PPS or at the 31-bit clock limit
                        rollover_count += 1
                        # FIXME disable this sync packets fill as it may overlap with what is already filled
                        #for io_group in io_groups:
                        #    packets.append(SyncPacket(sync_type=b'S',
                        #                              timestamp=detector.CLOCK_RESET_PERIOD-1, io_group=io_group))
                        #    packets_mc_evt.append([event])
                        #    packets_mc_trk.append([-1] * track_ids.shape[1])
                        #    packets_frac.append([0] * current_fractions.shape[2])
                        event_start_time_list[itick:] -= detector.CLOCK_RESET_PERIOD
                    else:
                        break
                
                event_t0 = event_t0 % detector.CLOCK_RESET_PERIOD
                time_tick = time_tick % detector.CLOCK_RESET_PERIOD

                # FIXME light.LIGHT_TRIG_MODE != 0 should also be here
                # This trigger packet block should only be activated with trigger forwarding scheme to individual modules
                if light.LIGHT_TRIG_MODE != 1:
                    # new event, insert light triggers and timestamp flag
                    if event != last_event:
                        for io_group in io_groups:
                            packets.append(TimestampPacket(timestamp=event_start_times[unique_events_inv[itick]] * units.mus / units.s))
                            packets[-1].chip_key = Key(io_group,0,0)

                            packets_mc_evt.append([event])
                            packets_mc_trk.append([-1] * track_ids.shape[1])
                            packets_mc_trj.append([-1] * traj_ids.shape[1])
                            packets_frac.append([0] * current_fractions.shape[2])

                            packets.append(SyncPacket(sync_type=b'S', timestamp=time_tick , io_group=io_group))
                            packets_mc_evt.append([event])
                            packets_mc_trk.append([-1] * track_ids.shape[1])
                            packets_mc_trj.append([-1] * traj_ids.shape[1])
                            packets_frac.append([0] * current_fractions.shape[2])

                        trig_mask = light_trigger_event_id == event
                        if any(trig_mask):
                            for t_trig, module_trig in zip(light_trigger_times[trig_mask], light_trigger_modules[trig_mask]):
                                t_trig = int(np.floor(t_trig / detector.CLOCK_CYCLE + event_t0)) % detector.CLOCK_RESET_PERIOD
                                if light.LIGHT_TRIG_MODE == 0:
                                    for io_group in detector.MODULE_TO_IO_GROUPS[int(module_trig)]:
                                        packets.append(TriggerPacket(io_group=io_group, trigger_type=b'\x02', timestamp=t_trig))
                                        packets_mc_evt.append([event])
                                        packets_mc_trk.append([-1] * track_ids.shape[1])
                                        packets_mc_trj.append([-1] * traj_ids.shape[1])
                                        packets_frac.append([0] * current_fractions.shape[2])
                                # redundant here
                                elif light.LIGHT_TRIG_MODE == 1:
                                    if module_trig == 1 or module_trig == 0: #1, beam trigger; 2, threshold trigger
                                        io_group = get_trig_io()
                                    packets.append(TriggerPacket(io_group=io_group, trigger_type=b'\x02', timestamp=t_trig))
                                    packets_mc_evt.append([event])
                                    packets_mc_trk.append([-1] * track_ids.shape[1])
                                    packets_mc_trj.append([-1] * traj_ids.shape[1])
                                    packets_frac.append([0] * current_fractions.shape[2])
                        last_event = event

                # Instantiate correct packet version for LArPix; see consts/detector.py
                p = detector.PACKET_VERSION()

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
                p.packet_type = detector.PACKET_TYPE
                p.first_packet = 1
                p.assign_parity()

                if not time_tick==last_time_tick:
                    # timestamp packet every time there is a new "message"
                    # the logic in real data for when a timestamp packet is complicated and depends on pacman CPU speed, packet creation rate
                    # best simple approximation is that any group of packets with the same timestamp get a single timestamp packet
                    last_time_tick = time_tick
                    packets.append(TimestampPacket(timestamp=np.floor(event_start_time_list[0] * detector.CLOCK_CYCLE * units.mus/units.s)) ) # s
                    packets[-1].chip_key = Key(io_group,0,0)

                    packets_mc_evt.append([event])
                    packets_mc_trk.append([-1] * sim.MAX_TRACKS_PER_PIXEL)
                    packets_mc_trj.append([-1] * sim.MAX_TRACKS_PER_PIXEL)
                    packets_frac.append([0] * sim.MAX_TRACKS_PER_PIXEL)

                packets_mc_evt.append([event])
                packets_mc_trk.append(track_ids[itick])
                packets_mc_trj.append(traj_ids[itick])
                packets_frac.append(current_fractions[itick][iadc])
                packets.append(p)
                
            else:
                break

    if packets:
        packet_list = PacketCollection(packets, read_id=0, message='')
        hdf5format.to_file(filename, packet_list, workers=1, version=detector.LARPIX_HDF5_VERSION)
        dtype = np.dtype([('event_ids',f'(1,)i8'),
                          ('segment_ids',f'({sim.ASSOCIATION_COUNT_TO_STORE},)i8'),
                          ('fraction', f'({sim.ASSOCIATION_COUNT_TO_STORE},)f8'),
                          ('file_traj_ids',f'({sim.ASSOCIATION_COUNT_TO_STORE},)i8'),
                          ('fraction_traj',f'({sim.ASSOCIATION_COUNT_TO_STORE},)f8'),])

        packets_mc_ds = np.empty(len(packets), dtype=dtype)

        # First, sort the back-tracking information by the magnitude of the fraction
        packets_frac = np.array(packets_frac)
        packets_mc_trk = np.array(packets_mc_trk)
        packets_mc_trj = np.array(packets_mc_trj)
        packets_mc_evt = np.array(packets_mc_evt)
        
        frac_order = np.flip(np.argsort(packets_frac,axis=1),axis=1)
        ass_segment_ids = np.take_along_axis(packets_mc_trk,   frac_order, axis=1)
        ass_trajectory_ids = np.take_along_axis(packets_mc_trj, frac_order, axis=1)
        ass_fractions = np.take_along_axis(packets_frac, frac_order, axis=1)

        # Second, only store the relevant portion.
        if ass_segment_ids.shape[1] >= sim.ASSOCIATION_COUNT_TO_STORE:
            packets_mc_ds['segment_ids'] = ass_segment_ids[:,:sim.ASSOCIATION_COUNT_TO_STORE]
            packets_mc_ds['fraction' ] = ass_fractions[:,:sim.ASSOCIATION_COUNT_TO_STORE]
        else:
            num_to_pad = sim.ASSOCIATION_COUNT_TO_STORE - ass_segment_ids.shape[1]
            packets_mc_ds['segment_ids'] = np.pad(ass_segment_ids,
                pad_width=((0,0),(0,num_to_pad)),
                mode='constant',
                constant_values=-1)
            packets_mc_ds['fraction' ] = np.pad(ass_fractions,
                pad_width=((0,0),(0,num_to_pad)),
                mode='constant',
                constant_values=0.)


        ass_track_ids = np.full(ass_trajectory_ids.shape,fill_value=-1,dtype=np.int32)
        ass_fractions_track = np.full(ass_fractions.shape,fill_value=0.,dtype=np.float32)
        for pidx, tids in enumerate(ass_trajectory_ids):
            mask = tids > -1
            for tidx, unique_tid in enumerate(np.unique(tids[mask])):
                ass_track_ids[pidx][tidx] = unique_tid
                ass_fractions_track[pidx][tidx] = np.sum(ass_fractions[pidx][mask][tids[mask]==unique_tid])

        if ass_segment_ids.shape[1] >= sim.ASSOCIATION_COUNT_TO_STORE:
            packets_mc_ds['file_traj_ids'] = ass_track_ids[:,:sim.ASSOCIATION_COUNT_TO_STORE]
            packets_mc_ds['fraction_traj'] = ass_fractions_track[:,:sim.ASSOCIATION_COUNT_TO_STORE]
        else:
            num_to_pad = sim.ASSOCIATION_COUNT_TO_STORE - ass_track_ids.shape[1]
            packets_mc_ds['file_traj_ids'] = np.pad(ass_track_ids,
                pad_width=((0,0),(0,num_to_pad)),
                mode='constant',
                constant_values=-1)
            packets_mc_ds['fraction_traj' ] = np.pad(ass_fractions_track,
                pad_width=((0,0),(0,num_to_pad)),
                mode='constant',
                constant_values=0.)

        packets_mc_ds['event_ids'] = packets_mc_evt

        with h5py.File(filename, 'a') as f:
            if "mc_packets_assn" not in f.keys():
                f.create_dataset("mc_packets_assn", data=packets_mc_ds, maxshape=(None,), compression=compression)
            else:
                f['mc_packets_assn'].resize((f['mc_packets_assn'].shape[0] + packets_mc_ds.shape[0]), axis=0)
                f['mc_packets_assn'][-packets_mc_ds.shape[0]:] = packets_mc_ds

            f['configs'].attrs['vdrift'] = detector.V_DRIFT
            f['configs'].attrs['long_diff'] = detector.LONG_DIFF
            f['configs'].attrs['tran_diff'] = detector.TRAN_DIFF
            f['configs'].attrs['lifetime'] = detector.ELECTRON_LIFETIME
            f['configs'].attrs['drift_length'] = detector.DRIFT_LENGTH

    return packets, packets_mc_ds

def export_sync_to_hdf5(filename, event, sync_times, i_mod, compression=None):
    """
    Saves sync packets in the LArPix HDF5 format.
    Args:
        event (int): ID of the event
        sync_times (:obj:`numpy.ndarray`): list of sync timestamps [us]
    Returns:
        tuple: a tuple containing the list of LArPix sync packets and the list of entries for the `mc_packets_assn` dataset
    """
    io_groups = np.unique(np.array(list(detector.MODULE_TO_IO_GROUPS.values())))
    io_groups = detector.MODULE_TO_IO_GROUPS[i_mod] if i_mod > 0 else io_groups

    packets = []
    packets_mc_evt = []
    packets_mc_trk = []
    packets_frac = []
    packets_mc_trj = []
    packets_frac_trj =[]

    sync_ticks = sync_times / detector.CLOCK_CYCLE # us -> time tick
    for sync_tick in sync_ticks:
        if sync_tick % detector.CLOCK_RESET_PERIOD != 0:
            warnings.warn("The provided sync time is not a multiple of the reset period!")
            sync_tick = sync_tick // detector.CLOCK_RESET_PERIOD * detector.CLOCK_RESET_PERIOD
        for io_group in io_groups:
            packets.append(SyncPacket(sync_type=b'S', timestamp=sync_tick, io_group=io_group))
            packets_mc_evt.append(np.array([event]))

            packets_mc_trk.append(np.array([-1] * sim.ASSOCIATION_COUNT_TO_STORE))
            packets_frac.append(np.array([0] * sim.ASSOCIATION_COUNT_TO_STORE))
            packets_mc_trj.append(np.array([-1] * sim.ASSOCIATION_COUNT_TO_STORE))
            packets_frac_trj.append(np.array([0] * sim.ASSOCIATION_COUNT_TO_STORE))

    if packets:
        packet_list = PacketCollection(packets, read_id=0, message='')
        hdf5format.to_file(filename, packet_list, workers=1, version=detector.LARPIX_HDF5_VERSION)

        dtype = np.dtype([('event_ids',f'(1,)i8'),
                          ('segment_ids',f'({sim.ASSOCIATION_COUNT_TO_STORE},)i8'),
                          ('fraction', f'({sim.ASSOCIATION_COUNT_TO_STORE},)f8'),
                          ('file_traj_ids',f'({sim.ASSOCIATION_COUNT_TO_STORE},)i8'),
                          ('fraction_traj',f'({sim.ASSOCIATION_COUNT_TO_STORE},)f8'),])

        packets_mc_ds = np.empty(len(packets), dtype=dtype)

        packets_frac = np.array(packets_frac)
        packets_mc_trk   = np.array(packets_mc_trk)
        packets_mc_evt   = np.array(packets_mc_evt)
        packets_mc_trj = np.array(packets_mc_trj)
        packets_frac_trj = np.array(packets_frac_trj)

        packets_mc_ds['event_ids'] = packets_mc_evt
        packets_mc_ds['segment_ids'] = packets_mc_trk
        packets_mc_ds['fraction' ] = packets_frac
        packets_mc_ds['file_traj_ids'] = packets_mc_trj
        packets_mc_ds['fraction_traj'] = packets_frac_trj

        with h5py.File(filename, 'a') as f:
            if "mc_packets_assn" not in f.keys():
                f.create_dataset("mc_packets_assn", data=packets_mc_ds, maxshape=(None,), compression=compression)
            else:
                f['mc_packets_assn'].resize((f['mc_packets_assn'].shape[0] + packets_mc_ds.shape[0]), axis=0)
                f['mc_packets_assn'][-packets_mc_ds.shape[0]:] = packets_mc_ds

    return packets, packets_mc_ds

def export_timestamp_trigger_to_hdf5(filename, event_ids, event_start_times, i_mod, compression=None):
    """
    Saves timestamp and trigger packets in the LArPix HDF5 format.
    Args:
        event_ids (:obj:`numpy.ndarray`): list of IDs for each unique event
        event_start_times (:obj:`numpy.ndarray`): list of timestamps for start each unique event [in microseconds]
    Returns:
        tuple: a tuple containing the list of LArPix timestamp and trigger packets and the list of entries for the `mc_packets_assn` dataset
    """
    io_groups = np.unique(np.array(list(detector.MODULE_TO_IO_GROUPS.values())))
    io_groups = detector.MODULE_TO_IO_GROUPS[i_mod] if i_mod > 0 else io_groups

    packets = []
    packets_mc_evt = []
    packets_mc_trk = []
    packets_frac = []
    packets_mc_trj = []
    packets_frac_trj =[]

    for event, evt_time in zip(event_ids, event_start_times):

        t_trig = int(np.floor(evt_time / detector.CLOCK_CYCLE)) % detector.CLOCK_RESET_PERIOD # tick

        io_group = get_trig_io()

        # timestamp packets
        packets.append(TimestampPacket(timestamp=evt_time*units.mus/units.s)) # s
        packets[-1].chip_key = Key(io_group,0,0)
        packets_mc_evt.append(np.array([event]))
        packets_mc_trk.append(np.array([-1] * sim.ASSOCIATION_COUNT_TO_STORE))
        packets_frac.append(np.array([0] * sim.ASSOCIATION_COUNT_TO_STORE))
        packets_mc_trj.append(np.array([-1] * sim.ASSOCIATION_COUNT_TO_STORE))
        packets_frac_trj.append(np.array([0] * sim.ASSOCIATION_COUNT_TO_STORE))

        # trigger packets
        packets.append(TriggerPacket(io_group=io_group, trigger_type=b'\x02', timestamp=t_trig)) # tick
        packets_mc_evt.append(np.array([event]))
        packets_mc_trk.append(np.array([-1] * sim.ASSOCIATION_COUNT_TO_STORE))
        packets_frac.append(np.array([0] * sim.ASSOCIATION_COUNT_TO_STORE))
        packets_mc_trj.append(np.array([-1] * sim.ASSOCIATION_COUNT_TO_STORE))
        packets_frac_trj.append(np.array([0] * sim.ASSOCIATION_COUNT_TO_STORE))

    if packets:
        packet_list = PacketCollection(packets, read_id=0, message='')
        hdf5format.to_file(filename, packet_list, workers=1, version=detector.LARPIX_HDF5_VERSION)

        dtype = np.dtype([('event_ids',f'(1,)i8'),
                          ('segment_ids',f'({sim.ASSOCIATION_COUNT_TO_STORE},)i8'),
                          ('fraction', f'({sim.ASSOCIATION_COUNT_TO_STORE},)f8'),
                          ('file_traj_ids',f'({sim.ASSOCIATION_COUNT_TO_STORE},)i8'),
                          ('fraction_traj',f'({sim.ASSOCIATION_COUNT_TO_STORE},)f8'),])
        packets_mc_ds = np.empty(len(packets), dtype=dtype)

        packets_frac = np.array(packets_frac)
        packets_mc_trk   = np.array(packets_mc_trk)
        packets_mc_evt   = np.array(packets_mc_evt)
        packets_mc_trj = np.array(packets_mc_trj)
        packets_frac_trj = np.array(packets_frac_trj)

        packets_mc_ds['event_ids'] = packets_mc_evt
        packets_mc_ds['segment_ids'] = packets_mc_trk
        packets_mc_ds['fraction' ] = packets_frac
        packets_mc_ds['file_traj_ids'] = packets_mc_trj
        packets_mc_ds['fraction_traj'] = packets_frac_trj

        with h5py.File(filename, 'a') as f:
            if "mc_packets_assn" not in f.keys():
                f.create_dataset("mc_packets_assn", data=packets_mc_ds, maxshape=(None,), compression=compression)
            else:
                f['mc_packets_assn'].resize((f['mc_packets_assn'].shape[0] + packets_mc_ds.shape[0]), axis=0)
                f['mc_packets_assn'][-packets_mc_ds.shape[0]:] = packets_mc_ds

    return packets, packets_mc_ds

def digitize(integral_list, gain=detector.GAIN, pedestal=detector.V_PEDESTAL):
    """
    The function takes as input the integrated charge and returns the digitized
    ADC counts.

    Args:
        integral_list (:obj:`numpy.ndarray`): list of charge collected by each pixel
        gain (:obj:`numpy.ndarray`): list of gain values (or float) for each pixel

    Returns:
        :obj:`numpy.ndarray`: list of ADC values for each pixel
    """
    xp = cp.get_array_module(integral_list)
    adcs = xp.floor(xp.minimum(xp.maximum((integral_list * gain * mV / e + pedestal * mV - detector.V_CM * mV), 0)
                                * detector.ADC_COUNTS / ((detector.V_REF * mV - detector.V_CM * mV) * detector.ADC_SCALE_FACTOR)
                                + detector.ADC_OFFSET, detector.ADC_COUNTS-1))

    adcs[integral_list == 0] = 0

    return adcs

# Begin reworked get_adc_values
# RNG Kernel
@cuda.jit
def generate_noise(
    rng_states,
    noise_uncorr,
    noise_disc,
    noise_reset,
    periodic_reset_phase
):
    """Pre-compute all noise values used by downstream ADC kernels.

    Draws from the xoroshiro128p RNG to populate per-pixel, per-tick
    noise arrays for uncorrelated, discriminator, and reset noise, plus
    a per-pixel periodic reset phase offset.

    Args:
        rng_states (:obj:`numpy.ndarray`): array of per-thread RNG states.
            Shape (TPB*BPG,).
        noise_uncorr (:obj:`numpy.ndarray`): Output; pre-scaled uncorrelated
            noise per pixel per tick. Shape (n_pixels, n_ticks).
        noise_disc (:obj:`numpy.ndarray`): Output; pre-scaled discriminator
            noise per pixel per tick. Shape (n_pixels, n_ticks).
        noise_reset (:obj:`numpy.ndarray`): Output; pre-scaled reset noise
            per pixel per tick. Shape (n_pixels, n_ticks).
        periodic_reset_phase (:obj:`numpy.ndarray`): Output; integer phase
            offset for periodic resets per pixel. Shape (n_pixels,).
    """
    ip = cuda.grid(1)
    if ip >= noise_uncorr.shape[0]:
        return

    # 1) periodic reset phase (exactly once)
    periodic_reset_phase[ip] = int(
        xoroshiro128p_uniform_float32(rng_states, ip)
        * (detector.PERIODIC_RESET_CYCLES + 1)
    )

    # 2) initialize noise
    noise_reset[ip, 0] = (
        xoroshiro128p_normal_float32(rng_states, ip)
        * detector.RESET_NOISE_CHARGE * units.e
    )

    noise_uncorr[ip, 0] = (
        xoroshiro128p_normal_float32(rng_states, ip)
        * detector.UNCORRELATED_NOISE_CHARGE * units.e
    )

    noise_disc[ip, 0] = (
        xoroshiro128p_normal_float32(rng_states, ip)
        * detector.DISCRIMINATOR_NOISE * units.e
    )


    # 3) per-tick noise
    for t in range(1, noise_uncorr.shape[1]):
        noise_uncorr[ip, t] = (
            xoroshiro128p_normal_float32(rng_states, ip)
            * detector.UNCORRELATED_NOISE_CHARGE * units.e
        )

        noise_disc[ip, t] = (
            xoroshiro128p_normal_float32(rng_states, ip)
            * detector.DISCRIMINATOR_NOISE * units.e
        )

        noise_reset[ip, t] = (
            xoroshiro128p_normal_float32(rng_states, ip)
            * detector.RESET_NOISE_CHARGE * units.e
        )

TIME_SAMPLING = float32(0.1)   # µs
CONV_WINDOW   = int32(10 * detector.BUFFER_RISETIME / detector.TIME_SAMPLING)
# ---------------------------------------------------------------------------
# Constant-memory weight table
#
# WEIGHTS[k] = TIME_SAMPLING * exp(-k) * (1 - exp(-1))  for k in 0..10
#
# TIME_SAMPLING is folded in here so the inner loop is a single FMA:
#   q += WEIGHTS[k] * pixels_signals[jc, ip]
#
# cuda.const.array_like() places this in GPU __constant__ memory.
# ---------------------------------------------------------------------------
_WEIGHTS_HOST = np.array([
    0.0632120559,   # k= 0, offset=  0
    0.0232544158,   # k= 1, offset= -1
    0.0085548215,   # k= 2, offset= -2
    0.0031471429,   # k= 3, offset= -3
    0.0011577692,   # k= 4, offset= -4
    0.0004259195,   # k= 5, offset= -5
    0.0001566870,   # k= 6, offset= -6
    0.0000576419,   # k= 7, offset= -7
    0.0000212053,   # k= 8, offset= -8
    0.0000078010,   # k= 9, offset= -9
    0.0000028698,   # k=10, offset=-10
], dtype=np.float32)


@cuda.jit
def integrate_signal(
    pixels_signals,
    signal_charge
):
    """Convolve raw induced currents with the buffer risetime response.

    Computes the integrated charge for every (pixel, tick) pair. When
    BUFFER_RISETIME > 0, applies an exponential convolution backwards
    over a window of 10 * BUFFER_RISETIME / TIME_SAMPLING ticks.
    Otherwise, multiplies the raw current by TIME_SAMPLING directly.

    Args:
        pixels_signals (:obj:`numpy.ndarray`): raw induced currents for
            each pixel. Shape (n_pixels, n_ticks).
        signal_charge (:obj:`numpy.ndarray`): Output; integrated charge
            per pixel per tick. Shape (n_pixels, n_ticks).
    """
    ic, ip = cuda.grid(2)
    WEIGHTS = cuda.const.array_like(_WEIGHTS_HOST)

    if ip >= pixels_signals.shape[0]:
        return
    if ic >= pixels_signals.shape[1]:
        return

    q = float32(0.0)

    # NOTE (Issue 4 — accepted difference): The original kernel uses
    # conv_start = max(last_reset, ...) to bound the convolution window at
    # the most recent ADC reset tick. Because last_reset is computed by the
    # ADC state machine (Phase 5) which runs AFTER this kernel, we cannot
    # replicate that bound here. Using max(0, ...) instead means the
    # pre-computed signal_charge may include small contributions from before
    # the last reset that the original would have excluded. The exponential
    # weighting heavily attenuates these distant contributions.
    if detector.BUFFER_RISETIME > 0:
        conv_start = max(0, ic - CONV_WINDOW)
        for k in range(CONV_WINDOW + 1):
            jc = ic - k
            if jc < conv_start:
                break
            q = cuda.fma(WEIGHTS[k], pixels_signals[ip, jc], q)

    else:
        q = pixels_signals[ip, ic] * TIME_SAMPLING

    signal_charge[ip, ic] = q

# --- Phase 3b/4: Tracks Accumulation ---
@cuda.jit
def integrate_signal_tracks(
    pixels_signals_tracks,
    num_backtrack,
    offset_backtrack,
    signal_charge_track
):
    """Convolve per-track induced currents with the buffer risetime response.

    Applies the same convolution as integrate_signal to each track's
    signal contribution independently. Reads from a flat/jagged layout
    indexed by (total_backtracks * tick + offset + track).

    Args:
        pixels_signals_tracks (:obj:`numpy.ndarray`): per-track induced
            currents in flat jagged layout. Shape (n_ticks * total_backtracks,).
        num_backtrack (:obj:`numpy.ndarray`): number of backtracked track
            segments per pixel. Shape (n_pixels,).
        offset_backtrack (:obj:`numpy.ndarray`): offset into
            pixels_signals_tracks for each pixel's data. Shape (n_pixels,).
        signal_charge_track (:obj:`numpy.ndarray`): Output; integrated
            charge per pixel per tick per track.
            Shape (n_pixels, n_ticks, MAX_TRACKS_PER_PIXEL).
    """
    ic, ip = cuda.grid(2)
    WEIGHTS = cuda.const.array_like(_WEIGHTS_HOST)

    if ip >= num_backtrack.shape[0]:
        return
    if ic >= signal_charge_track.shape[1]:
        return

    ntrks = min(num_backtrack[ip], signal_charge_track.shape[2])
    off   = offset_backtrack[ip]

    # NOTE: The below line is the same as summing num_backtrack, but is more efficient and is preferred.
    # An addition of two array accesses is O(1) while a sum over an array is O(n).
    total_backtracks = offset_backtrack[-1] + num_backtrack[-1]
    #total_backtracks = num_backtrack.sum() <- numba no likey

    conv_start = max(0, ic - CONV_WINDOW)
    for itrk in range(ntrks):
        q = float32(0.0)

        # NOTE (Issue 4 — accepted difference): Same conv_start bound
        # difference as integrate_signal. See comment there for details.
        if detector.BUFFER_RISETIME > 0:
            conv_start = max(
                0,
                int(ic - 10 * detector.BUFFER_RISETIME / detector.TIME_SAMPLING)
            )

            for k in range(CONV_WINDOW + 1):
                jc = ic - k
                if jc < conv_start:
                    break
                idx = total_backtracks * jc + off + itrk
                q = cuda.fma(WEIGHTS[k], pixels_signals_tracks[idx], q)

        else:
            idx = total_backtracks * ic + off + itrk
            q = pixels_signals_tracks[idx] * TIME_SAMPLING

        signal_charge_track[ip, ic, itrk] = q

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
    adc_counts,
    adc_q_sum,
    adc_last_reset
):
    """Run the ADC self-trigger state machine to find trigger windows.

    Walks the tick axis per pixel, accumulating pre-computed charge and
    checking for threshold crossings. On a crossing, an inner loop
    walks the ADC hold window accumulating charge, then a second
    threshold check accepts or rejects the trigger. Accepted windows
    are recorded; rejected windows are discarded and scanning resumes.

    Args:
        signal_charge (:obj:`numpy.ndarray`): integrated charge per pixel
            per tick (from integrate_signal). Shape (n_pixels, n_ticks).
        noise_uncorr (:obj:`numpy.ndarray`): pre-scaled uncorrelated noise
            per pixel per tick (from generate_noise).
            Shape (n_pixels, n_ticks).
        noise_disc (:obj:`numpy.ndarray`): pre-scaled discriminator noise
            per pixel per tick (from generate_noise).
            Shape (n_pixels, n_ticks).
        noise_reset (:obj:`numpy.ndarray`): pre-scaled reset noise per
            pixel per tick (from generate_noise).
            Shape (n_pixels, n_ticks).
        periodic_reset_phase (:obj:`numpy.ndarray`): integer phase offset
            for periodic resets per pixel (from generate_noise).
            Shape (n_pixels,).
        pixel_thresholds (:obj:`numpy.ndarray`): discriminator threshold
            per pixel. Shape (n_pixels,).
        adc_start (:obj:`numpy.ndarray`): Output; first tick of each ADC
            integration window. Shape (n_pixels, MAX_ADC_VALUES).
        adc_end (:obj:`numpy.ndarray`): Output; last tick of each ADC
            integration window. Shape (n_pixels, MAX_ADC_VALUES).
        adc_ticks_idx (:obj:`numpy.ndarray`): Output; post-integration tick
            index for timestamp computation.
            Shape (n_pixels, MAX_ADC_VALUES).
        adc_counts (:obj:`numpy.ndarray`): Output; number of accepted ADC
            triggers per pixel. Shape (n_pixels,).
        adc_q_sum (:obj:`numpy.ndarray`): Output; accumulated charge
            including uncorrelated noise at end of integration window.
            Shape (n_pixels, MAX_ADC_VALUES).
        adc_last_reset (:obj:`numpy.ndarray`): Output; tick index of the
            most recent ADC reset at the time each window was accepted.
            Shape (n_pixels, MAX_ADC_VALUES).
    """
    ip = cuda.grid(1)

    if ip >= signal_charge.shape[0]:
        return

    n_ticks = signal_charge.shape[1]

    ic = 0
    iadc = 0
    adc_busy = 0
    true_q = 0.0
    last_reset = 0
    q_sum = noise_reset[ip, 0]

    apply_periodic_reset = detector.PERIODIC_RESET_CYCLES > 0
    reset_phase = periodic_reset_phase[ip] if apply_periodic_reset else -1

    while ic < n_ticks or adc_busy > 0:

        if iadc >= sim.MAX_ADC_VALUES:
            break

        if ic < n_ticks:
            q = signal_charge[ip, ic]
        else:
            q = 0.0

        if apply_periodic_reset and ic % detector.PERIODIC_RESET_CYCLES == reset_phase:
            q_sum = noise_reset[ip, min(ic, n_ticks - 1)]
            true_q = 0.0
            ic += 1
            continue

        q_sum += q
        true_q += q

        if adc_busy > 0:
            adc_busy -= 1

        q_noise = noise_uncorr[ip, min(ic, n_ticks - 1)]
        disc_noise = noise_disc[ip, min(ic, n_ticks - 1)]

        if q_sum + q_noise >= pixel_thresholds[ip] + disc_noise and adc_busy == 0:

            interval = round(
                (3 * detector.CLOCK_CYCLE + detector.ADC_HOLD_DELAY * detector.CLOCK_CYCLE) / detector.TIME_SAMPLING
            )
            start = ic
            integrate_end = ic + interval
            current_last_reset = last_reset

            ic += 1

            # Inner integration loop — walk through the ADC hold window,
            # accumulating charge into q_sum (matching original behavior).
            while ic <= integrate_end:
                if ic < n_ticks:
                    q = signal_charge[ip, ic]
                else:
                    q = 0.0

                q_sum += q
                true_q += q
                ic += 1

            # Second threshold check after integration (Issue 1).
            end_tick = min(integrate_end, n_ticks - 1)
            adc = q_sum + noise_uncorr[ip, end_tick]
            disc_noise_2 = noise_disc[ip, end_tick]

            if adc < pixel_thresholds[ip] + disc_noise_2:
                # REJECT — charge after integration fell below threshold.
                ic += round(detector.RESET_CYCLES * detector.CLOCK_CYCLE / detector.TIME_SAMPLING)
                q_sum = noise_reset[ip, min(ic, n_ticks - 1)]
                true_q = 0.0
                last_reset = ic
                continue

            # ACCEPT — record the window.
            end = min(integrate_end, n_ticks - 1)

            adc_start[ip, iadc] = start
            adc_end[ip, iadc] = end
            adc_ticks_idx[ip, iadc] = ic
            adc_q_sum[ip, iadc] = adc
            adc_last_reset[ip, iadc] = current_last_reset

            ic += round(detector.RESET_CYCLES * detector.CLOCK_CYCLE / detector.TIME_SAMPLING)
            last_reset = ic
            adc_busy = round(detector.ADC_BUSY_DELAY * detector.CLOCK_CYCLE / detector.TIME_SAMPLING)

            q_sum = noise_reset[ip, min(ic, n_ticks - 1)]
            true_q = 0.0

            iadc += 1
            continue

        ic += 1

    adc_counts[ip] = iadc

# --- Phase 5b: Integrate Windows ---
@cuda.jit
def integrate_windows(
    signal_charge,
    signal_charge_track,
    num_backtrack,
    adc_end,
    adc_counts,
    adc_ticks_idx,
    adc_q_sum,
    adc_last_reset,
    adc_list,
    adc_ticks_list,
    current_fractions,
    time_ticks,
    time_padding,
    periodic_reset_phase
):
    """Accumulate per-track current fractions and write final ADC outputs.

    For each accepted ADC window, iterates from the last reset tick
    through the window end, accumulating per-track charge from
    pre-computed arrays and normalizing to fractional contributions.
    Also writes the ADC value and timestamp for each window.

    Args:
        signal_charge (:obj:`numpy.ndarray`): integrated charge per pixel
            per tick (from integrate_signal). Shape (n_pixels, n_ticks).
        signal_charge_track (:obj:`numpy.ndarray`): integrated charge per
            pixel per tick per track (from integrate_signal_tracks).
            Shape (n_pixels, n_ticks, MAX_TRACKS_PER_PIXEL).
        num_backtrack (:obj:`numpy.ndarray`): number of backtracked track
            segments per pixel. Shape (n_pixels,).
        adc_end (:obj:`numpy.ndarray`): last tick of each ADC integration
            window (from discover_adc_windows).
            Shape (n_pixels, MAX_ADC_VALUES).
        adc_counts (:obj:`numpy.ndarray`): number of accepted ADC triggers
            per pixel (from discover_adc_windows). Shape (n_pixels,).
        adc_ticks_idx (:obj:`numpy.ndarray`): post-integration tick index
            for timestamp computation (from discover_adc_windows).
            Shape (n_pixels, MAX_ADC_VALUES).
        adc_q_sum (:obj:`numpy.ndarray`): accumulated charge including
            uncorrelated noise (from discover_adc_windows).
            Shape (n_pixels, MAX_ADC_VALUES).
        adc_last_reset (:obj:`numpy.ndarray`): tick of the most recent ADC
            reset for each window (from discover_adc_windows).
            Shape (n_pixels, MAX_ADC_VALUES).
        adc_list (:obj:`numpy.ndarray`): Output; ADC charge value for each
            pixel per trigger. Shape (n_pixels, MAX_ADC_VALUES).
        adc_ticks_list (:obj:`numpy.ndarray`): Output; timestamp for each
            ADC trigger. Shape (n_pixels, MAX_ADC_VALUES).
        current_fractions (:obj:`numpy.ndarray`): Output; fraction of
            charge induced by each track per trigger.
            Shape (n_pixels, MAX_ADC_VALUES, MAX_TRACKS_PER_PIXEL).
        time_ticks (:obj:`numpy.ndarray`): array of time tick values.
            Shape (n_ticks,).
        time_padding (float): time offset added to each timestamp.
        periodic_reset_phase (:obj:`numpy.ndarray`): integer phase offset
            for periodic resets per pixel (from generate_noise).
            Shape (n_pixels,).
    """
    ip, iadc = cuda.grid(2)
    if ip >= signal_charge.shape[0]:
        return
    if iadc >= adc_counts[ip]:
        return

    end = adc_end[ip, iadc]
    lr = adc_last_reset[ip, iadc]
    n_ticks = signal_charge.shape[1]

    apply_periodic_reset = detector.PERIODIC_RESET_CYCLES > 0
    reset_phase = periodic_reset_phase[ip] if apply_periodic_reset else -1

    true_q = 0.0

    ntrks = min(
        num_backtrack[ip],
        signal_charge_track.shape[2]
    )

    # Accumulate track fractions from last_reset through end,
    # handling periodic resets which zero the accumulation.
    for ic in range(lr, end + 1):
        if ic >= n_ticks:
            break

        if apply_periodic_reset and ic % detector.PERIODIC_RESET_CYCLES == reset_phase:
            true_q = 0.0
            for itrk in range(ntrks):
                current_fractions[ip, iadc, itrk] = 0.0
            continue

        q = signal_charge[ip, ic]
        true_q += q

        for itrk in range(ntrks):
            current_fractions[ip, iadc, itrk] += signal_charge_track[ip, ic, itrk]

    if true_q > 0:
        for itrk in range(ntrks):
            current_fractions[ip, iadc, itrk] /= true_q

    # ADC value already computed by discover_adc_windows (q_sum + uncorr noise).
    adc_list[ip, iadc] = adc_q_sum[ip, iadc]

    # Timestamp with overflow handling (replicates original logic).
    tick_raw = adc_ticks_idx[ip, iadc]
    crossing_time_tick = min(tick_raw, len(time_ticks) - 1)
    post_adc_ticks = max(tick_raw - crossing_time_tick, 0)
    adc_ticks_list[ip, iadc] = (
        time_ticks[crossing_time_tick]
        + time_padding
        + post_adc_ticks * detector.TIME_SAMPLING
    )
# End reworked get_adc_values

@cuda.jit(max_registers=128)
def get_adc_values(pixels_signals,
                   pixels_signals_tracks,
                   num_backtrack,
                   offset_backtrack,
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
            each pixel. Shape (n_unique_pix, n_ticks).
        pixels_signals_tracks (:obj:`numpy.ndarray`): list of induced currents
            for each track that induces current on each pixel.
            Jagged; "shape" (n_unique_pix, n_ticks, num_backtrack[ipix]).
        num_backtrack (:obj:`numpy.ndarray`): For a given pixel, the number of
            backtracked track segments. Shape (n_unique_pix,).
        offset_backtrack (:obj:`numpy.ndarray`): For a given pixel, the offset
            into `pixels_signals_tracks` for that pixel's data.
            Shape (n_unique_pix,).
        time_ticks (:obj:`numpy.ndarray`): list of time ticks for each pixel.
            Shape (n_ticks+1,).
        adc_list (:obj:`numpy.ndarray`): Output; list of integrated charges for each
            pixel. Shape (n_unique_pix, max_adc_vals).
        adc_ticks_list (:obj:`numpy.ndarray`): Output; list of the time ticks that
            correspond to each integrated charge. Shape (n_unique_pix, max_adc_vals).
        time_padding (float): time interval to add to each time tick.
        rng_states (:obj:`numpy.ndarray`): array of random states for noise
            generation. Shape (TPB*BPG,).
        current_fractions (:obj:`numpy.ndarray`): Output; 2D array that will contain
            the fraction of current induced on the pixel by each track.
            Shape (n_unique_pix, max_adc_vals, MAX_TRACKS_PER_PIXEL).
            TODO: Jaggedize.
        pixel_thresholds(: obj: `numpy.ndarray`): list of discriminator
            thresholds for each pixel. Shape (n_unique_pix,)
    """
    ip = cuda.grid(1)

    # equivalent to num_backtrack.sum()
    total_backtracks = offset_backtrack[-1] + num_backtrack[-1]

    ntrks = min(num_backtrack[ip], current_fractions.shape[2])

    if ip < pixels_signals.shape[0]:
        curre = pixels_signals[ip]
        ic = 0
        iadc = 0
        adc_busy = 0
        last_reset = 0
        last_periodic_reset = 0
        true_q = 0
        apply_periodic_reset = detector.PERIODIC_RESET_CYCLES > 0
        periodic_reset_phase = int(xoroshiro128p_uniform_float32(rng_states, ip) * (detector.PERIODIC_RESET_CYCLES + 1)) if apply_periodic_reset else 0
        q_sum = xoroshiro128p_normal_float32(rng_states, ip) * detector.RESET_NOISE_CHARGE * e

                
        while ic < curre.shape[0] or adc_busy > 0:

            if iadc >= sim.MAX_ADC_VALUES:
                print("More ADC values than possible,", sim.MAX_ADC_VALUES)
                break

            q = 0

            if apply_periodic_reset and ic % detector.PERIODIC_RESET_CYCLES == periodic_reset_phase:
                q_sum = xoroshiro128p_normal_float32(rng_states, ip) * detector.RESET_NOISE_CHARGE * e
                true_q = 0
                last_periodic_reset = ic
                for itrk in range(current_fractions.shape[2]):
                    current_fractions[ip][iadc][itrk] = 0
                ic += 1
                continue
            
            if detector.BUFFER_RISETIME > 0:
                conv_start = max(last_reset, floor(ic - 10*detector.BUFFER_RISETIME/detector.TIME_SAMPLING))
                for jc in range(conv_start, min(ic+1, curre.shape[0])):
                    w = exp((jc - ic) * detector.TIME_SAMPLING / detector.BUFFER_RISETIME) * (1 - exp(-detector.TIME_SAMPLING/detector.BUFFER_RISETIME))
                    q += curre[jc] * detector.TIME_SAMPLING * w

                    for itrk in range(ntrks):
                        idx = total_backtracks * jc + offset_backtrack[ip] + itrk
                        current_fractions[ip][iadc][itrk] += pixels_signals_tracks[idx] * detector.TIME_SAMPLING * w

            elif ic < curre.shape[0]:
                q += curre[ic] * detector.TIME_SAMPLING
                for itrk in range(ntrks):
                    idx = total_backtracks * ic + offset_backtrack[ip] + itrk
                    current_fractions[ip][iadc][itrk] += pixels_signals_tracks[idx] * detector.TIME_SAMPLING

            q_sum += q
            true_q += q
            
            q_noise = xoroshiro128p_normal_float32(rng_states, ip) * detector.UNCORRELATED_NOISE_CHARGE * e
            disc_noise = xoroshiro128p_normal_float32(rng_states, ip) * detector.DISCRIMINATOR_NOISE * e

            if adc_busy > 0:
                adc_busy -= 1

            if q_sum + q_noise >= pixel_thresholds[ip] + disc_noise and adc_busy == 0:
                interval = round((3 * detector.CLOCK_CYCLE + detector.ADC_HOLD_DELAY * detector.CLOCK_CYCLE) / detector.TIME_SAMPLING)
                integrate_end = ic+interval

                ic+=1

                while ic <= integrate_end:
                    q = 0

                    if detector.BUFFER_RISETIME > 0:
                        conv_start = max(last_reset, floor(ic - 10*detector.BUFFER_RISETIME/detector.TIME_SAMPLING))
                        for jc in range(conv_start, min(ic+1, curre.shape[0])):
                            w = exp((jc - ic) * detector.TIME_SAMPLING / detector.BUFFER_RISETIME) * (1 - exp(-detector.TIME_SAMPLING/detector.BUFFER_RISETIME))
                            q += curre[jc] * detector.TIME_SAMPLING * w

                            for itrk in range(ntrks):
                                idx = total_backtracks * jc + offset_backtrack[ip] + itrk
                                current_fractions[ip][iadc][itrk] += pixels_signals_tracks[idx] * detector.TIME_SAMPLING * w

                    elif ic < curre.shape[0]:
                        q += curre[ic] * detector.TIME_SAMPLING
                        for itrk in range(ntrks):
                            idx = total_backtracks * ic + offset_backtrack[ip] + itrk
                            current_fractions[ip][iadc][itrk] += pixels_signals_tracks[idx] * detector.TIME_SAMPLING


                    q_sum += q
                    true_q += q
                    ic+=1

                adc = q_sum + xoroshiro128p_normal_float32(rng_states, ip) * detector.UNCORRELATED_NOISE_CHARGE * e
                disc_noise = xoroshiro128p_normal_float32(rng_states, ip) * detector.DISCRIMINATOR_NOISE * e

                if adc < pixel_thresholds[ip] + disc_noise:
                    ic += round(detector.RESET_CYCLES * detector.CLOCK_CYCLE / detector.TIME_SAMPLING)
                    q_sum = xoroshiro128p_normal_float32(rng_states, ip) * detector.RESET_NOISE_CHARGE * e
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
                adc_ticks_list[ip][iadc] = time_ticks[crossing_time_tick]+time_padding+(post_adc_ticks*detector.TIME_SAMPLING)

                ic += round(detector.RESET_CYCLES * detector.CLOCK_CYCLE / detector.TIME_SAMPLING)
                last_reset = ic
                adc_busy = round(detector.ADC_BUSY_DELAY * detector.CLOCK_CYCLE / detector.TIME_SAMPLING)

                q_sum = xoroshiro128p_normal_float32(rng_states, ip) * detector.RESET_NOISE_CHARGE * e
                true_q = 0

                iadc += 1
                continue

            ic += 1
