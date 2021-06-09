"""
Module that si mulates the front-end electronics (triggering, ADC)
"""

import numpy as np
import h5py

from numba import cuda
from numba.cuda.random import xoroshiro128p_normal_float32

from larpix.packet import Packet_v2, TimestampPacket, TriggerPacket
from larpix.packet import PacketCollection
from larpix.format import hdf5format
from tqdm import tqdm
from . import consts, detsim

#: Maximum number of ADC values stored per pixel
MAX_ADC_VALUES = 10
#: Discrimination threshold
DISCRIMINATION_THRESHOLD = 7e3*consts.e_charge
#: ADC hold delay in clock cycles
ADC_HOLD_DELAY = 15
#: Clock cycle time in :math:`\mu s`
CLOCK_CYCLE = 0.1
#: Front-end gain in :math:`mV/ke-`
GAIN = 4/1e3
#: Common-mode voltage in :math:`mV`
V_CM = 288
#: Reference voltage in :math:`mV`
V_REF = 1300
#: Pedestal voltage in :math:`mV`
V_PEDESTAL = 580
#: Number of ADC counts
ADC_COUNTS = 2**8
#: Reset noise in e-
RESET_NOISE_CHARGE = 900
#: Uncorrelated noise in e-
UNCORRELATED_NOISE_CHARGE = 500

nonrouted_channels=[6,7,8,9,22,23,24,25,38,39,40,54,55,56,57]
routed_channels=[i for i in range(64) if i not in nonrouted_channels]
top_row_channels=[3,2,1,63,62,61,60]
bottom_row_channels=[28,29,30,31,33,34,35]
inner_edge_channels=[60,52,53,48,45,41,35]
top_row_chip_ids=[11,12,13,14,15,16,17,18,19,20]
bottom_row_chip_ids=[101,102,103,104,105,106,107,108,109,110]
inner_edge_chip_ids=[20,30,40,50,60,70,80,90,100,110]

def rotate_tile(pixel_id, tile_id):
    axes = consts.tile_orientations[tile_id-1]
    x_axis = axes[2]
    y_axis = axes[1]

    pix_x = pixel_id[0]
    if x_axis < 0:
        pix_x = consts.n_pixels_per_tile[0]-pixel_id[0]-1
        
    pix_y = pixel_id[1]
    if y_axis < 0:
        pix_y = consts.n_pixels_per_tile[1]-pixel_id[1]-1
    
    return pix_x, pix_y
        
def export_to_hdf5(adc_list, adc_ticks_list, unique_pix, track_ids, filename):
    """
    Saves the ADC counts in the LArPix HDF5 format.

    Args:
        adc_list (:obj:`numpy.ndarray`): list of ADC values for each pixel
        adc_ticks_list (:obj:`numpy.ndarray`): list of time ticks for each pixel
        unique_pix (:obj:`numpy.ndarray`): list of pixel IDs
        filename (str): filename of HDF5 output file

    Returns:
        list: list of LArPix packets
    """

    dtype = np.dtype([('track_ids','(5,)i8')])
    packets = [TimestampPacket()]
    packets_mc = [[-1]*5]
    packets_mc_ds = []
    last_event = -1
    
    for itick, adcs in enumerate(tqdm(adc_list, desc="Writing to HDF5...")):
        ts = adc_ticks_list[itick]
        pixel_id = unique_pix[itick]

        plane_id = int(pixel_id[0] // consts.n_pixels[0])
        tile_x = int((pixel_id[0] - consts.n_pixels[0] * plane_id) // consts.n_pixels_per_tile[1])
        tile_y = int(pixel_id[1] // consts.n_pixels_per_tile[1])
        tile_id = consts.tile_map[plane_id][tile_x][tile_y]

        for iadc, adc in enumerate(adcs):
            t = ts[iadc]

            if adc > digitize(0):
                event = t // (consts.time_interval[1]*3)
                time_tick = int(np.floor(t/CLOCK_CYCLE))

                if event != last_event:
                    packets.append(TriggerPacket(io_group=1,trigger_type=b'\x02',timestamp=int(event*consts.time_interval[1]/consts.t_sampling*3)))
                    packets_mc.append([-1]*5)
                    packets.append(TriggerPacket(io_group=2,trigger_type=b'\x02',timestamp=int(event*consts.time_interval[1]/consts.t_sampling*3)))
                    packets_mc.append([-1]*5)
                    last_event = event
                
                p = Packet_v2()

                try:
                    chip, channel = consts.pixel_connection_dict[rotate_tile(pixel_id%70, tile_id)]
                except KeyError:
                    print("Pixel ID not valid", pixel_id)
                    continue
                
                # disabled channels near the borders of the tiles
                if chip in top_row_chip_ids and channel in top_row_channels: continue
                if chip in bottom_row_chip_ids and channel in bottom_row_channels: continue
                if chip in inner_edge_chip_ids and channel in inner_edge_channels: continue 
                    
                p.dataword = int(adc)
                p.timestamp = time_tick

                try:
                    io_group_io_channel = consts.tile_chip_to_io[tile_id][chip]
                except KeyError:
#                     print("Chip %i on tile %i not found" % (chip, tile_id))
                    continue
                    
                io_group, io_channel = io_group_io_channel // 1000, io_group_io_channel % 1000
                p.chip_key = "%i-%i-%i" % (io_group, io_channel, chip)
                p.channel_id = channel
                p.packet_type = 0
                p.first_packet = 1
                p.assign_parity()

                packets_mc.append(track_ids[itick][iadc])
                packets.append(p)
            else:
                break
        
    packet_list = PacketCollection(packets, read_id=0, message='')
    
    hdf5format.to_file(filename, packet_list)

    if packets:
        packets_mc_ds = np.empty(len(packets), dtype=dtype)
        packets_mc_ds['track_ids'] = packets_mc

    with h5py.File(filename, 'a') as f:
        if "mc_packets_assn" in f.keys():
            del f['mc_packets_assn']
        f.create_dataset("mc_packets_assn", data=packets_mc_ds)

    return packets, packets_mc_ds

def digitize(integral_list):
    """
    The function takes as input the integrated charge and returns the digitized
    ADC counts.

    Args:
        integral_list (:obj:`numpy.ndarray`): list of charge collected by each pixel

    Returns:
        numpy.ndarray: list of ADC values for each pixel
    """
    import cupy as cp
    xp = cp.get_array_module(integral_list)
    adcs = xp.minimum(xp.floor(xp.maximum((integral_list*GAIN/consts.e_charge+V_PEDESTAL - V_CM), 0) \
                      * ADC_COUNTS/(V_REF-V_CM)), ADC_COUNTS)

    return adcs

@cuda.jit
def get_adc_values(pixels_signals, time_ticks, adc_list, adc_ticks_list, time_padding, rng_states):
    """
    Implementation of self-trigger logic

    Args:
        pixels_signals (:obj:`numpy.ndarray`): list of induced currents for
            each pixel
        time_ticks (:obj:`numpy.ndarray`): list of time ticks for each pixel
        adc_list (:obj:`numpy.ndarray`): list of integrated charges for each
            pixel
        adc_ticks_list (:obj:`numpy.ndarray`): list of the time ticks that
            correspond to each integrated charge.
    """
    ip = cuda.grid(1)

    if ip < pixels_signals.shape[0]:
        curre = pixels_signals[ip]
        ic = 0
        iadc = 0
        q_sum = xoroshiro128p_normal_float32(rng_states, ip) * RESET_NOISE_CHARGE * consts.e_charge

        while ic < curre.shape[0]:

            q = curre[ic]*consts.t_sampling

            q_sum += q
            q_noise = xoroshiro128p_normal_float32(rng_states, ip) * UNCORRELATED_NOISE_CHARGE * consts.e_charge

            if q_sum + q_noise >= DISCRIMINATION_THRESHOLD:
                interval = round((3 * CLOCK_CYCLE + ADC_HOLD_DELAY * CLOCK_CYCLE) / consts.t_sampling)
                integrate_end = ic+interval

                while ic <= integrate_end and ic < curre.shape[0]:
                    q = curre[ic] * consts.t_sampling
                    q_sum += q
                    ic += 1

                adc = q_sum + xoroshiro128p_normal_float32(rng_states, ip) * UNCORRELATED_NOISE_CHARGE * consts.e_charge

                if adc < DISCRIMINATION_THRESHOLD:
                    ic += round(CLOCK_CYCLE / consts.t_sampling)
                    continue

                if iadc >= MAX_ADC_VALUES:
                    print("More ADC values than possible, ", MAX_ADC_VALUES)
                    break

                adc_list[ip][iadc] = adc
                adc_ticks_list[ip][iadc] = time_ticks[ic]+time_padding

                ic += round(CLOCK_CYCLE / consts.t_sampling)
                q_sum = xoroshiro128p_normal_float32(rng_states, ip) * RESET_NOISE_CHARGE * consts.e_charge
                iadc += 1
            ic += 1
