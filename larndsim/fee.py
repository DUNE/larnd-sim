"""
Module that simulates the front-end electronics (triggering, ADC)
"""

import numpy as np
from numba import cuda
from numba.cuda.random import xoroshiro128p_normal_float32

from larpix.packet import Packet_v2, TimestampPacket
from larpix.packet import PacketCollection
from larpix.format import hdf5format
from tqdm import tqdm
from . import consts, detsim

#: Maximum number of ADC values stored per pixel
MAX_ADC_VALUES = 10
#: Discrimination threshold
DISCRIMINATION_THRESHOLD = 5e3*consts.e_charge
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

def export_to_hdf5(adc_list, adc_ticks_list, unique_pix, filename):
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

    packets = {}
    for ic in range(len(consts.tpc_centers)):
        packets[ic] = []

    for itick, adcs in enumerate(tqdm(adc_list,desc="Writing to HDF5...")):
        ts = adc_ticks_list[itick]
        pixel_id = unique_pix[itick]
        plane_id = pixel_id[0] // consts.n_pixels[0]
        pix_x, pix_y = detsim.get_pixel_coordinates_cpu(pixel_id)
        pix_x -= consts.tpc_centers[int(plane_id)][0]
        pix_y -= consts.tpc_centers[int(plane_id)][1]
        pix_x *= 10
        pix_y *= 10

        for iadc, adc in enumerate(adcs):
            t = ts[iadc] 
            
            if adc > digitize(0):
                p = Packet_v2()

                try:
                    channel, chip = consts.pixel_connection_dict[(round(pix_x/consts.pixel_size[0]),round(pix_y/consts.pixel_size[1]))]
                except KeyError:
                    print("Pixel coordinates not valid", pix_x, pix_y, pixel_id, adc)
                    continue

                p.dataword = int(adc)
                p.timestamp = int(np.floor(t/CLOCK_CYCLE))
                p.chip_key = chip
                p.channel_id = channel
                p.packet_type = 0
                p.first_packet = 1
                p.assign_parity()
                
                if not packets[ic]:
                    packets[ic].append(TimestampPacket())
                    
                packets[ic].append(p)
            else:
                break

    for ipc in packets:
        packet_list = PacketCollection(packets[ipc], read_id=0, message='')
        
        if "." in filename:
            pre_extension, post_extension = filename.rsplit('.', 1)
            filename_ext = "%s-%i.%s" % (pre_extension, ipc, post_extension)
        else:
            filename_ext = "%s-%i" % (filename, ipc)
            
        hdf5format.to_file(filename_ext, packet_list)

    return packets

def digitize(integral_list):
    """
    The function takes as input the integrated charge and returns the digitized
    ADC counts.

    Args:
        integral_list (:obj:`numpy.ndarray`): list of charge collected by each pixel

    Returns:
        numpy.ndarray: list of ADC values for each pixel
    """

    adcs = np.minimum(np.floor(np.maximum((integral_list*GAIN/consts.e_charge+V_PEDESTAL - V_CM), 0) \
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
        q_sum = 0
        adc = 0
        iadc = 0
        
        while ic < curre.shape[0]:

            q = curre[ic]*consts.t_sampling
            
            if ic == 0:
                q += 0#xoroshiro128p_normal_float32(rng_states, ip) * RESET_NOISE_CHARGE * consts.e_charge 
                
            q_sum += q
#             adc += q_sum
#             print(q_sum)
            if q_sum >= DISCRIMINATION_THRESHOLD:
                
                interval = round((3 * CLOCK_CYCLE + ADC_HOLD_DELAY * CLOCK_CYCLE) / consts.t_sampling)
                integrate_end = ic+interval

                while ic <= integrate_end and ic <= curre.shape[0]:
                    q = curre[ic] * consts.t_sampling
                    q_sum += q
                    ic += 1

                adc = q_sum
                
                if iadc >= MAX_ADC_VALUES:
                    print("More ADC values than possible, ", MAX_ADC_VALUES)
                    break

                adc_list[ip][iadc] = adc
                adc_ticks_list[ip][iadc] = time_ticks[ic]+time_padding

                ic += round(CLOCK_CYCLE / consts.t_sampling)
                q_sum = 0#xoroshiro128p_normal_float32(rng_states, ip) * RESET_NOISE_CHARGE * consts.e_charge 
                iadc += 1

            adc = 0
            ic += 1
