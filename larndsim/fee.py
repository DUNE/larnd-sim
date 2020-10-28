"""
Module that simulates the front-end electronics (triggering, ADC)
"""

import numpy as np
from numba import cuda

from . import consts

from larpix.packet import Packet_v2
from larpix.packet import PacketCollection
from larpix.format import hdf5format

MAX_ADC_VALUES = 5

#: Discrimination threshold :math:`\mu s`
DISCRIMINATION_THRESHOLD = 5e3*consts.e_charge#0.1e-15
#: ADC hold delay in clock cycles
ADC_HOLD_DELAY = 15
#: Clock cycle time in :math:`\mu s`
CLOCK_CYCLE = 0.1
#: Front-end gain in :math:`mV/ke-`
GAIN = 4/1e3
#: Common-mode voltage in :math:`mV`
V_CM = 77 
#: Reference voltage in :math:`mV`
V_REF = 1539 
#: Pedestal voltage in :math:`mV`
V_PEDESTAL = 550
#: Number of ADC counts
ADC_COUNTS = 2**8

def export_to_hdf5(adc_list, adc_ticks_list, unique_pix, filename):
    """
    Saves the ADC counts in the LArPix HDF5 format.
    """
    packet_dset_name = 'packets'
    pc = []
    first_packet = True

    for itick, adcs in enumerate(adc_list):
        ts = adc_ticks_list[itick]
        pixel_id = unique_pix[itick]
        pix_x, pix_y = consts.get_pixel_coordinates(pixel_id)
        pix_x *= 10
        pix_y *= 10
        for t, adc in zip(ts, adcs):
            if adc > digitize(0):
                p = Packet_v2()
                connection_list = consts.board.channels_where(lambda pixel: np.isclose(pixel.x,pix_x) and np.isclose(pixel.y,pix_y))
                
                try:
                    chip, channel = connection_list[0]
                except IndexError:
                    print("Pixel coordinates not valid", pix_x, pix_y, pixel_id, adc)
                    
                p.dataword = int(adc)
                p.timestamp = int(np.floor(t/CLOCK_CYCLE))
                p.chip_id = chip.chipid
                p.channel_id = channel
                p.packet_type = 0

                if first_packet:
                    p.first_packet = 1
                    first_packet = False
                p.assign_parity()
                pc.append(p)

    packet_list = PacketCollection(pc, read_id=0, message='')
    hdf5format.to_file(filename, packet_list)
    
    return packet_list
    
def digitize(adc_list):
    """
    The function takes as input the integrated charge and returns the digitized
    ADC counts.
    """
    adcs = np.minimum(np.floor(np.maximum((adc_list*GAIN/consts.e_charge+V_PEDESTAL - V_CM),0)*ADC_COUNTS/(V_REF-V_CM)), ADC_COUNTS)
    
    return adcs

@cuda.jit
def get_adc_values(pixels_signals, time_ticks, adc_list, adc_ticks_list):
    ip = cuda.grid(1)

    if ip < pixels_signals.shape[0]:
        curre = pixels_signals[ip]

        ic = 0
        q_sum = 0
        adc = 0
        iadc = 0

        while ic < curre.shape[0]:
            q = curre[ic]*consts.t_sampling
            q_sum += q
            adc += q_sum

            if q_sum >= DISCRIMINATION_THRESHOLD:     

                interval = round((3 * CLOCK_CYCLE + ADC_HOLD_DELAY * CLOCK_CYCLE) / consts.t_sampling)        
                integrate_end = ic+interval

                while ic <= integrate_end and ic <= curre.shape[0]:                
                    q = curre[ic]*consts.t_sampling
                    q_sum += q
                    ic += 1

                adc = q_sum
                if iadc >= MAX_ADC_VALUES:
                    print("More ADC values than possible, ", MAX_ADC_VALUES)
                    break

                adc_list[ip][iadc]= adc
                adc_ticks_list[ip][iadc] = time_ticks[ic]

                ic += round(CLOCK_CYCLE / consts.t_sampling)
                q_sum = 0
                iadc += 1

            adc = 0
            ic += 1

