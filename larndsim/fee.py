"""
Module that simulates the front-end electronics (triggering, ADC)
"""

import numpy as np
from numba import cuda

from . import consts

MAX_ADC_VALUES = 2
DISCRIMINATION_THRESHOLD = 1e-15
ADC_HOLD_DELAY = 12
CLOCK_CYCLE = 0.1

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

                interval = round((3 * CLOCK_CYCLE + ADC_HOLD_DELAY) / consts.t_sampling)        
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

