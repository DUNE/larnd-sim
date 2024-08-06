#!/usr/bin/env python3

NITER = 10
DEFAULT_INPUT_FILE = '/global/cfs/cdirs/dune/www/data/2x2/simulation/mkramer_dev/hackathon2024/miniapp-inputs/get_adc_values.pkl'

import argparse
from functools import partial
from math import ceil
import pickle

from pynvjitlink import patch
patch.patch_numba_linker()

import cupy as cp
from numba.cuda.random import create_xoroshiro128p_states

from larndsim import fee
from larndsim.util import CudaDict

# flush on every print
print = partial(print, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-file', default=DEFAULT_INPUT_FILE)
    args = ap.parse_args()

    print('Loading input... ', end='')
    with open(args.input_file, 'rb') as f:
        d = pickle.load(f)
    print('done')

    print('Copying input... ', end='')
    pixels_signals = cp.array(d['pixels_signals'])
    pixels_tracks_signals = cp.array(d['pixels_tracks_signals'])
    unique_pix = cp.array(d['unique_pix'])
    print('done')

    print('Initializing output... ', end='')
    time_ticks = cp.linspace(0, d['nevents'] * d['drift_window_usec'], pixels_signals.shape[1]+1)
    integral_list = cp.zeros((pixels_signals.shape[0], d['max_adc_values']))
    adc_ticks_list = cp.zeros((pixels_signals.shape[0], d['max_adc_values']))
    current_fractions = cp.zeros((pixels_signals.shape[0], d['max_adc_values'],
                                  d['max_tracks_per_pixel']))
    print('done')

    print('Running kernel... ', end='')
    TPB = 128
    BPG = ceil(pixels_signals.shape[0] / TPB)
    rng_states = create_xoroshiro128p_states(int(TPB * BPG), seed=321)
    pixel_thresholds_lut = CudaDict(cp.array([fee.DISCRIMINATION_THRESHOLD]), 1, 1)
    pixel_thresholds = pixel_thresholds_lut[unique_pix.ravel()].reshape(unique_pix.shape)

    for i in range(NITER):
        print(f'{i} ', end='')
        fee.get_adc_values[BPG, TPB](pixels_signals,
                                     pixels_tracks_signals,
                                     time_ticks,
                                     integral_list,
                                     adc_ticks_list,
                                     0,
                                     rng_states,
                                     current_fractions,
                                     pixel_thresholds)
    print('done')

if __name__ == '__main__':
    main()
