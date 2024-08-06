#!/usr/bin/env python3

NITER = 10
DEFAULT_INPUT_FILE = 'calc_light_det_response.pkl'

import argparse
from functools import partial
from math import ceil
import pickle

from pynvjitlink import patch
patch.patch_numba_linker()

import cupy as cp

from larndsim import light_sim


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
    light_sample_inc_disc = cp.array(d['light_sample_inc_disc'])
    light_sample_inc_scint_true_track_id = cp.array(d['light_sample_inc_scint_true_track_id'])
    light_sample_inc_scint_true_photons = cp.array(d['light_sample_inc_scint_true_photons'])
    print('done')

    print('Initializing output... ', end='')
    light_response = cp.zeros_like(light_sample_inc_disc)
    light_response_true_track_id = cp.full_like(light_sample_inc_scint_true_track_id, -1)
    light_response_true_photons = cp.zeros_like(light_sample_inc_scint_true_photons)
    print('done')

    print('Running kernel... ', end='')
    TPB = (1,64)
    BPG = (max(ceil(light_sample_inc_disc.shape[0] / TPB[0]),1),
           max(ceil(light_sample_inc_disc.shape[1] / TPB[1]),1))
    for i in range(NITER):
        print(f'{i} ', end='')
        light_sim.calc_light_detector_response[BPG, TPB](
            light_sample_inc_disc, light_sample_inc_scint_true_track_id, light_sample_inc_scint_true_photons,
            light_response, light_response_true_track_id, light_response_true_photons)
    print('done')

if __name__ == '__main__':
    main()
