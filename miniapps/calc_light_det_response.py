#!/usr/bin/env python3

import argparse
from functools import partial
from math import ceil
import os
import pickle

from pynvjitlink import patch
patch.patch_numba_linker()

import cupy as cp
import numpy as np

from larndsim import consts, light_sim
from larndsim.config import get_config


# flush on every print
print = partial(print, flush=True)

NITER = int(os.getenv('LARNDSIM_MINIAPP_NUM_RUNS', '10'))
DEFAULT_INPUT_FILE = '/global/cfs/cdirs/dune/www/data/2x2/simulation/mkramer_dev/hackathon2024/miniapp-inputs/calc_light_det_response.pkl'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-file', default=DEFAULT_INPUT_FILE)
    ap.add_argument('--output-file')
    ap.add_argument('--config', default='2x2')
    args = ap.parse_args()

    cfg = get_config(args.config)
    consts.load_properties(cfg['DET_PROPERTIES'],
                           cfg['PIXEL_LAYOUT'],
                           cfg['SIM_PROPERTIES'])

    print('Loading input... ', end='')
    with open(args.input_file, 'rb') as f:
        d = pickle.load(f)
    print('done')

    print('Copying input... ', end='')
    light_sample_inc_disc = cp.array(d['light_sample_inc_disc'])
    light_sample_inc_scint_true_track_id = cp.array(d['light_sample_inc_scint_true_track_id'])
    light_sample_inc_scint_true_photons = cp.array(d['light_sample_inc_scint_true_photons'])
    print('done')

    TPB = (1,64)
    BPG = (max(ceil(light_sample_inc_disc.shape[0] / TPB[0]),1),
           max(ceil(light_sample_inc_disc.shape[1] / TPB[1]),1))

    for i in range(NITER):
        print(f'===== Iteration {i}')

        print('Initializing output... ', end='')
        light_response = cp.zeros_like(light_sample_inc_disc)
        light_response_true_track_id = cp.full_like(light_sample_inc_scint_true_track_id, -1)
        light_response_true_photons = cp.zeros_like(light_sample_inc_scint_true_photons)
        print('done')

        print('Running kernel... ', end='')
        light_sim.calc_light_detector_response[BPG, TPB](
            light_sample_inc_disc, light_sample_inc_scint_true_track_id, light_sample_inc_scint_true_photons,
            light_response, light_response_true_track_id, light_response_true_photons)
        print('done')

        if args.output_file and i == 0:
            print('Writing output... ', end='')
            out = {'light_response': np.array(light_response.get()),
                   'light_response_true_track_id': np.array(light_response_true_track_id.get()),
                   'light_response_true_photons': np.array(light_response_true_photons.get())}
            with open(args.output_file, 'wb') as f:
                pickle.dump(out, f)
            print('done')


if __name__ == '__main__':
    main()
