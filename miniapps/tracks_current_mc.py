#!/usr/bin/env python3

NITER = 10
DEFAULT_INPUT_FILE = '/global/cfs/cdirs/dune/www/data/2x2/simulation/mkramer_dev/hackathon2024/miniapp-inputs/tracks_current_mc.pkl'

import argparse
from functools import partial
from math import ceil
import pickle

from pynvjitlink import patch
patch.patch_numba_linker()

import cupy as cp
from numba.cuda import device_array
from numba.cuda.random import create_xoroshiro128p_states
import numpy as np

from larndsim import detsim

# flush on every print
print = partial(print, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-file', default=DEFAULT_INPUT_FILE)
    args = ap.parse_args()

    print('Loading input... ', end='')
    with open(args.input_file, 'rb') as f:
        d = pickle.load(f)
    response = cp.load('larnd-sim/larndsim/bin/response_44.npy')
    print('done')

    print('Copying input... ', end='')
    neighboring_pixels = cp.array(d['neighboring_pixels'])
    selected_tracks = d['selected_tracks']
    max_length = d['max_length']
    print('done')

    print('Initializing output... ', end='')
    signals = cp.zeros((selected_tracks.shape[0],
                        neighboring_pixels.shape[1],
                        max_length), dtype=np.float32)
    print('done')

    print('Running kernel... ', end='')
    TPB = (1,1,64)
    BPG_X = max(ceil(signals.shape[0] / TPB[0]),1)
    BPG_Y = max(ceil(signals.shape[1] / TPB[1]),1)
    BPG_Z = max(ceil(signals.shape[2] / TPB[2]),1)
    BPG = (BPG_X, BPG_Y, BPG_Z)
    N = int(np.prod(TPB[:2]) * np.prod(BPG[:2]))
    rng_states = create_xoroshiro128p_states(N, seed=321)
    # rng_states = device_array(N, dtype=rng_states_.dtype)
    # rng_states[:] = rng_states_
    for i in range(NITER):
        print(f'{i} ', end='')
        detsim.tracks_current_mc[BPG,TPB](signals, neighboring_pixels, selected_tracks, response, rng_states)
    print('done')

if __name__ == '__main__':
    main()
