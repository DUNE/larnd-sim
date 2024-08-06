#!/usr/bin/env python3

NITER = 10
DEFAULT_INPUT_FILE = 'tracks_current_mc.pkl'

from functools import partial
from math import ceil
import pickle

from pynvjitlink import patch
patch.patch_numba_linker()

import cupy as cp
from numba.cuda.random import create_xoroshiro128p_states

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
    print('done')

    print('Copying input... ', end='')
    neighboring_pixels = cp.array(d['neighboring_pixels'])
    selected_tracks = cp.array(d['selected_tracks'])
    max_length = d['max_length']
    print('done')

    print('Initializing output... ', end='')
    signals = cp.zeros((selected_tracks.shape[0],
                        neighboring_pixels.shape[1],
                        cp.asnumpy(max_length)[0]), dtype=np.float32)
    print('done')

    print('Running kernel... ', end='')
    TPB = (1,1,64)
    BPG_X = max(ceil(signals.shape[0] / TPB[0]),1)
    BPG_Y = max(ceil(signals.shape[1] / TPB[1]),1)
    BPG_Z = max(ceil(signals.shape[2] / TPB[2]),1)
    BPG = (BPG_X, BPG_Y, BPG_Z)
    rng_states = create_xoroshiro128p_states(int(np.prod(TPB[:2]) * np.prod(BPG[:2])))
    for i in range(NITER):
        print(f'{i} ', end='')
        detsim.tracks_current_mc[BPG,TPB](signals, neighboring_pixels, selected_tracks, response, rng_states)
    print('done')

if __name__ == '__main__':
    main()
