#!/usr/bin/env python3

import argparse

import h5py
import numpy as np
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('fname', help='larnd-sim output hdf5 file')
    args = ap.parse_args()

    f = h5py.File(args.fname, 'r+')
    packets = f['packets'][...]
    assn = f['mc_packets_assn'][...]

    event_ids = assn['event_ids'][:,0]
    uniq_event_ids = np.unique(event_ids)

    new_packets = np.zeros_like(packets)
    new_assn = np.zeros_like(assn)
    loc = 0

    for event in tqdm(uniq_event_ids):
        mask = event_ids == event
        size = np.sum(mask)
        new_packets[loc : loc+size] = packets[mask]
        new_assn[loc : loc+size] = assn[mask]
        loc += size

    f['packets'][...] = new_packets
    f['mc_packets_assn'][...] = new_assn


if __name__ == '__main__':
    main()
