#!/usr/bin/env python

import sys, time
import argparse
import h5py
import numpy as np

np.set_printoptions(precision=3)

parser = argparse.ArgumentParser()
parser.add_argument('--ref_file', default=None, type=str, help='path of the larnd-sim reference simulation file to be considered')
parser.add_argument('--sim_file', default=None, type=str, help='path of the larnd-sim output simulation file to be considered')
parser.add_argument('--strict', default=False, action='store_true', help='enable strict comparisons')
parser.add_argument('--verbose', default=False, action='store_true', help='print summary statistics in addition to warnings')
args = parser.parse_args()

def check_dataset(ref_file, sim_file, dset_name):
    print(f"Checking {dset_name} ...")
    ref_dataset = ref_file[dset_name]
    sim_dataset = sim_file[dset_name]
    struct_names = ref_dataset.dtype.names

    if struct_names is not None:
        for name in ref_dataset.dtype.names:
            if args.strict:
                indices = np.nonzero(ref_dataset[name, :] != sim_dataset[name, :])
            else:
                indices = np.nonzero(~np.isclose(ref_dataset[name, :], sim_dataset[name, :]))

            all_empty = np.all(np.array([arr.size == 0 for arr in indices]))
            if not all_empty:
                print(f"Mismatching {dset_name}/{name}!")
                print("Index locations of mismatch.")
                print(indices)
            else:
                print(f"{dset_name}/{name} match.")
    else:
        if args.strict:
            indices = np.nonzero(ref_dataset[:] != sim_dataset[:])
        else:
            indices = np.nonzero(~np.isclose(ref_dataset[:], sim_dataset[:]))

        all_empty = np.all(np.array([arr.size == 0 for arr in indices]))
        if not all_empty:
            print(f"Mismatching {dset_name}!")
            print("Index locations of mismatch.")
            print(indices)
        else:
            print(f"{dset_name} match.")
    print("---------------------------")

print("-----------------------------------------")
print("Comparing larnd-sim simulation outputs...")
print(f"Reference file : {args.ref_file}")
print(f"Simulation file: {args.sim_file}")

ref_file = h5py.File(args.ref_file)
sim_file = h5py.File(args.sim_file)
print("-------------------")
print("Opened files...")

t_start = time.time()
failed_tests = 0

# for dset in ref_file.keys():
    # check_dataset(ref_file, sim_file, dset)
print("Testing datasets...")
print("-------------------")
check_dataset(ref_file, sim_file, 'segments')
check_dataset(ref_file, sim_file, 'packets')
check_dataset(ref_file, sim_file, 'mc_packets_assn')
check_dataset(ref_file, sim_file, 'light_wvfm')
check_dataset(ref_file, sim_file, 'light_trig')
check_dataset(ref_file, sim_file, 'light_wvfm_mc_assn')

t_end = time.time()
t_elapse = t_end - t_start
print(f"Elapsed time: {t_elapse:.3f} s")
print(f"Failed tests: {failed_tests}")
print("Finished.")

if failed_tests > 0:
    sys.exit(121)
else:
    sys.exit(0)
