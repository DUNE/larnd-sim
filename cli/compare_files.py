#!/usr/bin/env python

import argparse
import h5py
import numpy as np
# import matplotlib.pyplot as plt
import time

parser = argparse.ArgumentParser()
parser.add_argument('--ref_file', default=None, type=str, help='path of the larnd-sim reference simulation file to be considered')
parser.add_argument('--sim_file', default=None, type=str, help='path of the larnd-sim output simulation file to be considered')
parser.add_argument('--rel_tol', default=0.1, type=float, help='relative tolerance for np.isclose()')
parser.add_argument('--abs_tol', default=10, type=float, help='absolute tolerance for np.isclose()')
parser.add_argument('--break_file', default=False, action='store_true', help='intentionally introduce errors into the file to check if tests are working')
parser.add_argument('--verbose', default=False, action='store_true', help='print summary statistics in addition to warnings')
args = parser.parse_args()

def get_packets(sim_file):
    packets = sim_file['packets']
    # packet_index = np.array(list(range(0,len(packets))))
    data_packet_mask = packets['packet_type'] == 0
    trig_packet_mask = packets['packet_type'] == 7
    timestamp_packet_mask = packets['packet_type'] == 4
    sync_packet_mask = (packets['packet_type'] == 6) & (packets['trigger_type'] == 83)
    other_packet_mask= ~(data_packet_mask | trig_packet_mask | sync_packet_mask | timestamp_packet_mask)

    packets_dict = {}
    packets_dict['data'] = packets[data_packet_mask]
    packets_dict['trig'] = packets[trig_packet_mask]
    packets_dict['time'] = packets[timestamp_packet_mask]
    packets_dict['sync'] = packets[sync_packet_mask]
    packets_dict['other'] = packets[other_packet_mask]

    return packets_dict

def describe(dataset):
    min = np.min(dataset)
    max = np.max(dataset)
    mean = np.mean(dataset)
    median = np.median(dataset)
    stdev = np.std(dataset)
    return (min, max, median, mean, stdev)

print("-----------------------------------------")
print("Comparing larnd-sim simulation outputs...")
print(f"Reference file : {args.ref_file}")
print(f"Simulation file: {args.sim_file}")

print("\nTolerances for np.isclose() method:")
print(f"Relative tolerance: {args.rel_tol}")
print(f"Absolute tolerance: {args.abs_tol}")

ref_file = h5py.File(args.ref_file)
sim_file = h5py.File(args.sim_file)
print("---------------")
print("Opened files...")

t_start = time.time()
ref_segments = ref_file['segments']
sim_segments = sim_file['segments']
ref_packets = get_packets(ref_file)
sim_packets = get_packets(sim_file)

if args.break_file:
    print("Corrupting packets.")
    sim_packets['data']['io_group'][111] = 11
    sim_packets['sync']['timestamp'][74] = 9e9
    sim_packets['trig']['timestamp'][74] = 9e9
    sim_packets['time']['timestamp'][48] = 9e9

if args.verbose:
    print("Packet length comparison (ref vs. sim):")
    print(f"Data: {ref_packets['data'].shape} vs. {sim_packets['data'].shape}")
    print(f"Trig: {ref_packets['trig'].shape} vs. {sim_packets['trig'].shape}")
    print(f"Time: {ref_packets['time'].shape} vs. {sim_packets['time'].shape}")
    print(f"Sync: {ref_packets['sync'].shape} vs. {sim_packets['sync'].shape}")

### Tests checking the 'timestamp' distributions of data/trig/time/sync packets
print("\nChecking 'timestamp' in data/trig/time/sync packets...")

## Data packet timestamps can vary slightly between runs
TS_CYCLE = 0.2e7
for c in range(0, 5):
    offset = c * TS_CYCLE
    ref_hist, _ = np.histogram(ref_packets['data']['timestamp']-offset, range=[0, 2000], bins=50)
    sim_hist, _ = np.histogram(sim_packets['data']['timestamp']-offset, range=[0, 2000], bins=50)
    test = np.all(np.isclose(ref_hist, sim_hist, args.rel_tol, args.abs_tol))
    if not test:
        print("Data packets timestamp distribution not matching")
        print(f"Difference in bins for range ({c*TS_CYCLE}, {(c+1)*TS_CYCLE})")
        print(ref_hist - sim_hist)

ref_data_hist, _ = np.histogram(ref_packets['data']['timestamp']%TS_CYCLE, range=[0, 2000], bins=100)
sim_data_hist, _ = np.histogram(sim_packets['data']['timestamp']%TS_CYCLE, range=[0, 2000], bins=100)
test_data_hist = np.all(np.isclose(ref_data_hist, sim_data_hist, args.rel_tol, args.abs_tol))

# np.all(ref_packets['sync']['timestamp'] == 1e7)
# np.all(sim_packets['sync']['timestamp'] == 1e7)

## Trigger and sync timestamps should be identical
test_trig_time = ref_packets['trig']['timestamp'] == sim_packets['trig']['timestamp']
if not np.all(test_trig_time):
    print("Trigger packets timestamp distribution not matching")
    print(f"Mismatched packets at {np.where(test_trig_time == False)}")

test_sync_time = ref_packets['sync']['timestamp'] == sim_packets['sync']['timestamp']
if not np.all(test_sync_time):
    print("Sync packets timestamp distribution not matching")
    print(f"Mismatched packets at {np.where(test_sync_time == False)}")

ref_time_hist, _ = np.histogram(ref_packets['time']['timestamp'], bins=100)
sim_time_hist, _ = np.histogram(sim_packets['time']['timestamp'], bins=100)
test_time_hist = np.isclose(ref_time_hist, sim_time_hist, args.rel_tol, args.abs_tol)
if not np.all(test_time_hist):
    print("Time packets timestamp distribution not matching")
    print("Difference in bins for time packets:")
    print(ref_time_hist - sim_time_hist)

### Test if all io_groups are present in data packets and similarly distributed
print("\nChecking 'io_groups' in data packets...")

ref_iog_id, ref_iog_count = np.unique(ref_packets['data']['io_group'], return_counts=True)
sim_iog_id, sim_iog_count = np.unique(sim_packets['data']['io_group'], return_counts=True)

if len(ref_iog_id) != len(sim_iog_id):
    print("Missing/wrong io_groups in data packets!")
    print("io_groups:", sim_iog_id)
else:
    test_iog_count = np.isclose(ref_iog_count, sim_iog_count, args.rel_tol, args.abs_tol)
    if not np.all(test_iog_count):
        print("Distribution of io_groups not matching")
        print(ref_iog_count - sim_iog_count)

### Tests checking the 'dataword' distribution of data packets
print("\nChecking 'dataword' in data packets...")

ref_dword_stat = describe(ref_packets['data']['dataword'])
sim_dword_stat = describe(sim_packets['data']['dataword'])
if args.verbose:
    print("Statistics  : (min, max, median, mean, std)")
    print(f"Ref dataword: {ref_dword_stat}")
    print(f"Sim dataword: {sim_dword_stat}")

test_dword_stat = np.isclose(ref_dword_stat, sim_dword_stat, args.rel_tol, args.abs_tol)
if not np.all(test_dword_stat):
    print("Dataword summary statistics not matching")

ref_dword_hist, _ = np.histogram(ref_packets['data']['dataword'], bins=100)
sim_dword_hist, _ = np.histogram(ref_packets['data']['dataword'], bins=100)
test_dword_hist = np.isclose(ref_dword_hist, sim_dword_hist, args.rel_tol, args.abs_tol)
if not np.all(test_dword_hist):
    print("Distribution (histogram) of dataword in data packets not matching")

### Tests checking the electron and photon distribution in the energy segments
print("\nChecking drift time, electron, and photon distributions in energy segments...")

## All three of these should be identical
test_dtime = np.isclose(ref_segments['t'], sim_segments['t'], 0.01, 10)
if not np.all(test_dtime):
    print("Drift time distribution does not match")

test_nelec = np.isclose(ref_segments['n_electrons'], sim_segments['n_electrons'], 0.01, 10)
if not np.all(test_nelec):
    print("Electron distribution does not match")

test_nphtn = np.isclose(ref_segments['n_photons'], sim_segments['n_photons'], 0.01, 10)
if not np.all(test_nphtn):
    print("Photon distribution does not match")

t_end = time.time()
t_elapse = t_end - t_start
print("-------------------")
print(f"Elapsed time: {t_elapse:.3f} s")
print("Finished.")
