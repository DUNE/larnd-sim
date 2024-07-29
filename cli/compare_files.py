#!/usr/bin/env python

import argparse
import h5py
import numpy as np
# import matplotlib.pyplot as plt
import time
import sys

np.set_printoptions(precision=3)

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

def get_light_wvfms(sim_file):
    larray_geom = np.array([1,1,1,1,1,1,0,0,0,0,0,0]*8*4)
    lcm_events = [sim_file['light_wvfm'][i][larray_geom==1] for i in range(NUM_LIGHT_EVENTS)]
    acl_events = [sim_file['light_wvfm'][i][larray_geom!=1] for i in range(NUM_LIGHT_EVENTS)]

    waveforms = {}
    waveforms['lcm'] = np.array(lcm_events).reshape((NUM_LIGHT_EVENTS * NUM_OPT_CH, SAMPLES))
    waveforms['acl'] = np.array(acl_events).reshape((NUM_LIGHT_EVENTS * NUM_OPT_CH, SAMPLES))
    return waveforms

def describe(dataset):
    min = np.min(dataset)
    max = np.max(dataset)
    mean = np.mean(dataset)
    median = np.median(dataset)
    stdev = np.std(dataset)
    return np.array([min, max, median, mean, stdev])
    # return (min, max, median, mean, stdev)

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

## Define some constants for the simulation (mostly light sim related)
TS_CYCLE = 0.2e7
BIT = 4
PRE_NOISE = 65
SAMPLES = 1000
NUM_OPT_CH = 192 # Number of ACL or LCM optical channels
NUM_LIGHT_EVENTS = 150 # Save processing time
THRESHOLD = 50 # change this if you want to exclude events from noise analysis
SAMPLE_RATE = 6.25e7

## Get the various datasets for comparison
ref_segments = ref_file['segments']
sim_segments = sim_file['segments']
ref_packets = get_packets(ref_file)
sim_packets = get_packets(sim_file)
ref_light_wvfms = get_light_wvfms(ref_file)
sim_light_wvfms = get_light_wvfms(sim_file)

failed_tests = 0

if args.break_file:
    print("Corrupting packets.")
    sim_packets['data']['io_group'][111] = 11
    sim_packets['sync']['timestamp'][74] = 9e9
    sim_packets['trig']['timestamp'][74] = 9e9
    sim_packets['time']['timestamp'][48] = 9e9

### Charge/packets data tests

if args.verbose:
    print("Packet length comparison (ref vs. sim):")
    print(f"Data: {ref_packets['data'].shape} vs. {sim_packets['data'].shape}")
    print(f"Trig: {ref_packets['trig'].shape} vs. {sim_packets['trig'].shape}")
    print(f"Time: {ref_packets['time'].shape} vs. {sim_packets['time'].shape}")
    print(f"Sync: {ref_packets['sync'].shape} vs. {sim_packets['sync'].shape}")

### Tests checking the 'timestamp' distributions of data/trig/time/sync packets
print("\nChecking 'timestamp' in data/trig/time/sync packets...")

## Data packet timestamps can vary slightly between runs
## Groups of packets come in bunches spaced out every 2e6 ticks
for c in range(0, 5):
    offset = c * TS_CYCLE
    ref_hist, _ = np.histogram(ref_packets['data']['timestamp']-offset, range=[0, 2000], bins=50)
    sim_hist, _ = np.histogram(sim_packets['data']['timestamp']-offset, range=[0, 2000], bins=50)
    test = np.isclose(ref_hist, sim_hist, args.rel_tol, args.abs_tol)
    if not np.all(test):
        print("Data packets timestamp distribution not matching")
        print(f"Difference in bins for range ({c*TS_CYCLE}, {(c+1)*TS_CYCLE})")
        abs_diff = (sim_hist - ref_hist)
        rel_diff = (sim_hist - ref_hist) / ref_hist
        print(abs_diff)
        print(rel_diff)

        fail_bins_idx = np.where(test == False)[0]
        fail_bins_val = sim_hist[fail_bins_idx]
        ref_bins_val = ref_hist[fail_bins_idx]
        print("Failed indices:", fail_bins_idx)
        print("Failed values :", fail_bins_val)
        print("Reference val :", ref_bins_val)
        failed_tests += 1

ref_data_hist, _ = np.histogram(ref_packets['data']['timestamp']%TS_CYCLE, range=[0, 2000], bins=100)
sim_data_hist, _ = np.histogram(sim_packets['data']['timestamp']%TS_CYCLE, range=[0, 2000], bins=100)
test_data_hist = np.all(np.isclose(ref_data_hist, sim_data_hist, args.rel_tol, args.abs_tol))

## Trigger and sync timestamps should be identical
if not ref_packets['trig']['timestamp'].shape == sim_packets['trig']['timestamp'].shape:
    print("Trigger packets shape do not match.")
    failed_test +=1
else:
    test_trig_time = ref_packets['trig']['timestamp'] == sim_packets['trig']['timestamp']
    if not np.all(test_trig_time):
        print("Trigger packets timestamp distribution not matching")
        print(f"Mismatched packets at {np.where(test_trig_time == False)}")
        failed_tests += 1

if not ref_packets['sync']['timestamp'].shape == sim_packets['sync']['timestamp'].shape:
    print("Sync packets shape do not match.")
    failed_test +=1
else:
    test_sync_time = ref_packets['sync']['timestamp'] == sim_packets['sync']['timestamp']
    if not np.all(test_sync_time):
        print("Sync packets timestamp distribution not matching")
        print(f"Mismatched packets at {np.where(test_sync_time == False)}")
        failed_tests += 1

ref_time_hist, _ = np.histogram(ref_packets['time']['timestamp'], bins=100)
sim_time_hist, _ = np.histogram(sim_packets['time']['timestamp'], bins=100)
test_time_hist = np.isclose(ref_time_hist, sim_time_hist, args.rel_tol, args.abs_tol)
if not np.all(test_time_hist):
    print("Time packets timestamp distribution not matching")
    print("Difference in bins for time packets:")
    print(ref_time_hist - sim_time_hist)
    failed_tests += 1

### Test if all io_groups are present in data packets and similarly distributed
print("\nChecking 'io_groups' in data packets...")

ref_iog_id, ref_iog_count = np.unique(ref_packets['data']['io_group'], return_counts=True)
sim_iog_id, sim_iog_count = np.unique(sim_packets['data']['io_group'], return_counts=True)
if args.verbose:
    print("Counts of each io_group (1-8):")
    print("Ref:", ref_iog_count)
    print("Sim:", sim_iog_count)

if len(ref_iog_id) != len(sim_iog_id):
    print("Missing/wrong io_groups in data packets!")
    print("io_groups:", sim_iog_id)
    failed_tests += 1
else:
    test_iog_count = np.isclose(ref_iog_count, sim_iog_count, args.rel_tol, args.abs_tol)
    if not np.all(test_iog_count):
        print("Distribution of io_groups not matching")
        print(ref_iog_count - sim_iog_count)
        failed_tests += 1

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
    failed_tests += 1

ref_dword_hist, _ = np.histogram(ref_packets['data']['dataword'], bins=100)
sim_dword_hist, _ = np.histogram(ref_packets['data']['dataword'], bins=100)
test_dword_hist = np.isclose(ref_dword_hist, sim_dword_hist, args.rel_tol, args.abs_tol)
if not np.all(test_dword_hist):
    print("Distribution (histogram) of dataword in data packets not matching")
    failed_tests += 1

### Tests checking the electron and photon distribution in the energy segments
print("\nChecking drift time, electron, and photon distributions in energy segments...")

## All three of these should be identical
## Using stricter tolerances for np.isclose()
## Not sure what is the best way to include different tolerances for different tests
## without hard coding them
test_dtime = np.isclose(ref_segments['t'], sim_segments['t'], 0.01, 10)
if not np.all(test_dtime):
    print("Drift time distribution does not match")
    failed_tests += 1

test_nelec = np.isclose(ref_segments['n_electrons'], sim_segments['n_electrons'], 0.01, 10)
if not np.all(test_nelec):
    print("Electron distribution does not match")
    failed_tests += 1

test_nphtn = np.isclose(ref_segments['n_photons'], sim_segments['n_photons'], 0.01, 10)
if not np.all(test_nphtn):
    print("Photon distribution does not match")
    failed_tests += 1

#This variable is an input to the simulation and is read-only, so it better be identical
test_dE = np.isclose(ref_segments['dE'], sim_segments['dE'], 0.001, 0)
if not np.all(test_dE):
    print("Deposited energy (dE) distribution does not match. This is really bad.")
    failed_tests += 1

### Light waveform comparisons

def noise_datasets(no_ped_adc):
    # Onky keep waveforms with a sample above the threshold
    max_abs_values = np.max(np.abs(no_ped_adc), axis=1)
    adc_signal_indices = np.flatnonzero(max_abs_values > THRESHOLD)
    adc_normal_pretrig = no_ped_adc[adc_signal_indices[0:3000], 0:PRE_NOISE]
    # Normalize waveforms to their max value
    norms = np.max(np.abs(adc_normal_pretrig), axis=1)
    ns_wvfms = np.divide(adc_normal_pretrig, norms[:, np.newaxis])

    # Calculate power spectra using FFT
    freqs = np.fft.fftfreq(PRE_NOISE, 1/SAMPLE_RATE)
    freqs = freqs[:PRE_NOISE//2] # Keep only positive frequencies
    freq_matrix = np.tile(freqs, (len(adc_normal_pretrig),1))
    frequencies = freq_matrix.flatten()
    spectrum_arr = np.fft.fft(ns_wvfms, axis=1)
    psds = np.abs(spectrum_arr[:,:PRE_NOISE//2])**2 / (PRE_NOISE * SAMPLE_RATE)
    psds[:,1:] *= 2 # Double the power except for the DC component
    power = psds.flatten()
    p_dbfs = 20 * np.log10(power[power != 0]) #Apparently 'power' can have zeroes, mask them out for now
    return adc_signal_indices, frequencies[power != 0], adc_normal_pretrig, p_dbfs

def power_hist_max(adc_dataset):
    adc_freq  = adc_dataset[1]
    adc_pdbfs = adc_dataset[3]
    hist, *edges = np.histogram2d(adc_freq[(adc_pdbfs)>-500]/1e6, adc_pdbfs[(adc_pdbfs)>-500], bins=32)
    ycenters = (edges[1][:-1] + edges[1][1:]) / 2
    xcenters = (edges[0][:-1] + edges[0][1:]) / 2
    maxes = []
    for array in hist:
        maxes.append(np.where(array == max(array))[0][0])
    max_bins = np.array([ycenters[i] for i in maxes])
    return xcenters, max_bins

ref_acl_dset = noise_datasets(-ref_light_wvfms['acl'])
ref_lcm_dset = noise_datasets(-ref_light_wvfms['lcm'])
sim_acl_dset = noise_datasets(-sim_light_wvfms['acl'])
sim_lcm_dset = noise_datasets(-sim_light_wvfms['lcm'])

ref_acl_max = power_hist_max(ref_acl_dset)
ref_lcm_max = power_hist_max(ref_lcm_dset)
sim_acl_max = power_hist_max(sim_acl_dset)
sim_lcm_max = power_hist_max(sim_lcm_dset)

### Tests checking the pre-trigger noise spectrum

## Check if the max power is similar distributed for each light readout
print("\nChecking pre-trigger noise max power...")
test_acl_max = np.isclose(ref_acl_max[1], sim_acl_max[1], args.rel_tol, args.abs_tol)
if not np.all(test_acl_max):
    print("ArCLight noise max power spectrum does not match")
    print(ref_acl_max[1] - sim_acl_max[1])

test_lcm_max = np.isclose(ref_lcm_max[1], sim_lcm_max[1], args.rel_tol, args.abs_tol)
if not np.all(test_lcm_max):
    print("LCM noise max power spectrum does not match")
    print(ref_lcm_max[1] - sim_lcm_max[1])

t_end = time.time()
t_elapse = t_end - t_start
print("-------------------")
print(f"Elapsed time: {t_elapse:.3f} s")
print(f"Failed tests: {failed_tests}")
print("Finished.")

if failed_tests > 0:
    sys.exit(121)
else:
    sys.exit(0)
