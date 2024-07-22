#!/usr/bin/env python

import argparse
import h5py
import numpy as np
# import matplotlib.pyplot as plt
import time

parser = argparse.ArgumentParser()
parser.add_argument('--ref_file', default=None, type=str, help='path of the larnd-sim reference simulation file to be considered')
parser.add_argument('--sim_file', default=None, type=str, help='path of the larnd-sim output simulation file to be considered')
parser.add_argument('--tolerance', default=None, type=float, help='tolerance for equality')
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

print(f"Reference file : {args.ref_file}")
print(f"Simulation file: {args.sim_file}")

ref_file = h5py.File(args.ref_file)
sim_file = h5py.File(args.sim_file)
print("Opened files...")

t_start = time.time()
ref_packets = get_packets(ref_file)
sim_packets = get_packets(sim_file)

t_end = time.time()
t_elapse = t_end - t_start
print(f"Elapsed time: {t_elapse:.3f} s")
print("Finished.")
