#!/usr/bin/env python
"""
Performance test for voxelization on real edep-sim file.
Mimics simulate_pixels.py structure but focuses on voxelization only.
"""
import os
import sys
import time
from math import ceil
import argparse

import numpy as np
import h5py
import cupy as cp

from larndsim import consts, quenching, drifting
from larndsim.active_volume import select_active_volume
from larndsim.far_field.voxelization import gpu_voxelize
from larndsim.config import get_config
import warnings
from numba.core.errors import NumbaPerformanceWarning

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


def swap_coordinates(tracks):
    """Swap x and z coordinates (edep-sim -> larnd-sim convention)."""
    x_start = np.copy(tracks['x_start'])
    x_end = np.copy(tracks['x_end'])
    x = np.copy(tracks['x'])
    
    tracks['x_start'] = np.copy(tracks['z_start'])
    tracks['x_end'] = np.copy(tracks['z_end'])
    tracks['x'] = np.copy(tracks['z'])
    
    tracks['z_start'] = x_start
    tracks['z_end'] = x_end
    tracks['z'] = x
    
    return tracks


def test_voxelization_performance(input_filename, config='2x2', n_events=None, output_file=None):
    """
    Test voxelization performance on edep-sim input.
    Flow mimics cli/simulate_pixels.py structure but stops at voxelization step.
    
    Args:
        input_filename: path to edep-sim HDF5 file
        config: detector configuration ('2x2', 'fsd', etc.)
        n_events: limit number of events (None = all)
        output_file: optional path to save voxel data
    """
    
    print("="*70)
    print("VOXELIZATION PERFORMANCE TEST")
    print("="*70)
    
    # Load configuration
    print(f"\n[1/7] Loading configuration: {config}")
    cfg = get_config(config)
    consts.load_properties(cfg['DET_PROPERTIES'], cfg['PIXEL_LAYOUT'], 
                          cfg['RESPONSE'], cfg['SIM_PROPERTIES'])
    from larndsim.consts import detector, sim, physics
    
    # Load input segments
    print(f"\n[2/7] Loading segments from {input_filename}")
    if not os.path.exists(input_filename):
        raise FileNotFoundError(f"Input file not found: {input_filename}")
    
    with h5py.File(input_filename, 'r') as f:
        segments = np.array(f['segments'])
    
    print(f"  Loaded {len(segments)} segments")
    
    # Add missing fields if needed
    if 't0' not in segments.dtype.names:
        print("  Adding missing timing fields (t0, t0_start, t0_end)")
        t0 = np.array(segments['t'].copy(), dtype=[('t0','f4')])
        t0_start = np.array(segments['t_start'].copy(), dtype=[('t0_start','f4')])
        t0_end = np.array(segments['t_end'].copy(), dtype=[('t0_end','f4')])
        segments = np.lib.recfunctions.merge_arrays((segments, t0, t0_start, t0_end), flatten=True)
        segments['t'] = np.zeros(segments.shape[0], dtype='f4')
        segments['t_start'] = np.zeros(segments.shape[0], dtype='f4')
        segments['t_end'] = np.zeros(segments.shape[0], dtype='f4')
    
    if 'segment_id' not in segments.dtype.names:
        print("  Adding segment_id field")
        dtype = [('segment_id','u4')] + segments.dtype.descr
        new = np.empty(segments.shape, dtype=np.dtype(dtype, align=True))
        new['segment_id'] = np.arange(segments.shape[0], dtype='u4')
        for name, fmt in segments.dtype.descr:
            new[name] = segments[name]
        segments = new
    # larnd-sim uses "t0" in a way that 0 is the "trigger" time (e.g spill time)
    # Therefore, to run the detector simulation we reset the t0 to reflect that
    # When storing the mc truth, revert this change and store the "real" segment time
    # The event times are added to segments in the spill building stage. This step is not needed for non-beam simulation
    if sim.IS_SPILL_SIM:
        # "Reset" the spill period so t0 is wrt the corresponding spill start time.
        # The spill starts are marking the start of
        # The space between spills will be accounted for in the
        # packet timestamps through the event_times array below
        localSpillIDs = segments[sim.EVENT_SEPARATOR] - (segments[sim.EVENT_SEPARATOR] // sim.MAX_EVENTS_PER_FILE) * sim.MAX_EVENTS_PER_FILE
        segments['t0_start'] = segments['t0_start'] - localSpillIDs*sim.SPILL_PERIOD
        segments['t0_end'] = segments['t0_end'] - localSpillIDs*sim.SPILL_PERIOD
        segments['t0'] = segments['t0'] - localSpillIDs*sim.SPILL_PERIOD
    
    # Preprocessing
    print(f"\n[3/7] Preprocessing")
    
    # Filter neutrals
    mask_neutral = (segments['pdg_id'] != 2112) & (segments['pdg_id'] != 22)
    segments = segments[mask_neutral]
    print(f"  After neutral filtering: {len(segments)} segments")
    
    # Filter delayed
    if 't0' in segments.dtype.names:
        segments = segments[segments['t0'] < sim.MAX_SEGMENT_T0]
        print(f"  After delayed filter: {len(segments)} segments")
    
    # Limit events
    all_events = np.unique(segments[sim.EVENT_SEPARATOR])
    if n_events is not None:
        all_events = all_events[:n_events]
        segments = segments[np.isin(segments[sim.EVENT_SEPARATOR], all_events)]
        print(f"  Processing {n_events} events: {len(segments)} segments")
    else:
        print(f"  Processing {len(all_events)} events")
    
    # Swap coordinates
    print("  Swapping coordinates (edep-sim -> larnd-sim)")
    segments = swap_coordinates(segments)
    
    # Per-event voxelization
    print(f"\n[4/7] Running voxelization per event/TPC")
    
    voxel_stats = {
        'total_events': 0,
        'total_tpcs': 0,
        'total_segments_processed': 0,
        'total_voxels': 0,
        'total_charge_input': 0.0,
        'total_charge_output': 0.0,
        'event_times': [],
        'quench_times': [],
        'drift_times': [],
        'voxel_times': [],
        'voxel_memory_kb': [],
    }
    
    all_voxel_data = []
    
    print(f"  Total events to process: {len(all_events)}")
    for event_id in all_events:
        t0_event = time.time()
        event_mask = segments[sim.EVENT_SEPARATOR] == event_id
        event_segs = segments[event_mask]
        # print(f"  Event {event_id}: {len(event_segs)} segments")
        
        if len(event_segs) == 0:
            print(f"  Event {event_id} has no segments, skipping")
            continue
        
        voxel_stats['total_events'] += 1
        
        # Process per TPC
        for i_tpc in range(len(detector.TPC_BORDERS)):
            tpc_borders = detector.TPC_BORDERS[i_tpc:i_tpc+1]
            idx_tpc = select_active_volume(event_segs, tpc_borders)
            tpc_segs = event_segs[idx_tpc]
            # print(f"    TPC {i_tpc}: {len(tpc_segs)} segments")
            
            if len(tpc_segs) == 0:
                # print(f"    TPC {i_tpc} has no segments, skipping")
                continue
            
            voxel_stats['total_tpcs'] += 1
            voxel_stats['total_segments_processed'] += len(tpc_segs)
            
            # Quenching
            t0_quench = time.time()
            TPB = 256
            BPG = max(ceil(len(tpc_segs) / TPB), 1)
            quenching.quench[BPG, TPB](tpc_segs, physics.BIRKS)
            cp.cuda.Stream.null.synchronize()
            voxel_stats['quench_times'].append(time.time() - t0_quench)
            
            # Drifting
            t0_drift = time.time()
            drifting.drift[BPG, TPB](tpc_segs)
            cp.cuda.Stream.null.synchronize()
            voxel_stats['drift_times'].append(time.time() - t0_drift)
            
            charge_before = float(np.sum(tpc_segs['n_electrons']))
            voxel_stats['total_charge_input'] += charge_before
            
            # Voxelization
            t0_voxel = time.time()
            vox_idx, vox_charge, grid_shape, voxel_size, bounds = gpu_voxelize(
                tpc_segs, tpc_borders=tpc_borders
            )
            cp.cuda.Stream.null.synchronize()
            voxel_stats['voxel_times'].append(time.time() - t0_voxel)
            
            n_voxels = len(vox_idx)
            charge_after = float(cp.sum(vox_charge))
            voxel_stats['total_voxels'] += n_voxels
            voxel_stats['total_charge_output'] += charge_after
            
            # Track memory usage (sparse arrays only: indices + charges)
            mem_kb = (vox_idx.nbytes + vox_charge.nbytes) / 1024  # KB
            voxel_stats['voxel_memory_kb'].append(mem_kb)
            
            # Optional: save voxel data
            if output_file:
                all_voxel_data.append({
                    'event_id': event_id,
                    'tpc_id': i_tpc,
                    'voxel_indices': cp.asnumpy(vox_idx),
                    'voxel_charges': cp.asnumpy(vox_charge),
                    'grid_shape': grid_shape,
                    'voxel_size': voxel_size,
                    'bounds': bounds,
                })
        
        voxel_stats['event_times'].append(time.time() - t0_event)
    
    # Print statistics
    print(f"\n[5/7] Voxelization Complete")
    print(f"  Events processed: {voxel_stats['total_events']}")
    print(f"  TPCs processed: {voxel_stats['total_tpcs']}")
    print(f"  Total segments: {voxel_stats['total_segments_processed']}")
    print(f"  Total voxels created: {voxel_stats['total_voxels']}")
    
    print(f"\n[6/7] Performance Metrics")
    if voxel_stats['event_times']:
        avg_event = np.mean(voxel_stats['event_times'])
        print(f"  Average event time: {avg_event*1000:.2f} ms")
    if voxel_stats['quench_times']:
        total_quench = np.sum(voxel_stats['quench_times'])
        print(f"  Total quenching time: {total_quench:.3f} s")
    if voxel_stats['drift_times']:
        total_drift = np.sum(voxel_stats['drift_times'])
        print(f"  Total drifting time: {total_drift:.3f} s")
    if voxel_stats['voxel_times']:
        total_voxel = np.sum(voxel_stats['voxel_times'])
        avg_voxel = np.mean(voxel_stats['voxel_times'])
        print(f"  Total voxelization time: {total_voxel:.3f} s")
        print(f"  Average voxelization per TPC: {avg_voxel*1000:.2f} ms")
        if voxel_stats['total_segments_processed'] > 0:
            segs_per_sec = voxel_stats['total_segments_processed'] / total_voxel
            print(f"  Segments per second: {segs_per_sec:.1f}")
    
    if voxel_stats['voxel_memory_kb']:
        mem_arr = np.array(voxel_stats['voxel_memory_kb'])
        print(f"\n  GPU Memory (sparse voxel arrays per TPC):")
        print(f"    Min:     {mem_arr.min():.2f} KB")
        print(f"    Max:     {mem_arr.max():.2f} KB ({mem_arr.max()/1024:.3f} MB)")
        print(f"    Average: {mem_arr.mean():.2f} KB ({mem_arr.mean()/1024:.3f} MB)")
    
    print(f"\n[7/7] Charge Conservation Check")
    charge_in = voxel_stats['total_charge_input']
    charge_out = voxel_stats['total_charge_output']
    if charge_in > 0:
        rel_diff = abs(charge_out - charge_in) / charge_in
        print(f"  Input charge:  {charge_in:.3e} e-")
        print(f"  Output charge: {charge_out:.3e} e-")
        print(f"  Relative diff: {rel_diff:.2e}")

        if rel_diff < 1e-3:
            print("  O Charge conserved within tolerance (< 0.1%)")
        else:
            print(f"  X Warning: charge difference {rel_diff*100:.3f}% exceeds 0.1% tolerance!")
    
    # Save output if requested
    if output_file and all_voxel_data:
        print(f"\nSaving voxel data to {output_file}")
        with h5py.File(output_file, 'w') as f:
            for i, data in enumerate(all_voxel_data):
                grp = f.create_group(f"event_{data['event_id']}_tpc_{data['tpc_id']}")
                grp.create_dataset('voxel_indices', data=data['voxel_indices'], compression='gzip')
                grp.create_dataset('voxel_charges', data=data['voxel_charges'], compression='gzip')
                grp.attrs['grid_shape'] = data['grid_shape']
                grp.attrs['voxel_size'] = data['voxel_size']
                grp.attrs['x_bounds'] = data['bounds'][0]
                grp.attrs['y_bounds'] = data['bounds'][1]
                grp.attrs['z_bounds'] = data['bounds'][2]
        print(f"  Saved {len(all_voxel_data)} TPC voxelizations")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    
    return voxel_stats


def main():
    parser = argparse.ArgumentParser(description='Voxelization performance test')
    parser.add_argument('input_file', help='Path to edep-sim HDF5 file')
    parser.add_argument('--config', default='2x2', help='Detector configuration (default: 2x2)')
    parser.add_argument('--n-events', type=int, default=None, help='Number of events to process (default: all)')
    parser.add_argument('--output', default=None, help='Output HDF5 file for voxel data (optional)')
    
    args = parser.parse_args()
    
    test_voxelization_performance(
        args.input_file,
        config=args.config,
        n_events=args.n_events,
        output_file=args.output
    )


if __name__ == '__main__':
    main()
