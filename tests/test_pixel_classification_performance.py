#!/usr/bin/env python
"""
Performance test for pixel classification on real edep-sim file.
Mimics test_voxelization_performance.py but applies pixel classification per event/TPC.
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
from larndsim.mesh_refinement.pixel_classifier import classify_pixels
from larndsim.mesh_refinement import PixelCategory
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


def test_pixel_classification_performance(input_filename, config='2x2', n_events=None, output_file=None):
    """
    Test pixel classification performance on edep-sim input.
    Flow mimics test_voxelization_performance.py but applies classify_pixels per TPC.
    
    Args:
        input_filename: path to edep-sim HDF5 file
        config: detector configuration ('2x2', 'fsd', etc.)
        n_events: limit number of events (None = all)
        output_file: optional path to save classification results
    """
    
    print("="*70)
    print("PIXEL CLASSIFICATION PERFORMANCE TEST")
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
    
    # Per-event pixel classification
    print(f"\n[4/7] Running pixel classification per event/TPC")
    
    pixel_stats = {
        'total_events': 0,
        'total_tpcs': 0,
        'total_segments_processed': 0,
        'total_pixels_classified': 0,
        'total_charge_pixels': 0,
        'total_neighbor_pixels': 0,
        'total_induction_pixels': 0,
        'total_active_pixels': 0,
        'event_times': [],
        'quench_times': [],
        'drift_times': [],
        'classify_times': [],
        'tpc_pixel_counts': [],
        'tpc_active_pixel_counts': [],
        'tpc_output_arrays_mem_mb': [],
    }
    
    all_classification_data = []
    
    print(f"  Total events to process: {len(all_events)}")
    for event_id in all_events:
        t0_event = time.time()
        event_mask = segments[sim.EVENT_SEPARATOR] == event_id
        event_segs = segments[event_mask]
        
        if len(event_segs) == 0:
            print(f"  Event {event_id} has no segments, skipping")
            continue
        
        pixel_stats['total_events'] += 1
        
        # Process per TPC
        for i_tpc in range(len(detector.TPC_BORDERS)):
            tpc_borders = detector.TPC_BORDERS[i_tpc:i_tpc+1]
            idx_tpc = select_active_volume(event_segs, tpc_borders)
            tpc_segs = event_segs[idx_tpc]
            
            if len(tpc_segs) == 0:
                continue
            
            pixel_stats['total_tpcs'] += 1
            pixel_stats['total_segments_processed'] += len(tpc_segs)
            
            # Quenching
            t0_quench = time.time()
            TPB = 256
            BPG = max(ceil(len(tpc_segs) / TPB), 1)
            quenching.quench[BPG, TPB](tpc_segs, physics.BIRKS)
            cp.cuda.Stream.null.synchronize()
            pixel_stats['quench_times'].append(time.time() - t0_quench)
            
            # Drifting
            t0_drift = time.time()
            drifting.drift[BPG, TPB](tpc_segs)
            cp.cuda.Stream.null.synchronize()
            pixel_stats['drift_times'].append(time.time() - t0_drift)
            
            # Add pixel_plane field (required by classify_pixels)
            if 'pixel_plane' not in tpc_segs.dtype.names:
                pixel_plane = np.full(len(tpc_segs), i_tpc, dtype=np.int32)
                pixel_plane_arr = np.array(pixel_plane, dtype=[('pixel_plane', 'i4')])
                tpc_segs = np.lib.recfunctions.merge_arrays((tpc_segs, pixel_plane_arr), flatten=True)
            
            # Pixel classification
            t0_classify = time.time()
            result = classify_pixels(tpc_segs, plane_id=i_tpc)
            cp.cuda.Stream.null.synchronize()
            pixel_stats['classify_times'].append(time.time() - t0_classify)
            
            # Measure memory footprint of returned device arrays
            outputs_bytes = (result.charge_pixels.nbytes +
                              result.neighbor_pixels.nbytes +
                              result.induction_pixels.nbytes)
            pixel_stats['tpc_output_arrays_mem_mb'].append(outputs_bytes / (1024 * 1024))

            # Extract statistics (convert CuPy to NumPy for counting)
            # Total pixels is full plane grid
            n_pixels_total = detector.N_PIXELS[0] * detector.N_PIXELS[1]
            n_charge = len(result.charge_pixels)
            n_neighbor = len(result.neighbor_pixels)
            n_induction = len(result.induction_pixels)
            n_active = n_charge + n_neighbor + n_induction
            
            pixel_stats['total_pixels_classified'] += n_pixels_total
            pixel_stats['total_charge_pixels'] += n_charge
            pixel_stats['total_neighbor_pixels'] += n_neighbor
            pixel_stats['total_induction_pixels'] += n_induction
            pixel_stats['total_active_pixels'] += n_active
            pixel_stats['tpc_pixel_counts'].append(n_pixels_total)
            pixel_stats['tpc_active_pixel_counts'].append(n_active)
            
            # Optional: save classification data
            if output_file:
                all_classification_data.append({
                    'event_id': event_id,
                    'tpc_id': i_tpc,
                    'n_segments': len(tpc_segs),
                    'n_pixels_total': n_pixels_total,
                    'n_charge_pixels': n_charge,
                    'n_neighbor_pixels': n_neighbor,
                    'n_induction_pixels': n_induction,
                    'n_active_pixels': n_active,
                    'charge_pixels': cp.asnumpy(result.charge_pixels),
                    'neighbor_pixels': cp.asnumpy(result.neighbor_pixels),
                    'induction_pixels': cp.asnumpy(result.induction_pixels),
                })
        
        pixel_stats['event_times'].append(time.time() - t0_event)
    
    # Print statistics
    print(f"\n[5/7] Pixel Classification Complete")
    print(f"  Events processed: {pixel_stats['total_events']}")
    print(f"  TPCs processed: {pixel_stats['total_tpcs']}")
    print(f"  Total segments: {pixel_stats['total_segments_processed']}")
    print(f"  Total pixels classified: {pixel_stats['total_pixels_classified']}")
    
    print(f"\n[6/7] Classification Results")
    print(f"  CHARGE_COLLECTION pixels: {pixel_stats['total_charge_pixels']} "
          f"({100*pixel_stats['total_charge_pixels']/max(pixel_stats['total_pixels_classified'],1):.2f}%)")
    print(f"  CHARGE_NEIGHBOR pixels:   {pixel_stats['total_neighbor_pixels']} "
          f"({100*pixel_stats['total_neighbor_pixels']/max(pixel_stats['total_pixels_classified'],1):.2f}%)")
    print(f"  INDUCTION_ONLY pixels:    {pixel_stats['total_induction_pixels']} "
          f"({100*pixel_stats['total_induction_pixels']/max(pixel_stats['total_pixels_classified'],1):.2f}%)")
    print(f"  ACTIVE (total) pixels:    {pixel_stats['total_active_pixels']} "
          f"({100*pixel_stats['total_active_pixels']/max(pixel_stats['total_pixels_classified'],1):.2f}%)")
    
    if pixel_stats['tpc_active_pixel_counts']:
        active_arr = np.array(pixel_stats['tpc_active_pixel_counts'])
        print(f"\n  Active pixels per TPC:")
        print(f"    Min:     {active_arr.min()}")
        print(f"    Max:     {active_arr.max()}")
        print(f"    Average: {active_arr.mean():.1f}")
    
    if pixel_stats['tpc_output_arrays_mem_mb']:
        out_mem_arr = np.array(pixel_stats['tpc_output_arrays_mem_mb'])
        print(f"\n  Output Arrays GPU Memory (CuPy .nbytes sum per TPC):")
        print(f"    Average: {out_mem_arr.mean():.2f} MB")
        print(f"    Min:     {out_mem_arr.min():.2f} MB")
        print(f"    Max:     {out_mem_arr.max():.2f} MB")
    
    print(f"\n[7/7] Performance Metrics")
    if pixel_stats['event_times']:
        avg_event = np.mean(pixel_stats['event_times'])
        print(f"  Average event time: {avg_event*1000:.2f} ms")
    if pixel_stats['quench_times']:
        total_quench = np.sum(pixel_stats['quench_times'])
        print(f"  Total quenching time: {total_quench:.3f} s")
    if pixel_stats['drift_times']:
        total_drift = np.sum(pixel_stats['drift_times'])
        print(f"  Total drifting time: {total_drift:.3f} s")
    if pixel_stats['classify_times']:
        total_classify = np.sum(pixel_stats['classify_times'])
        avg_classify = np.mean(pixel_stats['classify_times'])
        print(f"  Total classification time: {total_classify:.3f} s")
        print(f"  Average classification per TPC: {avg_classify*1000:.2f} ms")
        if pixel_stats['total_segments_processed'] > 0:
            segs_per_sec = pixel_stats['total_segments_processed'] / total_classify
            print(f"  Segments per second: {segs_per_sec:.1f}")
        if pixel_stats['total_pixels_classified'] > 0:
            pixels_per_sec = pixel_stats['total_pixels_classified'] / total_classify
            print(f"  Pixels per second: {pixels_per_sec:.1f}")
    
    # Save output if requested
    if output_file and all_classification_data:
        print(f"\nSaving classification data to {output_file}")
        with h5py.File(output_file, 'w') as f:
            for i, data in enumerate(all_classification_data):
                grp = f.create_group(f"event_{data['event_id']}_tpc_{data['tpc_id']}")
                grp.attrs['event_id'] = data['event_id']
                grp.attrs['tpc_id'] = data['tpc_id']
                grp.attrs['n_segments'] = data['n_segments']
                grp.attrs['n_pixels_total'] = data['n_pixels_total']
                grp.attrs['n_charge_pixels'] = data['n_charge_pixels']
                grp.attrs['n_neighbor_pixels'] = data['n_neighbor_pixels']
                grp.attrs['n_induction_pixels'] = data['n_induction_pixels']
                grp.attrs['n_active_pixels'] = data['n_active_pixels']
                grp.create_dataset('charge_pixels', data=data['charge_pixels'], compression='gzip')
                grp.create_dataset('neighbor_pixels', data=data['neighbor_pixels'], compression='gzip')
                grp.create_dataset('induction_pixels', data=data['induction_pixels'], compression='gzip')
        print(f"  Saved {len(all_classification_data)} TPC classifications")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    
    return pixel_stats


def main():
    parser = argparse.ArgumentParser(description='Pixel classification performance test')
    parser.add_argument('input_file', help='Path to edep-sim HDF5 file')
    parser.add_argument('--config', default='2x2', help='Detector configuration (default: 2x2)')
    parser.add_argument('--n-events', type=int, default=None, help='Number of events to process (default: all)')
    parser.add_argument('--output', default=None, help='Output HDF5 file for classification data (optional)')
    
    args = parser.parse_args()
    
    test_pixel_classification_performance(
        args.input_file,
        config=args.config,
        n_events=args.n_events,
        output_file=args.output
    )


if __name__ == '__main__':
    main()
