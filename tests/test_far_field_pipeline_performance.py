#!/usr/bin/env python
"""
Performance test for far-field induced current (only) + FEE on real edep-sim file.
Follows the full flow of cli/simulate_pixels.py:
- Applies quenching, drifting
- Processes one event and one TPC at a time
- Classifies pixels
- Coarse voxelizes segments
- Computes far-field induced currents on induction-only pixels
- Runs FEE self-trigger kernel (no backtracking for far-field)
- Digitizes and saves packets in an HDF5 file
- Logs GPU memory usage (.nbytes) and timings per stage
"""
import os
import time
from math import ceil
import argparse
import warnings
import tqdm

import numpy as np
import h5py
import cupy as cp

from larndsim import consts, quenching, drifting, fee
from larndsim.active_volume import select_active_volume
from larndsim.mesh_refinement.pixel_classifier import classify_pixels
from larndsim.mesh_refinement.voxelization import gpu_voxelize, voxel_id_to_coordinates
from larndsim.mesh_refinement.signal_calculation import launch_far_field_dipole_signal_calculation
from larndsim.config import get_config
from larndsim.consts import units, sim
from numba.cuda.random import create_xoroshiro128p_states
from numba.core.errors import NumbaPerformanceWarning

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


def swap_coordinates(tracks):
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


def maybe_nbytes_mb(*arrays):
    total = 0.0
    for a in arrays:
        if hasattr(a, 'nbytes'):
            total += a.nbytes / 1e6
    return total


def test_far_field_pipeline_performance(input_filename, config='2x2', n_events=None, output_file=None):
    
    print("="*70)
    print("FAR-FIELD PIPELINE PERFORMANCE TEST")
    print("="*70)

    # 1. Load configuration
    print(f"\n[1/9] Loading configuration: {config}")
    cfg = get_config(config)
    consts.load_properties(cfg['DET_PROPERTIES'], cfg['PIXEL_LAYOUT'], cfg['RESPONSE'], cfg['SIM_PROPERTIES'])
    from larndsim.consts import detector, sim, physics, mesh_params

    # 2. Load segments
    print(f"\n[2/9] Loading segments from {input_filename}")
    if not os.path.exists(input_filename):
        raise FileNotFoundError(f"Input file not found: {input_filename}")
    with h5py.File(input_filename, 'r') as f:
        segments = np.array(f['segments'])
    print(f"  Loaded {len(segments)} segments")

    # Add timing fields if missing (match CLI behavior)
    if 't0' not in segments.dtype.names:
        print("  Adding missing timing fields (t0, t0_start, t0_end)")
        t0 = np.array(segments['t'].copy(), dtype=[('t0','f4')])
        t0_start = np.array(segments['t_start'].copy(), dtype=[('t0_start','f4')])
        t0_end = np.array(segments['t_end'].copy(), dtype=[('t0_end','f4')])
        segments = np.lib.recfunctions.merge_arrays((segments, t0, t0_start, t0_end), flatten=True)
        # Zero out t, t_start, t_end (in larnd-sim, these are time at anode)
        segments['t'] = np.zeros(segments.shape[0], dtype='f4')
        segments['t_start'] = np.zeros(segments.shape[0], dtype='f4')
        segments['t_end'] = np.zeros(segments.shape[0], dtype='f4')
    # Add segment_id if missing (match CLI behavior)
    if 'segment_id' not in segments.dtype.names:
        print("  Adding segment_id field")
        dtype = [('segment_id','u4')] + segments.dtype.descr
        new = np.empty(segments.shape, dtype=np.dtype(dtype, align=True))
        new['segment_id'] = np.arange(segments.shape[0], dtype='u4')
        for field in dtype[1:]:
            new[field[0]] = segments[field[0]]
        segments = new

    # Spill handling
    if sim.IS_SPILL_SIM:
        localSpillIDs = segments[sim.EVENT_SEPARATOR] - (segments[sim.EVENT_SEPARATOR] // sim.MAX_EVENTS_PER_FILE) * sim.MAX_EVENTS_PER_FILE
        segments['t0_start'] = segments['t0_start'] - localSpillIDs*sim.SPILL_PERIOD
        segments['t0_end'] = segments['t0_end'] - localSpillIDs*sim.SPILL_PERIOD
        segments['t0'] = segments['t0'] - localSpillIDs*sim.SPILL_PERIOD

    # 3. Preprocessing
    print(f"\n[3/9] Preprocessing")
    mask_neutral = (segments['pdg_id'] != 2112) & (segments['pdg_id'] != 22)
    segments = segments[mask_neutral]
    print(f"  After neutral filtering: {len(segments)} segments")
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

    # Quenching and drifting globally (match CLI behavior)
    print("\n[4/9] Global quenching and drifting")
    TPB = 256
    BPG = max(ceil(len(segments) / TPB), 1)
    t0_quench = time.time()
    quenching.quench[BPG, TPB](segments, physics.BIRKS)
    cp.cuda.Stream.null.synchronize()
    quench_time_global = time.time() - t0_quench
    t0_drift = time.time()
    drifting.drift[BPG, TPB](segments)
    cp.cuda.Stream.null.synchronize()
    drift_time_global = time.time() - t0_drift

    # Generate event times lookup table (match CLI behavior)
    # Event IDs may have offset; take modulus to get zero-based local event IDs
    num_evids = (segments[sim.EVENT_SEPARATOR].max() % sim.MAX_EVENTS_PER_FILE) + 1
    if sim.IS_SPILL_SIM:
        event_times = cp.arange(num_evids) * sim.SPILL_PERIOD
    else:
        event_times = fee.gen_event_times(num_evids)

    # Accumulators for performance stats
    stats = {
        'total_events': 0,
        'total_tpcs': 0,
        'total_segments_processed': 0,
        'event_times': [],
        'classify_times': [],
        'voxel_times': [],
        'far_times': [],
        'fee_times': [],
        'voxel_memory_mb': [],
        'far_output_memory_mb': [],
        'fee_output_memory_mb': [],
        'total_pixels_induction': 0,
        'total_voxels_nonzero': 0,
        'total_adc_values': 0,
    }

    print(f"\n[5/9] Processing per event/TPC")
    for event_id in tqdm.tqdm(all_events):
        t0_event = time.time()
        event_mask = segments[sim.EVENT_SEPARATOR] == event_id
        event_segs = segments[event_mask]
        if len(event_segs) == 0:
            print(f"  Event {event_id} has no segments, skipping")
            continue
        stats['total_events'] += 1

        for i_tpc in tqdm.tqdm(range(len(detector.TPC_BORDERS)), leave=False):
            tpc_borders = detector.TPC_BORDERS[i_tpc:i_tpc+1]
            idx_tpc = select_active_volume(event_segs, tpc_borders)
            tpc_segs = event_segs[idx_tpc]
            if len(tpc_segs) == 0:
                continue
            stats['total_tpcs'] += 1
            stats['total_segments_processed'] += len(tpc_segs)

            # Classify
            t0_class = time.time()
            result = classify_pixels(tpc_segs, plane_id=i_tpc)
            cp.cuda.Stream.null.synchronize()
            stats['classify_times'].append(time.time() - t0_class)
            n_ind = len(result.induction_pixels)
            stats['total_pixels_induction'] += n_ind
            if n_ind == 0:
                continue

            # Voxelize
            t0_vox = time.time()
            vox_idx, vox_charge, grid_shape, voxel_size, bounds = gpu_voxelize(
                tpc_segs, tpc_borders=tpc_borders
            )
            cp.cuda.Stream.null.synchronize()
            stats['voxel_times'].append(time.time() - t0_vox)
            stats['total_voxels_nonzero'] += int(len(vox_idx))
            stats['voxel_memory_mb'].append(maybe_nbytes_mb(vox_idx, vox_charge))

            # Coordinates
            vox_x, vox_y, vox_z = voxel_id_to_coordinates(vox_idx, grid_shape, voxel_size, bounds)

            # Use induction pixel x,y from classification result
            ind_x_gpu = result.induction_pixels_x
            ind_y_gpu = result.induction_pixels_y

            # Far-field kernel (2D over pixels,ticks, sum over voxels)
            electron_pos = cp.stack([
                cp.asarray(vox_x, dtype=cp.float32),
                cp.asarray(vox_y, dtype=cp.float32),
                cp.asarray(vox_z, dtype=cp.float32),
            ], axis=1)
            pixel_pos = cp.stack([
                ind_x_gpu,
                ind_y_gpu,
            ], axis=1)

            z_anode = float(detector.TPC_BORDERS[i_tpc][2][0])
            z_cathode = float(detector.TPC_BORDERS[i_tpc][2][1])

            # Determine time sampling and number of ticks as simulate_pixels.py
            if detector.RESPONSE_MAX_TIME > detector.DRIFT_MAX_TIME:
                max_signal_time = (tpc_segs['t_end'] - tpc_segs['t0']).max() \
                                   + tpc_segs['long_diff'].max() / detector.V_DRIFT * detector.DIFF_N_SIGMAS \
                                   + detector.RESPONSE_MAX_TIME - detector.DRIFT_MAX_TIME
            else:
                max_signal_time = (tpc_segs['t_end'] - tpc_segs['t0']).max() \
                                   + tpc_segs['long_diff'].max() / detector.V_DRIFT * detector.DIFF_N_SIGMAS
            # Use detector.TIME_SAMPLING and ticks from max_signal_time
            tick_size = float(detector.TIME_SAMPLING)
            n_ticks_this = int(np.ceil(max_signal_time / tick_size))

            t0_far = time.time()
            induced_current = launch_far_field_dipole_signal_calculation(
                electron_pos,
                pixel_pos,
                cp.asarray(vox_charge, dtype=cp.float32),
                z_anode,
                z_cathode,
                float(detector.V_DRIFT),
                float(tick_size),
                int(n_ticks_this),
                5,
                None,  # Use INDUCED_CURRENT_SCALE from mesh_params
            )
            cp.cuda.Stream.null.synchronize()
            stats['far_times'].append(time.time() - t0_far)
            stats['far_output_memory_mb'].append(induced_current.nbytes / 1e6)

            pixels_signals = cp.ascontiguousarray(induced_current, dtype=cp.float32)
            n_pixels_fee = pixels_signals.shape[0]
            n_ticks_sig = pixels_signals.shape[1]

            # FEE (no backtracking)
            num_backtrack = cp.zeros(n_pixels_fee, dtype=cp.int32)
            offset_backtrack = cp.zeros(n_pixels_fee, dtype=cp.int32)
            pixels_signals_tracks = cp.array([], dtype=cp.float32)
            # Same time_ticks construction as simulate_pixels.py
            time_ticks = cp.arange(0, n_ticks_sig * tick_size, tick_size, dtype=cp.float32)

            integral_list = cp.zeros((n_pixels_fee, sim.MAX_ADC_VALUES), dtype=cp.float32)
            adc_ticks_list = cp.zeros((n_pixels_fee, sim.MAX_ADC_VALUES), dtype=cp.float32)
            current_fractions = cp.zeros((n_pixels_fee, sim.MAX_ADC_VALUES, 0), dtype=cp.float32)

            TPB = 4
            BPG = ceil(n_pixels_fee / TPB)
            rng_states = create_xoroshiro128p_states(int(TPB * BPG), seed=12345)
            pixel_thresholds = cp.full(n_pixels_fee, detector.DISCRIMINATION_THRESHOLD * units.e, dtype=cp.float32)

            t0_fee = time.time()
            fee.get_adc_values[BPG, TPB](
                pixels_signals,
                pixels_signals_tracks,
                num_backtrack,
                offset_backtrack,
                time_ticks,
                integral_list,
                adc_ticks_list,
                0,
                rng_states,
                current_fractions,
                pixel_thresholds,
            )
            cp.cuda.Stream.null.synchronize()
            stats['fee_times'].append(time.time() - t0_fee)

            gain_list = detector.GAIN * units.mV / units.e
            pedestal_list = detector.V_PEDESTAL
            adc_list = fee.digitize(integral_list, gain_list, pedestal_list)
            stats['fee_output_memory_mb'].append(maybe_nbytes_mb(adc_list, adc_ticks_list))
            stats['total_adc_values'] += int(cp.sum(adc_list > 0).get())

            # Export to HDF5 per TPC/event to clear GPU memory sooner
            if output_file:
                adc_np = cp.asnumpy(adc_list)
                adc_ticks_np = cp.asnumpy(adc_ticks_list)
                pixel_ids_np = np.asarray(result.induction_pixels.get() if hasattr(result.induction_pixels, 'get') else result.induction_pixels, dtype=np.int32)
                event_id_list_tpc = np.full((n_pixels_fee, adc_np.shape[1]), int(event_id), dtype=np.uint32)
                max_trks = int(sim.MAX_TRACKS_PER_PIXEL)
                current_fractions_tpc = np.zeros((n_pixels_fee, adc_np.shape[1], max_trks), dtype=np.float64)
                track_ids_tpc = np.zeros((n_pixels_fee, max_trks), dtype=np.int64)
                traj_ids_tpc = np.zeros((n_pixels_fee, max_trks), dtype=np.int64)

                # Extract event start times from global lookup (match CLI behavior)
                unique_events = np.unique(event_id_list_tpc[:, 0])
                uniq_event_times = cp.asnumpy(event_times[unique_events % sim.MAX_EVENTS_PER_FILE])

                fee.export_to_hdf5(
                    event_id_list_tpc,
                    adc_np,
                    adc_ticks_np,
                    pixel_ids_np,
                    current_fractions_tpc,
                    track_ids_tpc,
                    traj_ids_tpc,
                    output_file,
                    uniq_event_times,
                    compression='gzip'
                )

            # Free GPU memory after TPC
            del adc_list, adc_ticks_list, integral_list, pixels_signals
            cp.get_default_memory_pool().free_all_blocks()

        stats['event_times'].append(time.time() - t0_event)

    # 5. Summary
    print(f"\n[6/9] Summary")
    print(f"  Events processed: {stats['total_events']}")
    print(f"  TPCs processed: {stats['total_tpcs']}")
    print(f"  Total segments: {stats['total_segments_processed']}")
    print(f"  Total induction pixels: {stats['total_pixels_induction']}")
    print(f"  Total nonzero voxels: {stats['total_voxels_nonzero']}")
    print(f"  Total ADC values: {stats['total_adc_values']}")

    # 6. Timings
    def pr_arr(name, arr):
        if arr:
            total = np.sum(arr)
            avg = np.mean(arr)
            print(f"  {name}: total={total:.3f}s, avg={avg*1000:.2f} ms")
    print(f"\n[7/9] Timings per stage")
    print(f"  Global quenching: {quench_time_global:.3f}s")
    print(f"  Global drifting:  {drift_time_global:.3f}s")
    pr_arr('Classification', stats['classify_times'])
    pr_arr('Voxelization', stats['voxel_times'])
    pr_arr('Far-field kernel', stats['far_times'])
    pr_arr('FEE kernel', stats['fee_times'])

    # 7. Memory
    def pr_mem(name, arr):
        if arr:
            print(f"  {name}: avg={np.mean(arr):.2f} MB, min={np.min(arr):.2f} MB, max={np.max(arr):.2f} MB")
    print(f"\n[8/9] GPU Memory (CuPy .nbytes)")
    pr_mem('Voxel sparse arrays', stats['voxel_memory_mb'])
    pr_mem('Far-field output', stats['far_output_memory_mb'])
    pr_mem('FEE output (adc + ticks)', stats['fee_output_memory_mb'])

    # 8. Packets saved incrementally per TPC
    if output_file:
        print(f"\n[9/9] Packets saved incrementally to {output_file}")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)

    return stats


def main():
    parser = argparse.ArgumentParser(description='Far-field pipeline performance test')
    parser.add_argument('input_file', help='Path to edep-sim HDF5 file')
    parser.add_argument('--config', default='2x2', help='Detector configuration (default: 2x2)')
    parser.add_argument('--n-events', type=int, default=None, help='Number of events to process (default: all)')
    parser.add_argument('--output', default=None, help='Output HDF5 file for packets (optional)')
    args = parser.parse_args()

    test_far_field_pipeline_performance(
        args.input_file,
        config=args.config,
        n_events=args.n_events,
        output_file=args.output,
    )


if __name__ == '__main__':
    main()
