#!/usr/bin/env python

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from math import ceil
from time import time

import numpy as np
import cupy as cp
import fire
import h5py

from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states

from tqdm import tqdm

from larndsim import consts
from larndsim.cuda_dict import CudaDict

logo = """
  _                      _            _
 | |                    | |          (_)
 | | __ _ _ __ _ __   __| |______ ___ _ _ __ ___
 | |/ _` | '__| '_ \ / _` |______/ __| | '_ ` _ \\
 | | (_| | |  | | | | (_| |      \__ \ | | | | | |
 |_|\__,_|_|  |_| |_|\__,_|      |___/_|_| |_| |_|

"""

def cupy_unique_axis0(array):
    # axis is still not supported for cupy.unique, this
    # is a workaround
    if len(array.shape) != 2:
        raise ValueError("Input array must be 2D.")
    sortarr     = array[cp.lexsort(array.T[::-1])]
    mask        = cp.empty(array.shape[0], dtype=cp.bool_)
    mask[0]     = True
    mask[1:]    = cp.any(sortarr[1:] != sortarr[:-1], axis=1)
    return sortarr[mask]

def run_simulation(input_filename,
                   pixel_layout,
                   detector_properties,
                   output_filename='',
                   response_file='../larndsim/response_44.npy',
                   bad_channels=None,
                   n_tracks=100000,
                   pixel_thresholds_file=None):
    """
    Command-line interface to run the simulation of a pixelated LArTPC

    Args:
        input_filename (str): path of the edep-sim input file
        output_filename (str): path of the HDF5 output file. If not specified
            the output is added to the input file.
        pixel_layout (str): path of the YAML file containing the pixel
            layout and connection details.
        detector_properties (str): path of the YAML file containing
            the detector properties
            detector_properties (str): path of the YAML file containing
            the detector properties
        pixel_thresholds_file (str): path to npz file containing pixel thresholds
        n_tracks (int): number of tracks to be simulated
    """
    start_simulation = time()

    from cupy.cuda.nvtx import RangePush, RangePop

    RangePush("run_simulation")

    print(logo)
    print("**************************\nLOADING SETTINGS AND INPUT\n**************************")
    print("Pixel layout file:", pixel_layout)
    print("Detector propeties file:", detector_properties)
    print("edep-sim input file:", input_filename)
    if bad_channels:
        print("Disabled channel list: ", bad_channels)
    RangePush("load_detector_properties")
    consts.load_detector_properties(detector_properties, pixel_layout)
    RangePop()

    RangePush("load_larndsim_modules")
    # Here we load the modules after loading the detector properties
    # maybe can be implemented in a better way?
    from larndsim import quenching, drifting, detsim, pixels_from_track, fee
    RangePop()

    RangePush("load_pixel_thresholds")
    if pixel_thresholds_file is not None:
        print("Pixel thresholds file:", pixel_thresholds_file)
        pixel_thresholds_lut = CudaDict.load(pixel_thresholds_file, 1, 1)
    else:
        pixel_thresholds_lut = CudaDict(np.array([fee.DISCRIMINATION_THRESHOLD]), 1, 1)
    RangePop()

    RangePush("load_hd5_file")
    # First of all we load the edep-sim output
    # For this sample we need to invert $z$ and $y$ axes
    with h5py.File(input_filename, 'r') as f:
        tracks = np.array(f['segments'])
        try:
            trajectories = np.array(f['trajectories'])
            input_has_trajectories = True
        except KeyError:
            input_has_trajectories = False
    RangePop()
    
    if tracks.size == 0:
        print("Empty input dataset, exiting")
        return

    RangePush("slicing_and_swapping")
    tracks = tracks[:n_tracks]

    x_start = np.copy(tracks['x_start'] )
    x_end = np.copy(tracks['x_end'])
    x = np.copy(tracks['x'])

    tracks['x_start'] = np.copy(tracks['z_start'])
    tracks['x_end'] = np.copy(tracks['z_end'])
    tracks['x'] = np.copy(tracks['z'])

    tracks['z_start'] = x_start
    tracks['z_end'] = x_end
    tracks['z'] = x
    RangePop()
    
    is_inside = np.zeros((tracks.shape[0]))
    for itrk in range(tracks.shape[0]):
        track = tracks[itrk]
        for plane in consts.tpc_borders:
            if plane[0][0] <= track['x_start'] <= plane[0][1] and plane[0][0] <= track['x_end'] <= plane[0][1] and \
               plane[1][0] <= track['y_start'] <= plane[1][1] and plane[1][0] <= track['y_end'] <= plane[1][1] and \
               min(plane[2][0],plane[2][1]) <= track['z_start'] <= max(plane[2][0],plane[2][1]) and min(plane[2][0],plane[2][1]) <= track['z_end'] <= max(plane[2][0],plane[2][1]):
                is_inside[itrk] = 1
                break
    
    tracks = tracks[is_inside==1]
    
    response = cp.load(response_file)

    TPB = 256
    BPG = ceil(tracks.shape[0] / TPB)

    print("******************\nRUNNING SIMULATION\n******************")
    # We calculate the number of electrons after recombination (quenching module)
    # and the position and number of electrons after drifting (drifting module)
    print("Quenching electrons...",end='')
    start_quenching = time()
    RangePush("quench")
    quenching.quench[BPG,TPB](tracks, consts.birks)
    RangePop()
    end_quenching = time()
    print(f" {end_quenching-start_quenching:.2f} s")

    print("Drifting electrons...",end='')
    start_drifting = time()
    RangePush("drift")
    drifting.drift[BPG,TPB](tracks)
    RangePop()
    end_drifting = time()
    print(f" {end_drifting-start_drifting:.2f} s")
    
    # initialize lists to collect results from GPU
    event_id_list = []
    adc_tot_list = []
    adc_tot_ticks_list = []
    track_pixel_map_tot = []
    unique_pix_tot = []
    current_fractions_tot = []
    
    # create a lookup table that maps between unique event ids and the segments in the file
    tot_evids = np.unique(tracks['eventID'])
    _, _, start_idx = np.intersect1d(tot_evids, tracks['eventID'], return_indices=True)
    _, _, rev_idx = np.intersect1d(tot_evids, tracks['eventID'][::-1], return_indices=True)
    end_idx = len(tracks['eventID']) - 1 - rev_idx
    
    # We divide the sample in portions that can be processed by the GPU
    tracks_batch_runtimes = []
    step = 1
    tot_events = 0
    for ievd in tqdm(range(0, tot_evids.shape[0], step), desc='Simulating pixels...'):
        start_tracks_batch = time()
        first_event = tot_evids[ievd]
        last_event = tot_evids[min(ievd+step, tot_evids.shape[0]-1)]

        if first_event == last_event:
            last_event += 1

        # load a subset of segments from the file and process those that are from the current event
        track_subset = tracks[min(start_idx[ievd:ievd + step]):max(end_idx[ievd:ievd + step])+1]
        evt_tracks = track_subset[(track_subset['eventID'] >= first_event) & (track_subset['eventID'] < last_event)]
        first_trk_id = np.where(track_subset['eventID'] == evt_tracks['eventID'][0])[0][0] + min(start_idx[ievd:ievd + step])
        
        for itrk in range(0, evt_tracks.shape[0], 50):
            selected_tracks = evt_tracks[itrk:itrk+50]
            RangePush("event_id_map")
            # Here we build a map between tracks and event IDs
            event_ids = selected_tracks['eventID']
            unique_eventIDs = np.unique(event_ids)
            event_id_map = np.searchsorted(unique_eventIDs, event_ids)
            RangePop()

            # We find the pixels intersected by the projection of the tracks on
            # the anode plane using the Bresenham's algorithm. We also take into
            # account the neighboring pixels, due to the transverse diffusion of the charges.
            RangePush("pixels_from_track")
            longest_pix = ceil(max(selected_tracks["dx"])/consts.pixel_pitch)
#             max_radius = ceil(max(selected_tracks["tran_diff"])*5/consts.pixel_pitch)
            max_radius = 3.5
            MAX_PIXELS = int((longest_pix*4+6)*max_radius*1.5)
            MAX_ACTIVE_PIXELS = int(longest_pix*1.5)
            active_pixels = cp.full((selected_tracks.shape[0], MAX_ACTIVE_PIXELS), -1, dtype=np.int32)
            neighboring_pixels = cp.full((selected_tracks.shape[0], MAX_PIXELS), -1, dtype=np.int32)
            n_pixels_list = cp.zeros(shape=(selected_tracks.shape[0]))
            threadsperblock = 128
            blockspergrid = ceil(selected_tracks.shape[0] / threadsperblock)

            if not active_pixels.shape[1] or not neighboring_pixels.shape[1]:
                continue
     
            pixels_from_track.get_pixels[blockspergrid,threadsperblock](selected_tracks,
                                                                        active_pixels,
                                                                        neighboring_pixels,
                                                                        n_pixels_list,
                                                                        max_radius+1)
            RangePop()

            RangePush("unique_pix")
            shapes = neighboring_pixels.shape
            joined = neighboring_pixels.reshape(shapes[0] * shapes[1])
            #unique_pix = cupy_unique_axis0(joined)
            unique_pix = cp.unique(joined)
            unique_pix = unique_pix[(unique_pix != -1)]
            RangePop()
            
            if not unique_pix.shape[0]:
                continue

            RangePush("time_intervals")
            # Here we find the longest signal in time and we store an array with the start in time of each track
            max_length = cp.array([0])
            track_starts = cp.empty(selected_tracks.shape[0])
            threadsperblock = 128
            blockspergrid = ceil(selected_tracks.shape[0] / threadsperblock)
            detsim.time_intervals[blockspergrid,threadsperblock](track_starts, max_length, event_id_map, selected_tracks)
            RangePop()

            RangePush("tracks_current")
            # Here we calculate the induced current on each pixel
            signals = cp.zeros((selected_tracks.shape[0],
                                neighboring_pixels.shape[1],
                                cp.asnumpy(max_length)[0]), dtype=np.float32)
            threadsperblock = (1,1,64)
            blockspergrid_x = ceil(signals.shape[0] / threadsperblock[0])
            blockspergrid_y = ceil(signals.shape[1] / threadsperblock[1])
            blockspergrid_z = ceil(signals.shape[2] / threadsperblock[2])
            blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
            detsim.tracks_current[blockspergrid,threadsperblock](signals,
                                                                 neighboring_pixels,
                                                                 selected_tracks,
                                                                 response)
            
            RangePop()
            RangePush("pixel_index_map")
            # Here we create a map between tracks and index in the unique pixel array
            pixel_index_map = cp.full((selected_tracks.shape[0], neighboring_pixels.shape[1]), -1)
            for i_ in range(selected_tracks.shape[0]):
                compare = neighboring_pixels[i_, ..., cp.newaxis] == unique_pix
                indices = cp.where(compare)
                pixel_index_map[i_, indices[0]] = indices[1]
            RangePop()

            RangePush("track_pixel_map")
            # Mapping between unique pixel array and track array index
            track_pixel_map = cp.full((unique_pix.shape[0], detsim.MAX_TRACKS_PER_PIXEL), -1)
            TPB = 32
            BPG = ceil(unique_pix.shape[0] / TPB)
            detsim.get_track_pixel_map[BPG, TPB](track_pixel_map, unique_pix, neighboring_pixels)
            RangePop()

            RangePush("sum_pixels_signals")
            # Here we combine the induced current on the same pixels by different tracks
            threadsperblock = (8,8,8)
            blockspergrid_x = ceil(signals.shape[0] / threadsperblock[0])
            blockspergrid_y = ceil(signals.shape[1] / threadsperblock[1])
            blockspergrid_z = ceil(signals.shape[2] / threadsperblock[2])
            blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
            pixels_signals = cp.zeros((len(unique_pix), len(consts.time_ticks)))
            pixels_tracks_signals = cp.zeros((len(unique_pix),len(consts.time_ticks),track_pixel_map.shape[1]))
            detsim.sum_pixel_signals[blockspergrid,threadsperblock](pixels_signals,
                                                                    signals,
                                                                    track_starts,
                                                                    pixel_index_map,
                                                                    track_pixel_map,
                                                                    pixels_tracks_signals)
            RangePop()
            
            RangePush("get_adc_values")
            # Here we simulate the electronics response (the self-triggering cycle) and the signal digitization
            time_ticks = cp.linspace(0, len(unique_eventIDs)*consts.time_interval[1], pixels_signals.shape[1]+1)
            integral_list = cp.zeros((pixels_signals.shape[0], fee.MAX_ADC_VALUES))
            adc_ticks_list = cp.zeros((pixels_signals.shape[0], fee.MAX_ADC_VALUES))
            TPB = 128
            BPG = ceil(pixels_signals.shape[0] / TPB)
            
            current_fractions = cp.zeros((pixels_signals.shape[0], fee.MAX_ADC_VALUES, track_pixel_map.shape[1]))
            rng_states = create_xoroshiro128p_states(TPB * BPG, seed=ievd)
            pixel_thresholds_lut.tpb = TPB
            pixel_thresholds_lut.bpg = BPG
            orig_shape = unique_pix.shape
            pixel_thresholds = pixel_thresholds_lut[unique_pix.ravel()].reshape(orig_shape)

            fee.get_adc_values[BPG, TPB](pixels_signals,
                                         pixels_tracks_signals,
                                         time_ticks,
                                         integral_list,
                                         adc_ticks_list,
                                         0,
                                         rng_states,
                                         current_fractions,
                                         pixel_thresholds)


            adc_list = fee.digitize(integral_list)
            adc_event_ids = np.full(adc_list.shape, unique_eventIDs[0]) # FIXME: only works if looping on a single event
            RangePop()

            event_id_list.append(adc_event_ids)
            adc_tot_list.append(cp.asnumpy(adc_list))
            adc_tot_ticks_list.append(cp.asnumpy(adc_ticks_list))
            unique_pix_tot.append(cp.asnumpy(unique_pix))
            current_fractions_tot.append(cp.asnumpy(current_fractions))
            track_pixel_map[track_pixel_map != -1] += first_trk_id + itrk
            track_pixel_map = cp.repeat(track_pixel_map[:, cp.newaxis], fee.MAX_ADC_VALUES, axis=1)
            track_pixel_map_tot.append(cp.asnumpy(track_pixel_map))

        tot_events += step

        end_tracks_batch = time()
        tracks_batch_runtimes.append(end_tracks_batch - start_tracks_batch)

    print("*************\nSAVING RESULT\n*************")
    RangePush("Exporting to HDF5")
    # Here we export the result in a HDF5 file.
    event_id_list = np.concatenate(event_id_list, axis=0)
    adc_tot_list = np.concatenate(adc_tot_list, axis=0)
    adc_tot_ticks_list = np.concatenate(adc_tot_ticks_list, axis=0)
    unique_pix_tot = np.concatenate(unique_pix_tot, axis=0)
    current_fractions_tot = np.concatenate(current_fractions_tot, axis=0)
    track_pixel_map_tot = np.concatenate(track_pixel_map_tot, axis=0)
    fee.export_to_hdf5(event_id_list,
                       adc_tot_list,
                       adc_tot_ticks_list,
                       unique_pix_tot,
                       current_fractions_tot,
                       track_pixel_map_tot,
                       output_filename,
                       bad_channels=bad_channels)
    RangePop()

    with h5py.File(output_filename, 'a') as f:
        f.create_dataset("tracks", data=tracks)
        if input_has_trajectories:
            f.create_dataset("trajectories", data=trajectories)
        f['configs'].attrs['pixel_layout'] = pixel_layout

    print("Output saved in:", output_filename)

    RangePop()
    end_simulation = time()
    print(f"Elapsed time: {end_simulation-start_simulation:.2f} s")

if __name__ == "__main__":
    fire.Fire(run_simulation)
