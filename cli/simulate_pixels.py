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
                   response_filename='../larndsim/response_44.npy',
                   light_lut_filename='../larndsim/lightLUT.npy',
                   bad_channels=None,
                   n_tracks=100000):
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
    from larndsim import quenching, drifting, detsim, pixels_from_track, fee, lightLUT
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
    
    # Made a separate array to store number of photons generated from quenching.py. Should be added to tracks as another dtype
    phot_from_edep = np.zeros(len(tracks), dtype = [('n_photons_edep','f4')])

    light_sim_dat = np.zeros([len(tracks),consts.n_op_channel*2], dtype = [('n_photons_det','f4'),('t0_det','f4')])

    RangePop()
    
    if tracks.size == 0:
        print("Empty input dataset, exiting")
        return

    RangePush("slicing_and_swapping")
    tracks = tracks[:n_tracks]
    light_sim_dat = light_sim_dat[:n_tracks]

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
    
    response = cp.load(response_filename)

    TPB = 256
    BPG = ceil(tracks.shape[0] / TPB)

    print("******************\nRUNNING SIMULATION\n******************")
    # We calculate the number of electrons after recombination (quenching module)
    # and the position and number of electrons after drifting (drifting module)
    print("Quenching electrons...",end='')
    start_quenching = time()
    RangePush("quench")
    quenching.quench[BPG,TPB](tracks, phot_from_edep, consts.birks)
    RangePop()
    end_quenching = time()
    print(f" {end_quenching-start_quenching:.2f} s")

    print("Calculating optical responses...",end='')
    start_lightLUT = time()
    lightLUT.calculate_light_incidence(tracks, light_lut_filename, phot_from_edep, light_sim_dat)
    end_lightLUT = time()
    print(f" {end_lightLUT-start_lightLUT:.2f} s")

    print("Drifting electrons...",end='')
    start_drifting = time()
    RangePush("drift")
    drifting.drift[BPG,TPB](tracks)
    RangePop()
    end_drifting = time()
    print(f" {end_drifting-start_drifting:.2f} s")
    step = 1
    adc_tot_list = []
    adc_tot_ticks_list = []
    track_pixel_map_tot = []
    unique_pix_tot = []
    current_fractions_tot = []
    tot_events = 0
    
    tot_evids = np.unique(tracks['eventID'])
    # We divide the sample in portions that can be processed by the GPU
    tracks_batch_runtimes = []
    for ievd in tqdm(range(0, tot_evids.shape[0], step), desc='Simulating pixels...'):
        start_tracks_batch = time()
        first_event = tot_evids[ievd]
        last_event = tot_evids[min(ievd+step, tot_evids.shape[0]-1)]

        if first_event == last_event:
            last_event += 1

        evt_tracks = tracks[(tracks['eventID']>=first_event) & (tracks['eventID']<last_event)]
        first_trk_id = np.where(tracks['eventID']==evt_tracks['eventID'][0])[0][0]
        
        for itrk in range(0, evt_tracks.shape[0], 600):
            selected_tracks = evt_tracks[itrk:itrk+600]

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
            max_radius = ceil(max(selected_tracks["tran_diff"])*5/consts.pixel_pitch)
            MAX_PIXELS = int((longest_pix*4+6)*max_radius*1.5)
            MAX_ACTIVE_PIXELS = int(longest_pix*1.5)
            active_pixels = cp.full((selected_tracks.shape[0], MAX_ACTIVE_PIXELS, 2), -1, dtype=np.int32)
            neighboring_pixels = cp.full((selected_tracks.shape[0], MAX_PIXELS, 2), -1, dtype=np.int32)
            n_pixels_list = cp.zeros(shape=(selected_tracks.shape[0]))
            threadsperblock = 128
            blockspergrid = ceil(selected_tracks.shape[0] / threadsperblock)

            if not active_pixels.shape[1]:
                continue
                
            pixels_from_track.get_pixels[blockspergrid,threadsperblock](selected_tracks,
                                                                        active_pixels,
                                                                        neighboring_pixels,
                                                                        n_pixels_list,
                                                                        max_radius+1)
            RangePop()

            RangePush("unique_pix")
            shapes = neighboring_pixels.shape
            joined = neighboring_pixels.reshape(shapes[0]*shapes[1],2)
            unique_pix = cupy_unique_axis0(joined)
            unique_pix = unique_pix[(unique_pix[:,0] != -1) & (unique_pix[:,1] != -1),:]
            RangePop()
            
            if not unique_pix.shape[0]:
                continue

            RangePush("time_intervals")
            # Here we find the longest signal in time and we store an array with the start in time of each track
            max_length = cp.array([0])
            track_starts = cp.empty(selected_tracks.shape[0])
            threadsperblock = 128
            blockspergrid = ceil(selected_tracks.shape[0] / threadsperblock)
            detsim.time_intervals[blockspergrid,threadsperblock](track_starts, max_length,  event_id_map, selected_tracks)
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
            compare = neighboring_pixels[..., np.newaxis, :] == unique_pix
            indices = cp.where(cp.logical_and(compare[..., 0], compare[..., 1]))
            pixel_index_map[indices[0], indices[1]] = indices[2]
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
            pixels_signals = cp.zeros((len(unique_pix), len(consts.time_ticks)*3))
            pixels_tracks_signals = cp.zeros((len(unique_pix),len(consts.time_ticks)*3,track_pixel_map.shape[1]))
            detsim.sum_pixel_signals[blockspergrid,threadsperblock](pixels_signals,
                                                                    signals,
                                                                    track_starts,
                                                                    pixel_index_map,
                                                                    track_pixel_map,
                                                                    pixels_tracks_signals)
            RangePop()

            RangePush("get_adc_values")
            # Here we simulate the electronics response (the self-triggering cycle) and the signal digitization
            time_ticks = cp.linspace(0, len(unique_eventIDs)*consts.time_interval[1]*3, pixels_signals.shape[1]+1)
            integral_list = cp.zeros((pixels_signals.shape[0], fee.MAX_ADC_VALUES))
            adc_ticks_list = cp.zeros((pixels_signals.shape[0], fee.MAX_ADC_VALUES))
            TPB = 128
            BPG = ceil(pixels_signals.shape[0] / TPB)
            
            current_fractions = cp.zeros((pixels_signals.shape[0], fee.MAX_ADC_VALUES, track_pixel_map.shape[1]))
            rng_states = create_xoroshiro128p_states(TPB * BPG, seed=ievd)
           
            fee.get_adc_values[BPG,TPB](pixels_signals,
                                        pixels_tracks_signals,
                                        time_ticks,
                                        integral_list,
                                        adc_ticks_list,
                                        consts.time_interval[1]*3*tot_events,
                                        rng_states,
                                        current_fractions)

            adc_list = fee.digitize(integral_list)
            RangePop()

            adc_tot_list.append(cp.asnumpy(adc_list))
            adc_tot_ticks_list.append(cp.asnumpy(adc_ticks_list))
            unique_pix_tot.append(cp.asnumpy(unique_pix))
            current_fractions_tot.append(cp.asnumpy(current_fractions))
            track_pixel_map[track_pixel_map != -1] += first_trk_id + itrk
            track_pixel_map = cp.repeat(track_pixel_map[:, cp.newaxis], detsim.MAX_TRACKS_PER_PIXEL, axis=1)
            track_pixel_map_tot.append(cp.asnumpy(track_pixel_map))

        tot_events += step

        end_tracks_batch = time()
        tracks_batch_runtimes.append(end_tracks_batch - start_tracks_batch)

    print("*************\nSAVING RESULT\n*************")
    RangePush("Exporting to HDF5")
    # Here we export the result in a HDF5 file.
    adc_tot_list = np.concatenate(adc_tot_list, axis=0)
    adc_tot_ticks_list = np.concatenate(adc_tot_ticks_list, axis=0)
    unique_pix_tot = np.concatenate(unique_pix_tot, axis=0)
    current_fractions_tot = np.concatenate(current_fractions_tot, axis=0)
    track_pixel_map_tot = np.concatenate(track_pixel_map_tot, axis=0)
    fee.export_to_hdf5(adc_tot_list,
                       adc_tot_ticks_list,
                       unique_pix_tot,
                       current_fractions_tot,
                       track_pixel_map_tot,
                       output_filename,
                       bad_channels=bad_channels)
    RangePop()

    with h5py.File(output_filename, 'a') as f:
        f.create_dataset("tracks", data=tracks)
        f.create_dataset('light_dat', data = light_sim_dat)
        if input_has_trajectories:
            f.create_dataset("trajectories", data=trajectories)
        f['configs'].attrs['pixel_layout'] = pixel_layout

    print("Output saved in:", output_filename)

    RangePop()
    end_simulation = time()
    print(f"Elapsed time: {end_simulation-start_simulation:.2f} s")

if __name__ == "__main__":
    fire.Fire(run_simulation)
