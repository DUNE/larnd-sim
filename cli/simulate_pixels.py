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
    RangePush("load_detector_properties")
    consts.load_detector_properties(detector_properties, pixel_layout)
    RangePop()

    RangePush("load_larndsim_modules")
    # Here we load the modules after loading the detector properties
    # maybe can be implemented in a better way?
    from larndsim import quenching, drifting, detsim, pixels_from_track, fee
    RangePop()

    RangePush("load_hd5_file")
    # First of all we load the edep-sim output
    # For this sample we need to invert $z$ and $y$ axes
    with h5py.File(input_filename, 'r') as f:
        tracks = np.array(f['segments'])
    RangePop()

    RangePush("slicing_and_swapping")
    tracks = tracks[:n_tracks]

    y_start = np.copy(tracks['y_start'] )
    y_end = np.copy(tracks['y_end'])
    y = np.copy(tracks['y'])

    tracks['y_start'] = np.copy(tracks['z_start'])
    tracks['y_end'] = np.copy(tracks['z_end'])
    tracks['y'] = np.copy(tracks['z'])

    tracks['z_start'] = y_start
    tracks['z_end'] = y_end
    tracks['z'] = y
    RangePop()

    TPB = 256
    BPG = ceil(tracks.shape[0] / TPB)

    print("*******************\nSTARTING SIMULATION\n*******************")
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
    step = 200
    adc_tot_list = cp.empty((0,fee.MAX_ADC_VALUES))
    adc_tot_ticks_list = cp.empty((0,fee.MAX_ADC_VALUES))
    backtracked_id_tot = cp.empty((0,fee.MAX_ADC_VALUES,5))
    unique_pix_tot = cp.empty((0,2))
    tot_events = 0

    # We divide the sample in portions that can be processed by the GPU
    tracks_batch_runtimes = []
    for itrk in tqdm(range(0, tracks.shape[0], step), desc='Simulating pixels...'):
        start_tracks_batch = time()
        selected_tracks = tracks[itrk:itrk+step]

        RangePush("event_id_map")
        # Here we build a map between tracks and event IDs
        unique_eventIDs = cp.unique(selected_tracks['eventID'])
        event_id_map = cp.searchsorted(unique_eventIDs,cp.asarray(selected_tracks['eventID']))
        RangePop()

        # We find the pixels intersected by the projection of the tracks on
        # the anode plane using the Bresenham's algorithm. We also take into
        # account the neighboring pixels, due to the transverse diffusion of the charges.
        RangePush("pixels_from_track")
        longest_pix = ceil(max(selected_tracks["dx"])/consts.pixel_size[0])
        max_radius = ceil(max(selected_tracks["tran_diff"])*5/consts.pixel_size[0])
        MAX_PIXELS = int((longest_pix*4+6)*max_radius*1.5)
        MAX_ACTIVE_PIXELS = int(longest_pix*1.5)
        active_pixels = cp.full((selected_tracks.shape[0], MAX_ACTIVE_PIXELS, 2), -1, dtype=np.int32)
        neighboring_pixels = cp.full((selected_tracks.shape[0], MAX_PIXELS, 2), -1, dtype=np.int32)
        n_pixels_list = cp.zeros(shape=(selected_tracks.shape[0]))
        threadsperblock = 128
        blockspergrid = ceil(selected_tracks.shape[0] / threadsperblock)
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

        RangePush("time_intervals")
        # Here we find the longest signal in time and we store an array with the start in time of each track
        max_length = cp.array([0])
        track_starts = cp.empty(selected_tracks.shape[0])
        # d_track_starts = cuda.to_device(track_starts)
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
                                                             selected_tracks)
        RangePop()

        RangePush("pixel_index_map")
        # Here we create a map between tracks and index in the unique pixel array
        pixel_index_map = cp.full((selected_tracks.shape[0], neighboring_pixels.shape[1]), -1)
        compare = neighboring_pixels[..., np.newaxis, :] == unique_pix
        indices = cp.where(cp.logical_and(compare[..., 0], compare[..., 1]))
        pixel_index_map[indices[0], indices[1]] = indices[2]
        # Inverse mapping between pixels and track indices
        track_pixel_map = cp.full((unique_pix.shape[0], neighboring_pixels.shape[1]),-1)
        track_pixel_map[indices[2], indices[1]] = indices[0]
        RangePop()

        RangePush("sum_pixels_signals")
        # Here we combine the induced current on the same pixels by different tracks
        threadsperblock = (8,8,8)
        blockspergrid_x = ceil(signals.shape[0] / threadsperblock[0])
        blockspergrid_y = ceil(signals.shape[1] / threadsperblock[1])
        blockspergrid_z = ceil(signals.shape[2] / threadsperblock[2])
        blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
        pixels_signals = cp.zeros((len(unique_pix), len(consts.time_ticks)*len(unique_eventIDs)*2))
        detsim.sum_pixel_signals[blockspergrid,threadsperblock](pixels_signals,
                                                                signals,
                                                                track_starts,
                                                                pixel_index_map)
        RangePop()

        RangePush("get_adc_values")
        # Here we simulate the electronics response (the self-triggering cycle) and the signal digitization
        time_ticks = cp.linspace(0, len(unique_eventIDs)*consts.time_interval[1]*2, pixels_signals.shape[1]+1)
        integral_list = cp.zeros((pixels_signals.shape[0], fee.MAX_ADC_VALUES))
        adc_ticks_list = cp.zeros((pixels_signals.shape[0], fee.MAX_ADC_VALUES))
        TPB = 128
        BPG = ceil(pixels_signals.shape[0] / TPB)

        rng_states = create_xoroshiro128p_states(TPB * BPG, seed=itrk)
        fee.get_adc_values[BPG,TPB](pixels_signals,
                                    time_ticks,
                                    integral_list,
                                    adc_ticks_list,
                                    consts.time_interval[1]*2*tot_events,
                                    rng_states)
        adc_list = fee.digitize(integral_list)
        RangePop()

        RangePush("backtracking")
        # Here we backtrack the ADC counts to the Geant4 tracks
        TPB = 128
        BPG = ceil(adc_list.shape[0] / TPB)
        backtracked_id = cp.full((adc_list.shape[0], adc_list.shape[1], backtracked_id_tot.shape[2]), -1)
        detsim.backtrack_adcs[BPG,TPB](selected_tracks,
                                       adc_list,
                                       adc_ticks_list,
                                       track_pixel_map,
                                       event_id_map,
                                       backtracked_id)

        adc_tot_list = cp.concatenate((adc_tot_list, adc_list), axis=0)
        adc_tot_ticks_list = cp.concatenate((adc_tot_ticks_list, adc_ticks_list), axis=0)
        unique_pix_tot = cp.concatenate((unique_pix_tot, unique_pix), axis=0)
        backtracked_id_tot = cp.concatenate((backtracked_id_tot, backtracked_id), axis=0)
        tot_events += len(unique_eventIDs)
        RangePop()
        end_tracks_batch = time()
        tracks_batch_runtimes.append(end_tracks_batch - start_tracks_batch)

    print(f"- total time: {sum(tracks_batch_runtimes):.2f} s")
    if len(tracks_batch_runtimes) > 1:
        print(f"- excluding first iteration: {sum(tracks_batch_runtimes[1:]):.2f} s")

    RangePush("Exporting to HDF5")
    # Here we export the result in a HDF5 file.
    fee.export_to_hdf5(cp.asnumpy(adc_tot_list),
                       cp.asnumpy(adc_tot_ticks_list),
                       cp.asnumpy(unique_pix_tot),
                       cp.asnumpy(backtracked_id_tot),
                       output_filename if output_filename else input_filename)
    RangePop()

    print("Output saved in:", output_filename if output_filename else input_filename)

    RangePop()
    end_simulation = time()
    print(f"run_simulation elapsed time: {end_simulation-start_simulation:.2f} s")

if __name__ == "__main__":
    fire.Fire(run_simulation)
