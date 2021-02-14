#!/usr/bin/env python

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from larndsim import consts
import numpy as np
import fire
import h5py

import pickle
import numpy as np
import numba as nb
import pandas as pd

from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states

from math import ceil, sqrt, pi, exp, erf
from time import time

from tqdm import tqdm

def run_simulation(input_filename, 
                   output_filename='',
                   pixel_layout='../larndsim/pixel_layouts/layout-singlecube.yaml',
                   detector_properties='../larndsim/detector_properties/singlecube.yaml',
                   n_tracks=100000):
    
    consts.load_detector_properties(detector_properties, pixel_layout)
    
    from larndsim import quenching, drifting, detsim, pixels_from_track, fee

    with h5py.File(input_filename, 'r') as f:
        tracks = np.array(f['segments'])
        
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

    TPB = 256
    BPG = ceil(tracks.shape[0] / TPB)

    print("Quenching electrons...",end='')
    start_quenching = time()
    quenching.quench[BPG,TPB](tracks, consts.birks)
    end_quenching = time()
    print(f" {end_quenching-start_quenching:.2f} s")
    
    print("Drifting electrons...",end='')
    start_drifting = time()
    drifting.drift[BPG,TPB](tracks)
    end_drifting = time()
    print(f" {end_drifting-start_drifting:.2f} s")

    step = 800
    adc_tot_list = np.empty((1,fee.MAX_ADC_VALUES))
    adc_tot_ticks_list = np.empty((1,fee.MAX_ADC_VALUES))
    backtracked_id_tot = np.empty((1,fee.MAX_ADC_VALUES,5))
    unique_pix_tot = np.empty((1,2))
    tot_events = 0
    start_looping = time()
    
    for itrk in tqdm(range(0, tracks.shape[0], step), desc='Simulating pixels...'):
        selected_tracks = tracks[itrk:itrk+step]
        
        unique_eventIDs = np.unique(selected_tracks['eventID'])
        event_id_map = np.zeros_like(selected_tracks['eventID'])
        for iev, evID in enumerate(selected_tracks['eventID']):
            event_id_map[iev] = np.where(evID == unique_eventIDs)[0]
        d_event_id_map = cuda.to_device(event_id_map)
        
        longest_pix = ceil(max(selected_tracks["dx"])/consts.pixel_size[0])
        max_radius = ceil(max(selected_tracks["tran_diff"])*5/consts.pixel_size[0])

        MAX_PIXELS = (longest_pix*4+6)*max_radius*2
        MAX_ACTIVE_PIXELS = longest_pix*2
        active_pixels = np.full((selected_tracks.shape[0], MAX_ACTIVE_PIXELS, 2), -1, dtype=np.int32)
        neighboring_pixels = np.full((selected_tracks.shape[0], MAX_PIXELS, 2), -1, dtype=np.int32)
        n_pixels_list = np.zeros(shape=(selected_tracks.shape[0]))
        threadsperblock = 128
        blockspergrid = ceil(selected_tracks.shape[0] / threadsperblock)
        pixels_from_track.get_pixels[blockspergrid,threadsperblock](selected_tracks, 
                                                                    active_pixels, 
                                                                    neighboring_pixels, 
                                                                    n_pixels_list,
                                                                    max_radius+1)
        
        shapes = neighboring_pixels.shape
        joined = neighboring_pixels.reshape(shapes[0]*shapes[1],2)
        unique_pix = np.unique(joined, axis=0)
        unique_pix = unique_pix[(unique_pix[:,0] != -1) & (unique_pix[:,1] != -1),:]

        max_length = np.array([0])
        track_starts = np.empty(selected_tracks.shape[0])
        d_track_starts = cuda.to_device(track_starts)
        threadsperblock = 128
        blockspergrid = ceil(selected_tracks.shape[0] / threadsperblock)
        detsim.time_intervals[blockspergrid,threadsperblock](d_track_starts, max_length,  d_event_id_map, selected_tracks)

        signals = np.zeros((selected_tracks.shape[0], 
                            neighboring_pixels.shape[1], 
                            max_length[0]), dtype=np.float32)
        threadsperblock = (4,4,4)
        blockspergrid_x = ceil(signals.shape[0] / threadsperblock[0])
        blockspergrid_y = ceil(signals.shape[1] / threadsperblock[1])
        blockspergrid_z = ceil(signals.shape[2] / threadsperblock[2])
        blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
        d_signals = cuda.to_device(signals)

        detsim.tracks_current[blockspergrid,threadsperblock](d_signals, 
                                                             neighboring_pixels, 
                                                             selected_tracks)

        pixel_index_map = np.full((selected_tracks.shape[0], neighboring_pixels.shape[1]), -1)

        for itr in range(neighboring_pixels.shape[0]):
            for ipix in range(neighboring_pixels.shape[1]):
                pID = neighboring_pixels[itr][ipix]
                if pID[0] >= 0 and pID[1] >= 0:
                    try:
                        index = np.where((unique_pix[:,0] == pID[0]) & (unique_pix[:,1] == pID[1]))
                    except IndexError:
                        print(index,"More pixels than maximum value")
                    pixel_index_map[itr,ipix] = index[0]

        d_pixel_index_map = cuda.to_device(pixel_index_map)
        threadsperblock = (8,8,8)
        blockspergrid_x = ceil(d_signals.shape[0] / threadsperblock[0])
        blockspergrid_y = ceil(d_signals.shape[1] / threadsperblock[1])
        blockspergrid_z = ceil(d_signals.shape[2] / threadsperblock[2])
        blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
        pixels_signals = np.zeros((len(unique_pix),len(consts.time_ticks)*len(unique_eventIDs)*2))
        d_pixels_signals = cuda.to_device(pixels_signals)
        detsim.sum_pixel_signals[blockspergrid,threadsperblock](d_pixels_signals, 
                                                                d_signals, 
                                                                d_track_starts, 
                                                                d_pixel_index_map)
        
        time_ticks = np.linspace(0,len(unique_eventIDs)*consts.time_interval[1]*2,pixels_signals.shape[1]+1)
        integral_list = np.zeros((pixels_signals.shape[0], fee.MAX_ADC_VALUES))
        adc_ticks_list = np.zeros((pixels_signals.shape[0], fee.MAX_ADC_VALUES))
        TPB = 128
        BPG = ceil(pixels_signals.shape[0] / TPB)
        
        rng_states = create_xoroshiro128p_states(TPB * BPG, seed=itrk)
        fee.get_adc_values[BPG,TPB](d_pixels_signals, 
                                    time_ticks, 
                                    integral_list, 
                                    adc_ticks_list, 
                                    consts.time_interval[1]*2*tot_events,
                                    rng_states)
        adc_list = fee.digitize(integral_list)
        track_pixel_map = np.full((unique_pix.shape[0], 5),-1)
        backtracked_id = np.full((adc_list.shape[0], adc_list.shape[1], track_pixel_map.shape[1]), -1)

        detsim.get_track_pixel_map(track_pixel_map, unique_pix, neighboring_pixels)
        detsim.backtrack_adcs(selected_tracks, adc_list, adc_ticks_list, track_pixel_map, event_id_map, backtracked_id)
        adc_tot_list = np.append(adc_tot_list, adc_list, axis=0)
        adc_tot_ticks_list = np.append(adc_tot_ticks_list, adc_ticks_list, axis=0)
        unique_pix_tot = np.append(unique_pix_tot, unique_pix, axis=0)
        backtracked_id_tot = np.append(backtracked_id_tot, backtracked_id, axis=0)
        tot_events += len(unique_eventIDs)

    unique_pix_tot = unique_pix_tot[1:]
    adc_tot_list = adc_tot_list[1:]
    adc_tot_ticks_list = adc_tot_ticks_list[1:]
    backtracked_id_tot = backtracked_id_tot[1:]
    if output_filename:
        fee.export_to_hdf5(adc_tot_list, adc_tot_ticks_list, unique_pix_tot, backtracked_id_tot, output_filename)
    else:
        fee.export_to_hdf5(adc_tot_list, adc_tot_ticks_list, unique_pix_tot, backtracked_id_tot, input_filename)
    
if __name__ == "__main__":
    fire.Fire(run_simulation)