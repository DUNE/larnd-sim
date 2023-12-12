#!/usr/bin/env python
"""
Command-line interface to larnd-sim module.
"""
from math import ceil
from time import time
import warnings
from collections import defaultdict

import numpy as np
import numpy.lib.recfunctions as rfn

import cupy as cp
from cupy.cuda.nvtx import RangePush, RangePop

import fire
import h5py

from numba.cuda import device_array, to_device
from numba.cuda.random import create_xoroshiro128p_states
from numba.core.errors import NumbaPerformanceWarning

from tqdm import tqdm

from larndsim import consts
from larndsim.util import CudaDict, batching, memory_logger

import os

SEED = int(time())

LOGO = """
                             .-----.
                            /7  .  (
                           /   .-.  \
                          /   /   \  \
                         / `  )   (   )
                        / `   )   ).  \
                      .'  _.   \_/  . |
     .--.           .' _.' )`.        |
    (    `---...._.'   `---.'_)    ..  \
     \            `----....___    `. \  |
      `.           _ ----- _   `._  )/  |
        `.       /"  \   /"  \`.  `._   |
          `.    ((O)` ) ((O)` ) `.   `._\
            `-- '`---'   `---' )  `.    `-.
               /                  ` \      `-.
             .'                      `.       `.
            /                     `  ` `.       `-.
     .--.   \ ===._____.======. `    `   `. .___.--`     .''''.
    ' .` `-. `.                )`. `   ` ` \          .' . '  8)
   (8  .  ` `-.`.               ( .  ` `  .`\      .'  '    ' /
    \  `. `    `-.               ) ` .   ` ` \  .'   ' .  '  /
     \ ` `.  ` . \`.    .--.     |  ` ) `   .``/   '  // .  /
      `.  ``. .   \ \   .-- `.  (  ` /_   ` . / ' .  '/   .'
        `. ` \  `  \ \  '-.   `-'  .'  `-.  `   .  .'/  .'
          \ `.`.  ` \ \    ) /`._.`       `.  ` .  .'  /
    LGB    |  `.`. . \ \  (.'               `.   .'  .'
        __/  .. \ \ ` ) \                     \.' .. \__
 .-._.-'     '"  ) .-'   `.                   (  '"     `-._.--.
(_________.-====' / .' /\_)`--..__________..-- `====-. _________)
                 (.'(.'
      _                      _            _
     | |                    | |          (_)
     | | __ _ _ __ _ __   __| |______ ___ _ _ __ ___
     | |/ _` | '__| '_ \ / _` |______/ __| | '_ ` _ \\
     | | (_| | |  | | | | (_| |      \__ \ | | | | | |
     |_|\__,_|_|  |_| |_|\__,_|      |___/_|_| |_| |_|

"""

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

def swap_coordinates(tracks):
    """
    Swap x and z coordinates in tracks.
    This is because the convention in larnd-sim is different
    from the convention in edep-sim. FIXME.

    Args:
        tracks (:obj:`numpy.ndarray`): tracks array.

    Returns:
        :obj:`numpy.ndarray`: tracks with swapped axes.
    """
    x_start = np.copy(tracks['x_start'] )
    x_end = np.copy(tracks['x_end'])
    x = np.copy(tracks['x'])

    tracks['x_start'] = np.copy(tracks['z_start'])
    tracks['x_end'] = np.copy(tracks['z_end'])
    tracks['x'] = np.copy(tracks['z'])

    tracks['z_start'] = x_start
    tracks['z_end'] = x_end
    tracks['z'] = x

    return tracks

def maybe_create_rng_states(n, seed=0, rng_states=None):
    """Create or extend random states for CUDA kernel"""

    if rng_states is None:
        return create_xoroshiro128p_states(n, seed=seed)

    if n > len(rng_states):
        new_states = device_array(n, dtype=rng_states.dtype)
        new_states[:len(rng_states)] = rng_states
        new_states[len(rng_states):] = create_xoroshiro128p_states(n - len(rng_states), seed=seed)
        return new_states

    return rng_states




def run_simulation(input_filename,
                   pixel_layout,
                   detector_properties,
                   simulation_properties,
                   output_filename,
                   response_file='../larndsim/bin/response_44.npy',
                   light_lut_filename='../larndsim/bin/lightLUT.npz',
                   light_det_noise_filename='../larndsim/bin/light_noise-module0.npy',
                   light_simulated=None,
                   bad_channels=None,
                   n_events=None,
                   pixel_thresholds_file=None,
                   pixel_gains_file=None,
                   rand_seed=None,
                   save_memory=None):
    """
    Command-line interface to run the simulation of a pixelated LArTPC

    Args:
        input_filename (str): path of the edep-sim input file
        pixel_layout (str): path of the YAML file containing the pixel
            layout and connection details.
        detector_properties (str): path of the YAML file containing
            the detector properties
        simulation_properties (str): path of the YAML file containing
            the simulation properties
        output_filename (str): path of the HDF5 output file. If not specified
            the output is added to the input file.
        response_file (str, optional): path of the Numpy array containing the pre-calculated
            field responses. Defaults to ../larndsim/bin/response_44.npy.
        light_lut_file (str, optional): path of the Numpy array containing the light
            look-up table. Defaults to ../larndsim/bin/lightLUT.npy.
        bad_channels (str, optional): path of the YAML file containing the channels to be
            disabled. Defaults to None
        n_events (int, optional): number of events to be simulated. Defaults to None
            (all tracks).
        pixel_thresholds_file (str, optional): path to npz file containing pixel thresholds. Defaults
            to None.
        pixel_gains_file (str): path to npz file containing pixel gain values. Defaults to None (the value of fee.GAIN)
        rand_seed (int, optional): the random number generator seed that can be set through 
            a command-line
        save_memory (string path, optional): if non-empty, this is used as a filename to 
            store memory snapshot information
    """
    
    if not os.path.exists(input_filename):
        raise Exception(f'Input file {input_filename} does not exist.')
    if os.path.exists(output_filename):
        raise Exception(f'Output file {output_filename} already exists.')
    
    logger = memory_logger(save_memory is None)
    logger.start()
    logger.take_snapshot()
    start_simulation = time()

    RangePush("run_simulation")

    if not rand_seed: rand_seed = SEED

    print(LOGO)
    print("FROG - FROG - FROG")
    print("**************************\nLOADING SETTINGS AND INPUT\n**************************")
    print("Output file:", output_filename)

    print("Random seed:", rand_seed)
    print("Pixel layout file:", pixel_layout)
    print("Detector properties file:", detector_properties)
    print("Simulation properties file:", simulation_properties)
    print("edep-sim input file:", input_filename)
    print("Response file:", response_file)
    if bad_channels:
        print("Disabled channel list: ", bad_channels)
    if save_memory:
        print('Recording the process resource log:', save_memory)
    else:
        print('Memory resource log will not be recorded')


    RangePush("set_random_seed")
    cp.random.seed(rand_seed)
    # pre-allocate some random number states for custom kernels
    rng_states = maybe_create_rng_states(1024*256, seed=rand_seed)
    RangePop()

    RangePush("load_detector_properties")
    consts.load_properties(detector_properties, pixel_layout, simulation_properties)
    from larndsim.consts import light, detector, physics, sim
    RangePop()
    print("Event batch size:", sim.EVENT_BATCH_SIZE)
    print("Batch size:", sim.BATCH_SIZE)
    print("Write batch size:", sim.WRITE_BATCH_SIZE)

    RangePush("load_larndsim_modules")
    # Here we load the modules after loading the detector properties
    # maybe can be implemented in a better way?
    from larndsim import (active_volume, quenching, drifting, detsim, pixels_from_track, fee,
        lightLUT, light_sim)
    RangePop()

    RangePush("load_pixel_thresholds")
    if pixel_thresholds_file is not None:
        print("Pixel thresholds file:", pixel_thresholds_file)
        pixel_thresholds_lut = CudaDict.load(pixel_thresholds_file, 512)
    else:
        pixel_thresholds_lut = CudaDict(cp.array([fee.DISCRIMINATION_THRESHOLD]), 1, 1)
    RangePop()

    RangePush("load_pixel_gains")
    if pixel_gains_file is not None:
        print("Pixel gains file:", pixel_gains_file)
        pixel_gains_lut = CudaDict.load(pixel_gains_file, 512)
    RangePop()
    
    RangePush("load_hd5_file")
    print("Loading track segments..." , end="")
    start_load = time()
    # First of all we load the edep-sim output
    with h5py.File(input_filename, 'r') as f:
        tracks = np.array(f['segments'])
        if 'segment_id' in tracks.dtype.names:
            segment_ids = tracks['segment_id']
        else:
            dtype = tracks.dtype.descr
            dtype = [('segment_id','u4')] + dtype
            new_tracks = np.empty(tracks.shape, dtype=np.dtype(dtype, align=True))
            new_tracks['segment_id'] = np.arange(tracks.shape[0], dtype='u4')
            for field in dtype[1:]:
                new_tracks[field[0]] = tracks[field[0]]
            tracks = new_tracks
            segment_ids = tracks['segment_id']
        try:
            trajectories = np.array(f['trajectories'])
            input_has_trajectories = True
        except KeyError:
            input_has_trajectories = False

        try:
            vertices = np.array(f['vertices'])
            input_has_vertices = True
        except KeyError:
            print("Input file does not have true vertices info")
            input_has_vertices = False

        try:
            mc_hdr = np.array(f['mc_hdr'])
            input_has_mc_hdr = True
        except KeyError:
            print("Input file does not have MC event summary info")
            input_has_mc_hdr = False

        try:
            mc_stack = np.array(f['mc_stack'])
            input_has_mc_stack = True
        except KeyError:
            print("Input file does not have MC particle stack info")
            input_has_mc_stack = False

    if tracks.size == 0:
        print("Empty input dataset, exiting")
        return

    RangePop()
    end_load = time()
    logger.take_snapshot()
    logger.archive('loading')
    print(f" {end_load-start_load:.2f} s")

    response = cp.load(response_file)

    TPB = 256
    BPG = max(ceil(tracks.shape[0] / TPB),1)

    print("******************\nRUNNING SIMULATION\n******************")
    logger.start()
    logger.take_snapshot()
    # Reduce dataset if not all events are to be simulated, being careful of gaps
    if n_events:
        print(f'Selecting only the first {n_events} events for simulation.')
        max_eventID = np.unique(tracks[sim.EVENT_SEPARATOR])[n_events-1]
        segment_ids = segment_ids[tracks[sim.EVENT_SEPARATOR] <= max_eventID]
        tracks = tracks[tracks[sim.EVENT_SEPARATOR] <= max_eventID]
        
        if input_has_trajectories:
            trajectories = trajectories[trajectories[sim.EVENT_SEPARATOR] <= max_eventID]
        if input_has_vertices:
            vertices = vertices[vertices[sim.EVENT_SEPARATOR] <= max_eventID]
        if input_has_mc_hdr:
            mc_hdr = mc_hdr[mc_hdr[sim.EVENT_SEPARATOR] <= max_eventID]
        if input_has_mc_stack:
            mc_stack = mc_stack[mc_stack[sim.EVENT_SEPARATOR] <= max_eventID]

    # Here we swap the x and z coordinates of the tracks
    # because of the different convention in larnd-sim wrt edep-sim
    tracks = swap_coordinates(tracks)

    # Sub-select only segments in active volumes
    if sim.IF_ACTIVE_VOLUME_CHECK:
        print("Skipping non-active volumes..." , end="")
        start_mask = time()
        active_tracks = active_volume.select_active_volume(tracks, detector.TPC_BORDERS)
        tracks = tracks[active_tracks]
        segment_ids = segment_ids[active_tracks]
        end_mask = time()
        print(f" {end_mask-start_mask:.2f} s")

    if light_simulated is not None:
        light.LIGHT_SIMULATED = light_simulated

    RangePush("run_simulation")

    # Set up light simulation data objects
    if light.LIGHT_SIMULATED:
        light_sim_dat = np.zeros([len(tracks), light.N_OP_CHANNEL],
                                 dtype=[('segment_id', 'u4'), ('n_photons_det','f4'),('t0_det','f4')])
        light_sim_dat['segment_id'] = segment_ids[..., np.newaxis]
        track_light_voxel = np.zeros([len(tracks), 3], dtype='i4')

    if 'n_photons' not in tracks.dtype.names:
        n_photons = np.zeros(tracks.shape[0], dtype=[('n_photons', 'f4')])
        tracks = rfn.merge_arrays((tracks, n_photons), flatten=True)

    if 't0' not in tracks.dtype.names:
        # the t0 key refers to the time of energy deposition
        # in the input files, it is called 't'
        # this is only true for older edep inputs (which are included in `examples/`)
        t0 = np.array(tracks['t'].copy(), dtype=[('t0', 'f4')])
        t0_start = np.array(tracks['t_start'].copy(), dtype=[('t0_start', 'f4')])
        t0_end = np.array(tracks['t_end'].copy(), dtype=[('t0_end', 'f4')])
        tracks = rfn.merge_arrays((tracks, t0, t0_start, t0_end), flatten=True)

        # then, re-initialize the t key to zero
        # in larnd-sim, this key is the time at the anode
        tracks['t'] = np.zeros(tracks.shape[0], dtype=[('t', 'f4')])
        tracks['t_start'] = np.zeros(tracks.shape[0], dtype=[('t_start', 'f4')])
        tracks['t_end'] = np.zeros(tracks.shape[0], dtype=[('t_end', 'f4')])

    if sim.IS_SPILL_SIM:
        # "Reset" the spill period so t0 is wrt the corresponding spill start time.
        # The spill starts are marking the start of 
        # The space between spills will be accounted for in the
        # packet timestamps through the event_times array below
        localSpillIDs = localSpillIDs = tracks[sim.EVENT_SEPARATOR] - (tracks[sim.EVENT_SEPARATOR] // sim.MAX_EVENTS_PER_FILE) * sim.MAX_EVENTS_PER_FILE
        tracks['t0_start'] = tracks['t0_start'] - localSpillIDs*sim.SPILL_PERIOD
        tracks['t0_end'] = tracks['t0_end'] - localSpillIDs*sim.SPILL_PERIOD
        tracks['t0'] = tracks['t0'] - localSpillIDs*sim.SPILL_PERIOD

    logger.take_snapshot()
    logger.archive('preparation')

    # We calculate the number of electrons after recombination (quenching module)
    # and the position and number of electrons after drifting (drifting module)
    print("Quenching electrons..." , end="")
    logger.start()
    logger.take_snapshot()
    start_quenching = time()
    quenching.quench[BPG,TPB](tracks, physics.BIRKS)
    end_quenching = time()
    logger.take_snapshot()
    logger.archive('quenching')
    print(f" {end_quenching-start_quenching:.2f} s")

    print("Drifting electrons...", end="")
    start_drifting = time()
    logger.start()
    logger.take_snapshot()
    drifting.drift[BPG,TPB](tracks)
    end_drifting = time()
    logger.take_snapshot()
    logger.archive('drifting')
    print(f" {end_drifting-start_drifting:.2f} s")

    if light.LIGHT_SIMULATED:
        print("Calculating optical responses...", end="")
        start_light_time = time()
        logger.start()
        logger.take_snapshot()
        lut = np.load(light_lut_filename)['arr']

        # clip LUT so that no voxel contains 0 visibility
        mask = lut['vis'] > 0
        lut['vis'][~mask] = lut['vis'][mask].min()

        lut = to_device(lut)

        light_noise = cp.load(light_det_noise_filename)

        TPB = 256
        BPG = max(ceil(tracks.shape[0] / TPB),1)
        lightLUT.calculate_light_incidence[BPG,TPB](tracks, lut, light_sim_dat, track_light_voxel)
        logger.take_snapshot()
        logger.archive('light')
        print(f" {time()-start_light_time:.2f} s")

    # Restart the memory logger for the electronics simulation loop
    logger.start()
    logger.take_snapshot()

    # Create a lookup table for event timestamps.

    # Event IDs may have some offset (e.g. to make them globally unique within
    # an MC production), which we assume to be a multiple of
    # sim.MAX_EVENTS_PER_FILE. We remove this offset by taking the modulus with
    # sim.MAX_EVENTS_PER_FILE, which gives us zero-based "local" event IDs that
    # we can use when indexing into event_times. Note that num_evids is actually
    # an upper bound on the number of events, since there may be gaps due to
    # events that didn't deposit any energy in the LAr. Such gaps are harmless.
    num_evids = (tracks[sim.EVENT_SEPARATOR].max() % sim.MAX_EVENTS_PER_FILE) + 1
    if sim.IS_SPILL_SIM:
        event_times = cp.arange(num_evids) * sim.SPILL_PERIOD
    else:
        event_times = fee.gen_event_times(num_evids, 0)

    if input_has_vertices and not sim.IS_SPILL_SIM:
        # create "t_event" in vertices dataset in case it doesn't exist
        if 't_event' not in vertices.dtype.names:
            dtype = vertices.dtype.descr
            dtype = [("t_event","f4")] + dtype
            new_vertices = np.empty(vertices.shape, dtype=np.dtype(dtype, align=True))
            for field in dtype[1:]:
                new_vertices[field[0]] = vertices[field[0]]
            vertices = new_vertices
        uniq_ev, counts = np.unique(vertices[sim.EVENT_SEPARATOR], return_counts=True)
        event_times_in_use = cp.take(event_times, uniq_ev)
        vertices['t_event'] = np.repeat(event_times_in_use.get(),counts)

    if sim.IS_SPILL_SIM:
        # write the true timing structure to the file, not t0 wrt event time .....
        tracks['t0_start'] = tracks['t0_start'] + localSpillIDs*sim.SPILL_PERIOD
        tracks['t0_end'] = tracks['t0_end'] + localSpillIDs*sim.SPILL_PERIOD
        tracks['t0'] = tracks['t0'] + localSpillIDs*sim.SPILL_PERIOD

    # prep output file with truth datasets
    with h5py.File(output_filename, 'a') as output_file:
        # We previously called swap_coordinates(tracks), but we want to write
        # all truth info in the edep-sim convention (z = beam coordinate). So
        # temporarily undo the swap. It's easier than reorganizing the code!
        swap_coordinates(tracks)
        output_file.create_dataset(sim.TRACKS_DSET_NAME, data=tracks)
        # To distinguish from the "old" files that had z=drift in 'tracks':
        output_file[sim.TRACKS_DSET_NAME].attrs['zbeam'] = True
        swap_coordinates(tracks)

        if light.LIGHT_SIMULATED:
            output_file.create_dataset('light_dat', data=light_sim_dat)
        if input_has_trajectories:
            output_file.create_dataset("trajectories", data=trajectories)
        if input_has_vertices:
            output_file.create_dataset("vertices", data=vertices)
        if input_has_mc_hdr:
            output_file.create_dataset("mc_hdr", data=mc_hdr)
        if input_has_mc_stack:
            output_file.create_dataset("mc_stack", data=mc_stack)

    if sim.IS_SPILL_SIM:
        # ..... even thought larnd-sim does expect t0 to be given with respect to
        # the event time
        tracks['t0_start'] = tracks['t0_start'] - localSpillIDs*sim.SPILL_PERIOD
        tracks['t0_end'] = tracks['t0_end'] - localSpillIDs*sim.SPILL_PERIOD
        tracks['t0'] = tracks['t0'] - localSpillIDs*sim.SPILL_PERIOD


    # create a lookup table that maps between unique event ids and the segments in the file
    track_ids = cp.array(np.arange(len(tracks)), dtype='i4')
    # copy to device
    track_ids = cp.asarray(np.arange(segment_ids.shape[0], dtype=int))

    # We divide the sample in portions that can be processed by the GPU
    step = 1

    # accumulate results for periodic file saving
    results_acc = defaultdict(list)
    def save_results(event_times, is_first_batch, results):
        '''
        results is a dictionary with the following keys

         for the charge simulation
         - event_id: event id for each hit
         - adc_tot: adc value for each hit
         - adc_tot_ticks: timestamp for each hit
         - track_pixel_map: map from track to active pixels
         - unique_pix: all unique pixels (per track?)
         - current_fractions: fraction of charge associated with each true track

         for the light simulation (in addition to all keys for the charge simulation)
         - light_event_id: event_id for each light trigger
         - light_start_time: simulation start time for event
         - light_trigger_idx: time tick at which each trigger occurs
         - light_op_channel_idx: optical channel id for each waveform
         - light_waveforms: waveforms of each light trigger
         - light_waveforms_true_track_id: true track ids for each tick in each waveform
         - light_waveforms_true_photons: equivalent pe for each track at each tick in each waveform
        
        returns is_first_batch = False
        
        Note: can't handle empty inputs
        '''
        for key in list(results.keys()):
            results[key] = np.concatenate([cp.asnumpy(arr) for arr in results[key]], axis=0)

        uniq_events = cp.asnumpy(np.unique(results['event_id']))
        uniq_event_times = cp.asnumpy(event_times[uniq_events % sim.MAX_EVENTS_PER_FILE])
        if light.LIGHT_SIMULATED:
            # prep arrays for embedded triggers in charge data stream
            light_trigger_modules = np.array([detector.TPC_TO_MODULE[tpc] for tpc in light.OP_CHANNEL_TO_TPC[results['light_op_channel_idx']][:,0]])
            if light.LIGHT_TRIG_MODE == 1:
                light_trigger_modules = np.array(results['trigger_type']+1)
            light_trigger_times = results['light_start_time'] + results['light_trigger_idx'] * light.LIGHT_TICK_SIZE
            light_trigger_event_ids = results['light_event_id']
        else:
            # prep arrays for embedded triggers in charge data stream (each event triggers once at perfect t0)
            light_trigger_modules = np.ones(len(uniq_events))
            light_trigger_times = np.zeros_like(uniq_event_times)
            light_trigger_event_ids = uniq_events

        fee.export_to_hdf5(results['event_id'],
                           results['adc_tot'],
                           results['adc_tot_ticks'],
                           results['unique_pix'],
                           results['current_fractions'],
                           results['track_pixel_map'],
                           output_filename, # defined earlier in script
                           uniq_event_times,
                           is_first_batch=is_first_batch,
                           light_trigger_times=light_trigger_times,
                           light_trigger_event_id=light_trigger_event_ids,
                           light_trigger_modules=light_trigger_modules,
                           bad_channels=bad_channels) # defined earlier in script

        if light.LIGHT_SIMULATED and len(results['light_event_id']):
            light_sim.export_to_hdf5(results['light_event_id'],
                                     results['light_start_time'],
                                     results['light_trigger_idx'],
                                     results['light_op_channel_idx'],
                                     results['light_waveforms'],
                                     output_filename,
                                     #cp.asnumpy(event_times[np.unique(results['light_event_id'])]),
                                     uniq_event_times,
                                     results['light_waveforms_true_track_id'],
                                     results['light_waveforms_true_photons'])
            #print('results["light_waveforms_true_track_id"] shape:',results['light_waveforms_true_track_id'].shape)
            #print('results["light_waveforms_true_track_id"] first value:',results['light_waveforms_true_track_id'][0])
            #print('waveforms_true_track_id shape:',waveforms_true_track_id.shape)
            #print('waveforms_true_track_id size:',waveforms_true_track_id.size)
            #print('waveforms_true_photons shape:',waveforms_true_photons.shape)
            #print('waveforms_true_photons size:',waveforms_true_photons.size)
        if is_first_batch:
            is_first_batch = False
        return is_first_batch
    logger.take_snapshot()
    logger.archive('preparation2')


    is_first_batch = True
    logger.start()
    logger.take_snapshot([0])
    for batch_mask in tqdm(batching.TPCBatcher(tracks, sim.EVENT_SEPARATOR, tpc_batch_size=sim.EVENT_BATCH_SIZE, tpc_borders=detector.TPC_BORDERS),
                           desc='Simulating batches...', ncols=80, smoothing=0):
        # grab only tracks from current batch
        track_subset = tracks[batch_mask]
        if len(track_subset) == 0:
            continue
        ievd = int(track_subset[0][sim.EVENT_SEPARATOR])
        evt_tracks = track_subset
        first_trk_id = np.argmax(batch_mask) # first track in batch

        for itrk in tqdm(range(0, evt_tracks.shape[0], sim.BATCH_SIZE),
                         delay=1, desc='  Simulating event %i batches...' % ievd, leave=False, ncols=80):
            if itrk > 0:
                warnings.warn(f"Entered sub-batch loop, results may not be accurate! Consider increasing batch_size (currently {sim.BATCH_SIZE}) in the simulation_properties file.")
                
            selected_tracks = evt_tracks[itrk:itrk+sim.BATCH_SIZE]

            RangePush("event_id_map")
            event_ids = selected_tracks[sim.EVENT_SEPARATOR]
            unique_eventIDs = np.unique(event_ids)
            RangePop()

            # We find the pixels intersected by the projection of the tracks on
            # the anode plane using the Bresenham's algorithm. We also take into
            # account the neighboring pixels, due to the transverse diffusion of the charges.
            RangePush("pixels_from_track")
            max_radius = ceil(max(selected_tracks["tran_diff"])*5/detector.PIXEL_PITCH)

            TPB = 128
            BPG = max(ceil(selected_tracks.shape[0] / TPB),1)
            max_pixels = np.array([0])
            pixels_from_track.max_pixels[BPG,TPB](selected_tracks, max_pixels)

            # This formula tries to estimate the maximum number of pixels which can have
            # a current induced on them.
            max_neighboring_pixels = (2*max_radius+1)*max_pixels[0]+(1+2*max_radius)*max_radius*2

            active_pixels = cp.full((selected_tracks.shape[0], max_pixels[0]), -1, dtype=np.int32)
            neighboring_pixels = cp.full((selected_tracks.shape[0], max_neighboring_pixels), -1, dtype=np.int32)
            n_pixels_list = cp.zeros(shape=(selected_tracks.shape[0]))

            if not active_pixels.shape[1] or not neighboring_pixels.shape[1]:
                continue

            pixels_from_track.get_pixels[BPG,TPB](selected_tracks,
                                                  active_pixels,
                                                  neighboring_pixels,
                                                  n_pixels_list,
                                                  max_radius)
            RangePop()

            RangePush("unique_pix")
            shapes = neighboring_pixels.shape
            joined = neighboring_pixels.reshape(shapes[0] * shapes[1])
            unique_pix = cp.unique(joined)
            unique_pix = unique_pix[(unique_pix != -1)]
            RangePop()

            if not unique_pix.shape[0]:
                continue

            RangePush("time_intervals")
            # Here we find the longest signal in time and we store an array with the start in time of each track
            max_length = cp.array([0])
            track_starts = cp.empty(selected_tracks.shape[0])
            detsim.time_intervals[BPG,TPB](track_starts, max_length, selected_tracks)
            RangePop()

            RangePush("tracks_current")
            # Here we calculate the induced current on each pixel
            signals = cp.zeros((selected_tracks.shape[0],
                                neighboring_pixels.shape[1],
                                cp.asnumpy(max_length)[0]), dtype=np.float32)
            TPB = (1,1,64)
            BPG_X = max(ceil(signals.shape[0] / TPB[0]),1)
            BPG_Y = max(ceil(signals.shape[1] / TPB[1]),1)
            BPG_Z = max(ceil(signals.shape[2] / TPB[2]),1)
            BPG = (BPG_X, BPG_Y, BPG_Z)
            rng_states = maybe_create_rng_states(int(np.prod(TPB[:2]) * np.prod(BPG[:2])), seed=rand_seed+ievd+itrk, rng_states=rng_states)
            detsim.tracks_current_mc[BPG,TPB](signals, neighboring_pixels, selected_tracks, response, rng_states)
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
            BPG = max(ceil(unique_pix.shape[0] / TPB),1)
            detsim.get_track_pixel_map[BPG, TPB](track_pixel_map, unique_pix, neighboring_pixels)
            RangePop()

            RangePush("sum_pixels_signals")
            # Here we combine the induced current on the same pixels by different tracks
            TPB = (1,1,64)
            BPG_X = max(ceil(signals.shape[0] / TPB[0]),1)
            BPG_Y = max(ceil(signals.shape[1] / TPB[1]),1)
            BPG_Z = max(ceil(signals.shape[2] / TPB[2]),1)
            BPG = (BPG_X, BPG_Y, BPG_Z)
            pixels_signals = cp.zeros((len(unique_pix), len(detector.TIME_TICKS)))
            pixels_tracks_signals = cp.zeros((len(unique_pix),
                                              len(detector.TIME_TICKS),
                                              track_pixel_map.shape[1]))
            detsim.sum_pixel_signals[BPG,TPB](pixels_signals,
                                              signals,
                                              track_starts,
                                              pixel_index_map,
                                              track_pixel_map,
                                              pixels_tracks_signals)
            RangePop()

            RangePush("get_adc_values")
            # Here we simulate the electronics response (the self-triggering cycle) and the signal digitization
            time_ticks = cp.linspace(0, len(unique_eventIDs) * detector.TIME_INTERVAL[1], pixels_signals.shape[1]+1)
            integral_list = cp.zeros((pixels_signals.shape[0], fee.MAX_ADC_VALUES))
            adc_ticks_list = cp.zeros((pixels_signals.shape[0], fee.MAX_ADC_VALUES))
            current_fractions = cp.zeros((pixels_signals.shape[0], fee.MAX_ADC_VALUES, track_pixel_map.shape[1]))

            TPB = 128
            BPG = ceil(pixels_signals.shape[0] / TPB)
            rng_states = maybe_create_rng_states(int(TPB * BPG), seed=rand_seed+ievd+itrk, rng_states=rng_states)
            pixel_thresholds_lut.tpb = TPB
            pixel_thresholds_lut.bpg = BPG
            pixel_thresholds = pixel_thresholds_lut[unique_pix.ravel()].reshape(unique_pix.shape)
            
            fee.get_adc_values[BPG, TPB](pixels_signals,
                                         pixels_tracks_signals,
                                         time_ticks,
                                         integral_list,
                                         adc_ticks_list,
                                         0,
                                         rng_states,
                                         current_fractions,
                                         pixel_thresholds)
            
            # get list of adc values
            if pixel_gains_file is not None:
                pixel_gains = cp.array(pixel_gains_lut[unique_pix.ravel()])
                gain_list = pixel_gains[:, cp.newaxis] * cp.ones((1, fee.MAX_ADC_VALUES)) # makes array the same shape as integral_list
                adc_list = fee.digitize(integral_list, gain_list)
            else:
                adc_list = fee.digitize(integral_list)
            
            adc_event_ids = np.full(adc_list.shape, unique_eventIDs[0]) # FIXME: only works if looping on a single event
            RangePop()

            results_acc['event_id'].append(adc_event_ids)
            results_acc['adc_tot'].append(adc_list)
            results_acc['adc_tot_ticks'].append(adc_ticks_list)
            results_acc['unique_pix'].append(unique_pix)
            results_acc['current_fractions'].append(current_fractions)
            #track_pixel_map[track_pixel_map != -1] += first_trk_id + itrk
            track_pixel_map[track_pixel_map != -1] = track_ids[batch_mask][track_pixel_map[track_pixel_map != -1] + itrk]
            results_acc['track_pixel_map'].append(track_pixel_map)

            # ~~~ Light detector response simulation ~~~
            if light.LIGHT_SIMULATED:
                RangePush("sum_light_signals")
                light_inc = light_sim_dat[batch_mask][itrk:itrk+sim.BATCH_SIZE]
                selected_track_id = track_ids[batch_mask][itrk:itrk+sim.BATCH_SIZE]
                n_light_ticks, light_t_start = light_sim.get_nticks(light_inc)
                n_light_ticks = min(n_light_ticks,int(5E4))
                op_channel = light_sim.get_active_op_channel(light_inc)

                n_light_det = op_channel.shape[0]
                light_sample_inc = cp.zeros((n_light_det,n_light_ticks), dtype='f4')
                #### CHANGING THINGS HERE!!!! ####
                tick_seg_backtrack_array = 3
                light_segment_inc = cp.zeros((n_light_det, tick_seg_backtrack_array), dtype='f4')
                ####
                light_sample_inc_true_track_id = cp.full((n_light_det, n_light_ticks, light.MAX_MC_TRUTH_IDS), -1, dtype='i8')
                light_sample_inc_true_photons = cp.zeros((n_light_det, n_light_ticks, light.MAX_MC_TRUTH_IDS), dtype='f8')
                # On the CPU CHANGE HERE
                n_idet = light_inc.shape[1]
                sorted_indices = cp.zeros((n_idet, light_inc.shape[0]), dtype=np.int32)

                for idet in range(n_idet):
                  sorted_indices[idet] = np.argsort(light_inc[:,idet]['n_photons_det'])[::-1]
                print(sorted_indices)
                TPB = (1,64)
                BPG = (max(ceil(light_sample_inc.shape[0] / TPB[0]),1),
                       max(ceil(light_sample_inc.shape[1] / TPB[1]),1))
                light_sim.sum_light_signals[BPG, TPB](
                    selected_tracks, track_light_voxel[batch_mask][itrk:itrk+sim.BATCH_SIZE], selected_track_id,
                    light_inc, op_channel, lut, light_t_start, light_sample_inc, light_sample_inc_true_track_id,
                    light_sample_inc_true_photons, sorted_indices)

                RangePop()
                if light_sample_inc_true_track_id.shape[-1] > 0 and cp.any(light_sample_inc_true_track_id[...,-1] != -1):
                    warnings.warn(f"Maximum number of true segments ({light.MAX_MC_TRUTH_IDS}) reached in backtracking info, consider increasing MAX_MC_TRUTH_IDS (larndsim/consts/light.py)")

                RangePush("sim_scintillation")
                light_sample_inc_scint = cp.zeros_like(light_sample_inc)
                light_sample_inc_scint_true_track_id = cp.full_like(light_sample_inc_true_track_id, -1)
                light_sample_inc_scint_true_photons = cp.zeros_like(light_sample_inc_true_photons)
                light_sim.calc_scintillation_effect[BPG, TPB](
                    light_sample_inc, light_sample_inc_true_track_id, light_sample_inc_true_photons, light_sample_inc_scint,
                    light_sample_inc_scint_true_track_id, light_sample_inc_scint_true_photons)

                light_sample_inc_disc = cp.zeros_like(light_sample_inc)
                rng_states = maybe_create_rng_states(int(np.prod(TPB) * np.prod(BPG)),
                                                     seed=rand_seed+ievd+itrk, rng_states=rng_states)
                light_sim.calc_stat_fluctuations[BPG, TPB](light_sample_inc_scint, light_sample_inc_disc, rng_states)
                RangePop()

                RangePush("sim_light_det_response")
                light_response = cp.zeros_like(light_sample_inc)
                #### CHANGING THINGS!!!! ####
                tick_segment_response = cp.zeros_like(light_segment_inc)
                ####
                light_response_true_track_id = cp.full_like(light_sample_inc_true_track_id, -1)
                light_response_true_photons = cp.zeros_like(light_sample_inc_true_photons)
                light_sim.calc_light_detector_response[BPG, TPB](
                    light_sample_inc_disc, light_sample_inc_scint_true_track_id, light_sample_inc_scint_true_photons,
                    light_response, light_response_true_track_id, light_response_true_photons)
                light_response += cp.array(light_sim.gen_light_detector_noise(light_response.shape, light_noise[op_channel.get()]))
                RangePop()

                RangePush("sim_light_triggers")
                light_threshold = cp.repeat(cp.array(light.LIGHT_TRIG_THRESHOLD)[...,np.newaxis], light.OP_CHANNEL_PER_TRIG, axis=-1)
                light_threshold = light_threshold.ravel()[op_channel.get()].copy()
                light_threshold = light_threshold.reshape(-1, light.OP_CHANNEL_PER_TRIG)[...,0]
                trigger_idx, trigger_op_channel_idx, trigger_type = light_sim.get_triggers(light_response, light_threshold, op_channel)
                digit_samples = ceil((light.LIGHT_TRIG_WINDOW[1] + light.LIGHT_TRIG_WINDOW[0]) / light.LIGHT_DIGIT_SAMPLE_SPACING)
                TPB = (1,1,64)
                BPG = (max(ceil(trigger_idx.shape[0] / TPB[0]),1),
                       max(ceil(trigger_op_channel_idx.shape[1] / TPB[1]),1),
                       max(ceil(digit_samples / TPB[2]),1))

                light_digit_signal, light_digit_signal_true_track_id, light_digit_signal_true_photons = light_sim.sim_triggers(
                    BPG, TPB, light_response, op_channel, light_response_true_track_id, light_response_true_photons, trigger_idx, trigger_op_channel_idx,
                    digit_samples, light_noise)
                RangePop()

                results_acc['light_event_id'].append(cp.full(trigger_idx.shape[0], unique_eventIDs[0])) # FIXME: only works if looping on a single event
                results_acc['light_start_time'].append(cp.full(trigger_idx.shape[0], light_t_start))
                results_acc['light_trigger_idx'].append(trigger_idx)
                results_acc['trigger_type'].append(trigger_type)
                results_acc['light_op_channel_idx'].append(trigger_op_channel_idx)
                results_acc['light_waveforms'].append(light_digit_signal)
                #print('light_digit_signal_true_track_id shape:',light_digit_signal_true_track_id.shape)
                #print('light_digit_signal_true_track_id first indice:',light_digit_signal_true_track_id[0])
                results_acc['light_waveforms_true_track_id'].append(light_digit_signal_true_track_id)
                #print('light_digit_signal_true_photons [0,0,0]:',light_digit_signal_true_photons[0,0,0])
                #print('light_digit_signal_true_photons last dimension shape:',light_digit_signal_true_photons[:,:,0].shape)
                results_acc['light_waveforms_true_photons'].append(light_digit_signal_true_photons)
        
        if len(results_acc['event_id']) >= sim.WRITE_BATCH_SIZE and len(np.concatenate(results_acc['event_id'], axis=0)) > 0:
            is_first_batch = save_results(event_times, is_first_batch, results=results_acc)
            results_acc = defaultdict(list)

        logger.take_snapshot([len(logger.log)])

    # Always save results after last iteration
    if len(results_acc['event_id']) >0 and len(np.concatenate(results_acc['event_id'], axis=0)) > 0:
        is_first_batch = save_results(event_times, is_first_batch, results=results_acc)

    logger.take_snapshot([len(logger.log)])

    with h5py.File(output_filename, 'a') as output_file:
        if 'configs' in output_file.keys():
            output_file['configs'].attrs['pixel_layout'] = pixel_layout

    print("Output saved in:", output_filename)

    RangePop()
    end_simulation = time()
    logger.take_snapshot([len(logger.log)])
    print(f"Elapsed time: {end_simulation-start_simulation:.2f} s")
    logger.archive('loop',['loop'])
    logger.store(save_memory)

if __name__ == "__main__":
    fire.Fire(run_simulation)
