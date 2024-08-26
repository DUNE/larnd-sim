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
from larndsim import active_volume, quenching, drifting, detsim, pixels_from_track, fee, lightLUT, light_sim
import importlib

from larndsim.util import CudaDict, batching, memory_logger
from larndsim.config import get_config

import os

SEED = int(time())

LOGO = """
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
                   output_filename,
                   config='2x2_mod2mod_variation',
                   mod2mod_variation=None,
                   pixel_layout=None,
                   detector_properties=None,
                   simulation_properties=None,
                   response_file=None,
                   light_simulated=None,
                   light_lut_filename=None,
                   light_det_noise_filename=None,
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
        output_filename (str): path of the HDF5 output file. If not specified
            the output is added to the input file.
        config (str, optional): a keyword to specify a configuration (all necessary meta data files)
        mod2mod_variation (bool): a flag indicating if load different configurations for different LArTPC modules
        pixel_layout (str): path of the YAML file containing the pixel
            layout and connection details.
        detector_properties (str): path of the YAML file containing
            the detector properties
        simulation_properties (str): path of the YAML file containing
            the simulation properties
        response_file (str): path of the Numpy array containing the pre-calculated
            field responses. 
        light_lut_file (str, optional): path of the Numpy array containing the light
            look-up table. 
        light_det_noise_filename (str, optional): path of the Numpy array containning the light noise information
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
    # Define a nested function to save the results
    def save_results(event_times, results, i_trig, i_mod=-1, light_only=False):
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
        
        Note: can't handle empty inputs
        '''
        for key in list(results.keys()):
            if isinstance(results[key], list) and len(results[key]) > 0: # we may have empty lists (e.g. for event_id) when light_only
                results[key] = np.concatenate([cp.asnumpy(arr) for arr in results[key]], axis=0)

        uniq_events = cp.asnumpy(np.unique(results['event_id'])) if not light_only else cp.asnumpy(np.unique(results['light_event_id']))
        uniq_event_times = cp.asnumpy(event_times[uniq_events % sim.MAX_EVENTS_PER_FILE])

        if not light_only:
            if light.LIGHT_SIMULATED:
                # prep arrays for embedded triggers in charge data stream
                light_trigger_modules = np.array([detector.TPC_TO_MODULE[tpc] for tpc in light.OP_CHANNEL_TO_TPC[results['light_op_channel_idx']][:,0]])
                if light.LIGHT_TRIG_MODE == 1:
                    light_trigger_modules = np.array(results['trigger_type'])
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
                               results['traj_pixel_map'],
                               output_filename, # defined earlier in script
                               uniq_event_times,
                               light_trigger_times=light_trigger_times,
                               light_trigger_event_id=light_trigger_event_ids,
                               light_trigger_modules=light_trigger_modules,
                               bad_channels=bad_channels, # defined earlier in script
                               i_mod=i_mod)

        if light.LIGHT_SIMULATED and len(results['light_event_id']):
            if light.LIGHT_TRIG_MODE == 0:
                light_sim.export_to_hdf5(results['light_event_id'],
                                         results['light_start_time'],
                                         results['light_trigger_idx'],
                                         results['light_op_channel_idx'],
                                         results['light_waveforms'],
                                         output_filename,
                                         uniq_event_times,
                                         results['light_waveforms_true_track_id'],
                                         results['light_waveforms_true_photons'],
                                         i_trig,
                                         i_mod)
            elif light.LIGHT_TRIG_MODE == 1:
                light_sim.export_light_wvfm_to_hdf5(results['light_event_id'],
                                                    results['light_waveforms'],
                                                    output_filename,
                                                    results['light_waveforms_true_track_id'],
                                                    results['light_waveforms_true_photons'],
                                                    i_trig,
                                                    i_mod)
    ###########################################################################################

    print(LOGO)
    print("**************************\nLOADING SETTINGS AND INPUT\n**************************")

    if not os.path.exists(input_filename):
        raise Exception(f'Input file {input_filename} does not exist.')
    if os.path.exists(output_filename):
        raise Exception(f'Output file {output_filename} already exists.')

    # Set the input (meta data) files
    cfg = get_config(config)
    if pixel_layout is None:
        pixel_layout = cfg['PIXEL_LAYOUT']
    if detector_properties is None:
        detector_properties = cfg['DET_PROPERTIES']
    if response_file is None:
        response_file = cfg['RESPONSE']
    if simulation_properties is None:
        simulation_properties = cfg['SIM_PROPERTIES']
    if light_simulated is None:
        try:
            light_simulated = cfg['LIGHT_SIMULATED']
        except:
            print("The configuration has not specify wether to simulate light. By default the light simulation is activated.")
    if light_simulated is True or None:
        if light_lut_filename is None:
            try:
                light_lut_filename = cfg['LIGHT_LUT']
                if isinstance(light_lut_filename, list):
                    for i_light_lut, f_light_lut in enumerate(light_lut_filename):
                        if not os.path.isfile(f_light_lut):
                            light_lut_filename[i_light_lut] = "larndsim/bin/lightLUT.npz" # the default 2x2 module light lookup table
                            warnings.warn("Path to light LUT in the configuration file is not valid. Switching to the default 2x2 module light LUT in larnd-sim now...")
                else:
                    if not os.path.isfile(light_lut_filename):
                        light_lut_filename = "larndsim/bin/lightLUT.npz" # the default 2x2 module light lookup table
                        warnings.warn("Path to light LUT in the configuration file is not valid. Switching to the default 2x2 module light LUT in larnd-sim now...")
            except:
                print("light_lut_filename is not provided (required if light_simulated is True)")
        if light_det_noise_filename is None:
            try:
                light_det_noise_filename = cfg['LIGHT_DET_NOISE']
            except:
                print("light_det_noise_filename is not provided (required if light_simulated is True)")

    # Assert necessary ones
    assert pixel_layout, 'pixel_layout (file) must be specified.'
    assert simulation_properties, 'simulation_properties (file) must be specified'
    assert detector_properties, 'detector_properties (file) must be specified'
    assert response_file, 'response_file must be specified'

    # Print configuration files
    # Shall we give an option to turn of the print out?
    print("")
    print("edep-sim input file:", input_filename)
    print("larnd-sim output file:", output_filename)
    print("")
    print("Random seed:", rand_seed)
    print("Simulation properties file:", simulation_properties)
    print("Detector properties file:", detector_properties)
    print("Pixel layout file:", pixel_layout)
    print("Response file:", response_file)
    if light_lut_filename:
        print("Light LUT:", light_lut_filename)
    if light_det_noise_filename:
        print("Light detector noise: ", light_det_noise_filename)
    if bad_channels:
        print("Disabled channel list: ", bad_channels)
    if save_memory:
        print('Recording the process resource log:', save_memory)
    else:
        print('Memory resource log will not be recorded')

    # Get number of modules in the simulation
    mod_ids = consts.detector.get_n_modules(detector_properties)
    n_modules = len(mod_ids)

    if mod2mod_variation is None:
        try:
            mod2mod_variation = cfg['MOD2MOD_VARIATION']
        except:
            print("The configuration has not specify wether to load different configurations for different modules. By default all the modules (if more than one simulated) are loaded with the same configuration.")

    if mod2mod_variation is True:
        if n_modules == 1:
            warnings.warn("Simulating one module with module variation activated! \nDeactivating module variation...")
            mod2mod_variation = False
        if (isinstance(pixel_layout, str) or len(pixel_layout) == 1) and (isinstance(response_file, str) or len(response_file) == 1) and (isinstance(light_lut_filename, str) or len(light_lut_filename) == 1):
            warnings.warn("Simulation with module variation activated, but only provided a single set of configuration files of pixel layout, induction response and light lookup table! \nDeactivating module variation...")
            mod2mod_variation = False

    if mod2mod_variation == True:
        # Load the index for pixel layout, response and LUT
        try:
            pixel_layout_id = cfg['PIXEL_LAYOUT_ID']
            if not isinstance(pixel_layout, list) or len(pixel_layout_id) != n_modules or max(pixel_layout_id) >= len(pixel_layout):
                raise KeyError("Simulation with module variation activated, but the number of pointer for pixel layout is incorrect!")
            else:
                module_pixel_layout = [pixel_layout[idx] for idx in pixel_layout_id]
                pixel_layout = module_pixel_layout
        except:
            if isinstance(pixel_layout, list) and len(pixel_layout) != n_modules:
                raise KeyError("Simulation with module variation activated, but the number of pixel layout files is incorrect!")
            elif isinstance(pixel_layout, list) and len(pixel_layout) == n_modules:
                warnings.warn("Simulation with module variation activated, using default orders for the pixel layout files.")

        try:
            response_id = cfg['RESPONSE_ID']
            if not isinstance(response_file, list) or len(response_id) != n_modules or max(response_id) >= len(response_file):
                raise KeyError("Simulation with module variation activated, but the number of pointer for response files is incorrect!")
            else:
                module_response_file = [response_file[idx] for idx in response_id]
                response_file = module_response_file
        except:
            if isinstance(response_file, list) and len(response_file) != n_modules:
                raise KeyError("Simulation with module variation activated, but the number of response files is incorrect!")
            elif isinstance(response_file, list) and len(response_file) == n_modules:
                warnings.warn("Simulation with module variation activated, using default orders for the response files.")

        if light_simulated is True or None:
            try:
                light_lut_id = cfg['LIGHT_LUT_ID']
                if not isinstance(light_lut_filename, list) or len(light_lut_id) != n_modules or max(light_lut_id) >= len(light_lut_filename):
                    raise KeyError("Simulation with module variation activated, but the number of pointer for light LUT is incorrect!")
                else:
                    module_light_lut_filename = [light_lut_filename[idx] for idx in light_lut_id]
                    light_lut_filename = module_light_lut_filename
            except:
                if isinstance(light_lut_filename, list) and len(light_lut_filename) != n_modules:
                    raise KeyError("Simulation with module variation activated, but the number of light LUT is incorrect!")
                elif isinstance(light_lut_filename, list) and len(light_lut_filename) == n_modules:
                    warnings.warn("Simulation with module variation activated, using default orders for the light LUT.")
        
        if cfg['PIXEL_LAYOUT_ID'] and cfg['RESPONSE_ID']:
            if cfg['PIXEL_LAYOUT_ID'] != cfg['RESPONSE_ID']:
                warnings.warn("Simulation with module variation activated, the pixel layout and response files may not be consistent with each other. Please double check!")

    logger = memory_logger(save_memory is None)
    logger.start()
    logger.take_snapshot()
    start_simulation = time()

    RangePush("set_random_seed")
    # set up random seed for larnd-sim
    if not rand_seed: rand_seed = SEED
    cp.random.seed(rand_seed)
    # pre-allocate some random number states for custom kernels
    rng_states = maybe_create_rng_states(1024*256, seed=rand_seed)
    RangePop()

    RangePush("load_properties")

    # if n_modules == 1, mod2mod_variation would have already been set to False
    if not mod2mod_variation:
        # Check if the configrations are consistent
        # Allow configuration to be provided as a string or a single element list
        if isinstance(pixel_layout, list) and len(pixel_layout) > 1:
            raise KeyError("Provided more than one pixel layout file for the simulation with no module variation.")
        elif isinstance(pixel_layout, list) and len(pixel_layout) == 1:
            pixel_layout = pixel_layout[0]

        if isinstance(response_file, list) and len(response_file) > 1:
            raise KeyError("Provided more than one response file for the simulation with no module variation.")
        elif isinstance(response_file, list) and len(response_file) == 1:
            response_file = response_file[0]

        if isinstance(pixel_thresholds_file, list) and len(pixel_thresholds_file) > 1:
            raise KeyError("Provided more than one pixel threshold file for the simulation with no module variation.")
        elif isinstance(pixel_thresholds_file, list) and len(pixel_thresholds_file) == 1:
            pixel_thresholds_file = pixel_thresholds_file[0]

        if isinstance(pixel_gains_file, list) and len(pixel_gains_file) > 1:
            raise KeyError("Provided more than one pixel gain file for the simulation with no module variation.")
        elif isinstance(pixel_gains_file, list) and len(pixel_gains_file) == 1:
            pixel_gains_file = pixel_gains_file[0]

        if isinstance(light_lut_filename, list) and len(light_lut_filename) > 1:
            raise KeyError("Provided more than one light lookup table for the simulation with no module variation.")
        elif isinstance(light_lut_filename, list) and len(light_lut_filename) == 1:
            light_lut_filename = light_lut_filename[0]

        RangePush("load_detector_properties")
        consts.load_properties(detector_properties, pixel_layout, simulation_properties)
        from larndsim.consts import light, detector, physics, sim
        RangePop()

        RangePush("load_induction_response")
        response = cp.load(response_file)
        RangePop()
    else:
        consts.light.set_light_properties(detector_properties)
        consts.sim.set_simulation_properties(simulation_properties)
        from larndsim.consts import light, physics, sim

    # set the value for the global variable MOD2MOD_VARIATION
    sim.MOD2MOD_VARIATION = mod2mod_variation

    # reload after the variables been defined (has been loaded for the first time at the top)
    importlib.reload(pixels_from_track)
    importlib.reload(active_volume)
    importlib.reload(detsim)
    importlib.reload(light_sim)
    importlib.reload(lightLUT)
    importlib.reload(fee)

    #if light.LIGHT_TRIG_MODE == 1 and not sim.IS_SPILL_SIM:
    #    raise ValueError("The simulation property indicates it is not beam simulation, but the light trigger mode is set to the beam trigger mode!")

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

    RangePush("set_if_simulate_light")
    if light_simulated is not None:
        light.LIGHT_SIMULATED = light_simulated
    RangePop()

    RangePop()                  # load_properties

    RangePush("load_hd5_file")
    print("Loading track segments..." , end="")
    start_load = time()
    # First of all we load the edep-sim output
    with h5py.File(input_filename, 'r') as f:
        tracks = np.array(f['segments'])
        if 'segment_id' in tracks.dtype.names:
            segment_ids = tracks['segment_id']
            trajectory_ids = tracks['file_traj_id']
        else:
            dtype = tracks.dtype.descr
            dtype = [('segment_id','u4')] + dtype
            new_tracks = np.empty(tracks.shape, dtype=np.dtype(dtype, align=True))
            new_tracks['segment_id'] = np.arange(tracks.shape[0], dtype='u4')
            for field in dtype[1:]:
                new_tracks[field[0]] = tracks[field[0]]
            tracks = new_tracks
            segment_ids = tracks['segment_id']
            trajectory_ids = tracks['file_traj_id']

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

    logger.take_snapshot()
    logger.archive('loading')

    logger.start()
    logger.take_snapshot()
    # Reduce dataset if not all events are to be simulated, being careful of gaps
    if n_events:
        print(f'Selecting only the first {n_events} events for simulation.')
        max_eventID = np.unique(tracks[sim.EVENT_SEPARATOR])[n_events-1]
        segment_ids = segment_ids[tracks[sim.EVENT_SEPARATOR] <= max_eventID]
        trajectory_ids = trajectory_ids[tracks[sim.EVENT_SEPARATOR] <= max_eventID]
        tracks = tracks[tracks[sim.EVENT_SEPARATOR] <= max_eventID]

        if input_has_trajectories:
            trajectories = trajectories[trajectories[sim.EVENT_SEPARATOR] <= max_eventID]
        if input_has_vertices:
            vertices = vertices[vertices[sim.EVENT_SEPARATOR] <= max_eventID]
        if input_has_mc_hdr:
            mc_hdr = mc_hdr[mc_hdr[sim.EVENT_SEPARATOR] <= max_eventID]
        if input_has_mc_stack:
            mc_stack = mc_stack[mc_stack[sim.EVENT_SEPARATOR] <= max_eventID]

    # Make "n_photons" attribute, if it doesn't exist
    if 'n_photons' not in tracks.dtype.names:
        n_photons = np.zeros(tracks.shape[0], dtype=[('n_photons', 'f4')])
        tracks = rfn.merge_arrays((tracks, n_photons), flatten=True)

    # Make "t0" attribute, if it doesn't exist
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

    # larnd-sim uses "t0" in a way that 0 is the "trigger" time (e.g spill time)
    # Therefore, to run the detector simulation we reset the t0 to reflect that
    # When storing the mc truth, revert this change and store the "real" segment time
    # The event times are added to segments in the spill building stage. This step is not needed for non-beam simulation
    if sim.IS_SPILL_SIM:
        # "Reset" the spill period so t0 is wrt the corresponding spill start time.
        # The spill starts are marking the start of
        # The space between spills will be accounted for in the
        # packet timestamps through the event_times array below
        localSpillIDs = tracks[sim.EVENT_SEPARATOR] - (tracks[sim.EVENT_SEPARATOR] // sim.MAX_EVENTS_PER_FILE) * sim.MAX_EVENTS_PER_FILE
        tracks['t0_start'] = tracks['t0_start'] - localSpillIDs*sim.SPILL_PERIOD
        tracks['t0_end'] = tracks['t0_end'] - localSpillIDs*sim.SPILL_PERIOD
        tracks['t0'] = tracks['t0'] - localSpillIDs*sim.SPILL_PERIOD

    # Here we swap the x and z coordinates of the tracks
    # because of the different convention in larnd-sim wrt edep-sim
    # When storing the mc truth, revert this change to have z as the beam direction and x as the drift axis
    tracks = swap_coordinates(tracks)
    
    logger.take_snapshot()
    logger.archive('preparation')

    RangePop()                  # load_hdf5_file
    end_load = time()
    print(f"Data preparation time: {end_load-start_load:.2f} s")

    print("******************\nRUNNING SIMULATION\n******************")
    RangePush("prep_simulation")
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

    # broadcast the event times to vertices
    if input_has_vertices and not sim.IS_SPILL_SIM:
        # create "t_event" in vertices dataset in case it doesn't exist
        if 't_event' not in vertices.dtype.names:
            dtype = vertices.dtype.descr
            dtype = [("t_event","f4")] + dtype
            new_vertices = np.empty(vertices.shape, dtype=np.dtype(dtype, align=True))
            for field in dtype[1:]:
                if len(field[0]) == 0: continue
                new_vertices[field[0]] = vertices[field[0]]
            vertices = new_vertices
        uniq_ev, counts = np.unique(vertices[sim.EVENT_SEPARATOR], return_counts=True)
        event_times_in_use = cp.take(event_times, uniq_ev)
        vertices['t_event'] = np.repeat(event_times_in_use.get(),counts)

    # copy the event times to mc_hdr
    if input_has_mc_hdr and input_has_vertices:
        if 't_event' not in mc_hdr.dtype.names:
            dtype = mc_hdr.dtype.descr
            dtype = [("t_event","f4")] + dtype
            new_mc_hdr = np.empty(mc_hdr.shape, dtype=np.dtype(dtype, align=True))
            for field in dtype[1:]:
                if len(field[0]) == 0: continue
                new_mc_hdr[field[0]] = mc_hdr[field[0]]
            mc_hdr = new_mc_hdr
        mc_hdr['t_event'] = vertices['t_event']
        if len(vertices[sim.EVENT_SEPARATOR]) != len(mc_hdr[sim.EVENT_SEPARATOR]):
            raise ValueError("vertices and mc_hdr datasets have different number of vertices! The number should be the same.")

    # accumulate results for periodic file saving
    results_acc = defaultdict(list)
    light_sim_dat_acc = list()

    # Allow module to module variance in the configuration files
    # First copy all tracks and segment_ids
    all_mod_tracks = tracks
    all_mod_segment_ids = segment_ids
    all_mod_trajectory_ids = trajectory_ids
    if mod2mod_variation == None or mod2mod_variation == False:
        mod_ids = [-1]
    else:
        mod_ids = consts.detector.get_n_modules(detector_properties)

    # If mod2mod variation, we load detector properties to get detector.TPC_BORDERS
    # For this purpose, it doesn't matter which pixel_layout to use
    if mod2mod_variation:
        consts.detector.set_detector_properties(detector_properties, pixel_layout[0])
        from larndsim.consts import detector

    # Sub-select only segments in active volumes
    print("Skipping non-active volumes..." , end="")
    start_mask = time()
    active_tracks_mask = active_volume.select_active_volume(all_mod_tracks, detector.TPC_BORDERS)
    tracks = all_mod_tracks = all_mod_tracks[active_tracks_mask]
    segment_ids = all_mod_segment_ids = all_mod_segment_ids[active_tracks_mask]
    trajectory_ids = all_mod_trajectory_ids[active_tracks_mask]
    end_mask = time()
    print(f" {end_mask-start_mask:.2f} s")

    RangePop()                  # prep_simulation

    # Convention module counting start from 1
    # Loop over all modules
    for i_mod in mod_ids:
        print(f'Simulating module {i_mod-1}')
        if mod2mod_variation:
            consts.detector.set_detector_properties(detector_properties, pixel_layout, i_mod)
            # Currently shouln't be necessary to reload light props, but if
            # someone later updates `set_light_properties` to use stuff from the
            # `consts.detector` module, we'll be glad for this line:
            consts.light.set_light_properties(detector_properties)
            from larndsim.consts import detector
            # reload after the variables been defined/updated; first imported at the top
            importlib.reload(pixels_from_track)
            importlib.reload(active_volume)
            importlib.reload(detsim)
            importlib.reload(light_sim)
            importlib.reload(lightLUT)
            importlib.reload(fee)

            RangePush("load_module_induction_response")
            response = cp.load(response_file[i_mod-1])
            RangePop()

            RangePush("load_segments_in_module")
            module_borders = detector.TPC_BORDERS[(i_mod-1)*2: i_mod*2]
            module_tracks_mask = active_volume.select_active_volume(all_mod_tracks, module_borders)
            tracks = all_mod_tracks[module_tracks_mask]
            segment_ids = all_mod_segment_ids[module_tracks_mask]
            trajectory_ids = all_mod_trajectory_ids[module_tracks_mask]
            RangePop()

        # find the module that triggers
        io_groups = np.array(list(consts.detector.MODULE_TO_IO_GROUPS.values()))
        if light.LIGHT_TRIG_MODE == 0 or light.LIGHT_TRIG_MODE == 1:
            trig_module = np.argwhere(io_groups==fee.get_trig_io())[0][0] + 1 # module id (i_mod) counts from 1

        RangePush("run_simulation")
        TPB = 256
        BPG = max(ceil(tracks.shape[0] / TPB),1)

        # We calculate the number of electrons after recombination (quenching module)
        # and the position and number of electrons after drifting (drifting module)
        print("Quenching electrons..." , end="")
        logger.start()
        logger.take_snapshot()
        start_quenching = time()
        quenching.quench[BPG,TPB](tracks, physics.BIRKS)
        end_quenching = time()
        logger.take_snapshot()
        logger.archive(f'quenching_mod{i_mod}')
        print(f" {end_quenching-start_quenching:.2f} s")

        print("Drifting electrons...", end="")
        start_drifting = time()
        logger.start()
        logger.take_snapshot()
        drifting.drift[BPG,TPB](tracks)
        end_drifting = time()
        logger.take_snapshot()
        logger.archive(f'drifting_mod{i_mod}')
        print(f" {end_drifting-start_drifting:.2f} s")

        # Set up light simulation data objects and calculate the optical responses
        if light.LIGHT_SIMULATED:
            n_light_channel = int(light.N_OP_CHANNEL/len(mod_ids)) if mod2mod_variation else light.N_OP_CHANNEL
            light_sim_dat = np.zeros([len(tracks), n_light_channel],
                                     dtype=[('segment_id', 'u4'), ('n_photons_det','f4'),('t0_det','f4')])
            light_sim_dat['segment_id'] = segment_ids[..., np.newaxis]
            track_light_voxel = np.zeros([len(tracks), 3], dtype='i4')

            print("Calculating optical responses...", end="")
            start_light_time = time()
            logger.start()
            logger.take_snapshot()
            light_lut = light_lut_filename[i_mod-1] if mod2mod_variation else light_lut_filename
            lut = np.load(light_lut)['arr']

            # check if the light LUT matches with the number of optical channels
            # lut (x, y, z, n_op_ch) for one TPC
            # n_light_channel is for one module or all modules depending if the mod2mod_variation is enabled
            if mod2mod_variation:
                warn_n_op_ch = (n_light_channel != lut.shape[3]*2)
            else:
                warn_n_op_ch = (n_light_channel != lut.shape[3]*2*n_modules)
            if warn_n_op_ch:
                warnings.warn("The light LUT has different number of optical channels than we expected in one TPC!")

            # clip LUT so that no voxel contains 0 visibility
            mask = lut['vis'] > 0
            lut['vis'][~mask] = lut['vis'][mask].min()

            lut = to_device(lut)

            if mod2mod_variation:
                light_noise = cp.load(light_det_noise_filename)[n_light_channel*(i_mod-1):n_light_channel*i_mod]
            else:
                light_noise = cp.load(light_det_noise_filename)

            RangePush('calculate_light_incidence')
            TPB = 256
            BPG = max(ceil(tracks.shape[0] / TPB),1)
            lightLUT.calculate_light_incidence[BPG,TPB](tracks, lut, light_sim_dat, track_light_voxel)
            RangePop()

            light_sim_dat_acc.append(light_sim_dat)

            logger.take_snapshot()
            logger.archive(f'light_mod{i_mod}')

            # Prepare the light waveform padding
            if light.LIGHT_SIMULATED and (light.LIGHT_TRIG_MODE == 0 or light.LIGHT_TRIG_MODE == 1):
                null_light_results_acc = defaultdict(list)
                trigger_idx = cp.array([0], dtype=int)
                op_channel = light.TPC_TO_OP_CHANNEL[:2].ravel() if mod2mod_variation else light.TPC_TO_OP_CHANNEL[:].ravel()
                op_channel = cp.array(op_channel)
                trigger_op_channel_idx = cp.repeat(np.expand_dims(op_channel, axis=0), len(trigger_idx), axis=0)
                digit_samples = ceil((light.LIGHT_TRIG_WINDOW[1] + light.LIGHT_TRIG_WINDOW[0]) / light.LIGHT_DIGIT_SAMPLE_SPACING)

                n_light_det = op_channel.shape[0]
                n_light_ticks = int((light.LIGHT_WINDOW[1] + light.LIGHT_WINDOW[0])/light.LIGHT_TICK_SIZE)

                light_response = cp.zeros((n_light_det,n_light_ticks), dtype='f4')
                #light_response += cp.array(light_sim.gen_light_detector_noise(light_response.shape, light_noise[op_channel.get()]))
                light_response_true_track_id = cp.full((n_light_det, n_light_ticks, light.MAX_MC_TRUTH_IDS), -1, dtype='i8')
                light_response_true_photons = cp.zeros((n_light_det, n_light_ticks, light.MAX_MC_TRUTH_IDS), dtype='f8')

                RangePush('light_sim_triggers')
                TPB = (1,1,64)
                BPG = (max(ceil(trigger_idx.shape[0] / TPB[0]),1),
                       max(ceil(len(op_channel) / TPB[1]),1),
                       max(ceil(digit_samples / TPB[2]),1))
                light_digit_signal, light_digit_signal_true_track_id, light_digit_signal_true_photons = light_sim.sim_triggers(
                    BPG, TPB, light_response, op_channel, light_response_true_track_id, light_response_true_photons, trigger_idx, trigger_op_channel_idx,
                    digit_samples, light_noise)
                RangePop()

                light_t_start = 0
                trigger_type = cp.full(trigger_idx.shape[0], light.LIGHT_TRIG_MODE, dtype = int)

                #null_light_results_acc['light_event_id'].append(cp.full(trigger_idx.shape[0], ievd)) # FIXME: only works if looping on a single event
                null_light_results_acc['light_start_time'].append(cp.full(trigger_idx.shape[0], light_t_start))
                null_light_results_acc['light_trigger_idx'].append(trigger_idx)
                null_light_results_acc['trigger_type'].append(trigger_type)
                null_light_results_acc['light_op_channel_idx'].append(trigger_op_channel_idx)
                null_light_results_acc['light_waveforms'].append(light_digit_signal)
                null_light_results_acc['light_waveforms_true_track_id'].append(light_digit_signal_true_track_id)
                null_light_results_acc['light_waveforms_true_photons'].append(light_digit_signal_true_photons)

            print(f" {time()-start_light_time:.2f} s")

        # Restart the memory logger for the electronics simulation loop
        logger.start()
        logger.take_snapshot()

        segment_ids_arr = cp.asarray(segment_ids)
        trajectory_ids_arr = cp.asarray(trajectory_ids)

        # We divide the sample in portions that can be processed by the GPU
        is_new_event = True
        event_id_buffer = -1
        logger.start()
        logger.take_snapshot([0])
        i_batch = 0
        i_trig = 0
        sync_start = event_times[0] // (fee.CLOCK_RESET_PERIOD * fee.CLOCK_CYCLE) * (fee.CLOCK_RESET_PERIOD * fee.CLOCK_CYCLE) +  (fee.CLOCK_RESET_PERIOD * fee.CLOCK_CYCLE)
        det_borders = module_borders if mod2mod_variation else detector.TPC_BORDERS
        # Batching is carried out by simulating detector response with selected segments from X number of tpcs per event
        # X is set by "sim.EVENT_BATCH_SIZE" and can be any number
        for ievd, batch_mask in tqdm(batching.TPCBatcher(all_mod_tracks, tracks, sim.EVENT_SEPARATOR, tpc_batch_size=sim.EVENT_BATCH_SIZE, tpc_borders=det_borders),
                               desc='Simulating batches...', ncols=80, smoothing=0):
            i_batch = i_batch+1
            # Grab segments from the current batch
            # If there are no segments in the batch, we still check if we need to generate null light signals
            track_subset = tracks[batch_mask]
            evt_tracks = track_subset
            #first_trk_id = np.argmax(batch_mask) # first track in batch

            # this relies on that batching is done in the order of events
            if ievd > event_id_buffer:
                is_new_event = True
            else:
                is_new_event = False
            this_event_time = [event_times[ievd % sim.MAX_EVENTS_PER_FILE]]
            if is_new_event:
                # forward sync packets
                if this_event_time[0] - sync_start >= 0: # this is duplicate to "is_new_event"
                    sync_times = cp.arange(sync_start, this_event_time[0]+1, fee.CLOCK_RESET_PERIOD * fee.CLOCK_CYCLE) #us
                    #PSS Sync also resets the timestamp in the PACMAN controller, so all of the timestamps in the packs should read 1e7 (for PPS)
                    sync_times_export = cp.full( sync_times.shape, fee.CLOCK_RESET_PERIOD * fee.CLOCK_CYCLE) 
                    if len(sync_times) > 0:
                        fee.export_sync_to_hdf5(output_filename, sync_times_export, i_mod)
                        sync_start = sync_times[-1] + fee.CLOCK_RESET_PERIOD * fee.CLOCK_CYCLE
                # beam trigger is only forwarded to one specific pacman (defined in fee)
                if (light.LIGHT_TRIG_MODE == 0 or light.LIGHT_TRIG_MODE == 1) and (i_mod == trig_module or i_mod == -1):
                    fee.export_timestamp_trigger_to_hdf5(output_filename, this_event_time, i_mod)

            # generate light waveforms for null signal in the module
            # so we can have light waveforms in this case (if the whole detector is triggered together)
            if len(track_subset) == 0:
                if light.LIGHT_SIMULATED and (light.LIGHT_TRIG_MODE == 0 or light.LIGHT_TRIG_MODE == 1):
                    null_light_results_acc['light_event_id'].append(cp.full(1, ievd)) # one event
                    save_results(event_times, null_light_results_acc, i_trig, i_mod, light_only=True)
                    i_trig += 1 # add to the trigger counter
                    del null_light_results_acc['light_event_id']
                # Nothing to simulate for charge readout?
                continue
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
                RangePush("max_pixels")
                max_radius = ceil(max(selected_tracks["tran_diff"])*5/detector.PIXEL_PITCH)

                TPB = 128
                BPG = max(ceil(selected_tracks.shape[0] / TPB),1)
                max_pixels = np.array([0])
                pixels_from_track.max_pixels[BPG,TPB](selected_tracks, max_pixels)
                RangePop()

                # This formula tries to estimate the maximum number of pixels which can have
                # a current induced on them.
                max_neighboring_pixels = (2*max_radius+1)*max_pixels[0]+(1+2*max_radius)*max_radius*2

                active_pixels = cp.full((selected_tracks.shape[0], max_pixels[0]), -1, dtype=np.int32)
                neighboring_pixels = cp.full((selected_tracks.shape[0], max_neighboring_pixels), -1, dtype=np.int32)
                neighboring_radius = cp.full((selected_tracks.shape[0], max_neighboring_pixels), -1, dtype=np.int32)
                n_pixels_list = cp.zeros(shape=(selected_tracks.shape[0]))

                if not active_pixels.shape[1] or not neighboring_pixels.shape[1]:
                    if light.LIGHT_SIMULATED and (light.LIGHT_TRIG_MODE == 0 or light.LIGHT_TRIG_MODE == 1):
                        null_light_results_acc['light_event_id'].append(cp.full(1, ievd)) # one event
                        save_results(event_times, null_light_results_acc, i_trig, i_mod, light_only=True)
                        i_trig += 1 # add to the trigger counter n_max_pixels
                        del null_light_results_acc['light_event_id']
                    continue

                RangePush("get_pixels")
                pixels_from_track.get_pixels[BPG,TPB](selected_tracks,
                                                      active_pixels,
                                                      neighboring_pixels,
                                                      neighboring_radius,
                                                      n_pixels_list,
                                                      max_radius)
                RangePop()

                RangePush("unique_pix")
                shapes = neighboring_pixels.shape
                joined = neighboring_pixels.reshape(shapes[0] * shapes[1])
                unique_pix = cp.unique(joined)
                unique_pix = unique_pix[(unique_pix != -1)]
                RangePop()

                ###################################
                # Kazu 2024-07-01 Useful if we modify the output to store all contributions
                ###################################
                #def invert_array_map(in_map,pix_set):
                #    '''
                #    Invert the map of unique segment id => a set of unique pixel IDs to a map of unique
                #    pixel index => a set of segment IDs.

                #    Args:
                #        in_map  (:obj:`numpy.ndarray`): 2D array where segment index => list of pixel IDs
                #        pix_set (:obj:`numpy.ndarray`): 1D array containing all unique pixel IDs
                #    Returns:
                #        ndarray: 2D array where pixel index => list of segment index
                #    '''
                #    pixids,counts=cp.unique(in_map[in_map>=0].flatten(),return_counts=True)
                #    
                #    pix_id2idx = {val.item():i for i,val in enumerate(pix_set)}
                #    
                #    mymap=cp.full(shape=(pix_set.shape[0],counts.max().item()),fill_value=-1,dtype=int)
                #    curr_idx=cp.zeros(shape=(len(pix_id2idx),),dtype=int)
                #    for seg_idx in range(in_map.shape[0]):
                #        ass = in_map[seg_idx]
                #        for pixid in ass:
                #            if pixid<0: break
                #            pix_idx = pix_id2idx[pixid.item()]
                #            mymap[pix_idx][curr_idx[pix_idx]]=seg_idx
                #            curr_idx[pix_idx] += 1
                #    return mymap
                #
                #assmap_pix2seg = invert_array_map(active_pixels,unique_pix)

                if not unique_pix.shape[0]:
                    if light.LIGHT_SIMULATED and (light.LIGHT_TRIG_MODE == 0 or light.LIGHT_TRIG_MODE == 1):
                        null_light_results_acc['light_event_id'].append(cp.full(1, ievd)) # one event
                        save_results(event_times, null_light_results_acc, i_trig, i_mod, light_only=True)
                        i_trig += 1 # add to the trigger counter
                        del null_light_results_acc['light_event_id']
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
                #max_segments_to_trace = max(assmap_pix2seg.shape[1],detsim.MAX_TRACKS_PER_PIXEL) # currently it doesn't work; see the comment for invert_array_map()
                max_segments_to_trace = detsim.MAX_TRACKS_PER_PIXEL
                track_pixel_map = cp.full((unique_pix.shape[0], max_segments_to_trace), -1)
                TPB = 32
                BPG = max(ceil(unique_pix.shape[0] / TPB),1)
                detsim.get_track_pixel_map2[BPG, TPB](track_pixel_map,
                    unique_pix,
                    #active_pixels,
                    neighboring_pixels,
                    neighboring_radius,
                    neighboring_radius.max().item()+1,
                    )
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
                overflow_flag = cp.zeros(len(unique_pix))
                detsim.sum_pixel_signals[BPG,TPB](pixels_signals,
                                                  signals,
                                                  track_starts,
                                                  pixel_index_map,
                                                  track_pixel_map,
                                                  pixels_tracks_signals,
                                                  overflow_flag)
                if cp.any(overflow_flag):
                    warnings.warn("More segments per pixel than the set MAX_TRACKS_PER_PIXEL value, "
                                  + f"{detsim.MAX_TRACKS_PER_PIXEL}")

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
                traj_pixel_map = cp.full(track_pixel_map.shape,-1)
                traj_pixel_map[:] = track_pixel_map
                traj_pixel_map[traj_pixel_map != -1] = trajectory_ids_arr[batch_mask][traj_pixel_map[traj_pixel_map != -1] + itrk]
                track_pixel_map[track_pixel_map != -1] = segment_ids_arr[batch_mask][track_pixel_map[track_pixel_map != -1] + itrk]
                results_acc['traj_pixel_map'].append(traj_pixel_map)
                results_acc['track_pixel_map'].append(track_pixel_map)

                # ~~~ Light detector response simulation ~~~
                if light.LIGHT_SIMULATED:
                    RangePush("sum_light_signals")
                    light_inc = light_sim_dat[batch_mask][itrk:itrk+sim.BATCH_SIZE]
                    selected_track_id = segment_ids_arr[batch_mask][itrk:itrk+sim.BATCH_SIZE]#cp.array(selected_tracks["segment_id"])
                    n_light_ticks, light_t_start = light_sim.get_nticks(light_inc)
                    n_light_ticks = min(n_light_ticks,int(5E4))
                    # at least the optical channels from a whole module are activated together

                    # in the mod2mod case, just take the channel indices of the first module (first two TPCs)
                    # e.g. for the 2x2, op_channel = [0..96) in mod2mod mode, [0..384) otherwise
                    # likewise light_inc etc. will have ndet=96 for mod2mod, ndet=384 otherwise
                    op_channel = light.TPC_TO_OP_CHANNEL[:2].ravel() if mod2mod_variation else light.TPC_TO_OP_CHANNEL[:].ravel()
                    op_channel = cp.array(op_channel)
                    #op_channel = light_sim.get_active_op_channel(light_inc)
                    n_light_det = op_channel.shape[0]
                    light_sample_inc = cp.zeros((n_light_det,n_light_ticks), dtype='f4')
                    light_sample_inc_true_track_id = cp.full((n_light_det, n_light_ticks, light.MAX_MC_TRUTH_IDS), -1, dtype='i8')
                    light_sample_inc_true_photons = cp.zeros((n_light_det, n_light_ticks, light.MAX_MC_TRUTH_IDS), dtype='f8')

                    ### TAKE LIMITED SEGMENTS FOR LIGHT TRUTH ###
                    ### FIXME: this is a temporary fix to avoid memory issues ###
                    sorted_indices = np.zeros((n_light_det, selected_tracks.shape[0]), dtype=np.int32)

                    for idet in range(n_light_det):
                        sorted_indices[idet] = np.argsort(light_inc[:,idet]['n_photons_det'])[::-1] # get the order in which to loop over tracks
                    ### END OF TEMPORARY FIX ###

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
                    light_response_true_track_id = cp.full_like(light_sample_inc_true_track_id, -1)
                    light_response_true_photons = cp.zeros_like(light_sample_inc_true_photons)
                    light_sim.calc_light_detector_response[BPG, TPB](
                        light_sample_inc_disc, light_sample_inc_scint_true_track_id, light_sample_inc_scint_true_photons,
                        light_response, light_response_true_track_id, light_response_true_photons)
                    #light_response += cp.array(light_sim.gen_light_detector_noise(light_response.shape, light_noise[op_channel.get()]))
                    RangePop()

                    RangePush("sim_light_triggers")
                    light_threshold = cp.repeat(cp.array(light.LIGHT_TRIG_THRESHOLD)[...,np.newaxis], light.OP_CHANNEL_PER_TRIG, axis=-1)
                    light_threshold = light_threshold.ravel()[op_channel.get()].copy()
                    light_threshold = light_threshold.reshape(-1, light.OP_CHANNEL_PER_TRIG)[...,0]
                    trigger_idx, trigger_op_channel_idx, trigger_type = light_sim.get_triggers(light_response, light_threshold, op_channel, itrk)
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
                    results_acc['light_waveforms_true_track_id'].append(light_digit_signal_true_track_id)
                    results_acc['light_waveforms_true_photons'].append(light_digit_signal_true_photons)

            if len(results_acc['event_id']) >= sim.WRITE_BATCH_SIZE:
                if len(results_acc['event_id']) > 0 and len(np.concatenate(results_acc['event_id'], axis=0)) > 0:
                    save_results(event_times, results_acc, i_trig, i_mod, light_only=False)
                    i_trig += 1 # add to the trigger counter
                elif len(results_acc['light_event_id']) > 0 and len(np.concatenate(results_acc['light_event_id'], axis=0)) > 0:
                    save_results(event_times, results_acc, i_trig, i_mod, light_only=True)
                    i_trig += 1 # add to the trigger counter
                results_acc = defaultdict(list) # reinitialize after each save_results

            logger.take_snapshot([len(logger.log)])
        RangePop()                  # run_simulation

        RangePush('save_results')
        # Always save results after last iteration
        if len(results_acc['event_id']) > 0 and len(np.concatenate(results_acc['event_id'], axis=0)) > 0:
            save_results(event_times, results_acc, i_trig, i_mod, light_only=False)
            i_trig += 1 # add to the trigger counter
        elif len(results_acc['light_event_id']) > 0 and len(np.concatenate(results_acc['light_event_id'], axis=0)) > 0:
            save_results(event_times, results_acc, i_trig, i_mod, light_only=True)
            i_trig += 1 # add to the trigger counter
        results_acc = defaultdict(list) # reinitialize after each save_results
        RangePop()

        # Collect updated true segments
        if i_mod <= 1: # i_mod counts from 1 for module to module variation, otherwise i_mod is set to -1
            segments_to_files = tracks # segments are only updated in quenching and drifting, otherwise this part should be in the batching loop
        else:
            segments_to_files = np.append(segments_to_files, tracks)

    logger.take_snapshot([len(logger.log)])

    # revert the mc truth information modified for larnd-sim consumption 
    # if the event time is generated by larndsim (non-beam cases), then the t0 is relative to the event time (0 ish, assume the edep particle window is O(us))
    # so there is no need to remove the event time
    if sim.IS_SPILL_SIM:
        # write the true timing structure to the file, not t0 wrt event time .....
        localSpillIDs = segments_to_files[sim.EVENT_SEPARATOR] - (segments_to_files[sim.EVENT_SEPARATOR] // sim.MAX_EVENTS_PER_FILE) * sim.MAX_EVENTS_PER_FILE
        segments_to_files['t0_start'] = segments_to_files['t0_start'] + localSpillIDs*sim.SPILL_PERIOD
        segments_to_files['t0_end'] = segments_to_files['t0_end'] + localSpillIDs*sim.SPILL_PERIOD
        segments_to_files['t0'] = segments_to_files['t0'] + localSpillIDs*sim.SPILL_PERIOD

    # store light triggers altogether if it's beam trigger (all light channels are forced to trigger)
    # FIXME one can merge the beam + threshold for LIGHT_TRIG_MODE = 1 in future
    # once mod2mod variation is enabled, the light threshold triggering does not work properly
    # compare the light trigger between different module and digitize afterwards should solve the issue
    if light.LIGHT_TRIG_MODE == 1:
        light_event_id = np.unique(localSpillIDs) if sim.IS_SPILL_SIM else vertices['event_id']
        light_start_times = np.full(len(light_event_id), 0) # if it is beam trigger it is set to 0
        light_trigger_idx = np.full(len(light_event_id), 0) # one beam spill, one trigger
        light_op_channel_idx = light.TPC_TO_OP_CHANNEL[:].ravel()
        light_event_times = light_event_id * sim.SPILL_PERIOD if sim.IS_SPILL_SIM else event_times.get() # us

        light_sim.export_light_trig_to_hdf5(light_event_id, light_start_times, light_trigger_idx, light_op_channel_idx, output_filename, light_event_times)
        #fee.export_pacman_trigger_to_hdf5(output_filename, light_event_times)

    # FIXME
    #if light.LIGHT_TRIG_MODE == 0:
    #    fee.export_pacman_trigger_to_hdf5(light_event_times_something_different)

    # merge light waveforms per module
    # correspond to light_sim.export_light_wvfm_to_hdf5
    if light.LIGHT_SIMULATED and mod2mod_variation:
        light_sim.merge_module_light_wvfm_same_trigger(output_filename)

    # prep output file with truth datasets
    with h5py.File(output_filename, 'a') as output_file:
        # We previously called swap_coordinates(tracks), but we want to write
        # all truth info in the edep-sim convention (z = beam coordinate).
        swap_coordinates(segments_to_files)

        # Store all tracks in the gdml module volume, could have small differences because of the active volume check
        output_file.create_dataset(sim.TRACKS_DSET_NAME, data=segments_to_files)

        # To distinguish from the "old" files that had z=drift in 'tracks':
        output_file[sim.TRACKS_DSET_NAME].attrs['zbeam'] = True

        if light.LIGHT_SIMULATED:
            # It seems unnecessary to store (all tracks, all channels) given the modules are light tight
            if mod2mod_variation:
                for i_mod in mod_ids:
                    output_file.create_dataset(f'light_dat/light_dat_module{i_mod-1}', data=light_sim_dat_acc[i_mod-1])
            else:
                output_file.create_dataset(f'light_dat/light_dat_allmodules', data=light_sim_dat_acc[0])
        if input_has_trajectories:
            output_file.create_dataset("trajectories", data=trajectories)
        if input_has_vertices:
            output_file.create_dataset("vertices", data=vertices)
        if input_has_mc_hdr:
            output_file.create_dataset("mc_hdr", data=mc_hdr)
        if input_has_mc_stack:
            output_file.create_dataset("mc_stack", data=mc_stack)

    with h5py.File(output_filename, 'a') as output_file:
        if 'configs' in output_file.keys():
            output_file['configs'].attrs['pixel_layout'] = pixel_layout

    print("Output saved in:", output_filename)

    end_simulation = time()
    logger.take_snapshot([len(logger.log)])
    print(f"Elapsed time: {end_simulation-start_simulation:.2f} s")
    logger.archive('loop',['loop'])
    logger.store(save_memory)

if __name__ == "__main__":
    fire.Fire(run_simulation)
