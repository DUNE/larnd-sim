#!/usr/bin/env python
"""
Command-line interface to larnd-sim module.
"""
from collections import defaultdict
from functools import lru_cache
import importlib
from math import ceil, floor
import os
from time import time
from typing import Any, Optional
import warnings

import cupy as cp
from cupy.cuda.nvtx import RangePush, RangePop # ty: ignore[unresolved-import]
import cupy.typing as cpt
import fire
import h5py
import numba as nb
from numba.cuda import device_array, to_device
import numpy as np
import numpy.typing as npt
import numpy.lib.recfunctions as rfn
from tqdm import tqdm

from larndsim import _version
from larndsim import consts
from larndsim import active_volume, quenching, drifting, detsim
from larndsim import pixels_from_track, fee, lightLUT, light_sim
from larndsim.config import get_config, load_mod2mod_prop, reload_modules
from larndsim.consts import detector, ff_induction, light, physics, sim
from larndsim.consts.detector import load_thresholds, load_gains, load_pedestals
import larndsim.consts.units
from larndsim.far_field import voxelization, signal_calculation, pixel_classifier, PixelClassificationResult
from larndsim.far_field.voxelization import VoxelDict
from larndsim.pixels_from_track import invert_array_map
from larndsim.util import batching
from larndsim.util.cuda_dict import CudaDict
from larndsim.util.misc import *




def do_save_results(
    event_times: cpt.NDArray[cp.float64],
    results: dict[str, Any],
    i_trig: int,
    i_mod: int,
    light_only: bool,
    output_filename: str,
    compression: Optional[str],
    bad_channels: dict[str, list[int]],
    light_simulated: bool,
):
    """
    Save the accumulated results of the simulation.

    `results` keys for the charge simulation:
        - event_id: event id for each hit
        - adc_tot: adc value for each hit
        - adc_tot_ticks: timestamp for each hit
        - track_pixel_map: map from track to active pixels
        - unique_pix: all unique pixels (per track?)
        - current_fractions: fraction of charge associated with each true track

        For the light simulation (in addition to all above keys):
        - light_event_id: event_id for each light trigger
        - light_start_time: simulation start time for event
        - light_trigger_idx: time tick at which each trigger occurs
        - light_op_channel_idx: optical channel id for each waveform
        - light_waveforms: waveforms of each light trigger
        - light_waveforms_true_track_id: true track ids for each tick in each
              waveform
        - light_waveforms_true_photons: equivalent pe for each track at each
              tick in each waveform

    Note:
        Can't handle empty inputs

    Args:
        event_times: (n_events,) array of event timestamps
        results: Dictionary of accumulated results (see below)
        i_trig: Index of light trigger (if any)
        i_mod: Detector module being simulated
        light_only: Whether only light data is to be saved
        output_filename: Output filename
        compression: Optional file compression mode (e.g. 'lzf', 'gzip')
        bad_channels: Optional dict with list of bad channels for each chip key
        light_simulated: Whether the light sim is enabled
    """
    from larndsim.consts import detector, light, sim

    for key in list(results.keys()):
        if isinstance(results[key], list) and len(results[key]) > 0: # we may have empty lists (e.g. for event_id) when light_only
            results[key] = np.concatenate([cp.asnumpy(arr) for arr in results[key]], axis=0)

    uniq_events = cp.asnumpy(np.unique(results['event_id'])) if not light_only else cp.asnumpy(np.unique(results['light_event_id']))
    uniq_event_times = cp.asnumpy(event_times[uniq_events % sim.MAX_EVENTS_PER_FILE])

    if not light_only:
        if light_simulated:
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
                           i_mod=i_mod,
                           compression=compression)

    if light_simulated and len(results['light_event_id']):
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
                                     i_mod,
                                     compression=compression)
        elif light.LIGHT_TRIG_MODE == 1:
            light_sim.export_light_wvfm_to_hdf5(results['light_event_id'],
                                                results['light_waveforms'],
                                                output_filename,
                                                results['light_waveforms_true_track_id'],
                                                results['light_waveforms_true_photons'],
                                                i_trig,
                                                i_mod,
                                                compression=compression)


def prep_null_light_results(light_noise):
    null_light_results_acc = defaultdict(list)
    trigger_idx = cp.array([0], dtype=int)
    # FIXME: mod2mod_var?
    op_channel = light.TPC_TO_OP_CHANNEL[:2].ravel()
    op_channel = cp.array(op_channel)
    trigger_op_channel_idx = cp.repeat(np.expand_dims(op_channel, axis=0), len(trigger_idx), axis=0)
    digit_samples = ceil(round(light.LIGHT_TRIG_WINDOW[1] + light.LIGHT_TRIG_WINDOW[0], 3) / light.LIGHT_DIGIT_SAMPLE_SPACING)

    n_light_det = op_channel.shape[0]
    n_light_ticks = int((light.LIGHT_WINDOW[1] + light.LIGHT_WINDOW[0])/light.LIGHT_TICK_SIZE)

    light_response = cp.zeros((n_light_det,n_light_ticks), dtype='f4')
    #light_response += cp.array(light_sim.gen_light_detector_noise(light_response.shape, light_noise[op_channel.get()]))
    light_response_true_track_id = cp.full((n_light_det, n_light_ticks, sim.MAX_MC_TRUTH_IDS), -1, dtype='i8')
    light_response_true_photons = cp.zeros((n_light_det, n_light_ticks, sim.MAX_MC_TRUTH_IDS), dtype='f8')

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

    return null_light_results_acc


class LArND_Sim:
    def __init__(self,
                 input_filename,
                 output_filename,
                 config='2x2',
                 n_events=None,
                 rand_seed: int | None = None,
                 compression=None):
        """
        Command-line interface to run the simulation of a pixelated LArTPC

        Args:
            input_filename (str): path of the edep-sim input file
            output_filename (str): path of the HDF5 output file. If not specified
                the output is added to the input file.
            config (str): a keyword to specify a configuration (all necessary meta data files)
            n_events (int, optional): number of events to be simulated. Defaults to None
                (all tracks).
            rand_seed (int, optional): the random number generator seed that can be set through 
                a command-line
            compression (str, optional): enable file compression of the output HDF5 datasets. Defaults to None,
                supported options are 'lzf' and 'gzip'
        """
        print(LOGO)
        print("**************************\nLOADING SETTINGS AND INPUT\n**************************")

        self.input_filename = input_filename
        self.output_filename = output_filename
        self.config = config
        self.compression = compression

        self.check_files()
        self.load_config()
        self.print_config()
        self.init_rng(rand_seed)
        self.load_input(n_events)
        self.cupaify_const_arrays()

        self.event_times, self.sync_start = prep_event_times(self.all_mod_tracks)

        # output accumulators:
        self.results_acc = defaultdict(list)
        self.light_sim_dat_acc = list()

    def check_files(self):
        if not os.path.exists(self.input_filename):
            raise Exception(f'Input file {self.input_filename} does not exist.')
        if os.path.exists(self.output_filename):
            raise Exception(f'Output file {self.output_filename} already exists.')

    def load_config(self):
        cfg = get_config(self.config)
        self.detector_properties = cfg['DET_PROPERTIES']
        self.simulation_properties = cfg['SIM_PROPERTIES']
        self.light_simulated = cfg.get('LIGHT_SIMULATED', True)
        self.bad_channels = cfg.get('BAD_CHANNELS')

        self.pixel_layout = load_mod2mod_prop(cfg, 'PIXEL_LAYOUT')
        self.response_file = load_mod2mod_prop(cfg, 'RESPONSE_FILE')
        self.pixel_thresholds_file = load_mod2mod_prop(cfg, 'PIXEL_THRESHOLDS_FILE')
        self.pixel_gains_file = load_mod2mod_prop(cfg, 'PIXEL_GAINS_FILE')
        self.pixel_pedestals_file = load_mod2mod_prop(cfg, 'PIXEL_PEDESTALS_FILE')

        # FIXME: these mod_ids start at 1 instead of 0
        self.mod_ids = consts.detector.get_module_ids(self.detector_properties)
        self.n_modules = len(self.mod_ids)

        consts.light.set_light_properties(self.detector_properties)
        consts.sim.set_simulation_properties(self.simulation_properties)

        # We load detector properties to get detector.TPC_BORDERS
        # For this purpose, it doesn't matter which pixel_layout to use
        consts.detector.set_detector_properties(self.detector_properties, self.pixel_layout[0], geo_only=True)

        if self.light_simulated:
            self.light_lut_filename = load_mod2mod_prop(cfg, 'LIGHT_LUT')
            self.light_det_noise_filename = cfg['LIGHT_DET_NOISE']

    def print_config(self):
        print("")
        print("edep-sim input file:", self.input_filename)
        print("larnd-sim output file:", self.output_filename)
        print("Config:", self.config)
        print("")
        print("Compression:", self.compression)
        print("Random seed:", self.rand_seed)
        print("Simulation properties file:", self.simulation_properties)
        print("Detector properties file:", self.detector_properties)
        print("Pixel layout file:", self.pixel_layout)
        print("Response file:", self.response_file)
        print("Disabled channel list: ", self.bad_channels)
        print("Pixel threshold file: ", self.pixel_thresholds_file)
        print("Pixel gain file: ", self.pixel_gains_file)
        print("Pixel pedestals file: ", self.pixel_pedestals_file)

        if sim.FARFIELD_ENABLED:
            print("Far-field mode:", sim.FARFIELD_MODE)

        if self.light_simulated:
            print("Light LUT:", self.light_lut_filename)
            print("Light detector noise: ", self.light_det_noise_filename)

    def init_rng(self, rand_seed):
        if rand_seed is None:
            self.rand_seed = int(time())
        else:
            self.rand_seed = int(rand_seed)
        cp.random.seed(self.rand_seed)
        # pre-allocate some random number states for custom kernels
        self.rng_states = maybe_create_rng_states(1024*256, seed=self.rand_seed)

    def load_input(self, n_events):
        RangePush("load_hdf5_file")
        print("Loading track segments...")
        start_load = time()
        # First of all we load the edep-sim output
        with h5py.File(self.input_filename, 'r') as f:
            tracks = np.array(f['segments'])

            tracks = maybe_add_t0(tracks)
            maybe_shift_times(tracks)
            tracks = remove_neutrals(tracks)
            tracks = remove_delayed_segments(tracks)
            tracks = maybe_add_segment_ids(tracks)

            segment_ids = tracks['segment_id']

            trajectories = maybe_read_array(f, 'trajectories')
            vertices = maybe_read_array(f, 'vertices')
            mc_hdr = maybe_read_array(f, 'mc_hdr')
            mc_stack = maybe_read_array(f, 'mc_stack')

        if tracks.size == 0:
            print("Empty input dataset, exiting")
            return

        # Reduce dataset if not all events are to be simulated, being careful of gaps
        if n_events:
            print(f'Selecting only the first {n_events} events for simulation.')
            max_eventID = np.unique(tracks[sim.EVENT_SEPARATOR])[n_events-1]
            segment_ids = segment_ids[tracks[sim.EVENT_SEPARATOR] <= max_eventID]
            tracks = tracks[tracks[sim.EVENT_SEPARATOR] <= max_eventID]

            if trajectories:
                trajectories = trajectories[trajectories[sim.EVENT_SEPARATOR] <= max_eventID]
            if vertices:
                vertices = vertices[vertices[sim.EVENT_SEPARATOR] <= max_eventID]
            if mc_hdr:
                mc_hdr = mc_hdr[mc_hdr[sim.EVENT_SEPARATOR] <= max_eventID]
            if mc_stack:
                mc_stack = mc_stack[mc_stack[sim.EVENT_SEPARATOR] <= max_eventID]

        tracks = maybe_add_n_photons(tracks)
        tracks = swap_coordinates(tracks)
        vertices = maybe_set_vertex_times(vertices, self.event_times)
        mc_hdr = maybe_set_mc_hdr_times(mc_hdr, vertices)

        self.all_mod_tracks = tracks
        self.all_mod_segment_ids = segment_ids
        self.segments_to_files = np.empty_like(tracks)

        self.trajectories = trajectories
        self.vertices = vertices
        self.mc_hdr = mc_hdr
        self.mc_stack = mc_stack

        RangePop()                  # load_hdf5_file
        end_load = time()
        print(f"Data preparation time: {end_load-start_load:.2f} s")

    def cupaify_const_arrays(self):
        # We need to make cupy arrays of these and pass them to the kernels;
        # otherwise numba will try to use the GPU's "global constant" memory
        # which (at 64 kB) is not large enough for ND-LAr
        self.op_channel_efficiency = cp.array(light.OP_CHANNEL_EFFICIENCY)
        self.op_channel_to_tpc = cp.array(light.OP_CHANNEL_TO_TPC)
        self.light_gain = cp.array(light.LIGHT_GAIN)

    def init_module(self, i_mod):
        # FIXME: i_mod starts at 1 (see call to get_module_ids above)
        print(f'Simulating module {i_mod-1}')
        reload_modules(self.detector_properties, self.pixel_layout, self.response_file, i_mod)

        self.response = consts.detector.load_response(self.response_file[i_mod-1])
        self.pixel_thresholds_lut = load_thresholds(self.pixel_thresholds_file, i_mod)
        self.pixel_gains_lut = load_gains(self.pixel_gains_file, i_mod)
        self.pixel_pedestals_lut = load_pedestals(self.pixel_pedestals_file, i_mod)

        RangePush("load_segments_in_module")
        self.module_borders = detector.TPC_BORDERS[(i_mod-1)*2: i_mod*2]
        module_tracks_mask = active_volume.select_active_volume(self.all_mod_tracks, self.module_borders)
        self.tracks = self.all_mod_tracks[module_tracks_mask]
        self.segment_ids = self.all_mod_segment_ids[module_tracks_mask]
        self.segment_ids_arr = cp.asarray(self.segment_ids)
        RangePop()

        # find the module that triggers
        if light.LIGHT_TRIG_MODE == 0 or light.LIGHT_TRIG_MODE == 1:
            io_groups = np.array(list(consts.detector.MODULE_TO_IO_GROUPS.values()))
            self.trig_module = np.argwhere(io_groups==fee.get_trig_io())[0][0] + 1 # module id (i_mod) counts from 1

        self.i_trig = 0

    def load_light_info(self, i_mod):
        RangePush("load_light_info")
        n_light_channel = int(light.N_OP_CHANNEL/len(self.mod_ids))
        self.light_sim_dat = np.zeros([len(self.tracks), n_light_channel],
                                      dtype=[('segment_id', 'u4'), ('n_photons_det','f4'),('t0_det','f4')])
        self.light_sim_dat['segment_id'] = self.segment_ids[..., np.newaxis]
        self.track_light_voxel = np.zeros([len(self.tracks), 3], dtype='i4')

        print("Calculating optical responses...", end="")
        self.start_light_time = time()

        light_lut = self.light_lut_filename[i_mod-1]

        if i_mod == 1 or light_lut != self.light_lut_filename[i_mod-2]:
            lut = np.load(light_lut)['arr']

            # check if the light LUT matches with the number of optical channels
            # lut (x, y, z, n_op_ch) for one TPC
            # n_light_channel is for one module or all modules depending if the mod2mod_variation is enabled
            warn_n_op_ch = (n_light_channel != lut.shape[3]*2)
            if warn_n_op_ch:
                warnings.warn("The light LUT has different number of optical channels than we expected in one TPC!")

            # clip LUT so that no voxel contains 0 visibility
            mask = lut['vis'] > 0
            lut['vis'][~mask] = lut['vis'][mask].min()

            # get length of the t0 time profile
            self.t0_profile_length = lut['time_dist'].shape[-1]

            self.lut = to_device(lut)

        light_noise = cp.load(self.light_det_noise_filename)

        if light_noise.shape[0] == self.n_modules * n_light_channel:
            self.light_noise = light_noise[n_light_channel*(i_mod-1):n_light_channel*i_mod]
        else:
            assert light_noise.shape[0] >= n_light_channel
            self.light_noise = light_noise[:n_light_channel]
            warnings.warn(f"Light noise file {self.light_det_noise_filename} does not span all modules. " +
                            f"Using noise from first {n_light_channel} channels for all modules.")
        RangePop() # load_light_info

    def calc_light_inc(self):
        RangePush('calculate_light_incidence')
        TPB = 256
        BPG = max(ceil(self.tracks.shape[0] / TPB),1)
        lightLUT.calculate_light_incidence[BPG,TPB](self.tracks, self.lut,
                                                    self.light_sim_dat, self.track_light_voxel,
                                                    self.op_channel_efficiency, self.op_channel_to_tpc)
        RangePop()

        self.light_sim_dat_acc.append(self.light_sim_dat)
        self.null_light_results_acc = prep_null_light_results(self.light_noise)

        print(f" {time()-self.start_light_time:.2f} s")

    def save_results(self, i_mod, light_only):
        do_save_results(event_times=self.event_times,
                        results=self.results_acc,
                        i_trig=self.i_trig,
                        i_mod=i_mod,
                        light_only=light_only,
                        output_filename=self.output_filename,
                        compression=self.compression,
                        bad_channels=self.bad_channels,
                        light_simulated=self.light_simulated)
        self.i_trig += 1

    def maybe_save_null_light_results(self, ievd, i_mod):
        if self.light_simulated:
            self.null_light_results_acc['light_event_id'].append(cp.full(1, ievd)) # one event
            self.save_results(i_mod, light_only=True)
            del self.null_light_results_acc['light_event_id']
            self.i_trig += 1

    def setup_event_batch(self, i_mod, ievd, batch_mask, is_new_event) -> bool:
        "Returns True if there's anything to process"
        this_event_time = [self.event_times[ievd % sim.MAX_EVENTS_PER_FILE]]
        if is_new_event:
            # forward sync packets
            if this_event_time[0] - self.sync_start >= 0: # this is duplicate to "is_new_event"
                sync_times = cp.arange(self.sync_start, this_event_time[0]+1, detector.CLOCK_RESET_PERIOD * detector.CLOCK_CYCLE) #us
                #PSS Sync also resets the timestamp in the PACMAN controller, so all of the timestamps in the packs should read 1e7 (for PPS)
                sync_times_export = cp.full( sync_times.shape, detector.CLOCK_RESET_PERIOD * detector.CLOCK_CYCLE) 
                if len(sync_times) > 0:
                    fee.export_sync_to_hdf5(self.output_filename, ievd, sync_times_export, i_mod, compression=self.compression)
                    self.sync_start = sync_times[-1] + detector.CLOCK_RESET_PERIOD * detector.CLOCK_CYCLE
            # beam trigger is only forwarded to one specific pacman (defined in fee)
            if (light.LIGHT_TRIG_MODE == 0 or light.LIGHT_TRIG_MODE == 1) and (i_mod == self.trig_module or i_mod == -1):
                fee.export_timestamp_trigger_to_hdf5(self.output_filename, [ievd], this_event_time, i_mod, compression=self.compression)

        if np.sum(batch_mask) == 0:
            self.maybe_save_null_light_results(ievd, i_mod)
            return False

        self.prepare_all_pixels(batch_mask)

        if not self.all_active_pixels.shape[1] or not self.all_neighboring_pixels.shape[1]:
            self.maybe_save_null_light_results(ievd, i_mod)
            return False

        self.get_pixels()

        if not self.all_unique_pix.shape[0]:
            self.maybe_save_null_light_results(ievd, i_mod)
            return False

        # Track all pixels processed in near-field batches for this event batch
        self.processed_pixels_event = cp.array([], dtype=cp.int32)

        return True

    def prepare_all_pixels(self, batch_mask):
        RangePush("event_id_map")
        track_subset = self.tracks[batch_mask]
        event_ids = track_subset[sim.EVENT_SEPARATOR]
        self.unique_eventIDs = np.unique(event_ids)
        RangePop()

        # Filter out tracks with invalid pixel_plane (outside TPCs)
        valid_plane_mask = track_subset['pixel_plane'] != detector.DEFAULT_PLANE_INDEX
        self.all_selected_tracks = track_subset[valid_plane_mask]

        # Create combined mask for light simulation arrays that are indexed by batch_mask
        # We need indices into the full tracks array for light data
        batch_indices = np.where(batch_mask)[0]
        valid_batch_indices = batch_indices[valid_plane_mask]
        self.valid_batch_mask = np.zeros_like(batch_mask, dtype=bool)
        self.valid_batch_mask[valid_batch_indices] = True

        # We find the pixels intersected by the projection of the tracks on
        # the anode plane using the Bresenham's algorithm. We also take into
        # account the neighboring pixels, due to the transverse diffusion of the charges.
        RangePush("max_pixels")
        TPB = 128
        BPG = max(ceil(self.all_selected_tracks.shape[0] / TPB),1)
        all_max_pixels = np.array([0])
        pixels_from_track.max_pixels[BPG,TPB](self.all_selected_tracks, all_max_pixels)
        RangePop()

        # This formula tries to estimate the maximum number of pixels which can have
        # a current induced on them.
        all_max_neighboring_pixels = (2*detector.MAX_RADIUS+1)*all_max_pixels[0]+(1+2*detector.MAX_RADIUS)*detector.MAX_RADIUS*2
        all_max_neighboring_pixels = np.clip([all_max_neighboring_pixels], all_max_neighboring_pixels, detector.N_PIXELS[0]*detector.N_PIXELS[1])[0] # limiting the all_max_neighboring_pixels by the total number of pixels in a TPC

        self.all_active_pixels = cp.full((self.all_selected_tracks.shape[0], all_max_pixels[0]), -1, dtype=np.int32)
        self.all_neighboring_pixels = cp.full((self.all_selected_tracks.shape[0], all_max_neighboring_pixels), -1, dtype=np.int32)
        self.all_neighboring_radius = cp.full((self.all_selected_tracks.shape[0], all_max_neighboring_pixels), -1, dtype=np.int32)
        self.all_n_pixels_list = cp.zeros(shape=(self.all_selected_tracks.shape[0]))

    def prepare_subbatch_pixels(self, start_pix, stop_pix) -> bool:
        self.unique_pix = self.all_unique_pix[start_pix:stop_pix]

        selected_track_idcs = np.unique(self.assmap_pix2seg[start_pix:stop_pix])
        selected_track_idcs = selected_track_idcs[selected_track_idcs != -1]
        if selected_track_idcs.size == 0:
            return False
        selected_track_idcs = to_device(selected_track_idcs)
        self.selected_tracks = self.all_selected_tracks[selected_track_idcs]
        RangePop() # setup_pixel_batch

        # We find the pixels intersected by the projection of the tracks on
        # the anode plane using the Bresenham's algorithm. We also take into
        # account the neighboring pixels, due to the transverse diffusion of the charges.
        RangePush("max_pixels")
        TPB = 128
        BPG = max(ceil(self.selected_tracks.shape[0] / TPB),1)
        max_pixels = np.array([0])
        pixels_from_track.max_pixels[BPG,TPB](self.selected_tracks, max_pixels)
        RangePop()
        # This formula tries to estimate the maximum number of pixels which can have
        # a current induced on them.
        max_neighboring_pixels = (2*detector.MAX_RADIUS+1)*max_pixels[0]+(1+2*detector.MAX_RADIUS)*detector.MAX_RADIUS*2
        max_neighboring_pixels = np.clip([max_neighboring_pixels], max_neighboring_pixels, detector.N_PIXELS[0]*detector.N_PIXELS[1])[0] # limiting the max_neighboring_pixels by the total number of pixels in a TPC

        self.active_pixels = cp.full((self.selected_tracks.shape[0], max_pixels[0]), -1, dtype=np.int32)
        self.neighboring_pixels = cp.full((self.selected_tracks.shape[0], max_neighboring_pixels), -1, dtype=np.int32)
        self.neighboring_radius = cp.full((self.selected_tracks.shape[0], max_neighboring_pixels), -1, dtype=np.float32)
        self.n_pixels_list = cp.zeros(shape=(self.selected_tracks.shape[0]))

        RangePush("get_pixels", 7)
        pixels_from_track.get_pixels[BPG,TPB](self.selected_tracks,
                                              self.active_pixels,
                                              self.neighboring_pixels,
                                              self.neighboring_radius,
                                              self.n_pixels_list)
        RangePop()

        active_pixels_isin_unique_pix = np.isin(self.active_pixels, self.unique_pix)
        self.active_pixels[~active_pixels_isin_unique_pix] = -1

        isin_unique_pix = np.isin(self.neighboring_pixels, self.unique_pix)
        self.neighboring_pixels[~isin_unique_pix] = -1
        self.neighboring_radius[~isin_unique_pix] = -1

        return True

    def get_pixels(self):
        RangePush("get_pixels", 7)
        TPB = 128
        BPG = max(ceil(self.all_selected_tracks.shape[0] / TPB),1)
        pixels_from_track.get_pixels[BPG,TPB](self.all_selected_tracks,
                                              self.all_active_pixels,
                                              self.all_neighboring_pixels,
                                              self.all_neighboring_radius,
                                              self.all_n_pixels_list)
        RangePop()

        RangePush("unique_pix")
        shapes = self.all_neighboring_pixels.shape
        joined = self.all_neighboring_pixels.reshape(shapes[0] * shapes[1])
        all_unique_pix = cp.unique(joined)
        self.all_unique_pix = all_unique_pix[(all_unique_pix != -1)]
        RangePop()

        RangePush("invert_array_map")
        # global pixel ID -> [segment IDs] (fixed-size; padded w/ -1)
        self.assmap_pix2seg = invert_array_map(self.all_neighboring_pixels,self.all_unique_pix)
        RangePop() # invert_array_map

    def maybe_farfield_precompute(self):
        # ~~~ Precompute far-field helper data once per event batch ~~~
        RangePush("event_farfield_precompute")
        self.classification_cache: dict[int, PixelClassificationResult] = {}
        self.voxel_cache: dict[int, VoxelDict] = {}
        if sim.FARFIELD_ENABLED:
            self.classification_cache = \
                pixel_classifier.get_classification_cache(self.all_selected_tracks)
            if sim.FARFIELD_MODE == 'voxels':
                self.voxel_cache = voxelization.get_voxel_cache(self.all_selected_tracks)
        RangePop()

    def call_tracks_current_mc(self):
        RangePush("tracks_current", 2)
        # Here we find the longest signal in time
        # Pad if RESPONSE_MAX_TIME is longer than DRIFT_MAX_TIME
        # remove t0 and account it later
        if detector.RESPONSE_MAX_TIME > detector.DRIFT_MAX_TIME:
            self.max_signal_time = (self.selected_tracks['t_end'] - self.selected_tracks['t0']).max() + self.selected_tracks['long_diff'].max() / detector.V_DRIFT * detector.DIFF_N_SIGMAS + detector.RESPONSE_MAX_TIME - detector.DRIFT_MAX_TIME
        else:
            self.max_signal_time = (self.selected_tracks['t_end'] - self.selected_tracks['t0']).max() + self.selected_tracks['long_diff'].max() / detector.V_DRIFT * detector.DIFF_N_SIGMAS
        self.signals_ticks = ceil(self.max_signal_time / detector.TIME_SAMPLING)  # signal span in time ticks

        # Here we calculate the induced current on each pixel
        self.signals = cp.zeros((self.selected_tracks.shape[0],
                                 self.neighboring_pixels.shape[1],
                                 self.signals_ticks), dtype=np.float32)
        TPB = (1,1,64)
        BPG_X = max(ceil(self.signals.shape[0] / TPB[0]),1)
        BPG_Y = max(ceil(self.signals.shape[1] / TPB[1]),1)
        BPG_Z = max(ceil(self.signals.shape[2] / TPB[2]),1)
        BPG = (BPG_X, BPG_Y, BPG_Z)

        # To conserve memory, we break up the signals calculation into subbatches.
        # The subbatches are sized so that rng_states array won't have to be expanded.
        # This is achieved by choosing a BPG_X_subbatch <= BPG_X that lets
        # the existing length of rng_states be able to accommodate the number of threads (BPG_X_subbatch, BPG_Y, BPG_Z) x TPB.

        # In the case that there are not enough rng states for even BPG_X_subbatch = 1, we will have to expand rng_states.
        if len(self.rng_states) < (np.prod(TPB) * BPG[1] * BPG[2]):
            self.rng_states = maybe_create_rng_states(int(np.prod(TPB) * BPG[1] * BPG[2]), seed=self.rand_seed, rng_states=self.rng_states)

        BPG_X_subbatch = min(floor(len(self.rng_states) / (np.prod(TPB) * BPG[1] * BPG[2])), BPG_X)
        BPG_subbatch = (BPG_X_subbatch, BPG_Y, BPG_Z)
        subbatches_size = BPG_X_subbatch
        n_subbatches = ceil(BPG_X/subbatches_size)

        for i_subbatch in range(n_subbatches):
            start = i_subbatch*subbatches_size
            end = (i_subbatch+1)*subbatches_size
            detsim.tracks_current_mc[BPG_subbatch,TPB](self.signals[start:end],
                                                       self.neighboring_pixels[start:end],
                                                       self.selected_tracks[start:end],
                                                       self.response, self.rng_states)

        RangePop()              # tracks_current

    def call_sum_pixel_signals(self):
        RangePush("pixel_index_map")
        # Here we create a map between tracks and index in the unique pixel array
        # First, create a lookup table for unique_pix values to their indices
        max_pix_val = int(cp.max(self.unique_pix)) + 1
        pix_lookup = cp.full((max_pix_val,), -1, dtype=cp.int32)
        pix_lookup[self.unique_pix] = cp.arange(self.unique_pix.shape[0], dtype=cp.int32)

        # Now directly map neighboring_pixels to pixel indices using lookup
        pixel_index_map = pix_lookup[self.neighboring_pixels]
        # Some elements of neighboring_pixels can have values of -1.
        # We want to make sure these pixels are also removed in pixel_index_map.`
        pixel_index_map[self.neighboring_pixels==-1] = -1
        RangePop()

        RangePush("track_pixel_map")
        # Mapping between unique pixel array and track array index
        #max_segments_to_trace = max(assmap_pix2seg.shape[1],detsim.MAX_TRACKS_PER_PIXEL) # currently it doesn't work; see the comment for invert_array_map()
        max_segments_to_trace = sim.MAX_TRACKS_PER_PIXEL
        self.track_pixel_map = cp.full((self.unique_pix.shape[0], max_segments_to_trace), -1)

        TPB = 32
        BPG = max(ceil(self.unique_pix.shape[0] / TPB),1)
        detsim.get_track_pixel_map2[BPG, TPB](
            self.track_pixel_map, self.unique_pix, self.neighboring_pixels, self.neighboring_radius)
        RangePop()

        (self.pixels_signals, self.pixels_tracks_signals,
         self.num_backtrack, self.offset_backtrack) = \
            detsim.launch_sum_pixel_signals(self.signals, self.selected_tracks,
                                            pixel_index_map, self.track_pixel_map)

    def digitize_and_update(self,
                            unique_pix: cpt.NDArray[cp.int32],
                            pixels_signals: cp.ndarray[tuple[int, int], cp.float64],
                            pixels_tracks_signals: cpt.NDArray[cp.float32],
                            num_backtrack: cpt.NDArray[cp.float32],
                            offset_backtrack: cpt.NDArray[cp.float32],
    ):
        """
        Helper function for calling get_adc_values and updating the results.

        Args:
            unique_pix: (n_pixels,) array of unique pixel IDs
            pixels_signals: (n_pixels,n_ticks) array of summed signals on each pixel
            pixels_tracks_signals: (n_ticks, n_pixels, n_backtracks) jagged array
                (represented as a 1D array) of signals on each pixel for each
                backtracked edep-sim segment
            num_backtrack: (len(pixels_tracks_signals),) array of the count of
                backtracked segments for each (pixel, tick)
            offset_backtrack: (len(pixels_tracks_signals),) cumulative sum of
                num_backtrack
        """
        time_ticks = cp.arange(0, len(self.unique_eventIDs) * self.max_signal_time,
                            detector.TIME_SAMPLING)
        integral_list = cp.zeros((pixels_signals.shape[0], sim.MAX_ADC_VALUES))
        adc_ticks_list = cp.zeros((pixels_signals.shape[0], sim.MAX_ADC_VALUES))
        current_fractions = cp.zeros((pixels_signals.shape[0], sim.MAX_ADC_VALUES,
                                    sim.MAX_TRACKS_PER_PIXEL))

        TPB = 4
        BPG = ceil(pixels_signals.shape[0] / TPB)
        rng_states = maybe_create_rng_states(int(TPB * BPG), seed=self.rand_seed,
                                            rng_states=self.rng_states)


        if self.pixel_thresholds_lut is not None:
            self.pixel_thresholds_lut.tpb = 128
            self.pixel_thresholds_lut.bpg = ceil(pixels_signals.shape[0]
                                            / self.pixel_thresholds_lut.tpb)
            pixel_thresholds = \
                self.pixel_thresholds_lut[unique_pix.ravel()].reshape(unique_pix.shape)
        else:
            default_threshold = detector.DISCRIMINATION_THRESHOLD * consts.units.e
            pixel_thresholds = cp.full(pixels_signals.shape[0], default_threshold)


        fee.get_adc_values[BPG, TPB](pixels_signals,
                                    pixels_tracks_signals,
                                    num_backtrack,
                                    offset_backtrack,
                                    time_ticks,
                                    integral_list,
                                    adc_ticks_list,
                                    0,
                                    rng_states,
                                    current_fractions,
                                    pixel_thresholds)

        if self.pixel_gains_lut is not None:
            pixel_gains = cp.array(self.pixel_gains_lut[unique_pix.ravel()])
            gain_list = pixel_gains[:, cp.newaxis] * cp.ones((1, sim.MAX_ADC_VALUES)) # makes array the same shape as integral_list
        else:
            gain_list = detector.GAIN

        if self.pixel_pedestals_lut is not None:
            pixel_pedestals = cp.array(self.pixel_pedestals_lut[unique_pix.ravel()])
            pedestal_list = pixel_pedestals[:, cp.newaxis] * cp.ones((1, sim.MAX_ADC_VALUES)) # makes array the same shape as integral_list
        else:
            pedestal_list = detector.V_PEDESTAL

        adc_list = fee.digitize(integral_list, gain_list, pedestal_list)

        adc_event_ids = np.full(adc_list.shape, self.unique_eventIDs[0]) # FIXME: only works if looping on a single event

        self.results_acc['event_id'].append(adc_event_ids)
        self.results_acc['adc_tot'].append(adc_list)
        self.results_acc['adc_tot_ticks'].append(adc_ticks_list)
        self.results_acc['unique_pix'].append(unique_pix)
        self.results_acc['current_fractions'].append(current_fractions)

    def maybe_compute_hybrid_ffe(self):
        if not sim.FARFIELD_ENABLED:
            return

        RangePush("far_field_contribution", 3)
        # Get pixel coordinates from pixel layout
        # unique_pix contains pixel indices; convert to (x, y) coordinates
        unique_pix_np = cp.asnumpy(self.unique_pix)
        px_idx = unique_pix_np % detector.N_PIXELS[0]
        py_idx = (unique_pix_np // detector.N_PIXELS[0]) % detector.N_PIXELS[1]
        plane_idx = unique_pix_np // (detector.N_PIXELS[0] * detector.N_PIXELS[1])
        x_min = detector.TPC_BORDERS[plane_idx.astype(int), 0, 0]
        y_min = detector.TPC_BORDERS[plane_idx.astype(int), 1, 0]
        pixel_x = cp.asarray(x_min + (px_idx + 0.5) * detector.PIXEL_PITCH, dtype=cp.float32)
        pixel_y = cp.asarray(y_min + (py_idx + 0.5) * detector.PIXEL_PITCH, dtype=cp.float32)

        active_tpc_indices = np.unique(self.all_selected_tracks['pixel_plane'].astype(np.int32))

        for tpc_idx in active_tpc_indices:
            tpc_mask = (plane_idx == tpc_idx)
            if not np.any(tpc_mask):
                continue

            ff_signals_tpc = signal_calculation.launch_ffe_kernel(
                tpc_idx=tpc_idx,
                tracks=self.all_selected_tracks,
                pixel_x=pixel_x[tpc_mask],
                pixel_y=pixel_y[tpc_mask],
                n_ticks=self.pixels_signals.shape[1],
                category=1,
                voxel_cache=self.voxel_cache)

            self.pixels_signals[tpc_mask, :] += ff_signals_tpc

        RangePop()

    def digitize_active_pix(self):
        RangePush("get_adc_values", 3)

        self.digitize_and_update(unique_pix=self.unique_pix,
                                 pixels_signals=self.pixels_signals,
                                 pixels_tracks_signals=self.pixels_tracks_signals,
                                 num_backtrack=self.num_backtrack,
                                 offset_backtrack=self.offset_backtrack)

        # Accumulate pixels processed via near-field path for this event batch
        self.processed_pixels_event = cp.unique(cp.concatenate([self.processed_pixels_event, self.unique_pix]))
        traj_pixel_map = cp.full(self.track_pixel_map.shape,-1)
        traj_pixel_map[:] = self.track_pixel_map
        traj_pixel_map[traj_pixel_map != -1] = self.selected_tracks['traj_id'][traj_pixel_map[traj_pixel_map != -1].get()]
        self.track_pixel_map[self.track_pixel_map != -1] = \
            self.selected_tracks['segment_id'][self.track_pixel_map[self.track_pixel_map != -1].get()]
        self.results_acc['traj_pixel_map'].append(traj_pixel_map)
        self.results_acc['track_pixel_map'].append(self.track_pixel_map)

    def maybe_compute_exclusive_ffe_and_digitize(self):
        if not sim.FARFIELD_ENABLED:
            return

        RangePush("far_field_induction_only", 2)
        # Pixels already processed via near-field path
        processed_pixels = self.processed_pixels_event
        # Unique TPCs in this batch (use all_selected_tracks to capture full event batch)
        active_tpc_indices_all = np.unique(self.all_selected_tracks['pixel_plane'].astype(np.int32))
        for tpc_idx in active_tpc_indices_all:
            cls = self.classification_cache.get(int(tpc_idx))
            if (cls is None or cls.induction_pixels is None or len(cls.induction_pixels) == 0):
                continue
            # Induction-only pixels for this TPC; preserve ID/coordinate ordering
            induction_pix_all = cp.asarray(cls.induction_pixels, dtype=cp.int32)
            pixel_x_all = cp.asarray(cls.induction_pixels_x, dtype=cp.float32)
            pixel_y_all = cp.asarray(cls.induction_pixels_y, dtype=cp.float32)

            if induction_pix_all.size == 0:
                continue

            if processed_pixels is not None and processed_pixels.size > 0:
                keep_mask = ~cp.isin(induction_pix_all, processed_pixels)
                induction_pix_ids = induction_pix_all[keep_mask]
                pixel_x_ff = pixel_x_all[keep_mask]
                pixel_y_ff = pixel_y_all[keep_mask]
            else:
                induction_pix_ids = induction_pix_all
                pixel_x_ff = pixel_x_all
                pixel_y_ff = pixel_y_all

            if induction_pix_ids.size == 0:
                continue

            # Use event-wide t0 max for tick extension (far-field only)
            t0_array = cp.asnumpy(self.all_selected_tracks['t0'])
            signals_ticks_t0_ff = self.signals_ticks + int(np.ceil(t0_array.max() / detector.TIME_SAMPLING))

            ff_signals = signal_calculation.launch_ffe_kernel(
                tpc_idx=tpc_idx,
                tracks=self.all_selected_tracks,
                pixel_x=pixel_x_ff,
                pixel_y=pixel_y_ff,
                n_ticks=signals_ticks_t0_ff,
                category=0,
                voxel_cache=self.voxel_cache)

            # Offset FF by event's min(t0) before digitization; normalize units if needed
            min_t0_event = float(t0_array.min())
            z_anode = float(detector.TPC_BORDERS[tpc_idx, 2, 0])
            z_cathode = float(detector.TPC_BORDERS[tpc_idx, 2, 1])
            z_span_us = abs(z_cathode - z_anode) / detector.V_DRIFT
            min_t0_event_used_us = min_t0_event / 1000.0 if (z_span_us > 0 and min_t0_event / max(z_span_us, 1e-9) > 500) else min_t0_event
            if min_t0_event_used_us != min_t0_event:
                warnings.warn("FF timing (induction-only): min(t0) appears in ns; converting to us for offset.")
            offset_ticks_ff = int(np.clip(np.ceil(min_t0_event_used_us / detector.TIME_SAMPLING), 0, signals_ticks_t0_ff))
            pixels_signals = cp.zeros_like(ff_signals)
            if offset_ticks_ff > 0:
                usable_ff = signals_ticks_t0_ff - offset_ticks_ff
                if usable_ff > 0:
                    pixels_signals[:, offset_ticks_ff:offset_ticks_ff+usable_ff] = ff_signals[:, :usable_ff]
            else:
                pixels_signals = ff_signals
            num_backtrack = cp.zeros(len(induction_pix_ids), dtype=cp.int32)
            offset_backtrack = cp.zeros(len(induction_pix_ids), dtype=cp.int32)
            pixels_tracks_signals = cp.zeros(1, dtype=cp.float32)

            self.digitize_and_update(unique_pix=induction_pix_ids,
                                     pixels_signals=pixels_signals,
                                     pixels_tracks_signals=pixels_tracks_signals,
                                     num_backtrack=num_backtrack,
                                     offset_backtrack=offset_backtrack)

            dummy_map = cp.full((len(induction_pix_ids), sim.MAX_TRACKS_PER_PIXEL), -1, dtype=cp.int32)
            self.results_acc['traj_pixel_map'].append(dummy_map)
            self.results_acc['track_pixel_map'].append(dummy_map)

    def run_light_sim(self):
        if not self.light_simulated:
            return

        RangePush("sum_light_signals", 4)
        light_inc = self.light_sim_dat[self.valid_batch_mask]
        selected_track_id = self.segment_ids_arr[self.valid_batch_mask]#cp.array(selected_tracks["segment_id"])
        n_light_ticks, light_t_start = light_sim.get_nticks(light_inc)
        n_light_ticks = min(n_light_ticks,int(5E4))
        # at least the optical channels from a whole module are activated together

        # in the mod2mod case, just take the channel indices of the first module (first two TPCs)
        # e.g. for the 2x2, op_channel = [0..96) in mod2mod mode, [0..384) otherwise
        # likewise light_inc etc. will have ndet=96 for mod2mod, ndet=384 otherwise
        # FIXME: mod2mod var?
        op_channel = light.TPC_TO_OP_CHANNEL[:2].ravel()
        op_channel = cp.array(op_channel)
        #op_channel = light_sim.get_active_op_channel(light_inc)
        n_light_det = op_channel.shape[0]
        light_sample_inc = cp.zeros((n_light_det,n_light_ticks), dtype='f4')
        light_sample_inc_true_track_id = cp.full((n_light_det, n_light_ticks, sim.MAX_MC_TRUTH_IDS), -1, dtype='i8')
        light_sample_inc_true_photons = cp.zeros((n_light_det, n_light_ticks, sim.MAX_MC_TRUTH_IDS), dtype='f8')

        ### TAKE LIMITED SEGMENTS FOR LIGHT TRUTH ###
        ### FIXME: this is a temporary fix to avoid memory issues ###
        sorted_indices = np.zeros((n_light_det, self.all_selected_tracks.shape[0]), dtype=np.int32)

        for idet in range(n_light_det):
            sorted_indices[idet] = np.argsort(light_inc[:,idet]['n_photons_det'])[::-1] # get the order in which to loop over tracks
        ### END OF TEMPORARY FIX ###

        TPB = (1,64)
        BPG = (max(ceil(light_sample_inc.shape[0] / TPB[0]),1),
                max(ceil(light_sample_inc.shape[1] / TPB[1]),1))
        light_sim.sum_light_signals[BPG, TPB](
            self.all_selected_tracks, self.track_light_voxel[self.valid_batch_mask], selected_track_id,
            light_inc, op_channel, self.lut, light_t_start, light_sample_inc, light_sample_inc_true_track_id,
            light_sample_inc_true_photons, sorted_indices, self.t0_profile_length)
        RangePop()
        if light_sample_inc_true_track_id.shape[-1] > 0 and cp.any(light_sample_inc_true_track_id[...,-1] != -1):
            warnings.warn(f"Maximum number of true segments ({sim.MAX_MC_TRUTH_IDS}) reached in backtracking info, consider increasing MAX_MC_TRUTH_IDS (larndsim/consts/light.py)")

        RangePush("sim_scintillation", 4)
        light_sample_inc_scint = cp.zeros_like(light_sample_inc)
        light_sample_inc_scint_true_track_id = cp.full_like(light_sample_inc_true_track_id, -1)
        light_sample_inc_scint_true_photons = cp.zeros_like(light_sample_inc_true_photons)
        scint_model = np.zeros(n_light_ticks, dtype=np.float32)
        light_sim.scintillation_array(scint_model)
        light_sim.calc_scintillation_effect[BPG, TPB](
            light_sample_inc, light_sample_inc_true_track_id, light_sample_inc_true_photons, light_sample_inc_scint,
            light_sample_inc_scint_true_track_id, light_sample_inc_scint_true_photons, scint_model)

        light_sample_inc_disc = cp.zeros_like(light_sample_inc)
        self.rng_states = maybe_create_rng_states(int(np.prod(TPB) * np.prod(BPG)),
                                                    seed=self.rand_seed, rng_states=self.rng_states)
        light_sim.calc_stat_fluctuations[BPG, TPB](light_sample_inc_scint, light_sample_inc_disc, self.rng_states)
        RangePop()

        RangePush("sim_light_det_response", 4)
        light_response = cp.zeros_like(light_sample_inc)
        light_response_true_track_id = cp.full_like(light_sample_inc_true_track_id, -1)
        light_response_true_photons = cp.zeros_like(light_sample_inc_true_photons)
        sipm_response = np.zeros(n_light_ticks, dtype=np.float32)
        light_sim.sipm_response_array(sipm_response) #precalculate the sipm_response
        light_sim.calc_light_detector_response[BPG, TPB](
            light_sample_inc_disc, light_sample_inc_scint_true_track_id, light_sample_inc_scint_true_photons,
            light_response, light_response_true_track_id, light_response_true_photons, self.light_gain, sipm_response)
        #light_response += cp.array(light_sim.gen_light_detector_noise(light_response.shape, light_noise[op_channel.get()]))
        RangePop()

        RangePush("sim_light_triggers", 4)
        light_threshold = cp.repeat(cp.array(light.LIGHT_TRIG_THRESHOLD)[...,np.newaxis], light.OP_CHANNEL_PER_TRIG, axis=-1)
        light_threshold = light_threshold.ravel()[op_channel.get()].copy()
        light_threshold = light_threshold.reshape(-1, light.OP_CHANNEL_PER_TRIG)[...,0]
        trigger_idx, trigger_op_channel_idx, trigger_type = light_sim.get_triggers(light_response, light_threshold, op_channel, 0)
        digit_samples = ceil(round(light.LIGHT_TRIG_WINDOW[1] + light.LIGHT_TRIG_WINDOW[0], 3) / light.LIGHT_DIGIT_SAMPLE_SPACING)
        TPB = (1,1,64)
        BPG = (max(ceil(trigger_idx.shape[0] / TPB[0]),1),
                max(ceil(trigger_op_channel_idx.shape[1] / TPB[1]),1),
                max(ceil(digit_samples / TPB[2]),1))

        light_digit_signal, light_digit_signal_true_track_id, light_digit_signal_true_photons = light_sim.sim_triggers(
            BPG, TPB, light_response, op_channel, light_response_true_track_id, light_response_true_photons, trigger_idx, trigger_op_channel_idx,
            digit_samples, self.light_noise)
        RangePop()

        self.results_acc['light_event_id'].append(cp.full(trigger_idx.shape[0], self.unique_eventIDs[0])) # FIXME: only works if looping on a single event
        self.results_acc['light_start_time'].append(cp.full(trigger_idx.shape[0], light_t_start))
        self.results_acc['light_trigger_idx'].append(trigger_idx)
        self.results_acc['trigger_type'].append(trigger_type)
        self.results_acc['light_op_channel_idx'].append(trigger_op_channel_idx)
        self.results_acc['light_waveforms'].append(light_digit_signal)
        self.results_acc['light_waveforms_true_track_id'].append(light_digit_signal_true_track_id)
        self.results_acc['light_waveforms_true_photons'].append(light_digit_signal_true_photons)

    def save(self, i_mod, final=False):
        RangePush('save_results', 1)
        if final or len(self.results_acc['event_id']) >= sim.WRITE_BATCH_SIZE:
            if len(self.results_acc['event_id']) > 0 \
               and len(np.concatenate(self.results_acc['event_id'], axis=0)) > 0:
                self.save_results(i_mod, light_only=False)
            elif len(self.results_acc['light_event_id']) > 0 \
                 and len(np.concatenate(self.results_acc['light_event_id'], axis=0)) > 0:
                self.save_results(i_mod, light_only=True)
            self.results_acc = defaultdict(list) # reinitialize after each save_results
        RangePop() # save_results

        if final:               # append segments for this module
            self.segments_to_files = np.append(self.segments_to_files, self.tracks)

    def finalize_light_output(self):
        if light.LIGHT_TRIG_MODE == 1:
            light_sim.export_merged_light_trig_to_hdf5(
                self.all_mod_tracks, self.event_times, self.output_filename,
                compression=self.compression)

        light_sim.merge_module_light_wvfm_same_trigger(self.output_filename, compression=self.compression)

    def save_truth(self):
        with h5py.File(self.output_filename, 'a') as output_file:
            # We previously called swap_coordinates(tracks), but we want to write
            # all truth info in the edep-sim convention (z = beam coordinate).
            maybe_unshift_times(self.segments_to_files)
            swap_coordinates(self.segments_to_files)

            # Store all tracks in the gdml module volume, could have small differences because of the active volume check
            output_file.create_dataset(sim.TRACKS_DSET_NAME, data=self.segments_to_files, compression=self.compression)

            # To distinguish from the "old" files that had z=drift in 'tracks':
            output_file[sim.TRACKS_DSET_NAME].attrs['zbeam'] = True

            if self.light_simulated:
                # It seems unnecessary to store (all tracks, all channels) given the modules are light tight
                for i_mod in self.mod_ids:
                    output_file.create_dataset(f'light_dat/light_dat_module{i_mod-1}',
                                               data=self.light_sim_dat_acc[i_mod-1],
                                               compression=self.compression)
            if self.trajectories:
                output_file.create_dataset("trajectories", data=self.trajectories,
                                           compression=self.compression)
            if self.vertices:
                output_file.create_dataset("vertices", data=self.vertices,
                                           compression=self.compression)
            if self.mc_hdr:
                output_file.create_dataset("mc_hdr", data=self.mc_hdr,
                                           compression=self.compression)
            if self.mc_stack:
                output_file.create_dataset("mc_stack", data=self.mc_stack,
                                           compression=self.compression)

    def save_metadata(self):
        with h5py.File(self.output_filename, 'a') as output_file:
            if 'configs' in output_file.keys():
                output_file['configs'].attrs['pixel_layout'] = self.pixel_layout

            # Store Git/version information as attributes in the output file
            output_file.attrs['VERSION'] = _version.__version__
            output_file.attrs['GIT_COMMIT'] = _version.GIT_COMMIT[1:] # Remove the leading 'g' char to get pure commit hash
            output_file.attrs['GIT_BRANCH'] = _version.GIT_BRANCH
            output_file.attrs['GIT_DISTANCE'] = _version.GIT_DISTANCE
            output_file.attrs['GIT_TAG'] = _version.GIT_TAG

    def bye(self):
        print("Output saved in:", self.output_filename)
        end_simulation = time()
        print(f"Elapsed time: {end_simulation-self.start_simulation:.2f} s")

    def run(self):
        print("******************\nRUNNING SIMULATION\n******************")
        self.start_simulation = time()

        for i_mod in self.mod_ids: # Conventional module counting starts from 1
            self.init_module(i_mod)

            RangePush("run_simulation")

            quenching.launch_quench(self.tracks, physics.BIRKS)
            drifting.launch_drift(self.tracks)

            if self.light_simulated:
                self.load_light_info(i_mod)
                self.calc_light_inc()
                
            batcher = batching.TPCBatcher(
                self.all_mod_tracks, self.tracks, sim.EVENT_SEPARATOR,
                tpc_batch_size=sim.EVENT_BATCH_SIZE,
                tpc_borders=self.module_borders)

            for ievd, batch_mask, is_new_event in \
                    tqdm(batcher, desc='Simulating batches...', ncols=80, smoothing=0):
                non_empty = self.setup_event_batch(i_mod, ievd, batch_mask, is_new_event)
                if not non_empty:
                    continue

                self.maybe_farfield_precompute()

                pixel_ranges = batching.subbatch_pixel_ranges(self.assmap_pix2seg,
                                                              sim.SEGMENT_BATCH_SIZE)

                for start_pix, stop_pix in \
                        tqdm(pixel_ranges, delay=1,
                             desc='  Simulating event %i batches...' % ievd,
                             leave=False, ncols=80):
                    non_empty = self.prepare_subbatch_pixels(start_pix, stop_pix)
                    if not non_empty:
                        continue

                    self.call_tracks_current_mc()
                    self.call_sum_pixel_signals()
                    self.maybe_compute_hybrid_ffe()
                    self.digitize_active_pix()
                    self.maybe_compute_exclusive_ffe_and_digitize()
                    # End loop over sub-batches (pixels)

                if self.light_simulated:
                    self.run_light_sim()

                self.save(i_mod)
                # End loop over batches (events, either TPC-by-TPC or whole-module)
            RangePop()                      # run_simulation
            self.save(i_mod, final=True)
            # End loop over modules

        if self.light_simulated:
            self.finalize_light_output()
        self.save_truth()
        self.save_metadata()
        self.bye()


def main():
    maybe_disable_cupy_mempool()
    configure_warnings()

    RangePush("simulate_pixels")
    app = fire.Fire(LArND_Sim)
    app.run()
    RangePop()


if __name__ == "__main__":
    main()
    
