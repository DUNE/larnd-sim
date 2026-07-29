#!/usr/bin/env python3

import os
import warnings

import cupy as cp
from numba.core.errors import NumbaPerformanceWarning
from numba.cuda import device_array
from numba.cuda.random import create_xoroshiro128p_states
import numpy as np
import numpy.lib.recfunctions as rfn
import tqdm as tqdm_mod

from larndsim import fee
from larndsim.consts import sim


LOGO = r"""
  _                      _            _
 | |                    | |          (_)
 | | __ _ _ __ _ __   __| |______ ___ _ _ __ ___
 | |/ _` | '__| '_ \ / _` |______/ __| | '_ ` _ \
 | | (_| | |  | | | | (_| |      \__ \ | | | | | |
 |_|\__,_|_|  |_| |_|\__,_|      |___/_|_| |_| |_|

"""


def maybe_disable_cupy_mempool():
    """Disable the CuPy memory pool if requested.

    If the environment variable `LARNDSIM_DISABLE_CUPY_MEMPOOL` is set (to
    anything), the CuPy memory pool will be disabled. Disabling the memory pool
    is useful when profiling, making it easier to match memory spikes to the
    responsible allocations.
    """
    if os.getenv('LARNDSIM_DISABLE_CUPY_MEMPOOL'):
        # Disable memory pool for device memory (GPU):
        cp.cuda.set_allocator(None)
        # Disable memory pool for pinned memory (CPU):
        cp.cuda.set_pinned_memory_allocator(None)


def configure_warnings():
    """Improve the appearance of warnings; play nice with tqdm."""
    def warning_str_format(message, category, filename, lineno, line=None):
        # Get the last few parts of the filepath for less clutter when printing
        splitname = "/".join(filename.split('/')[-3:])
        return f"\033[33m{splitname}:{lineno}: {category.__name__}: {message}\033[0m"

    # Play nice with loops wrapped with tqdm; using tqdm.write() prints the warning on its own line
    def tqdm_show_warning(message, category, filename, lineno, file=None, line=None):
        tqdm_mod.write(warning_str_format(str(message), category, filename, lineno))

    warnings.formatwarning = warning_str_format
    warnings.showwarning = tqdm_show_warning

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
        new_states[len(rng_states):] = create_xoroshiro128p_states(n - len(rng_states), seed=seed, subsequence_start = len(rng_states))
        return new_states

    return rng_states


def maybe_add_t0(tracks):
    """Make "t0" attribute, if it doesn't exist"""
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
    return tracks


def maybe_shift_times(tracks):
    """Subtract time of spill-start.

    larnd-sim uses "t0" in a way that 0 is the "trigger" time (e.g spill time)
    Therefore, to run the detector simulation we reset the t0 to reflect that
    When storing the mc truth, revert this change and store the "real" segment time
    The event times are added to segments in the spill building stage. This step is not needed for non-beam simulation
    """
    if sim.IS_SPILL_SIM:
        # "Reset" the spill period so t0 is wrt the corresponding spill start time.
        # The spill starts are marking the start of
        # The space between spills will be accounted for in the
        # packet timestamps through the event_times array below
        localSpillIDs = tracks[sim.EVENT_SEPARATOR] - (tracks[sim.EVENT_SEPARATOR] // sim.MAX_EVENTS_PER_FILE) * sim.MAX_EVENTS_PER_FILE
        tracks['t0_start'] = tracks['t0_start'] - localSpillIDs*sim.SPILL_PERIOD
        tracks['t0_end'] = tracks['t0_end'] - localSpillIDs*sim.SPILL_PERIOD
        tracks['t0'] = tracks['t0'] - localSpillIDs*sim.SPILL_PERIOD


def maybe_unshift_times(tracks):
    """Revert the mc truth information modified for larnd-sim consumption .

    If the event time is generated by larndsim (non-beam cases), then the t0 is
    relative to the event time (0 ish, assume the edep particle window is O(us))
    so there is no need to remove the event time.
    """
    if sim.IS_SPILL_SIM:
        # write the true timing structure to the file, not t0 wrt event time .....
        localSpillIDs = tracks[sim.EVENT_SEPARATOR] - (tracks[sim.EVENT_SEPARATOR] // sim.MAX_EVENTS_PER_FILE) * sim.MAX_EVENTS_PER_FILE
        tracks['t0_start'] = tracks['t0_start'] + localSpillIDs*sim.SPILL_PERIOD
        tracks['t0_end'] = tracks['t0_end'] + localSpillIDs*sim.SPILL_PERIOD
        tracks['t0'] = tracks['t0'] + localSpillIDs*sim.SPILL_PERIOD


def maybe_read_array(f, key):
    if key in f:
        return np.array(f[key])
    print(f'Input file does not have {key}')


def remove_neutrals(tracks):
    """Filter out neutrons and gammas, which will not directly create visible charge or light.

    (excluding these segments here results in a modest ~10% improvement to
    memory usage later on, since this reduces the size of the arrays CUDA must
    initialize for pixel current calculations)
    """
    neutrals_mask = (tracks['pdg_id'] != 2112) & (tracks['pdg_id'] != 22)
    if sum(~neutrals_mask) > 0:
        print("Rejected ",sum(~neutrals_mask), "track segments from neutral particles")
    return tracks[neutrals_mask]


def remove_delayed_segments(tracks):
    """Filter out highly-delayed segments"""
    t0_delay_mask = (tracks['t0'] < sim.MAX_SEGMENT_T0)
    if sum(~t0_delay_mask) > 0:
        print("Rejected ",sum(~t0_delay_mask)," highly-delayed segments with T0 > ",sim.MAX_SEGMENT_T0," us: ")
    return tracks[t0_delay_mask] 


def maybe_add_segment_ids(tracks):
    if 'segment_id' in tracks.dtype.names:
        return tracks

    dtype = tracks.dtype.descr
    dtype = [('segment_id','u4')] + dtype
    new_tracks = np.empty(tracks.shape, dtype=np.dtype(dtype, align=True))
    new_tracks['segment_id'] = np.arange(tracks.shape[0], dtype='u4')
    for field in dtype[1:]:
        new_tracks[field[0]] = tracks[field[0]]
    return new_tracks


def maybe_add_n_photons(tracks):
    """Make "n_photons" attribute, if it doesn't exist."""
    if 'n_photons' not in tracks.dtype.names:
        n_photons = np.zeros(tracks.shape[0], dtype=[('n_photons', 'f4')])
        return rfn.merge_arrays((tracks, n_photons), flatten=True)


def maybe_set_vertex_times(vertices, event_times):
    """Broadcast the event times to vertices."""
    if vertices and not sim.IS_SPILL_SIM:
        # create "t_event" in vertices dataset in case it doesn't exist
        if 't_event' not in vertices.dtype.names:
            dtype = vertices.dtype.descr
            dtype = [("t_event","f4")] + dtype
            new_vertices = np.empty(vertices.shape, dtype=np.dtype(dtype, align=True))
            for field in dtype[1:]:
                if len(field[0]) == 0:
                    continue
                new_vertices[field[0]] = vertices[field[0]]
            vertices = new_vertices
        uniq_ev, counts = np.unique(vertices[sim.EVENT_SEPARATOR], return_counts=True)
        event_times_in_use = cp.take(event_times, uniq_ev)
        vertices['t_event'] = np.repeat(event_times_in_use.get(),counts)
    return vertices


def maybe_set_mc_hdr_times(mc_hdr, vertices):
    """Copy the event times to mc_hdr."""
    if mc_hdr and vertices:
        if 't_event' not in mc_hdr.dtype.names:
            dtype = mc_hdr.dtype.descr
            dtype = [("t_event","f4")] + dtype
            new_mc_hdr = np.empty(mc_hdr.shape, dtype=np.dtype(dtype, align=True))
            for field in dtype[1:]:
                if len(field[0]) == 0
                continue
                new_mc_hdr[field[0]] = mc_hdr[field[0]]
            mc_hdr = new_mc_hdr
        mc_hdr['t_event'] = vertices['t_event']
        if len(vertices[sim.EVENT_SEPARATOR]) != len(mc_hdr[sim.EVENT_SEPARATOR]):
            raise ValueError("vertices and mc_hdr datasets have different number of vertices! The number should be the same.")
    return mc_hdr


# Event IDs may have some offset (e.g. to make them globally unique within
# an MC production), which we assume to be a multiple of
# sim.MAX_EVENTS_PER_FILE. We remove this offset by taking the modulus with
# sim.MAX_EVENTS_PER_FILE, which gives us zero-based "local" event IDs that
# we can use when indexing into event_times. Note that num_evids is actually
# an upper bound on the number of events, since there may be gaps due to
# events that didn't deposit any energy in the LAr. Such gaps are harmless.
def prep_event_times(tracks):
    num_evids = (tracks[sim.EVENT_SEPARATOR].max() % sim.MAX_EVENTS_PER_FILE) + 1
    if sim.IS_SPILL_SIM:
        return cp.arange(num_evids) * sim.SPILL_PERIOD
    else:
        return fee.gen_event_times(num_evids) # change non-beam event time offset with detector.NON_BEAM_EVENT_GAP
