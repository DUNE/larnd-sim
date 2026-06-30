#!/usr/bin/env python3

import os
import warnings

import cupy as cp
from numba.core.errors import NumbaPerformanceWarning
from numba.cuda import device_array
from numba.cuda.random import create_xoroshiro128p_states
import numpy as np
import tqdm


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
        tqdm.write(warning_str_format(str(message), category, filename, lineno))

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


def load_mod2mod_variation_properties(cfg_files, ids, n_modules, message=""):
    if cfg_files is None:
        return None

    if ids is None:
        if isinstance(cfg_files, list) and len(cfg_files) != n_modules:
            raise KeyError(f"Simulation with module variation activated, but the number of {message} is incorrect!")
        elif isinstance(cfg_files, list) and len(cfg_files) == n_modules:
            warnings.warn("Simulation with module variation activated, using default orders for the {message}.")
    else:
        if not isinstance(cfg_files, list) or len(ids) != n_modules or max(ids) >= len(cfg_files):
            raise KeyError(f"Simulation with module variation activated, but the number of pointer for {message} is incorrect!")
        else:
            module_files = [cfg_files[idx] for idx in ids]
            cfg_files = module_files

    return cfg_files
