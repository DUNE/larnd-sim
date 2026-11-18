"""Recreate configuration-dependent consumers after installing effective inputs."""

import importlib
import sys

# Providers precede consumers of directly imported functions (e.g. id2pixel).
_CONSUMERS = (
    'pixels_from_track',
    'active_volume',
    'quenching',
    'drifting',
    'detsim',
    'light_sim',
    'lightLUT',
    'fee',
    'far_field.voxelization',
    'far_field.pixel_classifier',
    'far_field.signal_calculation',
)


def prepare_simulation():
    """Refresh imported CPU/CUDA dispatchers and import-time defaults.

    Synchronise outstanding GPU work before installing new constants. Call this
    afterwards, before invoking any configuration-dependent compiled consumer.
    Module objects are retained, but old directly imported function references
    must not be used by callers across this boundary.
    """
    for name in _CONSUMERS:
        module = sys.modules.get(f'larndsim.{name}')
        if module is not None:
            importlib.reload(module)
