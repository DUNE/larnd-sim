"""
Set simulation options
"""

import numpy as np
import yaml

from collections import defaultdict

from .units import mm, cm, V, kV

BATCH_SIZE = 10000    # units = track segments
EVENT_BATCH_SIZE = 1  # units = N tpcs
WRITE_BATCH_SIZE = 1  # units = N batches
EVENT_SEPARATOR = 'vertexID'  # 'spillID' or 'vertexID'

IS_SPILL_SIM = True
SPILL_PERIOD = 1.2e6  # units = microseconds

# We mod event IDs by MAX_EVENTS_PER_FILE to get zero-based IDs for indexing
# purposes; see comments in simulate_pixels.py
MAX_EVENTS_PER_FILE = 1000

def set_simulation_properties(simprop_file):
    """
    The function loads the detector properties and
    the pixel geometry YAML files and stores the constants
    as global variables

    Args:
        detprop_file (str): detector properties YAML
            filename
        pixel_file (str): pixel layout YAML filename
    """
    global BATCH_SIZE
    global EVENT_BATCH_SIZE
    global WRITE_BATCH_SIZE
    global EVENT_SEPARATOR
    global IS_SPILL_SIM
    global SPILL_PERIOD
    global MAX_EVENTS_PER_FILE
    

    with open(simprop_file) as df:
        simprop = yaml.load(df, Loader=yaml.FullLoader)

    BATCH_SIZE = simprop['batch_size']
    EVENT_BATCH_SIZE = simprop['event_batch_size']
    WRITE_BATCH_SIZE = simprop['write_batch_size']
    EVENT_SEPARATOR = simprop['event_separator']
    IS_SPILL_SIM = simprop['is_spill_sim']
    SPILL_PERIOD = float(simprop['spill_period'])
    MAX_EVENTS_PER_FILE = simprop['max_events_per_file']
