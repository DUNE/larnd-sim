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
EVENT_SEPARATOR = 'event_id'  # 'spillID' or 'vertexID'

IS_SPILL_SIM = True
IF_ACTIVE_VOLUME_CHECK = False
SPILL_PERIOD = 1.2e6  # units = microseconds
TRACKS_DSET_NAME = 'segments'
LRS_TRIG_TO_SINGLE_PACMAN = True

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
    global IF_ACTIVE_VOLUME_CHECK
    global SPILL_PERIOD
    global MAX_EVENTS_PER_FILE
    global TRACKS_DSET_NAME
    global MOD2MOD_VARIATION
    global LRS_TRIG_TO_SINGLE_PACMAN

    with open(simprop_file) as df:
        simprop = yaml.load(df, Loader=yaml.FullLoader)

    try:
        BATCH_SIZE = simprop.get('batch_size', BATCH_SIZE)
        EVENT_BATCH_SIZE = simprop.get('event_batch_size', EVENT_BATCH_SIZE)
        WRITE_BATCH_SIZE = simprop.get('write_batch_size', WRITE_BATCH_SIZE)
        EVENT_SEPARATOR = simprop.get('event_separator', EVENT_SEPARATOR)
        IS_SPILL_SIM = bool(simprop.get('is_spill_sim', IS_SPILL_SIM))
        IF_ACTIVE_VOLUME_CHECK = bool(simprop.get('if_active_volume_check', IF_ACTIVE_VOLUME_CHECK))
        SPILL_PERIOD = float(simprop.get('spill_period', SPILL_PERIOD))
        MAX_EVENTS_PER_FILE = simprop.get('max_events_per_file', MAX_EVENTS_PER_FILE)
        TRACKS_DSET_NAME = simprop.get('tracks_dset_name', TRACKS_DSET_NAME)
        LRS_TRIG_TO_SINGLE_PACMAN = bool(simprop.get('lrs_trig_to_single_pacman', LRS_TRIG_TO_SINGLE_PACMAN))
    except:
        print("Check if all the necessary simulation properties are set. Taking some default values")
