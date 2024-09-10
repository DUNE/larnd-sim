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
SPILL_PERIOD = 1.2e6  # units = microseconds
TRACKS_DSET_NAME = 'segments'

# We mod event IDs by MAX_EVENTS_PER_FILE to get zero-based IDs for indexing
# purposes; see comments in simulate_pixels.py
MAX_EVENTS_PER_FILE = 1000

# See larndsim/detsim.py
MAX_TRACKS_PER_PIXEL = 50
MIN_STEP_SIZE = 0.001 # cm
MC_SAMPLE_MULTIPLIER = 1

#: Number of true segments to track for each time tick (`MAX_MC_TRUTH_IDS=0` to disable complete truth tracking)
MAX_MC_TRUTH_IDS = 0 # higher is better, but file size increases
#: Threshold for propogating truth information on a given SiPM
MC_TRUTH_THRESHOLD = 0.1 # pe/us lower is better, but memory usage increases

def set_simulation_properties(simprop_file):
    """
    The function loads the detector properties and
    the pixel geometry YAML files and stores the constants
    as global variables

    Args:
        simprop_file (str): detector properties YAML
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
    global TRACKS_DSET_NAME
    global MOD2MOD_VARIATION

    global MAX_TRACKS_PER_PIXEL
    global MIN_STEP_SIZE
    global MC_SAMPLE_MULTIPLIER

    global MAX_MC_TRUTH_IDS
    global MC_TRUTH_THRESHOLD

    with open(simprop_file) as df:
        simprop = yaml.load(df, Loader=yaml.FullLoader)

    try:
        BATCH_SIZE = simprop.get('batch_size', BATCH_SIZE)
        EVENT_BATCH_SIZE = simprop.get('event_batch_size', EVENT_BATCH_SIZE)
        WRITE_BATCH_SIZE = simprop.get('write_batch_size', WRITE_BATCH_SIZE)
        EVENT_SEPARATOR = simprop.get('event_separator', EVENT_SEPARATOR)
        IS_SPILL_SIM = bool(simprop.get('is_spill_sim', IS_SPILL_SIM))
        SPILL_PERIOD = float(simprop.get('spill_period', SPILL_PERIOD))
        MAX_EVENTS_PER_FILE = simprop.get('max_events_per_file', MAX_EVENTS_PER_FILE)
        TRACKS_DSET_NAME = simprop.get('tracks_dset_name', TRACKS_DSET_NAME)

        MAX_TRACKS_PER_PIXEL = simprop.get('max_tracks_per_pixel', MAX_TRACKS_PER_PIXEL)
        MIN_STEP_SIZE = simprop.get('min_step_size', MIN_STEP_SIZE)
        MC_SAMPLE_MULTIPLIER = simprop.get('mc_sample_multiplier', MC_SAMPLE_MULTIPLIER)

        MAX_MC_TRUTH_IDS = simprop.get('max_light_truth_ids', MAX_MC_TRUTH_IDS)
        MC_TRUTH_THRESHOLD = simprop.get('mc_truth_threshold', MC_TRUTH_THRESHOLD)
    except:
        print("Check if all the necessary simulation properties are set. Taking some default values")
