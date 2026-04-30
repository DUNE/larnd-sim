"""
Set simulation options
"""

import numpy as np
import yaml

from collections import defaultdict

from .units import mm, cm, V, kV

SEGMENT_BATCH_SIZE = 10000    # units = track segments
EVENT_BATCH_SIZE = 1          # units = N tpcs
WRITE_BATCH_SIZE = 1          # units = N batches
EVENT_SEPARATOR = 'event_id'  # 'spillID' or 'vertexID'

# Filter out highly-delayed track segments
MAX_SEGMENT_T0 = 30 # microseconds

IS_SPILL_SIM = True
SPILL_PERIOD = 1.2e6  # units = microseconds
TRACKS_DSET_NAME = 'segments'

# We mod event IDs by MAX_EVENTS_PER_FILE to get zero-based IDs for indexing
# purposes; see comments in simulate_pixels.py
MAX_EVENTS_PER_FILE = 1000

# See larndsim/detsim.py
MAX_TRACKS_PER_PIXEL = 50

#: Number of back-tracked segments to be recorded
ASSOCIATION_COUNT_TO_STORE = 20
#: Maximum number of ADC values stored per pixel
MAX_ADC_VALUES = 30

#: Number of true segments to track for each time tick (`MAX_MC_TRUTH_IDS=0` to disable complete truth tracking)
MAX_MC_TRUTH_IDS = 0 # higher is better, but file size increases
#: Threshold for propogating truth information on a given SiPM
MC_TRUTH_THRESHOLD = 0.1 # pe/us lower is better, but memory usage increases

FARFIELD_ENABLED = False
FARFIELD_MODE = 'segments'

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
    global SEGMENT_BATCH_SIZE
    global EVENT_BATCH_SIZE
    global WRITE_BATCH_SIZE
    global EVENT_SEPARATOR
    global MAX_SEGMENT_T0
    global IS_SPILL_SIM
    global SPILL_PERIOD
    global MAX_EVENTS_PER_FILE
    global TRACKS_DSET_NAME
    global MOD2MOD_VARIATION

    global MAX_TRACKS_PER_PIXEL

    global ASSOCIATION_COUNT_TO_STORE
    global MAX_ADC_VALUES

    global MAX_MC_TRUTH_IDS
    global MC_TRUTH_THRESHOLD

    global FARFIELD_ENABLED
    global FARFIELD_MODE

    with open(simprop_file) as df:
        simprop = yaml.load(df, Loader=yaml.FullLoader)

    SEGMENT_BATCH_SIZE = simprop.get('segment_batch_size', SEGMENT_BATCH_SIZE)
    EVENT_BATCH_SIZE = simprop.get('event_batch_size', EVENT_BATCH_SIZE)
    WRITE_BATCH_SIZE = simprop.get('write_batch_size', WRITE_BATCH_SIZE)
    EVENT_SEPARATOR = simprop.get('event_separator', EVENT_SEPARATOR)
    MAX_SEGMENT_T0 = simprop.get('max_segment_t0', MAX_SEGMENT_T0)
    IS_SPILL_SIM = bool(simprop.get('is_spill_sim', IS_SPILL_SIM))
    SPILL_PERIOD = float(simprop.get('spill_period', SPILL_PERIOD))
    MAX_EVENTS_PER_FILE = simprop.get('max_events_per_file', MAX_EVENTS_PER_FILE)
    TRACKS_DSET_NAME = simprop.get('tracks_dset_name', TRACKS_DSET_NAME)

    MAX_TRACKS_PER_PIXEL = simprop.get('max_tracks_per_pixel', MAX_TRACKS_PER_PIXEL)

    ASSOCIATION_COUNT_TO_STORE = simprop.get('association_count_to_store', ASSOCIATION_COUNT_TO_STORE)
    MAX_ADC_VALUES = simprop.get('max_adc_values', MAX_ADC_VALUES)

    MAX_MC_TRUTH_IDS = simprop.get('max_light_truth_ids', MAX_MC_TRUTH_IDS)
    MC_TRUTH_THRESHOLD = simprop.get('mc_truth_threshold', MC_TRUTH_THRESHOLD)

    FARFIELD_ENABLED = bool(simprop.get('farfield_enabled', FARFIELD_ENABLED))

    FARFIELD_MODE = simprop.get('farfield_mode', FARFIELD_MODE)
    options = ['voxels', 'segments', 'segments-srwf']
    if FARFIELD_MODE not in options:
        raise RuntimeError(f"Invalid farfield_mode {FARFIELD_MODE}; " +
                            f"must be one of {options}")
