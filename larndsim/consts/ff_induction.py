"""
Set far-field induction options
"""

import yaml

from .units import cm

###################
# Voxel Resolution
###################

#: Coarse voxel size in X (transverse to drift)
COARSE_VOXEL_SIZE_X = 0.5 * cm  # cm

#: Coarse voxel size in Y (transverse to drift)
COARSE_VOXEL_SIZE_Y = 0.5 * cm  # cm

#: Coarse voxel size in Z (drift direction)
COARSE_VOXEL_SIZE_Z = 0.5 * cm  # cm, matches XY default

#####################################
# Pixel pre-classification parameters
#####################################

# Charge collection pixel radius
CHARGE_COLLECTION_RADIUS = 1  # pixels

# Neighbor pixel radius
CHARGE_NEIGHBOR_RADIUS = 2  # pixels

# Maximum induction consideration distance
INDUCTION_CUTOFF_RADIUS = 50 # cm

# Far-field segment split step size (1 mm)
FAR_FIELD_SEGMENT_STEP_CM = 0.1  # cm

# Minimum induction signal threshold (electrons)
INDUCTION_SIGNAL_THRESHOLD = 2000.0  # e-

####################################
# Induced Current Normalization
####################################

# How many terms to include in the dipole expansion
DIPOLE_N_TERMS = 5


def set_ff_induction_properties(ffprop_file: str):
    """Load far-field induction properties.

    Args:
        ffprop_file: YAML filename
    """
    global COARSE_VOXEL_SIZE_X
    global COARSE_VOXEL_SIZE_Y
    global COARSE_VOXEL_SIZE_Z
    global CHARGE_COLLECTION_RADIUS
    global CHARGE_NEIGHBOR_RADIUS
    global INDUCTION_CUTOFF_RADIUS
    global FAR_FIELD_SEGMENT_STEP_CM
    global INDUCTION_SIGNAL_THRESHOLD
    global DIPOLE_N_TERMS

    with open(ffprop_file) as df:
        ffprop = yaml.load(df, Loader=yaml.FullLoader)

    COARSE_VOXEL_SIZE_X = ffprop.get('coarse_voxel_size_x', COARSE_VOXEL_SIZE_X)
    COARSE_VOXEL_SIZE_Y = ffprop.get('coarse_voxel_size_y', COARSE_VOXEL_SIZE_Y)
    COARSE_VOXEL_SIZE_Z = ffprop.get('coarse_voxel_size_z', COARSE_VOXEL_SIZE_Z)
    CHARGE_COLLECTION_RADIUS = ffprop.get('charge_collection_radius', CHARGE_COLLECTION_RADIUS)
    CHARGE_NEIGHBOR_RADIUS = ffprop.get('charge_neighbor_radius', CHARGE_NEIGHBOR_RADIUS)
    INDUCTION_CUTOFF_RADIUS = ffprop.get('induction_cutoff_radius', INDUCTION_CUTOFF_RADIUS)
    FAR_FIELD_SEGMENT_STEP_CM = ffprop.get('far_field_segment_step_cm', FAR_FIELD_SEGMENT_STEP_CM)
    INDUCTION_SIGNAL_THRESHOLD = ffprop.get('induction_signal_threshold', INDUCTION_SIGNAL_THRESHOLD)
    DIPOLE_N_TERMS = ffprop.get('dipole_n_terms', DIPOLE_N_TERMS)
