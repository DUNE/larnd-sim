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

# How many terms to include in the z-dipole expansion
DIPOLE_N_TERMS = 5

# Normalization constant for dipole; found emperically
# ~1.50 for infinite-plane dipole
# ~1.38 for lattice dipole
DIPOLE_SCALE = 1.5

# Detector half-width in X/Y for the lattice image calculation
# FIXME: Should use geometry info rather than (FSD Cube) values specified here
LATTICE_LX = 23.80              # cm
LATTICE_LY = 14.90              # cm

# Number of X/Y terms to include in the dipole expansion
LATTICE_N_TERMS_XY = 10

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
    global DIPOLE_SCALE
    global LATTICE_LX
    global LATTICE_LY
    global LATTICE_N_TERMS_XY

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
    DIPOLE_SCALE = ffprop.get('dipole_scale', DIPOLE_SCALE)
    LATTICE_LX = ffprop.get('lattice_lx', LATTICE_LX)
    LATTICE_LY = ffprop.get('lattice_ly', LATTICE_LY)
    LATTICE_N_TERMS_XY = ffprop.get('lattice_n_terms_xy', LATTICE_N_TERMS_XY)
