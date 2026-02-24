from .units import cm, microsecond, e

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

# Minimum induction signal threshold (electrons)
INDUCTION_SIGNAL_THRESHOLD = 1000.0  # e-

####################################
# Induced Current Normalization
####################################

#: Global scale factor applied to induced current calculation (dimensionless)
INDUCED_CURRENT_SCALE = 0.03


def set_mesh_parameters(voxel_size_x=None, voxel_size_y=None, voxel_size_t=None, 
                        charge_collection_radius=None, charge_neighbor_radius=None, 
                        induction_cutoff_radius=None, induction_signal_threshold=None,
                        induced_current_scale=None):
    """
    To be updated to read from YAML
    """
    global COARSE_VOXEL_SIZE_X, COARSE_VOXEL_SIZE_Y, COARSE_VOXEL_SIZE_Z
    global CHARGE_COLLECTION_RADIUS, CHARGE_NEIGHBOR_RADIUS
    global INDUCTION_CUTOFF_RADIUS, INDUCTION_SIGNAL_THRESHOLD, INDUCED_CURRENT_SCALE
    
    if voxel_size_x is not None:
        COARSE_VOXEL_SIZE_X = voxel_size_x
    if voxel_size_y is not None:
        COARSE_VOXEL_SIZE_Y = voxel_size_y
    if voxel_size_t is not None:
        COARSE_VOXEL_SIZE_Z = voxel_size_t
    if charge_collection_radius is not None:
        CHARGE_COLLECTION_RADIUS = charge_collection_radius
    if charge_neighbor_radius is not None:          
        CHARGE_NEIGHBOR_RADIUS = charge_neighbor_radius
    if induction_cutoff_radius is not None:
        INDUCTION_CUTOFF_RADIUS = induction_cutoff_radius
    if induction_signal_threshold is not None:
        INDUCTION_SIGNAL_THRESHOLD = induction_signal_threshold
    if induced_current_scale is not None:
        INDUCED_CURRENT_SCALE = induced_current_scale
