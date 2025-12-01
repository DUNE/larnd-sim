from .units import cm, microsecond, e

###################
# Voxel Resolution
###################

#: Coarse voxel size in X (transverse to drift)
COARSE_VOXEL_SIZE_X = 5.0 * cm  # cm

#: Coarse voxel size in Y (transverse to drift)
COARSE_VOXEL_SIZE_Y = 5.0 * cm  # cm

#: Coarse voxel size in drift time (can try different values)
COARSE_VOXEL_SIZE_T = 0.1 * microsecond  # us, matches detector.TIME_SAMPLING

def set_mesh_parameters(voxel_size_x=None, voxel_size_y=None, voxel_size_t=None):
    """
    To be updated to read from YAML
    """
    global COARSE_VOXEL_SIZE_X, COARSE_VOXEL_SIZE_Y, COARSE_VOXEL_SIZE_T
    
    if voxel_size_x is not None:
        COARSE_VOXEL_SIZE_X = voxel_size_x
    if voxel_size_y is not None:
        COARSE_VOXEL_SIZE_Y = voxel_size_y
    if voxel_size_t is not None:
        COARSE_VOXEL_SIZE_T = voxel_size_t
