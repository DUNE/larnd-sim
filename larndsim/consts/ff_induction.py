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

# Far-field segment split step size (1 mm)
FAR_FIELD_SEGMENT_STEP_CM = 0.5  # cm

# Minimum induction signal threshold (electrons)
INDUCTION_SIGNAL_THRESHOLD = 2000.0  # e-

####################################
# Induced Current Normalization
####################################

# How many terms to include in the z-dipole expansion
DIPOLE_N_TERMS = 5

# Normalization constant for dipole; found emperically
# ~1.50 for infinite-plane dipole
# ~1.38 for lattice dipole
DIPOLE_SCALE = 1.5

# Detector half-width in X/Y for the lattice image calculation
# FIXME: Should use geometry info rather than values specified here
LATTICE_LX = 238.0
LATTICE_LY = 149.0

# Number of X/Y terms to include in the dipole expansion
LATTICE_N_TERMS_XY = 10
