"""
Mesh refinement constants and parameters

Defines the mesh scales and thresholds for the multi-scale signal calculation.
These parameters control the boundaries between very-near, near, and far field
calculations.

References:
- Refactoring document section on "Refactoring Pseudocode"
- Physics assumptions in "Validating Assumptions"
"""

from .units import cm, microsecond, e

###################
# Mesh Scale Parameters
###################

#: Very near-field region (full diffusion)
#: Distance in drift direction where diffusion matters (~2cm)
VERY_NEAR_FIELD_DRIFT = 2.0 * cm  # cm

#: Transverse distance for very near-field (~1cm)
VERY_NEAR_FIELD_TRANSVERSE = 1.0 * cm  # cm

#: Near-field region (coarse calculation, no diffusion)
#: Distance in drift direction for discrete segment effects (~10cm)
NEAR_FIELD_DRIFT = 10.0 * cm  # cm

#: Transverse distance for near-field (~3cm)
NEAR_FIELD_TRANSVERSE = 3.0 * cm  # cm

#: Far-field region (aggregate calculation)
#: Distance in drift direction for far-field (~50cm)
FAR_FIELD_DRIFT = 50.0 * cm  # cm

#: Transverse distance for far-field (~1m = 100cm)
FAR_FIELD_TRANSVERSE = 100.0 * cm  # cm

###################
# Voxel Resolution
###################

#: Coarse voxel size in X (transverse to drift) (~5cm)
COARSE_VOXEL_SIZE_X = 5.0 * cm  # cm

#: Coarse voxel size in Y (transverse to drift) (~5cm)
COARSE_VOXEL_SIZE_Y = 5.0 * cm  # cm

#: Coarse voxel size in time (1 time tick, will be set from detector constants)
COARSE_VOXEL_SIZE_T = 0.1 * microsecond  # us, matches detector.TIME_SAMPLING

#: Fine voxel size for very near-field (smaller than coarse)
FINE_VOXEL_SIZE_X = 0.5 * cm  # cm
FINE_VOXEL_SIZE_Y = 0.5 * cm  # cm
FINE_VOXEL_SIZE_T = 0.05 * microsecond  # us, matches detector.RESPONSE_SAMPLING

###################
# Signal Thresholds
###################

#: Threshold for induction-only pixel consideration (peak signal amplitude)
#: Risk of triggering on induction-only signal
INDUCTION_THRESHOLD = 1.0e3 * e  # e- (electrons)

#: Threshold for charge-collection pixels (noise/bias in charge measurement)
COLLECTION_THRESHOLD = 100.0 * e  # e- (electrons)

#: Upper bound calculation uses 1/r^2 heuristic for quick prefiltering
#: This is a conservative estimate for induction signal screening
USE_INVERSE_R2_HEURISTIC = True

###################
# Mesh Refinement Strategy
###################

#: Enable mesh refinement (if False, use legacy MC method)
ENABLE_MESH_REFINEMENT = False  # Set to False by default, enable via config

#: Use analytic Gaussian convolution for diffusion (vs MC sampling)
USE_GAUSSIAN_DIFFUSION = True

#: Use Fourier convolution if signal convolution is expensive
USE_FOURIER_CONVOLUTION = False

#: Gauss-Hermite quadrature order for potential future use
GAUSS_HERMITE_ORDER = 5

###################
# Memory and Performance
###################

#: Maximum number of voxels to process in a single batch
MAX_VOXELS_PER_BATCH = 100000

#: Sparse voxel storage (only store non-empty voxels)
USE_SPARSE_VOXELS = True

#: Maximum number of segments per voxel for efficient storage
MAX_SEGMENTS_PER_VOXEL = 50


def set_mesh_parameters(voxel_size_x=None, voxel_size_y=None, voxel_size_t=None,
                       near_field_drift=None, far_field_drift=None):
    """
    Update mesh refinement parameters at runtime.
    
    Args:
        voxel_size_x: Coarse voxel size in X direction (cm)
        voxel_size_y: Coarse voxel size in Y direction (cm)
        voxel_size_t: Coarse voxel size in time (us)
        near_field_drift: Near-field boundary in drift direction (cm)
        far_field_drift: Far-field boundary in drift direction (cm)
    """
    global COARSE_VOXEL_SIZE_X, COARSE_VOXEL_SIZE_Y, COARSE_VOXEL_SIZE_T
    global NEAR_FIELD_DRIFT, FAR_FIELD_DRIFT
    
    if voxel_size_x is not None:
        COARSE_VOXEL_SIZE_X = voxel_size_x
    if voxel_size_y is not None:
        COARSE_VOXEL_SIZE_Y = voxel_size_y
    if voxel_size_t is not None:
        COARSE_VOXEL_SIZE_T = voxel_size_t
    if near_field_drift is not None:
        NEAR_FIELD_DRIFT = near_field_drift
    if far_field_drift is not None:
        FAR_FIELD_DRIFT = far_field_drift


def validate_mesh_parameters():
    """
    Validate that mesh parameters are physically reasonable.
    
    Checks:
    - Very near < near < far field boundaries
    - Voxel sizes are positive
    - Thresholds are sensible
    
    Raises:
        ValueError: If parameters are invalid
    """
    if not (VERY_NEAR_FIELD_DRIFT < NEAR_FIELD_DRIFT < FAR_FIELD_DRIFT):
        raise ValueError("Field boundaries must satisfy: very_near < near < far")
    
    if not (VERY_NEAR_FIELD_TRANSVERSE < NEAR_FIELD_TRANSVERSE < FAR_FIELD_TRANSVERSE):
        raise ValueError("Transverse boundaries must satisfy: very_near < near < far")
    
    if COARSE_VOXEL_SIZE_X <= 0 or COARSE_VOXEL_SIZE_Y <= 0 or COARSE_VOXEL_SIZE_T <= 0:
        raise ValueError("Voxel sizes must be positive")
    
    if INDUCTION_THRESHOLD < 0 or COLLECTION_THRESHOLD < 0:
        raise ValueError("Thresholds must be non-negative")
    
    print("Mesh parameters validated successfully")
    print(f"  Very near-field: {VERY_NEAR_FIELD_DRIFT:.1f} cm drift, {VERY_NEAR_FIELD_TRANSVERSE:.1f} cm transverse")
    print(f"  Near-field: {NEAR_FIELD_DRIFT:.1f} cm drift, {NEAR_FIELD_TRANSVERSE:.1f} cm transverse")
    print(f"  Far-field: {FAR_FIELD_DRIFT:.1f} cm drift, {FAR_FIELD_TRANSVERSE:.1f} cm transverse")
    print(f"  Coarse voxel: {COARSE_VOXEL_SIZE_X:.1f} × {COARSE_VOXEL_SIZE_Y:.1f} × {COARSE_VOXEL_SIZE_T:.3f} (cm × cm × us)")
