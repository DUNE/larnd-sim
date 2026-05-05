import math
from typing import Optional

import cupy as cp
import cupy.typing as cpt
from numba import cuda
import numpy as np
import numpy.typing as npt

from ..consts import detector, ff_induction


Bounds = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


def voxel_id_to_coordinates(
    voxel_indices: cpt.NDArray[cp.int32],
    grid_shape: tuple[int, int, int],
    voxel_size: tuple[float, float, float],
    bounds: Bounds
):
    """
    Convert flattened voxel indices to physical (x, y, z) center coordinates.
    
    Args:
        voxel_indices: 1D array of flattened voxel indices
        grid_shape: (nx, ny, nz) tuple
        voxel_size: (dx, dy, dz) tuple in cm
        bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max)) in cm
    
    Returns:
        (x_centers, y_centers, z_centers): NumPy arrays of voxel center coordinates in cm
    """

    # Convert CuPy to NumPy if needed
    if hasattr(voxel_indices, 'get'):
        voxel_indices = cp.asnumpy(voxel_indices)
    
    nx, ny, _ = grid_shape
    x_min, y_min, z_min = bounds[0][0], bounds[1][0], bounds[2][0]
    dx, dy, dz = voxel_size
    
    # Decode flattened indices
    iz = voxel_indices // (nx * ny)
    rem = voxel_indices % (nx * ny)
    iy = rem // nx
    ix = rem % nx
    
    # Compute voxel center coordinates
    x_centers = x_min + (ix + 0.5) * dx
    y_centers = y_min + (iy + 0.5) * dy
    z_centers = z_min + (iz + 0.5) * dz
    
    return x_centers, y_centers, z_centers


def gpu_voxelize(
   tracks: cpt.NDArray,
   tpc_borders: Optional[npt.NDArray]=None,
   voxel_size: Optional[float]=None,
   z_padding: float=0.0
) -> tuple[cpt.NDArray[cp.int32], cpt.NDArray[cp.float32],
           tuple[float, float, float], float, Bounds]:
    """
    Voxelize track segments on GPU.

    Args:
        tracks: structured track array (fields: x_start, y_start, z_start,
            x_end, y_end, z_end, n_electrons, pixel_plane, ...)
        tpc_borders: TPC borders override (optional; defaults to
            detector.TPC_BORDERS)
        voxel_size: Voxel size override (optional; defaults to using voxel size
            from ff_induction)
        z_padding: Extra padding on the range of drift coordinatess to cover


    Returns:
        (voxel_indices, voxel_charges, grid_shape, voxel_size, bounds) where
        voxel_indices & voxel_charges are 1D CuPy arrays for non-zero voxels.
    """
    if tpc_borders is None:
        tpc_borders = detector.TPC_BORDERS
    if voxel_size is None:
        voxel_size = (
            ff_induction.COARSE_VOXEL_SIZE_X,
            ff_induction.COARSE_VOXEL_SIZE_Y,
            ff_induction.COARSE_VOXEL_SIZE_Z,
        )
    x_min = float(np.min(tpc_borders[:,0,0])); x_max = float(np.max(tpc_borders[:,0,1]))
    y_min = float(np.min(tpc_borders[:,1,0])); y_max = float(np.max(tpc_borders[:,1,1]))
    if len(tracks) > 0:
        z_min_tracks = float(min(tracks['z_start'].min(), tracks['z_end'].min()))
        z_max_tracks = float(max(tracks['z_start'].max(), tracks['z_end'].max()))
        z_min = min(z_min_tracks, float(np.min(tpc_borders[:,2,0]))) - z_padding
        z_max = max(z_max_tracks, float(np.max(tpc_borders[:,2,1]))) + z_padding
    else:
        z_min = float(np.min(tpc_borders[:,2,0]))
        z_max = float(np.max(tpc_borders[:,2,1]))
    bounds = ((x_min,x_max),(y_min,y_max),(z_min,z_max))
    grid_shape = (
        int(np.ceil((x_max - x_min)/voxel_size[0])),
        int(np.ceil((y_max - y_min)/voxel_size[1])),
        int(np.ceil((z_max - z_min)/voxel_size[2])),
    )
    total_voxels = grid_shape[0]*grid_shape[1]*grid_shape[2]
    voxel_charges = cp.zeros(total_voxels, dtype=cp.float32)

    TPB = 256
    BPG = (tracks.shape[0] + TPB - 1)//TPB
    voxelize_segments_kernel[BPG, TPB](tracks, voxel_charges,
                                      cp.asarray(voxel_size, dtype=cp.float32),
                                      cp.asarray(bounds, dtype=cp.float32),
                                      cp.asarray(grid_shape, dtype=cp.int32))
    
    # Transfer to CPU for numpy operation
    voxel_charges_np = cp.asnumpy(voxel_charges)
    
    nonzero_indices = np.where(voxel_charges_np > 0)[0]
    
    # Return as cupy arrays
    return cp.asarray(nonzero_indices), cp.asarray(voxel_charges_np[nonzero_indices]), grid_shape, voxel_size, bounds


@cuda.jit
def voxelize_segments_kernel(
   tracks: cpt.NDArray,
   voxel_charges: cpt.NDArray[cp.float32],
   voxel_size: cpt.NDArray[cp.float32],
   bounds: cp.ndarray[tuple[int, int], cp.float32],
   grid_shape: cpt.NDArray[cp.int32]
):
    """
    CUDA kernel for fast voxelization of track segments.
    
    Parallelizes the voxelization process across segments.
    
    Args:
        tracks: structured track array (fields: x_start, y_start, z_start,
            x_end, y_end, z_end, n_electrons, pixel_plane, ...)
        voxel_charges: (n_voxels,) output array for voxel charges (flattened 3D grid)
        voxel_size: (dx, dy, dt) voxel dimensions
        bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        grid_shape: (nx, ny, nt) grid dimensions
    """
    i = cuda.grid(1)
    
    if i >= tracks.shape[0]:
        return
    
    segment = tracks[i]
    # Extract from structured array using field names (Numba handles this)
    x_start = segment['x_start']; y_start = segment['y_start']; z_start = segment['z_start']
    x_end = segment['x_end'];     y_end = segment['y_end'];     z_end = segment['z_end']
    n_electrons = segment['n_electrons']
    
    # Convert to voxel indices
    ix_start = int((x_start - bounds[0][0]) / voxel_size[0])
    iy_start = int((y_start - bounds[1][0]) / voxel_size[1])
    iz_start = int((z_start - bounds[2][0]) / voxel_size[2])
    ix_end = int((x_end - bounds[0][0]) / voxel_size[0])
    iy_end = int((y_end - bounds[1][0]) / voxel_size[1])
    iz_end = int((z_end - bounds[2][0]) / voxel_size[2])
    
    # Clip to grid bounds
    ix_start = max(0, min(grid_shape[0]-1, ix_start))
    iy_start = max(0, min(grid_shape[1]-1, iy_start))
    iz_start = max(0, min(grid_shape[2]-1, iz_start))
    ix_end = max(0, min(grid_shape[0]-1, ix_end))
    iy_end = max(0, min(grid_shape[1]-1, iy_end))
    iz_end = max(0, min(grid_shape[2]-1, iz_end))
    
    # 3D DDA 
    # Compute segment direction and length
    dx = x_end - x_start
    dy = y_end - y_start
    dz = z_end - z_start
    seg_len = math.sqrt(dx*dx + dy*dy + dz*dz)
    if seg_len == 0:
        # Zero-length segment, deposit all charge in start voxel
        idx = ix_start + iy_start * grid_shape[0] + iz_start * grid_shape[0] * grid_shape[1]
        cuda.atomic.add(voxel_charges, idx, n_electrons)
        return

    # Parametrize segment: 
    # p(t) = (x_start, y_start, z_start) + t*(dx, dy, dz), t in [0,1]

    # Voxel grid bounds
    nx, ny, nz = grid_shape[0], grid_shape[1], grid_shape[2]
    x0, y0, z0 = bounds[0][0], bounds[1][0], bounds[2][0]
    sx, sy, sz = voxel_size[0], voxel_size[1], voxel_size[2]

    # Initial voxel indices
    ix = ix_start; iy = iy_start; iz = iz_start

    # Step direction
    step_x = 1 if dx > 0 else -1 if dx < 0 else 0
    step_y = 1 if dy > 0 else -1 if dy < 0 else 0
    step_z = 1 if dz > 0 else -1 if dz < 0 else 0

    # Compute tMax and tDelta for DDA
    if step_x != 0:
        next_x = x0 + (ix + (step_x > 0)) * sx
        tMaxX = (next_x - x_start) / dx if dx != 0 else math.inf
        tDeltaX = sx / abs(dx)
    else:
        tMaxX = math.inf
        tDeltaX = math.inf
    
    if step_y != 0:
        next_y = y0 + (iy + (step_y > 0)) * sy
        tMaxY = (next_y - y_start) / dy if dy != 0 else math.inf
        tDeltaY = sy / abs(dy)
    else:
        tMaxY = math.inf
        tDeltaY = math.inf
    
    if step_z != 0:
        next_z = z0 + (iz + (step_z > 0)) * sz
        tMaxZ = (next_z - z_start) / dz if dz != 0 else math.inf
        tDeltaZ = sz / abs(dz)
    else:
        tMaxZ = math.inf
        tDeltaZ = math.inf

    t = 0.0
    t_end = 1.0
    # Traverse voxels
    while (0 <= ix < nx) and (0 <= iy < ny) and (0 <= iz < nz):
        # Find next voxel boundary crossed
        if tMaxX < tMaxY and tMaxX < tMaxZ:
            t_next = min(tMaxX, t_end)
        elif tMaxY < tMaxZ:
            t_next = min(tMaxY, t_end)
        else:
            t_next = min(tMaxZ, t_end)
        # Fraction of segment in this voxel
        frac = t_next - t
        if frac > 0:
            charge = n_electrons * frac
            idx = ix + iy * nx + iz * nx * ny
            cuda.atomic.add(voxel_charges, idx, charge)
        if t_next >= t_end:
            break
        # Advance to next voxel
        if tMaxX < tMaxY and tMaxX < tMaxZ:
            ix += step_x
            t = tMaxX
            tMaxX += tDeltaX
        elif tMaxY < tMaxZ:
            iy += step_y
            t = tMaxY
            tMaxY += tDeltaY
        else:
            iz += step_z
            t = tMaxZ
            tMaxZ += tDeltaZ
