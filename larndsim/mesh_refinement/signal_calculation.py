"""
Signal calculation module for mesh refinement

Calculates induced currents on pixels at multiple scales:
1. Very near-field: with diffusion 
2. Near-field: no diffusion 
3. Far-field: Coarse mesh aggregate calculation 
"""

import numpy as np
import cupy as cp
from numba import cuda
import math
from larndsim.consts import mesh_params


@cuda.jit
def calculate_far_field_dipole_signal_time_kernel(
    voxel_x, voxel_y, voxel_z,   # (n_voxels,) arrays (initial positions)
    charges,                              # (n_voxels,) array (charge in each voxel)
    pixel_x, pixel_y,                     # (n_pixels,) arrays
    z_anode, z_cathode,                   # scalars
    v_drift,                              # scalar
    tick_size,                            # scalar (us)
    n_terms,                              # scalar (int)
    C,                                    # scalar (float)
    output                                # (n_pixels, n_ticks) array
):
    """
    CUDA kernel: Calculate far-field induced current for each pixel-tick, summing over all voxels.
    Uses 2D grid/block launch for (pixel, tick) indexing.
    Each thread sums contributions from all voxels for its pixel-tick pair.
    """
    p_idx, t_idx = cuda.grid(2)
    n_pixels = pixel_x.shape[0]
    n_ticks = output.shape[1]
    if p_idx >= n_pixels or t_idx >= n_ticks:
        return
    
    n_voxels = voxel_x.shape[0]
    x_pixel = pixel_x[p_idx]
    y_pixel = pixel_y[p_idx]
    t = t_idx * tick_size
    
    # Sum contributions from all voxels
    total_current = 0.0
    
    for v_idx in range(n_voxels):
        # Get initial positions
        x = voxel_x[v_idx]
        y = voxel_y[v_idx]
        z0 = voxel_z[v_idx]
        
        # Electron position at tick (drifting toward anode)
        # Drift direction depends on which side of anode the electron starts
        drift_distance = v_drift * t
        if z0 > z_anode:
            z = z0 - drift_distance  # Drift in -z direction
            # Check if electron is past the anode
            if z < z_anode:
                continue
        else:
            z = z0 + drift_distance  # Drift in +z direction
            # Check if electron is past the anode
            if z > z_anode:
                continue
        
        # Dipole field calculation (Eq. 3.21, 3.22 from P. Madigan's thesis)
        # Vector from electron to pixel (test point relative to dipole)
        dx = x - x_pixel
        dy = y - y_pixel
        dz = z - z_anode
        r_sq = dx*dx + dy*dy + dz*dz
        if r_sq < 1e-20:  # Avoid singularity
            continue
        r = math.sqrt(r_sq)
        # Direct dipole term: z-component of gradient
        # For dipole at origin aligned with z-axis: dW/dz = C x (r² - 3z²)/r⁵
        term0 = (r_sq - 3.0*dz*dz) / (r_sq*r_sq*r)
        # Image dipole terms
        l = abs(z_cathode - z_anode)
        term_sum = 0.0
        for n in range(1, n_terms+1):
            # Positive image: z + 2nl
            dz_p = dz + 2*n*l
            r_p_sq = dx*dx + dy*dy + dz_p*dz_p
            if r_p_sq > 1e-20:
                r_p = math.sqrt(r_p_sq)
                term_sum += (r_p_sq - 3.0*dz_p*dz_p) / (r_p_sq*r_p_sq*r_p)
            # Negative image: z - 2nl
            dz_m = dz - 2*n*l
            r_m_sq = dx*dx + dy*dy + dz_m*dz_m
            if r_m_sq > 1e-20:
                r_m = math.sqrt(r_m_sq)
                term_sum += (r_m_sq - 3.0*dz_m*dz_m) / (r_m_sq*r_m_sq*r_m)
        # Total z-component of weighting field gradient (Eq. 3.21)
        dWdz = C * (term0 + term_sum)
        # Induced current (Eq. 3.23): I = -q * v_d * dW/dz (negative sign for electron charge)
        # Scale by voxel charge
        q = charges[v_idx]
        total_current += -q * v_drift * dWdz
    
    output[p_idx, t_idx] = total_current


def launch_far_field_dipole_signal_calculation(
    voxel_pos, pixel_pos, voxel_charge, z_anode, z_cathode, v_drift, tick_size, n_ticks, n_terms=5, C=None,
    bx=16, by=16
):
    """
    Launch CUDA kernel for far-field dipole induced current calculation with time ticks.
    Uses 2D grid/block launch for (pixel, tick) and sums over voxels in each thread.
    
    Args:
        voxel_pos: (n_voxels, 3) array (initial positions)
        pixel_pos: (n_pixels, 2) array
        voxel_charge: (n_voxels,) array (charge in each voxel, in units of electrons)
        z_anode, z_cathode: float
        v_drift: float
        tick_size: float (us)
        n_ticks: int
        n_terms: int
        C: float
        bx, by: block dimensions for (pixels, ticks) (default 16x16)
    Returns:
        output: (n_pixels, n_ticks) CuPy array - total induced current per pixel per tick
    """
    n_pixels = pixel_pos.shape[0]
    output = cp.zeros((n_pixels, n_ticks), dtype=cp.float32)
    blockspergrid = (
        (n_pixels + bx - 1) // bx,
        (n_ticks + by - 1) // by
    )
    threadsperblock = (bx, by)
    # Use global induced current scale from mesh_params if C not provided
    C_val = mesh_params.INDUCED_CURRENT_SCALE if (C is None) else C

    calculate_far_field_dipole_signal_time_kernel[blockspergrid, threadsperblock](
        cp.asarray(voxel_pos[:,0]),
        cp.asarray(voxel_pos[:,1]),
        cp.asarray(voxel_pos[:,2]),
        cp.asarray(voxel_charge, dtype=cp.float32),
        cp.asarray(pixel_pos[:,0]),
        cp.asarray(pixel_pos[:,1]),
        float(z_anode), float(z_cathode), float(v_drift), float(tick_size), int(n_terms), float(C_val), output
    )
    return output
