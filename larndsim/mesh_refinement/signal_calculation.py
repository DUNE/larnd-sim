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
    pixel_categories,                     # (n_pixels,) array: 0=INDUCTION, 1=COLLECTION, 2=NEIGHBOR
    exclude_radius,                       # scalar (radius for COLLECTION/NEIGHBOR exclusion, 0 for INDUCTION)
    voxel_radius,                         # scalar (half-diagonal of coarse voxel in cm)
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
    Exclusion applies only to COLLECTION (cat=1) and NEIGHBOR (cat=2) pixels.
    """
    p_idx, t_idx = cuda.grid(2)
    n_pixels = pixel_x.shape[0]
    n_ticks = output.shape[1]
    if p_idx >= n_pixels or t_idx >= n_ticks:
        return
    
    n_voxels = voxel_x.shape[0]
    x_pixel = pixel_x[p_idx]
    y_pixel = pixel_y[p_idx]
    pixel_cat = pixel_categories[p_idx]
    # Apply exclusion only for COLLECTION (1) and NEIGHBOR (2) pixels
    r_exclude = exclude_radius if (pixel_cat == 1 or pixel_cat == 2) else 0.0
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
        
        # Exclusion weighting for voxels near collection/neighbor pixels
        # Compute radial distance in the readout plane
        dx_xy = x - x_pixel
        dy_xy = y - y_pixel
        r_c = math.sqrt(dx_xy * dx_xy + dy_xy * dy_xy)

        weight = 1.0
        x_eff = x
        y_eff = y

        if r_exclude > 0.0:
            # Fully inside exclusion region -> skip
            if r_c + voxel_radius < r_exclude:
                weight = 0.0
            # Fully outside -> keep
            elif r_c - voxel_radius > r_exclude:
                weight = 1.0
            else:
                # Straddling boundary: partial weight + shift center outward
                denom = 2.0 * voxel_radius
                if denom > 0.0:
                    weight = (r_c - (r_exclude - voxel_radius)) / denom
                    if weight < 0.0:
                        weight = 0.0
                    elif weight > 1.0:
                        weight = 1.0
                # Shift effective center outward to approximate carved volume
                if weight > 0.0 and r_c > 1e-12:
                    carved_depth = r_exclude - (r_c - voxel_radius)
                    if carved_depth < 0.0:
                        carved_depth = 0.0
                    shift = 0.5 * carved_depth
                    scale = shift / r_c
                    x_eff = x + dx_xy * scale
                    y_eff = y + dy_xy * scale

        if weight == 0.0:
            continue

        # Dipole field calculation (Eq. 3.21, 3.22 from P. Madigan's thesis)
        # Vector from electron to pixel (test point relative to dipole)
        dx = x_eff - x_pixel
        dy = y_eff - y_pixel
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
        q = charges[v_idx] * weight
        total_current += -q * v_drift * dWdz
    
    output[p_idx, t_idx] = total_current


def launch_far_field_dipole_signal_calculation(
    voxel_pos, pixel_pos, voxel_charge, pixel_categories, z_anode, z_cathode, v_drift, tick_size, n_ticks, 
    n_terms=5, C=None, bx=16, by=16, exclude_radius=0.0, voxel_size_xy=None
):
    """
    Launch CUDA kernel for far-field dipole induced current calculation with time ticks.
    Uses 2D grid/block launch for (pixel, tick) and sums over voxels in each thread.
    
    Args:
        voxel_pos: (n_voxels, 3) array (initial positions)
        pixel_pos: (n_pixels, 2) array
        voxel_charge: (n_voxels,) array (charge in each voxel, in units of electrons)
        pixel_categories: (n_pixels,) array with values 0=INDUCTION, 1=COLLECTION, 2=NEIGHBOR
        z_anode, z_cathode: float
        v_drift: float
        tick_size: float (us)
        n_ticks: int
        n_terms: int (default 5)
        C: float (default: use INDUCED_CURRENT_SCALE from mesh_params)
        bx, by: block dimensions for (pixels, ticks) (default 16x16)
        exclude_radius: float (exclusion radius for COLLECTION/NEIGHBOR pixels, in cm; default 0.0)
        voxel_size_xy: tuple (dx, dy) for voxel half-diagonal; default uses COARSE_VOXEL_SIZE_X/Y
        
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

    # Voxel half-diagonal in the readout plane (for boundary checks)
    if voxel_size_xy is None:
        voxel_rad = 0.5 * math.sqrt(mesh_params.COARSE_VOXEL_SIZE_X**2 + mesh_params.COARSE_VOXEL_SIZE_Y**2)
    else:
        voxel_rad = 0.5 * math.sqrt(voxel_size_xy[0]**2 + voxel_size_xy[1]**2)

    calculate_far_field_dipole_signal_time_kernel[blockspergrid, threadsperblock](
        cp.asarray(voxel_pos[:,0]),
        cp.asarray(voxel_pos[:,1]),
        cp.asarray(voxel_pos[:,2]),
        cp.asarray(voxel_charge, dtype=cp.float32),
        cp.asarray(pixel_pos[:,0]),
        cp.asarray(pixel_pos[:,1]),
        cp.asarray(pixel_categories, dtype=cp.int32),
        float(exclude_radius),
        float(voxel_rad),
        float(z_anode), float(z_cathode), float(v_drift), float(tick_size), int(n_terms), float(C_val), output
    )
    return output
