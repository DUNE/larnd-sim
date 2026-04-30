"""
Signal calculation module for mesh refinement

Calculates induced currents on pixels at multiple scales:
1. Very near-field: with diffusion 
2. Near-field: no diffusion 
3. Far-field: Coarse mesh aggregate calculation 
"""

from functools import lru_cache
import math
from typing import Any, Optional

import numpy as np
import cupy as cp
import cupy.typing as cpt
import numba as nb
from numba import cuda

from larndsim.consts import detector, mesh_params, sim


@cuda.jit
def calculate_ff_voxels(
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

        dx = x_eff - x_pixel
        dy = y_eff - y_pixel
        dz = z - z_anode
        l = abs(z_cathode - z_anode)
        dWdz = C * dipole_dWdz(dx, dy, dz, l, n_terms)
        # Induced current (Eq. 3.23): I = -q * v_d * dW/dz (negative sign for electron charge)
        # Scale by voxel charge
        q = charges[v_idx] * weight
        total_current += -q * v_drift * dWdz
    
    output[p_idx, t_idx] = total_current


# FIXME if it varies by module
@lru_cache
def get_srwf_Ez() -> cpt.NDArray[cp.float32]:
    """Get the E_z of the finite-element SR weighting field.

    Returns:
        Drift component of the Shockley-Ramo weighting field as
        a 3D array on a 1 mm grid. Pixel at origin.
    """
    print('Loading Shockley-Ramo weighting field...')
    data = np.loadtxt(detector.SRWF_FILE, skiprows=8)
    Ez = data[:, 5]
    # FIXME:
    return cp.asarray(Ez, dtype=cp.float32).reshape(239, 150, 469)


@nb.njit
def dipole_dWdz(dx: float, dy: float, dz: float, l: float, n_terms: int) -> float:
    """Dipole field calculation (Eq. 3.21, 3.22 from P. Madigan's thesis)

    Args:
        (dx,dy,dz): Vector from electron to pixel (test point relative to dipole)
        l: Drift length
        n_terms: Degree of calculation

    Returns:
        Calculated Shockley-Ramo weighting field for the current induced on
        the pixel
    """
    r_sq = dx*dx + dy*dy + dz*dz
    if r_sq < 1e-20:  # Avoid singularity
        return 0.
    r = math.sqrt(r_sq)
    # Direct dipole term: z-component of gradient
    # For dipole at origin aligned with z-axis: dW/dz = C x (r² - 3z²)/r⁵
    term0 = (r_sq - 3.0*dz*dz) / (r_sq*r_sq*r)
    # Image dipole terms
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
    return term0 + term_sum


@nb.njit
def srwf_dWdz(dx: float, dy: float, dz: float,
              Ez: cp.ndarray[tuple[int, int, int], cp.float32]) -> float:
    # FIXME assume grid is in mm
    ix, iy, iz = round(dx/10), round(dy/10), round(dz/10)
    # v_drift is in cm/microsec
    # Ez is in V/m per 1 V i.e. has units of 1/m
    # Ez [1/m] * v_drift [cm/mu_sec] * (1e-2 m/cm) => 1/mu_sec
    # ==> need a factor of 1e-2
    return 1e-2 * Ez[ix, iy, iz]


@cuda.jit
def calculate_ff_segments(
    tracks: cpt.NDArray,
    pixel_x: cpt.NDArray[cp.float32],
    pixel_y: cpt.NDArray[cp.float32],
    exclude_radius: float,
    split_step_cm: float,
    z_anode: float,
    z_cathode: float,
    v_drift: float,
    tick_size: float,
    n_terms: int,
    srwf_Ez: Optional[cpt.NDArray[cp.float32]],
    C: float,
    output: cp.ndarray[tuple[int, int], cp.float32]
):
    """
    CUDA kernel: Calculate far-field induced current for each (pixel, tick)
    by summing over segment contributions outside exclude_radius.

    Long segments are split into pieces of roughly split_step_cm and each
    piece contributes as a point charge located at the piece midpoint.

    Args:
        tracks: structured track array (fields: x_start, y_start, z_start,
                x_end, y_end, z_end, n_electrons, pixel_plane, ...)
    """
    p_idx, t_idx = cuda.grid(2)
    n_pixels = pixel_x.shape[0]
    n_ticks = output.shape[1]
    if p_idx >= n_pixels or t_idx >= n_ticks:
        return

    x_pixel = pixel_x[p_idx]
    y_pixel = pixel_y[p_idx]
    t = t_idx * tick_size
    total_current = 0.0

    n_segments = tracks.shape[0]
    l = abs(z_cathode - z_anode)

    for s_idx in range(n_segments):
        segment = tracks[s_idx]
        x0 = segment['x_start']
        y0 = segment['y_start']
        z0_seg = segment['z_start']
        x1 = segment['x_end']
        y1 = segment['y_end']
        z1 = segment['z_end']

        # skip before splitting into sub-pieces.
        if exclude_radius > 0.0:
            dx0 = abs(x0 - x_pixel)
            dy0 = abs(y0 - y_pixel)
            dx1 = abs(x1 - x_pixel)
            dy1 = abs(y1 - y_pixel)
            if max(dx0, dx1) <= exclude_radius and max(dy0, dy1) <= exclude_radius:
                continue

        vx = x1 - x0
        vy = y1 - y0
        vz = z1 - z0_seg
        seg_len_sq = vx*vx + vy*vy + vz*vz
        if seg_len_sq <= 1e-20:
            continue
        seg_len = math.sqrt(seg_len_sq)

        n_split = 1
        if split_step_cm > 0.0:
            n_split = int(math.ceil(seg_len / split_step_cm))
            if n_split < 1:
                n_split = 1

        q_piece = segment['n_electrons'] / n_split

        for i_split in range(n_split):
            frac = (i_split + 0.5) / n_split
            x = x0 + frac * vx
            y = y0 + frac * vy
            z_start_piece = z0_seg + frac * vz

            # far-field exclusion radius
            if exclude_radius > 0.0:
                dx_xy = abs(x - x_pixel)
                dy_xy = abs(y - y_pixel)
                if dx_xy <= exclude_radius or dy_xy <= exclude_radius:
                    continue

            drift_distance = v_drift * t
            if z_start_piece > z_anode:
                z = z_start_piece - drift_distance
                if z < z_anode:
                    continue
            else:
                z = z_start_piece + drift_distance
                if z > z_anode:
                    continue

            dx = x - x_pixel
            dy = y - y_pixel
            dz = z - z_anode

            if srwf_Ez is not None:
                dWdz = C * srwf_dWdz(dx, dy, dz, srwf_Ez)
            else:
                dWdz = C * dipole_dWdz(dx, dy, dz, l, n_terms)

            total_current += -q_piece * v_drift * dWdz

    output[p_idx, t_idx] = total_current


def get_current_scale() -> float:
    if C := mesh_params.INDUCED_CURRENT_SCALE is None:
        # The FFE kernel gives dQ per microsecond whereas the standard pixel
        # response (and the rest of larnd-sim) uses dQ per tick. The FFE signal
        # therefore needs to be scaled down according to the tick length.
        C = detector.RESPONSE_SAMPLING
    return C


def launch_ffe_kernel(
    tpc_idx: int,
    tracks: cpt.NDArray,
    pixel_x: cpt.NDArray[cp.float32],
    pixel_y: cpt.NDArray[cp.float32],
    n_ticks: int,
    category: int,
    voxel_cache: dict[int, dict[str, Optional[cpt.NDArray[cp.float32]]]],
) -> cp.ndarray[tuple[int, int], cp.float32]:
    """Launch CUDA kernel for far-field induced current calculation.

    This uses a 2D grid/block launch for (pixel, tick) and sums over
    voxels/segments in each thread.

    Args:
        tpc_idx: TPC index to consider
        tracks: 1D array of edep-sim track segments
        pixel_x: 1D array of pixel x-positions
        pixel_y: 1D array of pixel y-positions
        n_ticks: Length (in time ticks) of the calculated current
        category: Category to assign to the pixels (ignored in 'segments' mode)
        voxel_cache: Dict that maps from a TPC ID to a dict of voxel data for
            that TPC. The latter has keys of 'x', 'y', 'z', and 'q'; each one
            maps to a 1D array of the corresponding value for all the voxels.

    Returns:
        2D array that maps (pixel, tick) to dQ
    """
    n_pixels = pixel_x.shape[0]
    TPB = (16, 16)
    BPG = (math.ceil(n_pixels / TPB[0]), math.ceil(n_ticks / TPB[1]))

    output = cp.zeros((n_pixels, n_ticks), dtype=cp.float32)

    z_anode = detector.TPC_BORDERS[tpc_idx, 2, 0]
    z_cathode = detector.TPC_BORDERS[tpc_idx, 2, 1]
    exclude_radius = mesh_params.CHARGE_NEIGHBOR_RADIUS * detector.PIXEL_PITCH

    def launch_voxels():
        cache = voxel_cache.get(tpc_idx, None)
        if cache is None or cache['x'] is None:
            return

        pixel_categories = cp.full(n_pixels, category, dtype=cp.int32)
        voxel_radius = np.sqrt(
            (mesh_params.COARSE_VOXEL_SIZE_X / 2.0)**2 +
            (mesh_params.COARSE_VOXEL_SIZE_Y / 2.0)**2 +
            (mesh_params.COARSE_VOXEL_SIZE_Z / 2.0)**2)

        calculate_ff_voxels[BPG, TPB](
            cache["x"], cache["y"], cache["z"], cache["q"],
            pixel_x, pixel_y,
            pixel_categories,
            exclude_radius,
            voxel_radius,
            z_anode, z_cathode,
            detector.V_DRIFT,
            detector.TIME_SAMPLING,
            mesh_params.DIPOLE_N_TERMS,
            get_current_scale(),
            output)

    def launch_segments(srwf: bool):
        tpc_tracks = tracks[tracks['pixel_plane'] == tpc_idx]
        if len(tpc_tracks) == 0:
            return

        calculate_ff_segments[BPG, TPB](
            tpc_tracks,
            pixel_x, pixel_y,
            exclude_radius,
            mesh_params.FAR_FIELD_SEGMENT_STEP_CM,
            z_anode, z_cathode,
            detector.V_DRIFT,
            detector.TIME_SAMPLING,
            mesh_params.DIPOLE_N_TERMS,
            get_srwf_Ez() if srwf else cp.empty(0, dtype=cp.float32),
            get_current_scale(),
            output)

    match sim.FARFIELD_MODE:
        case 'voxels':
            launch_voxels()
        case 'segments':
            launch_segments(False)
        case 'segments-srwf':
            launch_segments(True)
        case _:
            e = f"Invalid farfield_mode '{sim.FARFIELD_MODE}'"
            raise RuntimeError(e)

    return output


def launch_far_field_dipole_signal_calculation(
    voxel_pos, pixel_pos, voxel_charge, pixel_categories, z_anode, z_cathode, v_drift, tick_size, n_ticks, 
    n_terms=5, C=None, bx=16, by=16, exclude_radius=0.0, voxel_size_xy=None
):
    """
    Launch CUDA kernel for far-field dipole induced current calculation with time ticks.
    Uses 2D grid/block launch for (pixel, tick) and sums over voxels in each thread.
    (USED FOR TESTING/VALIDATION ONLY)
    
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

    calculate_ff_voxels[blockspergrid, threadsperblock](
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
