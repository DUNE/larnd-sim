"""
Signal calculation module for far field

  consts.sim.FARFIELD_DIPOLE_MODE : str
      'infinite_plane' (default/current behavior, dipole_dWdz -- two
      infinite grounded planes, anode + cathode only) or 'box_lattice'
      (image_lattice_dWdz -- adds grounded side-wall reflections).
      Only affects calculate_ff_segments; calculate_ff_voxels is
      unchanged and always uses dipole_dWdz.

  consts.ff_induction.LATTICE_LX, LATTICE_LY : float
      Box half-widths for the grounded side walls (x=+/-LATTICE_LX,
      y=+/-LATTICE_LY), analogous to BOX_LX/BOX_LY in
      hybrid_pixel_response_boxed.py.

  consts.ff_induction.LATTICE_N_TERMS_XY : int
      Number of side-wall image reflections per direction (total
      (2*LATTICE_N_TERMS_XY+1)**2 lattice points evaluated per
      cathode image). ff_induction.DIPOLE_N_TERMS is reused for the
      cathode/z-image degree, same as in calculate_ff_segments.
"""

from functools import lru_cache
import math
from typing import Any, Optional

import numpy as np
import cupy as cp
import cupy.typing as cpt
import numba as nb
from numba import cuda

from larndsim.consts import detector, ff_induction, sim


@cuda.jit
def calculate_ff_voxels(
    voxel_x: cpt.NDArray[cp.float32],
    voxel_y: cpt.NDArray[cp.float32],
    voxel_z: cpt.NDArray[cp.float32],
    charges: cpt.NDArray[cp.float32],
    pixel_x: cpt.NDArray[cp.float32],
    pixel_y: cpt.NDArray[cp.float32],
    pixel_categories: cpt.NDArray[cp.float32],
    z_anode: float,
    z_cathode: float,
    output: cp.ndarray[tuple[int, int], cp.float32]
):
    """
    CUDA kernel: Calculate far-field induced current using voxels.

    Uses 2D grid/block launch for (pixel, tick) indexing. Each thread sums
    contributions from all voxels for its pixel-tick pair. Exclusion applies
    only to COLLECTION (cat=1) and NEIGHBOR (cat=2) pixels.

    Args:
        voxel_x/y/z: (n_voxels,) array of x/y/z-positions of each voxel's center
        charges: (n_voxels,) array of summed charge in each voxel
        pixel_x/y: (n_pixels,) array of x/y-positions of each pixel's center
        pixel_categories: (n_pixels,) array: 0=INDUCTION, 1=COLLECTION, 2=NEIGHBOR
        z_anode/cathode: Drift coordinate of the anode/cathode
        output: (n_pixels, n_ticks) array of current signals
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
    exclude_radius = ff_induction.CHARGE_NEIGHBOR_RADIUS * detector.PIXEL_PITCH
    r_exclude = exclude_radius if (pixel_cat == 1 or pixel_cat == 2) else 0.0
    t = t_idx * detector.TIME_SAMPLING
    voxel_radius = math.sqrt(
        (ff_induction.COARSE_VOXEL_SIZE_X / 2.0)**2 +
        (ff_induction.COARSE_VOXEL_SIZE_Y / 2.0)**2 +
        (ff_induction.COARSE_VOXEL_SIZE_Z / 2.0)**2)

    # Sum contributions from all voxels
    total_current = 0.0
    
    for v_idx in range(n_voxels):
        # Get initial positions
        x = voxel_x[v_idx]
        y = voxel_y[v_idx]
        z0 = voxel_z[v_idx]
        
        # Electron position at tick (drifting toward anode)
        # Drift direction depends on which side of anode the electron starts
        drift_distance = detector.V_DRIFT * t
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
        C = detector.RESPONSE_SAMPLING # scale to near-field reponse's time tick
        dWdz = C * dipole_dWdz(dx, dy, dz, l, ff_induction.DIPOLE_N_TERMS)
        # Induced current (Eq. 3.23): I = -q * v_d * dW/dz (negative sign for electron charge)
        # Scale by voxel charge
        q = charges[v_idx] * weight
        total_current += -q * detector.V_DRIFT * dWdz
    
    output[p_idx, t_idx] = total_current


@cuda.jit
def calculate_ff_segments(
    tracks: cpt.NDArray,
    pixel_x: cpt.NDArray[cp.float32],
    pixel_y: cpt.NDArray[cp.float32],
    z_anode: float,
    z_cathode: float,
    output: cp.ndarray[tuple[int, int], cp.float32]
):
    """
    CUDA kernel: Calculate far-field induced current using segments.

    Sums over segment contributions outside exclude_radius. Long segments are
    split into pieces of roughly FAR_FIELD_SEGMENT_STEP_CM and each piece
    contributes as a point charge located at the piece midpoint.

    Args:
        tracks: structured track array (fields: x_start, y_start, z_start,
            x_end, y_end, z_end, n_electrons, pixel_plane, ...)
        pixel_x/y: (n_pixels,) array of x/y-positions of each pixel's center
        z_anode/cathode: Drift coordinate of the anode/cathode
        output: (n_pixels, n_ticks) array of current signals
    """
    p_idx, t_idx = cuda.grid(2)
    n_pixels = pixel_x.shape[0]
    n_ticks = output.shape[1]
    if p_idx >= n_pixels or t_idx >= n_ticks:
        return

    x_pixel = pixel_x[p_idx]
    y_pixel = pixel_y[p_idx]
    t = t_idx * detector.TIME_SAMPLING
    total_current = 0.0

    n_segments = tracks.shape[0]
    l = abs(z_cathode - z_anode)
    exclude_radius = (0.5 + ff_induction.CHARGE_NEIGHBOR_RADIUS) * detector.PIXEL_PITCH

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

        n_split, step = 1, ff_induction.FAR_FIELD_SEGMENT_STEP_CM
        if step > 0.0:
            n_split = max(int(math.ceil(seg_len / step)), 1)

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
                if dx_xy <= exclude_radius and dy_xy <= exclude_radius:
                    continue

            drift_distance = detector.V_DRIFT * t
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

            C = ff_induction.DIPOLE_SCALE
            C *= 0.01 # something something units
            dWdz = C * dipole_dWdz(dx, dy, dz, l, ff_induction.DIPOLE_N_TERMS)

            total_current += -q_piece * detector.V_DRIFT * dWdz

    output[p_idx, t_idx] = total_current


@cuda.jit
def calculate_ff_segments_box_lattice(
    tracks: cpt.NDArray,
    pixel_x: cpt.NDArray[cp.float32],
    pixel_y: cpt.NDArray[cp.float32],
    z_anode: float,
    z_cathode: float,
    output: cp.ndarray[tuple[int, int], cp.float32]
):
    """
    CUDA kernel: Calculate far-field induced current using segments,
    with the box-aware image-lattice far field (image_lattice_dWdz) in
    place of the infinite-plane dipole (dipole_dWdz).

    Identical to calculate_ff_segments() in every respect (exclusion
    handling, segment splitting, drift geometry) except the per-piece
    dW/dz calculation, which additionally reflects images off grounded
    side walls at x=+/-ff_induction.LATTICE_LX, y=+/-ff_induction.LATTICE_LY
    (see image_lattice_dWdz docstring). Selected via
    sim.FARFIELD_DIPOLE_MODE = 'box_lattice' in launch_ffe_kernel().

    Args:
        tracks: structured track array (fields: x_start, y_start, z_start,
            x_end, y_end, z_end, n_electrons, pixel_plane, ...)
        pixel_x/y: (n_pixels,) array of x/y-positions of each pixel's center
        z_anode/cathode: Drift coordinate of the anode/cathode
        output: (n_pixels, n_ticks) array of current signals
    """
    p_idx, t_idx = cuda.grid(2)
    n_pixels = pixel_x.shape[0]
    n_ticks = output.shape[1]
    if p_idx >= n_pixels or t_idx >= n_ticks:
        return

    x_pixel = pixel_x[p_idx]
    y_pixel = pixel_y[p_idx]
    t = t_idx * detector.TIME_SAMPLING
    total_current = 0.0

    n_segments = tracks.shape[0]
    l = abs(z_cathode - z_anode)
    exclude_radius = ff_induction.CHARGE_NEIGHBOR_RADIUS * detector.PIXEL_PITCH

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

        n_split, step = 1, ff_induction.FAR_FIELD_SEGMENT_STEP_CM
        if step > 0.0:
            n_split = max(int(math.ceil(seg_len / step)), 1)

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
                if dx_xy <= exclude_radius and dy_xy <= exclude_radius:
                    continue

            drift_distance = detector.V_DRIFT * t
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

            C = ff_induction.DIPOLE_SCALE
            C *= 0.01 # something something units
            dWdz = C * image_lattice_dWdz(
                dx, dy, dz, l,
                ff_induction.LATTICE_LX, ff_induction.LATTICE_LY,
                ff_induction.DIPOLE_N_TERMS, ff_induction.LATTICE_N_TERMS_XY)

            total_current += -q_piece * detector.V_DRIFT * dWdz

    output[p_idx, t_idx] = total_current


@nb.njit
def dipole_dWdz(dx: float, dy: float, dz: float, l: float, n_terms: int) -> float:
    """
    Dipole field calculation (Eq. 3.21, 3.22 from P. Madigan's thesis)

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
        # Positive image: z + (2^n)l
        dz_p = dz + (2**n)*l
        r_p_sq = dx*dx + dy*dy + dz_p*dz_p
        if r_p_sq > 1e-20:
            r_p = math.sqrt(r_p_sq)
            term_sum += (r_p_sq - 3.0*dz_p*dz_p) / (r_p_sq*r_p_sq*r_p)
        # Negative image: z - (2^n)l
        dz_m = dz - (2**n)*l
        r_m_sq = dx*dx + dy*dy + dz_m*dz_m
        if r_m_sq > 1e-20:
            r_m = math.sqrt(r_m_sq)
            term_sum += (r_m_sq - 3.0*dz_m*dz_m) / (r_m_sq*r_m_sq*r_m)
    # Total z-component of weighting field gradient (Eq. 3.21)
    return term0 + term_sum


@nb.njit
def image_lattice_dWdz(dx: float, dy: float, dz: float, l: float,
                        Lx: float, Ly: float, n_terms_z: int,
                        n_terms_xy: int) -> float:
    """
    Box-aware dipole field calculation: combines a cathode image series
    with additional image reflections off grounded side walls at
    x=+/-Lx, y=+/-Ly.

    The pixel/dipole sits at the box center (x=y=0 in pixel-relative
    coordinates), so reflecting off a wall pair a distance L away
    generates images at x = 2*j*L for every integer j, with alternating
    sign (-1)**j (j=0 recovers the un-reflected term). Combining
    independent reflections off the x- and y-walls gives a 2D lattice
    of images with sign (-1)**(j+k), applied to *every* z-image.

    Args:
        (dx, dy, dz): Vector from electron to pixel (test point
            relative to the dipole), with the pixel taken as the box
            center in x, y
        l: Drift length (anode-cathode separation)
        Lx, Ly: Box half-widths -- grounded walls at x=+/-Lx, y=+/-Ly
        n_terms_z: Degree of the cathode image series (images at
            dz + 2*n*l for n = -n_terms_z .. n_terms_z)
        n_terms_xy: Degree of the side-wall image lattice in each of
            x and y (total (2*n_terms_xy+1)**2 lattice points per
            z-image)

    Returns:
        Calculated Shockley-Ramo weighting field for the current
        induced on the pixel, including both cathode and side-wall
        image reflections
    """
    total = 0.0
    for j in range(-n_terms_xy, n_terms_xy + 1):
        dx_j = dx - 2 * j * Lx
        for k in range(-n_terms_xy, n_terms_xy + 1):
            dy_k = dy - 2 * k * Ly
            sign = 1.0 if (j + k) % 2 == 0 else -1.0

            # Cathode (z) image series, linear 2*n*l spacing (matches
            # far_field_image_lattice_potential), reflected off the
            # same x/y wall lattice
            for n in range(-n_terms_z, n_terms_z + 1):
                dz_n = dz + 2 * n * l
                r_sq = dx_j*dx_j + dy_k*dy_k + dz_n*dz_n
                if r_sq > 1e-20:
                    r = math.sqrt(r_sq)
                    total += sign * (r_sq - 3.0*dz_n*dz_n) / (r_sq*r_sq*r)
    return total


@nb.njit
def _corner_angle_dz(a: float, b: float, zpos: float) -> float:
    """d/dzpos of _corner_angle(a, b, zpos), holding a, b fixed.

    _corner_angle(a, b, zpos) = arctan2(u, v) with u = a*b (independent
    of zpos) and v = zpos*r, r = sqrt(a^2+b^2+zpos^2). Using
    d/dt atan2(u, v) = (v u' - u v') / (u^2 + v^2), and u' = 0 here,
    this reduces to -u v' / (u^2 + v^2), with
    v' = dv/dzpos = r + zpos^2/r = (a^2+b^2+2*zpos^2) / r.
    """
    r = math.sqrt(a**2 + b**2 + zpos**2)
    u = a * b
    v = zpos * r
    dv_dzpos = (a**2 + b**2 + 2.0 * zpos**2) / r
    return -u * dv_dzpos / (u**2 + v**2)


@nb.njit
def _patch_solid_angle_potential_dz(x: float, y: float, z: float,
                                    wx: float, wy: float) -> float:
    """d/dzpos of _patch_solid_angle_potential(x, y, zpos, wx, wy),
    for zpos > 0 (same convention/singularity guard as that function)."""
    x1, x2 = -wx - x, wx - x
    y1, y2 = -wy - y, wy - y
    domega = (_corner_angle_dz(x2, y2, z)
              - _corner_angle_dz(x1, y2, z)
              - _corner_angle_dz(x2, y1, z)
              + _corner_angle_dz(x1, y1, z))
    return domega / (2.0 * np.pi)


@nb.njit
def near_field_dWdz(dx: float, dy: float, dz: float,
                    wx: float, wy: float, l: float, n_terms: int) \
                    -> float:
    """Analytic dW/dz of near_field_potential -- the z-component of the
    exact finite-size-pixel weighting field, for direct use in place of
    finite-differencing near_field_potential.

    Derivation: _patch_signed(x,y,z,wx,wy,z_src) = sign(zz) * g(|zz|)
    with zz = z - z_src and g = _patch_solid_angle_potential(x,y,.,wx,wy).
    Since d|zz|/dz = sign(zz), the chain rule gives
        d/dz [sign(zz) * g(|zz|)] = sign(zz) * g'(|zz|) * sign(zz)
                                   = g'(|zz|)
    i.e. the sign(zz) factors cancel and the derivative is just the
    *unsigned* zpos-derivative of the solid-angle formula (see
    _patch_solid_angle_potential_dz), evaluated at |zz|, for every
    image term -- no explicit sign/odd-extension bookkeeping needed
    here (verified numerically against central differences of
    near_field_potential to ~1e-10 relative error, both within and
    across image planes).

    Only the z-component is provided (no dW/dx, dW/dy), matching
    dipole_dWdz / image_lattice_dWdz in larndsim_ffe_signal.py, since
    the simulated drift is purely along z.
    """
    total = 0
    for n in range(-n_terms, n_terms + 1):
        total += _patch_solid_angle_potential_dz(
            dx, dy, abs(dz - 2 * n * l), wx, wy)
    return total


def launch_ffe_kernel(
    tpc_idx: int,
    tracks: cpt.NDArray,
    pixel_x: cpt.NDArray[cp.float32],
    pixel_y: cpt.NDArray[cp.float32],
    n_ticks: int,
    category: int,
    voxel_cache: dict[int, dict[str, Optional[cpt.NDArray[cp.float32]]]],
) -> cp.ndarray[tuple[int, int], cp.float32]:
    """
    Launch CUDA kernel for far-field induced current calculation.

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

    def launch_voxels():
        cache = voxel_cache.get(tpc_idx, None)
        if cache is None or cache['x'] is None:
            return

        pixel_categories = cp.full(n_pixels, category, dtype=cp.int32)

        calculate_ff_voxels[BPG, TPB](
            cache["x"], cache["y"], cache["z"], cache["q"],
            pixel_x, pixel_y, pixel_categories,
            z_anode, z_cathode, output)

    def launch_segments(kernel):
        tpc_tracks = tracks[tracks['pixel_plane'] == tpc_idx]
        if len(tpc_tracks) == 0:
            return

        kernel[BPG, TPB](
            tpc_tracks,
            pixel_x, pixel_y,
            z_anode, z_cathode, output)

    match sim.FARFIELD_MODE:
        case 'voxels':
            launch_voxels()
        case 'segments':
            match sim.FARFIELD_DIPOLE_MODE:
                case 'infinite_plane':
                    launch_segments(calculate_ff_segments)
                case 'box_lattice':
                    launch_segments(calculate_ff_segments_box_lattice)
                case _:
                    e = f"Invalid farfield_dipole_mode '{sim.FARFIELD_DIPOLE_MODE}'"
                    raise RuntimeError(e)
        case _:
            e = f"Invalid farfield_mode '{sim.FARFIELD_MODE}'"
            raise RuntimeError(e)

    return output
