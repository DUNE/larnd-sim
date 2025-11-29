"""
Classifies pixels into physics categories (CHARGE_COLLECTION, CHARGE_NEIGHBOR,
INDUCTION_ONLY, INACTIVE) based on:
- Perpendicular distance from segment line to pixel surface (point-to-box distance)
- Expected diffusion spread (sigma_T = sqrt(2 * D_T * t_drift))
- Far-field induction signal strength estimate

Example workflow:
    # Classify pixels by physics regime
    # For each pixel, all segments are processed to determine if
    # it's in CHARGE_COLLECTION, CHARGE_NEIGHBOR, INDUCTION_ONLY, 
    # or INACTIVE 
    result = classify_pixels(segments, plane_id=0)
    
    # Result contains pixel IDs for each category
    charge_pixel_ids = result.charge_pixels  # CuPy array on GPU
    neighbor_pixel_ids = result.neighbor_pixels
    induction_pixel_ids = result.induction_pixels
"""

import math

from numba import cuda
import numpy as np

import cupy as cp

from . import (
    PixelCategory,
    PixelClassificationResult,
)
from ..consts import detector, mesh_params


def generate_pixel_coordinates(plane_id):
    """
    Generate all pixel center coordinates for a given TPC plane.
    
    Uses the same grid structure as pixels_from_track.py for consistency.
    This ensures that pixel IDs from pixel2id() correspond to the correct
    physical positions.
    
    Args:
        plane_id (int): TPC plane index
        
    Returns:
        tuple: (pixel_coords_x, pixel_coords_y, pixel_ids)
            - pixel_coords_x: (n_pixels,) array of x-coordinates in cm
            - pixel_coords_y: (n_pixels,) array of y-coordinates in cm
            - pixel_ids: (n_pixels,) array of unique pixel IDs from pixel2id()
    """
    tpc_border = detector.TPC_BORDERS[plane_id]

    # Use full-plane index ranges from detector
    n_pixels_x = detector.N_PIXELS[0]
    n_pixels_y = detector.N_PIXELS[1]

    px_indices = np.arange(n_pixels_x, dtype=np.int32)
    py_indices = np.arange(n_pixels_y, dtype=np.int32)
    
    # Create grid
    px_grid, py_grid = np.meshgrid(px_indices, py_indices, indexing='ij')
    px_flat = px_grid.ravel()
    py_flat = py_grid.ravel()
    
    # Compute physical coordinates using the global pixel grid convention
    pixel_coords_x = (tpc_border[0][0] + (px_flat + 0.5) * detector.PIXEL_PITCH).astype(np.float32)
    pixel_coords_y = (tpc_border[1][0] + (py_flat + 0.5) * detector.PIXEL_PITCH).astype(np.float32)
    
    # Compute pixel IDs using the global indexing
    pixel_ids = (px_flat + detector.N_PIXELS[0] * 
                 (py_flat + detector.N_PIXELS[1] * plane_id)).astype(np.int32)
    
    return pixel_coords_x, pixel_coords_y, pixel_ids


@cuda.jit
def classify_pixels_kernel(
    segments,
    pixel_coords_x,
    pixel_coords_y,
    categories,
    thresholds_cc,
    thresholds_neighbor,
    thresholds_induction,
    induction_signal_threshold,
    diffusion_n_sigmas,
    z_anode,
):
    """
    CUDA kernel to classify each pixel by proximity to segments.

    For each pixel:
    1. Loop over all segments
    2. Calculate perpendicular distance from segment to pixel surface (point-to-box)
    3. Estimate diffusion spread: sigma_T = sqrt(2 * D_T * t_drift)
    4. Check proximity thresholds including diffusion spread
    5. Assign highest-priority category (CHARGE_COLLECTION > CHARGE_NEIGHBOR > INDUCTION_ONLY)

    Args:
        segments: (n_segments,) structured array with fields:
            - x_start, y_start, z_start (cm)
            - x_end, y_end, z_end (cm)
            - n_electrons (float)
            - pixel_plane (int)
        pixel_coords_x: (n_pixels,) pixel center x positions (cm)
        pixel_coords_y: (n_pixels,) pixel center y positions (cm)
        categories: (n_pixels,) output array of PixelCategory enum values
        thresholds_cc: charge collection radius (cm)
        thresholds_neighbor: charge neighbor radius (cm)
        thresholds_induction: induction cutoff radius (cm)
        induction_signal_threshold: minimum signal strength for induction-only (e-)
        diffusion_n_sigmas: number of sigmas to consider for diffusion extent
        z_anode: anode z-position for this plane (cm)
    """

    i_pixel = cuda.grid(1)

    if i_pixel >= categories.shape[0]:
        return

    px = pixel_coords_x[i_pixel]
    py = pixel_coords_y[i_pixel]
    # Plane is fixed per-kernel launch; z_anode provided by host

    # Start with INACTIVE, upgrade as we find relevant segments
    pixel_category = PixelCategory.INACTIVE
    
    # Accumulate induction signals from all segments
    # This handles high-multiplicity events where many small contributions add up
    total_induction_signal = 0.0

    # Loop over all segments
    for i_seg in range(segments.shape[0]):
        seg = segments[i_seg]

        # Segments are pre-filtered by plane on host

        # Skip if segment has no charge
        if seg["n_electrons"] <= 0:
            continue

        # Segment endpoints in x-y plane
        x_start = seg["x_start"]
        y_start = seg["y_start"]
        x_end = seg["x_end"]
        y_end = seg["y_end"]

        # Vector v from start to end (projected onto x-y plane)
        vx = x_end - x_start
        vy = y_end - y_start
        seg_length_xy = math.sqrt(vx * vx + vy * vy)

        # Calculate perpendicular distance from pixel to segment
        # Use point-to-box distance: measure distance to closest point on pixel square
        # Pixel box boundaries: [px - half_pitch, px + half_pitch] x [py - half_pitch, py + half_pitch]
        half_pitch = detector.PIXEL_PITCH * 0.5
        
        if seg_length_xy < 1e-6:
            # Vertical segment (no x-y extent) - use distance to start point
            # Clamp segment start point to pixel box boundary
            x_clamped = x_start
            if x_clamped < px - half_pitch:
                x_clamped = px - half_pitch
            elif x_clamped > px + half_pitch:
                x_clamped = px + half_pitch
                
            y_clamped = y_start
            if y_clamped < py - half_pitch:
                y_clamped = py - half_pitch
            elif y_clamped > py + half_pitch:
                y_clamped = py + half_pitch
            
            dx = x_start - x_clamped
            dy = y_start - y_clamped
            r_trans = math.sqrt(dx * dx + dy * dy)
            # Choose midpoint in z for vertical segment
            seg_z_closest = 0.5 * (seg["z_start"] + seg["z_end"])  # representative z
        else:
            # Non-vertical segment: find closest point on line segment to pixel box
            # First, find closest point on infinite line to pixel center
            dx_to_pixel = px - x_start
            dy_to_pixel = py - y_start

            # Project pixel center onto segment line: t = dot(pixel-start, v) / |v|^2
            t = (dx_to_pixel * vx + dy_to_pixel * vy) / (seg_length_xy * seg_length_xy)

            # Clamp t to [0, 1] to stay on segment
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0

            # Closest point on segment to pixel center
            x_closest = x_start + t * vx
            y_closest = y_start + t * vy
            seg_z_closest = seg["z_start"] + t * (seg["z_end"] - seg["z_start"])

            # Now clamp this point to the pixel box boundary
            # This gives us the closest point on the pixel surface to the segment
            x_clamped = x_closest
            if x_clamped < px - half_pitch:
                x_clamped = px - half_pitch
            elif x_clamped > px + half_pitch:
                x_clamped = px + half_pitch
                
            y_clamped = y_closest
            if y_clamped < py - half_pitch:
                y_clamped = py - half_pitch
            elif y_clamped > py + half_pitch:
                y_clamped = py + half_pitch

            # Distance from segment's closest point to pixel box boundary
            dx = x_closest - x_clamped
            dy = y_closest - y_clamped
            r_trans = math.sqrt(dx * dx + dy * dy)

        # Estimate drift distance and time
        # Note: edep-sim segments can have start/end in any order along z
        # We need to consider the drift time range for the segment
        # z_anode passed from host; represents anode position for this plane
        
        # For classification, use drift time at closest point on segment
        drift_distance_closest = abs(seg_z_closest - z_anode)
        
        # Also compute drift time range for the entire segment
        # This accounts for segments that span a large z-range
        drift_distance_start = abs(seg["z_start"] - z_anode)
        drift_distance_end = abs(seg["z_end"] - z_anode)
        drift_distance_max = max(drift_distance_start, drift_distance_end)
        
        # Use maximum drift time for diffusion estimate (most conservative)
        # This ensures we don't under-classify pixels that see diffused charge
        # from the farthest part of the segment
        t_drift = drift_distance_max / detector.V_DRIFT  # μs
        
        # Transverse diffusion spread
        sigma_T = math.sqrt(2.0 * detector.TRAN_DIFF * t_drift)

        # Effective radius including diffusion
        r_eff_cc = thresholds_cc + diffusion_n_sigmas * sigma_T
        r_eff_neighbor = thresholds_neighbor + diffusion_n_sigmas * sigma_T

        # Check charge collection first (highest priority)
        if r_trans <= r_eff_cc:
            pixel_category = PixelCategory.CHARGE_COLLECTION
            break  # Can't get higher priority, stop checking

        # Check charge neighbor
        if r_trans <= r_eff_neighbor:
            # Upgrade if we're currently INACTIVE or INDUCTION_ONLY
            if pixel_category < PixelCategory.CHARGE_NEIGHBOR:
                pixel_category = PixelCategory.CHARGE_NEIGHBOR
            continue

        # Accumulate induction signal (lowest priority)
        # We accumulate from all segments, then check threshold at the end
        if r_trans <= thresholds_induction:
            # 3D distance: include drift distance at closest point
            r3d = math.sqrt(r_trans * r_trans + drift_distance_closest * drift_distance_closest)
            if r3d > 0.05:  # avoid singularity
                # Normalization: characteristic scale remains PIXEL_PITCH
                r_normalized = r3d / detector.PIXEL_PITCH
                signal_estimate = seg["n_electrons"] / (r_normalized * r_normalized)
                total_induction_signal += signal_estimate

    # After checking all segments, apply induction classification if threshold met
    # Only upgrade to INDUCTION_ONLY if we're currently INACTIVE
    if total_induction_signal >= induction_signal_threshold:
        if pixel_category == PixelCategory.INACTIVE:
            pixel_category = PixelCategory.INDUCTION_ONLY

    # Write final category
    categories[i_pixel] = pixel_category


def classify_pixels(
    segments,
    plane_id
    ):
    """
    Classify pixels into physics regimes based on segment proximity and diffusion.

    This function is the main entry point for pixel classification. It:
    1. Generates pixel coordinates for the specified TPC plane
    2. Validates inputs and transfers data to GPU if needed
    3. Launches the classification kernel
    4. Extracts per-category pixel indices
    5. Returns a PixelClassificationResult with categorized pixel lists and pixel IDs

    Args:
        segments: (n_segments,) structured array with fields:
            - x_start, y_start, z_start, x_end, y_end, z_end (cm)
            - n_electrons (float)
            - pixel_plane (int)
        plane_id: TPC plane identifier (int)

    Returns:
        PixelClassificationResult with:
            - charge_pixels: pixel IDs (global) of CHARGE_COLLECTION pixels
            - neighbor_pixels: pixel IDs (global) of CHARGE_NEIGHBOR pixels
            - induction_pixels: pixel IDs (global) of INDUCTION_ONLY pixels

    Raises:
        ValueError: If input arrays have mismatched sizes or invalid data
    """

    # Generate pixel coordinates for this TPC plane
    px_np, py_np, pixel_ids = generate_pixel_coordinates(plane_id)
    n_pixels = len(px_np)
    
    # Filter segments to only those on the specified plane
    plane_mask = segments["pixel_plane"] == plane_id
    segments_np = segments[plane_mask]
    
    if len(segments_np) == 0:
        # No segments on this plane -> all pixels inactive, return empty arrays
        return PixelClassificationResult(
            charge_pixels=cp.array([], dtype=np.int32),
            neighbor_pixels=cp.array([], dtype=np.int32),
            induction_pixels=cp.array([], dtype=np.int32),
        )

    # Move data to device memory for kernel consumption
    segments_dev = cuda.to_device(segments_np)
    px_dev = cuda.to_device(px_np.astype(np.float32))
    py_dev = cuda.to_device(py_np.astype(np.float32))
    categories_dev = cuda.device_array(n_pixels, dtype=np.int32)

    # Launch kernel (1D grid)
    threads_per_block = 256
    blocks = (n_pixels + threads_per_block - 1) // threads_per_block
        
    # Determine z_anode on host and pass as scalar to kernel
    z_anode = float(detector.TPC_BORDERS[int(plane_id)][2][0])

    classify_pixels_kernel[blocks, threads_per_block](
        segments_dev,
        px_dev,
        py_dev,
        categories_dev,
        mesh_params.CHARGE_COLLECTION_RADIUS * detector.PIXEL_PITCH,
        mesh_params.CHARGE_NEIGHBOR_RADIUS * detector.PIXEL_PITCH,
        mesh_params.INDUCTION_CUTOFF_RADIUS,
        mesh_params.INDUCTION_SIGNAL_THRESHOLD,
        detector.DIFF_N_SIGMAS,
        z_anode,
    )

    # Copy back categories to host
    categories_host = categories_dev.copy_to_host()

    # Build masks and index arrays (NumPy first)
    charge_mask_h = categories_host == int(PixelCategory.CHARGE_COLLECTION)
    neighbor_mask_h = categories_host == int(PixelCategory.CHARGE_NEIGHBOR)
    induction_mask_h = categories_host == int(PixelCategory.INDUCTION_ONLY)

    charge_pixels_h = np.where(charge_mask_h)[0].astype(np.int32)
    neighbor_pixels_h = np.where(neighbor_mask_h)[0].astype(np.int32)
    induction_pixels_h = np.where(induction_mask_h)[0].astype(np.int32)
    
    # Map indices to pixel IDs
    charge_pixel_ids_h = pixel_ids[charge_pixels_h]
    neighbor_pixel_ids_h = pixel_ids[neighbor_pixels_h]
    induction_pixel_ids_h = pixel_ids[induction_pixels_h]

    # Convert outputs to CuPy arrays for downstream GPU use
    charge_pixel_ids_gpu = cp.asarray(charge_pixel_ids_h)
    neighbor_pixel_ids_gpu = cp.asarray(neighbor_pixel_ids_h)
    induction_pixel_ids_gpu = cp.asarray(induction_pixel_ids_h)

    # Return pixel IDs for the three categories
    return PixelClassificationResult(
        charge_pixels=charge_pixel_ids_gpu,
        neighbor_pixels=neighbor_pixel_ids_gpu,
        induction_pixels=induction_pixel_ids_gpu,
    )
