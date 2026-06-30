"""
Module that finds which pixels lie on the projection on the anode plane
of each track segment. It can eventually include also the neighboring
pixels.
"""

import cupy as cp
import numba as nb
from numba import cuda
import numpy as np
from .consts import detector

MAX_NEIGHBOR_BACKTRACK_DISTANCE=4

@nb.njit
def pixel2id(pixel_x, pixel_y, pixel_plane):
    """
    Convert the (x, y, plane) pixel coordinate tuple to a unique integer identifier.

    The identifier is computed as a linearised index over the pixel grid:
    ``pixel_x + N_PIXELS[0] * (pixel_y + N_PIXELS[1] * pixel_plane)``.

    Args:
        pixel_x (int): Pixel index along the x-dimension (number of pixel
            pitches from the TPC border).
        pixel_y (int): Pixel index along the y-dimension (number of pixel
            pitches from the TPC border).
        pixel_plane (int): Pixel plane number (anode plane index).

    Returns:
        int: Unique integer identifier for the pixel.
    """
    return pixel_x + detector.N_PIXELS[0] * (pixel_y + detector.N_PIXELS[1] * pixel_plane)

@nb.njit
def id2pixel(pid):
    """
    Convert a unique pixel identifier back to its (x, y, plane) coordinate tuple.

    This is the inverse of :func:`pixel2id`.

    Args:
        pid (int): Unique integer pixel identifier as returned by
            :func:`pixel2id`.

    Returns:
        tuple[int, int, int]: A 3-tuple ``(pixel_x, pixel_y, pixel_plane)``
        where

        * ``pixel_x`` – pixel index along the x-dimension,
        * ``pixel_y`` – pixel index along the y-dimension,
        * ``pixel_plane`` – anode plane index.
    """
    return (pid % detector.N_PIXELS[0], (pid // detector.N_PIXELS[0]) % detector.N_PIXELS[1],
            (pid // (detector.N_PIXELS[0] * detector.N_PIXELS[1])))

@cuda.jit
def max_pixels(tracks, n_max_pixels):
    """
    CUDA kernel that calculates the maximum number of pixels intercepted
    across all supplied track segments.

    Each CUDA thread handles one track. The per-track pixel count is
    computed with :func:`get_num_active_pixels` and the global maximum is
    updated atomically so the result is safe for concurrent execution.

    Args:
        tracks (:obj:`numpy.ndarray`): Structured array of track segments.
            Each element must contain at least the fields ``x_start``,
            ``x_end``, ``y_start``, ``y_end``, and ``pixel_plane``.
        n_max_pixels (:obj:`numpy.ndarray`): Single-element integer array
            used to accumulate the maximum pixel count via
            ``cuda.atomic.max``.  The result is written to index 0.
    """
    itrk = cuda.grid(1)

    if itrk < tracks.shape[0]:
        t = tracks[itrk]
        this_border = detector.TPC_BORDERS[int(t["pixel_plane"])]
        start_pixel = ((t["x_start"] - this_border[0][0]) // detector.PIXEL_PITCH,
                       (t["y_start"] - this_border[1][0]) // detector.PIXEL_PITCH)
        end_pixel = ((t["x_end"] - this_border[0][0]) // detector.PIXEL_PITCH,
                     (t["y_end"]- this_border[1][0]) // detector.PIXEL_PITCH)
        n_active_pixels = get_num_active_pixels(start_pixel[0], start_pixel[1],
                                                end_pixel[0], end_pixel[1], t["pixel_plane"])
        cuda.atomic.max(n_max_pixels, 0, n_active_pixels)

@cuda.jit
def get_pixels(tracks, active_pixels, neighboring_pixels, neighboring_radius, n_pixels_list):
    """
    CUDA kernel that maps every track segment to its set of active and
    neighboring pixels on the anode plane.

    For each track the kernel:

    1. Determines the start/end pixel coordinates from physical positions.
    2. Calls :func:`get_active_pixels` (Bresenham line) to find pixels
       directly under the projected track segment.
    3. Calls :func:`get_neighboring_pixels` to expand the set by including
       pixels within ``detector.MAX_RADIUS`` of each active pixel.

    Args:
        tracks (:obj:`numpy.ndarray`): Structured array of track segments.
            Each element must contain the fields ``x_start``, ``x_end``,
            ``y_start``, ``y_end``, and ``pixel_plane``.
        active_pixels (:obj:`numpy.ndarray`): 2-D integer array of shape
            ``(n_tracks, max_active_pixels)`` pre-filled with ``-1``.
            On return, row ``i`` contains the unique pixel IDs that lie
            directly below the projection of track ``i``.
        neighboring_pixels (:obj:`numpy.ndarray`): 2-D integer array of
            shape ``(n_tracks, max_neighboring_pixels)`` pre-filled with
            ``-1``.  On return, row ``i`` contains the unique pixel IDs of
            both the active pixels and their neighbours for track ``i``.
        neighboring_radius (:obj:`numpy.ndarray`): 2-D float array of shape
            ``(n_tracks, max_neighboring_pixels)``.  On return, entry
            ``[i, j]`` holds the Euclidean distance (in pixel-pitch units)
            from the nearest active pixel to ``neighboring_pixels[i, j]``.
        n_pixels_list (:obj:`numpy.ndarray`): 1-D integer array of length
            ``n_tracks``.  On return, element ``i`` contains the total
            number of valid entries in ``neighboring_pixels[i]``.
    """
    itrk = cuda.grid(1)
    if itrk < tracks.shape[0]:
        t = tracks[itrk]

        this_border = detector.TPC_BORDERS[int(t["pixel_plane"])]
        start_pixel = (
            int((t["x_start"] - this_border[0][0]) // detector.PIXEL_PITCH),
            int((t["y_start"] - this_border[1][0]) // detector.PIXEL_PITCH),
            t["pixel_plane"])
        end_pixel = (
            int((t["x_end"] - this_border[0][0]) // detector.PIXEL_PITCH),
            int((t["y_end"] - this_border[1][0]) // detector.PIXEL_PITCH),
            t["pixel_plane"])

        get_active_pixels(start_pixel[0], start_pixel[1], end_pixel[0], end_pixel[1],
                          t["pixel_plane"], active_pixels[itrk])
        n_pixels_list[itrk] = get_neighboring_pixels(active_pixels[itrk],
                                                     neighboring_pixels[itrk],
                                                     neighboring_radius[itrk])

@nb.njit
def get_num_active_pixels(x0, y0, x1, y1, plane_id):
    """
    Count the number of pixels intercepted by the projection of a track
    segment onto the anode plane.

    Uses an adapted Bresenham line algorithm (without diagonal steps) to
    traverse the pixel grid from ``(x0, y0)`` to ``(x1, y1)`` and counts
    only pixels that fall within the valid detector bounds.

    Args:
        x0 (int): Start pixel index along the x-dimension.
        y0 (int): Start pixel index along the y-dimension.
        x1 (int): End pixel index along the x-dimension.
        y1 (int): End pixel index along the y-dimension.
        plane_id (int): Anode plane index used for bounds checking against
            ``detector.TPC_BORDERS``.

    Returns:
        int: Number of valid (in-bounds) pixels intercepted by the line
        from ``(x0, y0)`` to ``(x1, y1)`` on the given plane.
    """
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    n = 0
    if 0 <= x0 < detector.N_PIXELS[0] and 0 <= y0 < detector.N_PIXELS[1] and 0 <= plane_id < detector.TPC_BORDERS.shape[0]:
        n += 1

    while x0 != x1 or y0 != y1:

        e2 = 2*err

        if e2 - dy > dx - e2:
            err += dy
            x0 += sx
        else:
            err += dx
            y0 += sy

        if 0 <= x0 < detector.N_PIXELS[0] and 0 <= y0 < detector.N_PIXELS[1] and 0 <= plane_id < detector.TPC_BORDERS.shape[0]:
            n += 1

    return n

@nb.njit
def get_active_pixels(x0, y0, x1, y1, plane_id, tot_pixels):
    """
    Fill ``tot_pixels`` with the unique IDs of pixels intercepted by the
    projection of a track segment onto the anode plane.

    Uses an adapted Bresenham line algorithm without diagonal movement to
    convert the line from ``(x0, y0)`` to ``(x1, y1)`` into a sequence of
    pixel-grid cells.  Only pixels within the valid detector bounds are
    recorded.  Inspired by
    https://stackoverflow.com/questions/8936183/bresenham-lines-w-o-diagonal-movement/28786538.

    Args:
        x0 (int): Start pixel index along the x-dimension.
        y0 (int): Start pixel index along the y-dimension.
        x1 (int): End pixel index along the x-dimension.
        y1 (int): End pixel index along the y-dimension.
        plane_id (int): Anode plane index used for bounds checking against
            ``detector.TPC_BORDERS``.
        tot_pixels (:obj:`numpy.ndarray`): 1-D integer array (pre-allocated,
            length ≥ expected number of active pixels, initialised to ``-1``)
            that will be populated in-place with the unique pixel IDs
            (as returned by :func:`pixel2id`) of each intercepted pixel.
    """
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    i = 0

    if 0 <= x0 < detector.N_PIXELS[0] and 0 <= y0 < detector.N_PIXELS[1] and 0 <= plane_id < detector.TPC_BORDERS.shape[0]:
        tot_pixels[i] = pixel2id(x0, y0, plane_id)

    while x0 != x1 or y0 != y1:
        i += 1

        e2 = 2*err

        if e2 - dy > dx - e2:
            err += dy
            x0 += sx
        else:
            err += dx
            y0 += sy

        if 0 <= x0 < detector.N_PIXELS[0] and 0 <= y0 < detector.N_PIXELS[1] and 0 <= plane_id < detector.TPC_BORDERS.shape[0]:
            tot_pixels[i] = pixel2id(x0, y0, plane_id)

@cuda.jit(device=True)
def get_neighboring_pixels(active_pixels, neighboring_pixels, neighboring_radius):
    """
    Expand a set of active pixels by including all other pixels within
    ``detector.MAX_RADIUS`` pixels, then record the results in-place.

    For every pixel in ``active_pixels`` the function iterates over the
    square neighbourhood of half-width ``detector.MAX_RADIUS`` and adds
    each in-bounds, not-yet-seen pixel to ``neighboring_pixels`` together
    with its Euclidean distance to the active pixel.  Duplicate entries are
    suppressed by a linear search over already-added pixels.

    This is a CUDA *device* function and must be called from another kernel
    or device function.

    Args:
        active_pixels (:obj:`numpy.ndarray`): 1-D integer array of unique
            pixel IDs (as from :func:`pixel2id`) that lie directly below
            the projected track segment.  Entries equal to ``-1`` are
            treated as empty and skipped.
        neighboring_pixels (:obj:`numpy.ndarray`): 1-D integer array
            (pre-allocated, initialised to ``-1``) that will be populated
            in-place with the unique pixel IDs of all active and
            neighbouring pixels.
        neighboring_radius (:obj:`numpy.ndarray`): 1-D float array of the
            same length as ``neighboring_pixels``, populated in-place with
            the Euclidean distance (in pixel-pitch units) from the nearest
            active pixel to each entry in ``neighboring_pixels``.

    Returns:
        int: Total number of valid pixel entries written into
        ``neighboring_pixels`` (i.e. the number of unique in-bounds pixels
        found).
    """
    count = 0

    for pix in range(active_pixels.shape[0]):

        if active_pixels[pix] == -1:
            continue

        for x_r in range(-detector.MAX_RADIUS, detector.MAX_RADIUS+1):
            for y_r in range(-detector.MAX_RADIUS, detector.MAX_RADIUS+1):
                active_x, active_y, plane_id = id2pixel(active_pixels[pix])
                new_x, new_y = active_x + x_r, active_y + y_r
                is_unique = True

                if 0 <= new_x < detector.N_PIXELS[0] and 0 <= new_y < detector.N_PIXELS[1] and 0 <= plane_id < detector.TPC_BORDERS.shape[0]:
                    new_pixel = pixel2id(new_x, new_y, plane_id)

                    for ipix in range(neighboring_pixels.shape[0]):
                        if new_pixel == neighboring_pixels[ipix]:
                            is_unique = False
                            break

                    if is_unique:
                        neighboring_pixels[count] = new_pixel
                        dist=pow(x_r**2+y_r**2,0.5)
                        neighboring_radius[count] = dist
                        count += 1

    return count


@nb.njit
def _invert_array_map_inner(in_map, pix_id2idx, curr_idx, out_map):
    for seg_idx in range(in_map.shape[0]):
        ass = in_map[seg_idx]
        for pixid in ass:
            if pixid<0: break
            pix_idx = pix_id2idx[pixid.item()]
            out_map[pix_idx][curr_idx[pix_idx]]=seg_idx
            curr_idx[pix_idx] += 1


def invert_array_map(in_map,pix_set):
    '''
    Invert the map of unique segment id => a set of unique pixel IDs to a map of unique
    pixel index => a set of segment indexes (not IDs).

    Args:
        in_map  (:obj:`numpy.ndarray`): 2D array where segment index => list of pixel IDs
        pix_set (:obj:`numpy.ndarray`): 1D array containing all unique pixel IDs
    Returns:
        ndarray: 2D array where pixel index => list of segment index
    '''
    pixids,counts=cp.unique(in_map[in_map>=0].flatten(),return_counts=True)

    pix_id2idx = nb.typed.Dict.empty(key_type=nb.types.int64,
                                     value_type=nb.types.int64)
    for i, val in enumerate(pix_set.get()):
        pix_id2idx[val] = i

    mymap=np.full(shape=(pix_set.shape[0],counts.max().item()),fill_value=-1,dtype=int)
    curr_idx=np.zeros(shape=(len(pix_id2idx),),dtype=int)
    _invert_array_map_inner(in_map.get(), pix_id2idx, curr_idx, mymap)
    return cp.array(mymap)
