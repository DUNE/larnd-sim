"""
Module that finds which pixels lie on the projection on the anode plane
of each track segment. It can eventually include also the neighboring
pixels.
"""

from numba import cuda
from .consts import pixel_pitch, n_pixels, tpc_borders

import logging
logging.basicConfig()
logger = logging.getLogger('pixels_from_track')
logger.setLevel(logging.WARNING)
logger.info("PIXEL_FROM_TRACK MODULE PARAMETERS")


@cuda.jit(device=True)
def pixel2id(pixel_x, pixel_y, pixel_plane):
    """
    Convert the x,y,plane tuple to a unique identifier

    Args:
        pixel_x (int): number of pixel pitches in x-dimension
        pixel_y (int): number of pixel pitches in y-dimension
        pixel_plane (int): pixel plane number

    Returns:
        unique integer id
    """
    if pixel_x < 0 or pixel_y < 0 or pixel_x > n_pixels[0] or pixel_y > n_pixels[1]:
        return -1
    return pixel_x + n_pixels[0] * (pixel_y + n_pixels[1] * pixel_plane)


def pixel2id_nojit(pixel_x, pixel_y, pixel_plane):
    if pixel_x < 0 or pixel_y < 0 or pixel_x > n_pixels[0] or pixel_y > n_pixels[1]:
        return -1
    return pixel_x + n_pixels[0] * (pixel_y + n_pixels[1] * pixel_plane)


@cuda.jit(device=True)
def id2pixel(id):
    """
    Convert the unique pixel identifer to an x,y,plane tuple

    Args:
        id (int): unique pixel identifier
    Returns:
        pixel_x (int): number of pixel pitches in x-dimension
        pixel_y (int): number of pixel pitches in y-dimension
        pixel_plane (int): pixel plane number
    """
    if id < 0:
        return -1, -1, -1
    return (id % n_pixels[0], (id // n_pixels[0]) % n_pixels[1],
            (id // (n_pixels[0] * n_pixels[1])))


def id2pixel_nojit(id):
    if id < 0:
        return -1, -1, -1
    return (id % n_pixels[0], (id // n_pixels[0]) % n_pixels[1],
            (id // (n_pixels[0] * n_pixels[1])))


@cuda.jit
def get_pixels(tracks, active_pixels, neighboring_pixels, n_pixels_list, radius):
    """
    For all tracks, takes the xy start and end position
    and calculates all impacted pixels by the track segment

    Args:
        track (:obj:`numpy.ndarray`): array where we store the
            track segments information
        active_pixels (:obj:`numpy.ndarray`): array where we store
            the IDs of the pixels directly below the projection of
            the segments
        neighboring_pixels (:obj:`numpy.ndarray`): array where we store
            the IDs of the pixels directly below the projection of
            the segments and the ones next to them
        n_pixels_list (:obj:`numpy.ndarray`): number of total involved
            pixels
        radius (int): number of pixels around the active pixels that
            we are considering
    """
    itrk = cuda.grid(1)
    if itrk < tracks.shape[0]:
        t = tracks[itrk]
        this_border = tpc_borders[int(t["pixel_plane"])]
        start_pixel = pixel2id(
            (t["x_start"] - this_border[0][0]) // pixel_pitch,
            (t["y_start"] - this_border[1][0]) // pixel_pitch,
            t["pixel_plane"])
        end_pixel = pixel2id(
            (t["x_end"] - this_border[0][0]) // pixel_pitch,
            (t["y_end"] - this_border[1][0]) // pixel_pitch,
            t["pixel_plane"])

        get_active_pixels(start_pixel, end_pixel, active_pixels[itrk])
        n_pixels_list[itrk] = get_neighboring_pixels(active_pixels[itrk],
                                                     radius,
                                                     neighboring_pixels[itrk])

@cuda.jit(device=True)
def get_active_pixels(start_id, end_id, tot_pixels):
    """
    Converts track segement to an array of active pixels
    using Bresenham algorithm used to convert line to grid.

    Args:
        start_id (int): end pixel id
        end_id (int): end pixel id
        tot_pixels (:obj:`numpy.ndarray`): array where we store
            the IDs of the pixels directly below the projection of
            the segments
    """
    x0, y0, plane_id0 = id2pixel(start_id)
    x1, y1, _ = id2pixel(end_id)
    
    dx = x1 - x0
    dy = y1 - y0
    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1

    dx = abs(dx)
    dy = abs(dy)

    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0

    D = 2*dy - dx
    y = 0

    for x in range(dx + 1):
        x_id = x0 + x*xx + y*yx
        y_id = y0 + x*xy + y*yy
        plane_id = x_id // n_pixels[0]
        
            
        if 0 <= x_id <= n_pixels[0] and 0 <= y_id <= n_pixels[1]:
            tot_pixels[x] = pixel2id(x_id, y_id, plane_id0)
        if D >= 0:
            y += 1
            D -= 2*dx
            
        D += 2*dy

@cuda.jit(device=True)
def get_neighboring_pixels(active_pixels, radius, neighboring_pixels):
    """
    For each active_pixel, it includes all
    neighboring pixels within a specified radius

    Args:
        active_pixels (:obj:`numpy.ndarray`): array where we store
            the IDs of the pixels directly below the projection of
            the segments
        radius (int): number of layers of neighboring pixels we
            want to consider
        neighboring_pixels (:obj:`numpy.ndarray`): array where we store
            the IDs of the pixels directly below the projection of
            the segments and the ones next to them

    Returns:
        int: number of total involved pixels
    """
    count = 0

    for pix in range(active_pixels.shape[0]):

        if (active_pixels[pix] == -1):
            continue

        for x_r in range(-radius, radius+1):
            for y_r in range(-radius, radius+1):
                active_x, active_y, plane_id = id2pixel(active_pixels[pix])
                new_x, new_y = active_x + x_r, active_y + y_r
                new_pixel = pixel2id(new_x, new_y, plane_id)
                is_unique = True

                for ipix in range(neighboring_pixels.shape[0]):
                    if new_pixel == neighboring_pixels[ipix]:
                        is_unique = False
                        break

                if is_unique and 0 <= new_x < n_pixels[0] and 0 <= new_y < n_pixels[1] and plane_id < tpc_borders.shape[0]:
                    neighboring_pixels[count] = new_pixel
                    count += 1

    return count
