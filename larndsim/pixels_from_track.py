"""
Module that finds which pixels lie on the projection on the anode plane
of each track segment. It can eventually include also the neighboring
pixels.
"""
import numba as nb
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
    return pixel_x + n_pixels[0] * (pixel_y + n_pixels[1] * pixel_plane)


def pixel2id_nojit(pixel_x, pixel_y, pixel_plane):
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
    return (id % n_pixels[0], (id // n_pixels[0]) % n_pixels[1],
            (id // (n_pixels[0] * n_pixels[1])))


def id2pixel_nojit(id):
    return (id % n_pixels[0], (id // n_pixels[0]) % n_pixels[1],
            (id // (n_pixels[0] * n_pixels[1])))


@cuda.jit
def max_pixels(tracks, n_max_pixels):
    itrk = cuda.grid(1)

    if itrk < tracks.shape[0]:
        t = tracks[itrk]
        this_border = tpc_borders[int(t["pixel_plane"])]
        start_pixel = ((t["x_start"] - this_border[0][0]) // pixel_pitch + n_pixels[0]*t["pixel_plane"],
                       (t["y_start"] - this_border[1][0]) // pixel_pitch)
        end_pixel = ((t["x_end"] - this_border[0][0]) // pixel_pitch + n_pixels[0]*t["pixel_plane"],
                     (t["y_end"]- this_border[1][0]) // pixel_pitch)
        n_active_pixels = get_num_active_pixels(start_pixel[0], start_pixel[1],
                                                end_pixel[0], end_pixel[1])
        cuda.atomic.max(n_max_pixels, 0, n_active_pixels)

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
        start_pixel = (
            (t["x_start"] - this_border[0][0]) // pixel_pitch,
            (t["y_start"] - this_border[1][0]) // pixel_pitch,
            t["pixel_plane"])
        end_pixel = (
            (t["x_end"] - this_border[0][0]) // pixel_pitch,
            (t["y_end"] - this_border[1][0]) // pixel_pitch,
            t["pixel_plane"])

        get_active_pixels(start_pixel[0], start_pixel[1], end_pixel[0], end_pixel[1], t["pixel_plane"], active_pixels[itrk])

        n_pixels_list[itrk] = get_neighboring_pixels(active_pixels[itrk],
                                                     radius,
                                                     neighboring_pixels[itrk])
        
@cuda.jit(device=True)
def get_num_active_pixels(x0, y0, x1, y1, plane_id):
    """
    Converts track segement to an array of active pixels
    using Bresenham algorithm used to convert line to grid.

    Args:
        x0 (float): start `x` coordinate
        y0 (float): start `y` coordinate
        x1 (float): end `x` coordinate
        y1 (float): end `y` coordinate
        plane_id (int): plane index
        tot_pixels (:obj:`numpy.ndarray`): array where we store
            the IDs of the pixels directly below the projection of
            the segments
    """
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
    
    num_pixels = 0
    
    for x in range(dx + 1):
        x_id = x0 + x*xx + y*yx
        y_id = y0 + x*xy + y*yy
        
        if 0 <= x_id < n_pixels[0] and 0 <= y_id < n_pixels[1]:
            num_pixels += 1
            
        if D >= 0:
            y += 1
            D -= 2*dx
            
        D += 2*dy
        
    return num_pixels

@cuda.jit(device=True)
def get_active_pixels(x0, y0, x1, y1, plane_id, tot_pixels):
    """
    Converts track segement to an array of active pixels
    using Bresenham algorithm used to convert line to grid.

    Args:
        x0 (float): start `x` coordinate
        y0 (float): start `y` coordinate
        x1 (float): end `x` coordinate
        y1 (float): end `y` coordinate
        plane_id (int): plane index
        tot_pixels (:obj:`numpy.ndarray`): array where we store
            the IDs of the pixels directly below the projection of
            the segments
    """
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
            
        if 0 <= x_id < n_pixels[0] and 0 <= y_id < n_pixels[1] and plane_id < tpc_borders.shape[0]:
            tot_pixels[x] = pixel2id(x_id, y_id, plane_id)

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

        if (active_pixels[pix][0] == -1) and (active_pixels[pix][1] == -1):
            continue

        for x_r in range(-radius, radius+1):
            for y_r in range(-radius, radius+1):
                active_x, active_y, plane_id = id2pixel(active_pixels[pix])
                new_x, new_y = active_x + x_r, active_y + y_r
                is_unique = True

                if 0 <= new_x < n_pixels[0] and 0 <= new_y < n_pixels[1] and plane_id < tpc_borders.shape[0]:
                    new_pixel = pixel2id(new_x, new_y, plane_id)
                    
                    for ipix in range(neighboring_pixels.shape[0]):
                        if new_pixel == neighboring_pixels[ipix]:
                            is_unique = False
                            break

                    if is_unique:
                        neighboring_pixels[count] = new_pixel
                        count += 1
                        
    return count
