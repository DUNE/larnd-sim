"""
Module that finds which pixels lie on the projection on the anode plane 
of each track segment. It can eventually include also the neighboring 
pixels.
"""

from numba import cuda
from .consts import tpc_borders, pixel_size, n_pixels, tpc_centers, tpc_size, module_borders
from . import indeces as i

import logging
logging.basicConfig()
logger = logging.getLogger('pixels_from_track')
logger.setLevel(logging.WARNING)
logger.info("PIXEL_FROM_TRACK MODULE PARAMETERS")

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
        this_border = module_borders[int(t[i.pixel_plane])]
        start_pixel = (int((t[i.x_start] - this_border[0][0]) // pixel_size[0] + n_pixels[0]*t[i.pixel_plane]),
                       int((t[i.y_start] - this_border[1][0]) // pixel_size[1]))
        end_pixel = (int((t[i.x_end] - this_border[0][0]) // pixel_size[0] + n_pixels[0]*t[i.pixel_plane]),
                     int((t[i.y_end] - this_border[1][0]) // pixel_size[1]))

        get_active_pixels(start_pixel[0], start_pixel[1],
                          end_pixel[0], end_pixel[1],
                          active_pixels[itrk])
        n_pixels_list[itrk] = get_neighboring_pixels(active_pixels[itrk],
                                                     radius,
                                                     neighboring_pixels[itrk])


@cuda.jit(device=True)
def get_active_pixels(x0, y0, x1, y1, tot_pixels):
    """
    Converts track segement to an array of active pixels
    using Bresenham algorithm used to convert line to grid.
    
    Args:
        x0 (float): start `x` coordinate
        y0 (float): start `y` coordinate
        x1 (float): end `x` coordinate
        y1 (float): end `y` coordinate
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
        plane_id = x_id // n_pixels[0]
        if 0 <= x_id < n_pixels[0]*(plane_id+1) and 0 <= y_id < n_pixels[1]:
            tot_pixels[x] = x_id, y_id
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
                new_pixel = (active_pixels[pix][0]+x_r, active_pixels[pix][1]+y_r)
                is_unique = True
                
                for ipix in range(neighboring_pixels.shape[0]):
                    if new_pixel[0] == neighboring_pixels[ipix][0] and new_pixel[1] == neighboring_pixels[ipix][1]:
                        is_unique = False
                        break
                
                plane_id = new_pixel[0] // n_pixels[0] 
                if is_unique and 0 <= new_pixel[0] < (plane_id+1)*n_pixels[0] and 0 <= new_pixel[1] < n_pixels[1]:
                    neighboring_pixels[count] = new_pixel
                    count += 1
                    
    return count
