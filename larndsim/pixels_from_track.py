from numba import cuda
from math import pi, ceil, sqrt, erf, exp
from .consts import *
from .indeces import *

@cuda.jit
def get_pixels(tracks, active_pixels, neighboring_pixels, n_pixels):
    '''
    For all tracks, takes the xy start and end position
    and calculates all impacted pixels by the track segment
    '''
    i = cuda.grid(1)
    if i < tracks.shape[0]:
        t = tracks[i]
        start_pixel = (int(round((t[x_start] - tpc_borders[0][0]) // x_pixel_size)),
                       int(round((t[y_start] - tpc_borders[1][0]) // y_pixel_size)))
        end_pixel = (int(round((t[x_end] - tpc_borders[0][0]) // x_pixel_size)),
                     int(round((t[y_end] - tpc_borders[1][0]) // y_pixel_size)))
        
        get_active_pixels(start_pixel[0], start_pixel[1], end_pixel[0], end_pixel[1], active_pixels[i])
        radius = 3
        n_pixels[i] = get_neighboring_pixels(active_pixels[i], radius, neighboring_pixels[i])


@cuda.jit(device=True)
def get_active_pixels(x0, y0, x1, y1, active_pixels):
    '''
    Converts track segement to an array of active pixels
    using Bresenham algorithm used to convert line to grid.
    '''
    
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
        active_pixels[x] = (x0 + x*xx + y*yx, y0 + x*xy + y*yy)
        if D >= 0:
            y += 1
            D -= 2*dx
        D += 2*dy
        

@cuda.jit(device=True)
def get_neighboring_pixels(active_pixels, radius, neighboring_pixels):
    '''
    For each active_pixel, it includes all 
    neighboring pixels within a specified radius
    '''
    count = 0
    for pix in range(active_pixels.shape[0]):
        if (active_pixels[pix][0] == 0) and (active_pixels[pix][1] == 0):
            break
        for x_r in range(-radius, radius+1):
            for y_r in range(-radius, radius+1):
                new_pixel = (active_pixels[pix][0]+x_r, active_pixels[pix][1]+y_r)
                isUnique = True
                for ipix in range(neighboring_pixels.shape[0]):
                    if new_pixel[0] == neighboring_pixels[ipix][0] and new_pixel[1] == neighboring_pixels[ipix][1]:
                        isUnique = False
                        break
                if isUnique:
                    neighboring_pixels[count] = new_pixel
                    count +=1
    return count
