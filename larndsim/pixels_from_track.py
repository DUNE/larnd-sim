from numba import cuda
from .consts import tpc_borders, pixel_size
from . import indeces as i

@cuda.jit
def get_pixels(tracks, active_pixels, neighboring_pixels, n_pixels_list):
    """
    For all tracks, takes the xy start and end position
    and calculates all impacted pixels by the track segment
    """
    itrk = cuda.grid(1)
    if itrk < tracks.shape[0]:
        t = tracks[itrk]
        start_pixel = (int(round((t[i.x_start] - tpc_borders[0][0]) // pixel_size[0])),
                       int(round((t[i.y_start] - tpc_borders[1][0]) // pixel_size[1])))
        end_pixel = (int(round((t[i.x_end] - tpc_borders[0][0]) // pixel_size[0])),
                     int(round((t[i.y_end] - tpc_borders[1][0]) // pixel_size[1])))

        get_active_pixels(start_pixel[0], start_pixel[1],
                          end_pixel[0], end_pixel[1],
                          active_pixels[itrk])
        radius = 3
        n_pixels_list[itrk] = get_neighboring_pixels(active_pixels[itrk],
                                                     radius,
                                                     neighboring_pixels[itrk])


@cuda.jit(device=True)
def get_active_pixels(x0, y0, x1, y1, active_pixels):
    """
    Converts track segement to an array of active pixels
    using Bresenham algorithm used to convert line to grid.
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
        active_pixels[x] = (x0 + x*xx + y*yx, y0 + x*xy + y*yy)
        if D >= 0:
            y += 1
            D -= 2*dx
        D += 2*dy


@cuda.jit(device=True)
def get_neighboring_pixels(active_pixels, radius, neighboring_pixels):
    """
    For each active_pixel, it includes all
    neighboring pixels within a specified radius
    """
    count = 0
    for pix in range(active_pixels.shape[0]):
        if (active_pixels[pix][0] == 0) and (active_pixels[pix][1] == 0):
            break
        for x_r in range(-radius, radius+1):
            for y_r in range(-radius, radius+1):
                new_pixel = (active_pixels[pix][0]+x_r, active_pixels[pix][1]+y_r)
                is_unique = True
                for ipix in range(neighboring_pixels.shape[0]):
                    if new_pixel[0] == neighboring_pixels[ipix][0] and new_pixel[1] == neighboring_pixels[ipix][1]:
                        is_unique = False
                        break
                if is_unique:
                    neighboring_pixels[count] = new_pixel
                    count += 1
    return count
