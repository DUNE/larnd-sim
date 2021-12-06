"""
Module that finds which pixels lie on the projection on the anode plane
of each track segment. It can eventually include also the neighboring
pixels.
"""
import numba as nb
from numba import cuda
from .consts.detector import PIXEL_PITCH, N_PIXELS, TPC_BORDERS

@nb.njit
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
    return pixel_x + N_PIXELS[0] * (pixel_y + N_PIXELS[1] * pixel_plane)

@nb.njit
def id2pixel(pid):
    """
    Convert the unique pixel identifer to an x,y,plane tuple

    Args:
        pid (int): unique pixel identifier
    Returns:
        tuple: number of pixel pitches in x-dimension,
            number of pixel pitches in y-dimension,
            pixel plane number
    """
    return (pid % N_PIXELS[0], (pid // N_PIXELS[0]) % N_PIXELS[1],
            (pid // (N_PIXELS[0] * N_PIXELS[1])))

@cuda.jit
def max_pixels(tracks, n_max_pixels):
    """
    CUDA kernels that calculates the maximum number of pixels in the
    selected tracks.

    Args:
        tracks (:obj:`numpy.ndarray`): tracks array.
        n_max_pixels (:obj:`numpy.ndarray`): array to store the maximum
            number of pixels in the selected tracks.
    """
    itrk = cuda.grid(1)

    if itrk < tracks.shape[0]:
        t = tracks[itrk]
        this_border = TPC_BORDERS[int(t["pixel_plane"])]
        start_pixel = ((t["x_start"] - this_border[0][0]) // PIXEL_PITCH,
                       (t["y_start"] - this_border[1][0]) // PIXEL_PITCH)
        end_pixel = ((t["x_end"] - this_border[0][0]) // PIXEL_PITCH,
                     (t["y_end"]- this_border[1][0]) // PIXEL_PITCH)
        n_active_pixels = get_num_active_pixels(start_pixel[0], start_pixel[1],
                                                end_pixel[0], end_pixel[1], t["pixel_plane"])
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
        this_border = TPC_BORDERS[int(t["pixel_plane"])]
        start_pixel = (
            int((t["x_start"] - this_border[0][0]) // PIXEL_PITCH),
            int((t["y_start"] - this_border[1][0]) // PIXEL_PITCH),
            t["pixel_plane"])
        end_pixel = (
            int((t["x_end"] - this_border[0][0]) // PIXEL_PITCH),
            int((t["y_end"] - this_border[1][0]) // PIXEL_PITCH),
            t["pixel_plane"])

        get_active_pixels(start_pixel[0], start_pixel[1], end_pixel[0], end_pixel[1],
                          t["pixel_plane"], active_pixels[itrk])
        n_pixels_list[itrk] = get_neighboring_pixels(active_pixels[itrk],
                                                     radius,
                                                     neighboring_pixels[itrk])

@nb.njit
def get_num_active_pixels(x0, y0, x1, y1, plane_id):
    """
    Counts number of pixels intercepted by the projection of the
    track on the anode plane

    Args:
        x0 (int): start `x` coordinate
        y0 (int): start `y` coordinate
        x1 (int): end `x` coordinate
        y1 (int): end `y` coordinate
        plane_id (int): plane index
        tot_pixels (:obj:`numpy.ndarray`): array where we store
            the IDs of the pixels directly below the projection of
            the segments

    Returns:
        int: number of pixels intercepted by the projection of the
            track on the anode plane.
    """
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    n = 0
    if 0 <= x0 < N_PIXELS[0] and 0 <= y0 < N_PIXELS[1] and plane_id < TPC_BORDERS.shape[0]:
        n += 1

    while x0 != x1 or y0 != y1:

        e2 = 2*err

        if e2 - dy > dx - e2:
            err += dy
            x0 += sx
        else:
            err += dx
            y0 += sy

        if 0 <= x0 < N_PIXELS[0] and 0 <= y0 < N_PIXELS[1] and plane_id < TPC_BORDERS.shape[0]:
            n += 1

    return n

@nb.njit
def get_active_pixels(x0, y0, x1, y1, plane_id, tot_pixels):
    """
    Converts track segement to an array of active pixels
    using an adapted version of the Bresenham algorithm
    used to convert line to grid. Inspired by
    https://stackoverflow.com/questions/8936183/bresenham-lines-w-o-diagonal-movement/28786538

    Args:
        x0 (int): start `x` coordinate
        y0 (int): start `y` coordinate
        x1 (int): end `x` coordinate
        y1 (int): end `y` coordinate
        plane_id (int): plane index
        tot_pixels (:obj:`numpy.ndarray`): array where we store
            the IDs of the pixels directly below the projection of
            the segments
    """
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    i = 0
    if 0 <= x0 < N_PIXELS[0] and 0 <= y0 < N_PIXELS[1] and plane_id < TPC_BORDERS.shape[0]:
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

        if 0 <= x0 < N_PIXELS[0] and 0 <= y0 < N_PIXELS[1] and plane_id < TPC_BORDERS.shape[0]:
            tot_pixels[i] = pixel2id(x0, y0, plane_id)

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

        if active_pixels[pix] == -1:
            continue

        for x_r in range(-radius, radius+1):
            for y_r in range(-radius, radius+1):
                active_x, active_y, plane_id = id2pixel(active_pixels[pix])
                new_x, new_y = active_x + x_r, active_y + y_r
                is_unique = True

                if 0 <= new_x < N_PIXELS[0] and 0 <= new_y < N_PIXELS[1] and plane_id < TPC_BORDERS.shape[0]:
                    new_pixel = pixel2id(new_x, new_y, plane_id)

                    for ipix in range(neighboring_pixels.shape[0]):
                        if new_pixel == neighboring_pixels[ipix]:
                            is_unique = False
                            break

                    if is_unique:
                        neighboring_pixels[count] = new_pixel
                        count += 1

    return count
