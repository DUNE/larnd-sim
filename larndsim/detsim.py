"""
Module that calculates the current induced by edep-sim track segments
on the pixels
"""

from math import pi, ceil, sqrt, erf, exp, log

import numba as nb
import numpy as np

from numba import cuda
from .consts import pixel_size, module_borders, time_interval, n_pixels
from . import indeces as i
from . import consts

import logging
logging.basicConfig()
logger = logging.getLogger('detsim')
logger.setLevel(logging.WARNING)
logger.info("DETSIM MODULE PARAMETERS")

@cuda.jit
def time_intervals(track_starts, time_max, event_id_map, tracks):
    """
    Find the value of the longest signal time and stores the start
    time of each segment.

    Args:
        track_starts (:obj:`numpy.ndarray`): array where
            we store the segments start time
        time_max (:obj:`numpy.ndarray`): array where we store
            the longest signal time
        event_id_map (:obj:`numpy.ndarray`): array containing
            the event ID corresponding to each track
        tracks (:obj:`numpy.ndarray`): array containing the segment
            information
    """
    itrk = cuda.grid(1)

    if itrk < tracks.shape[0]:
        track = tracks[itrk]
        t_end = min(time_interval[1], (track[i.t_end] + consts.time_padding) // consts.t_sampling * consts.t_sampling)
        t_start = max(time_interval[0], (track[i.t_start] - consts.time_padding) // consts.t_sampling * consts.t_sampling)
        t_length = t_end - t_start
        track_starts[itrk] = t_start + event_id_map[itrk] * time_interval[1] * 2
        cuda.atomic.max(time_max, 0, ceil(t_length / consts.t_sampling))

@cuda.jit(device=True)
def z_interval(start_point, end_point, x_p, y_p, tolerance):
    """
    Here we calculate the interval in the drift direction for the pixel pID
    using the impact factor

    Args:
        start_point (tuple): coordinates of the segment start
        end_point (tuple): coordinates of the segment end
        x_p (float): pixel center `x` coordinate
        y_p (float): pixel center `y` coordinate
        tolerance (float): maximum distance between the pixel center and
            the segment

    Returns:
        tuple: `z` coordinate of the point of closest approach (POCA),
        `z` coordinate of the first slice, `z` coordinate of the last slice.
        (0,0,0) if POCA > tolerance.
    """
    if start_point[0] > end_point[0]:
        start = end_point
        end = start_point
    elif start_point[0] < end_point[0]:
        start = start_point
        end = end_point
    else: # Limit case that we should probably manage better
        return 0, 0, 0

    xs, ys = start[0], start[1]
    xe, ye = end[0], end[1]

    m = (ye - ys) / (xe - xs)
    q = (xe * ys - xs * ye) / (xe - xs)

    a, b, c = m, -1, q

    x_poca = (b*(b*x_p-a*y_p) - a*c)/(a*a+b*b)

    length = sqrt((end[0]-start[0])**2+(end[1]-start[1])**2+(end[2]-start[2])**2)
    dir3D = (end[0] - start[0])/length, (end[1] - start[1])/length, (end[2] - start[2])/length

    if x_poca < start[0]:
        doca = sqrt((x_p - start[0])**2 + (y_p - start[1])**2)
        x_poca = start[0]
    elif x_poca > end[0]:
        doca = sqrt((x_p - end[0])**2 + (y_p - end[1])**2)
        x_poca = end[0]
    else:
        doca = abs(a*x_p+b*y_p+c)/sqrt(a*a+b*b)

    z_poca = start[2] + (x_poca - start[0])/dir3D[0]*dir3D[2]

    plusDeltaZ, minusDeltaZ = 0, 0

    if tolerance > doca:
        length2D = sqrt((xe-xs)**2 + (ye-ys)**2)
        dir2D = (end[0]-start[0])/length2D, (end[1]-start[1])/length2D
        deltaL2D = sqrt(tolerance**2 - doca**2) # length along the track in 2D

        x_plusDeltaL = x_poca + deltaL2D*dir2D[0] # x coordinates of the tolerance range
        x_minusDeltaL = x_poca - deltaL2D*dir2D[0]
        plusDeltaL = (x_plusDeltaL - start[0])/dir3D[0] # length along the track in 3D
        minusDeltaL = (x_minusDeltaL - start[0])/dir3D[0] # of the tolerance range

        plusDeltaZ = start[2] + dir3D[2] * plusDeltaL # z coordinates of the
        minusDeltaZ = start[2] + dir3D[2] * minusDeltaL # tolerance range

        return z_poca, min(minusDeltaZ, plusDeltaZ), max(minusDeltaZ, plusDeltaZ)
    else:
        return 0, 0, 0

@cuda.jit(device=True)
def _b(x, y, z, start, sigmas, segment, Deltar):
    return -((x-start[0]) / (sigmas[0]*sigmas[0]) * (segment[0]/Deltar) + \
             (y-start[1]) / (sigmas[1]*sigmas[1]) * (segment[1]/Deltar) + \
             (z-start[2]) / (sigmas[2]*sigmas[2]) * (segment[2]/Deltar))

@cuda.jit(device=True)
def rho(point, q, start, sigmas, segment):
    """
    Function that returns the amount of charge at a certain point in space

    Args:
        point (tuple): point coordinates
        q (float): total charge
        start (tuple): segment start coordinates
        sigmas (tuple): diffusion coefficients
        segment (tuple): segment sizes

    Returns:
        float: the amount of charge at `point`.
    """
    x, y, z = point
    Deltax, Deltay, Deltaz = segment[0], segment[1], segment[2]
    Deltar = sqrt(Deltax**2+Deltay**2+Deltaz**2)
    a = ((Deltax/Deltar) * (Deltax/Deltar) / (2*sigmas[0]*sigmas[0]) + \
         (Deltay/Deltar) * (Deltay/Deltar) / (2*sigmas[1]*sigmas[1]) + \
         (Deltaz/Deltar) * (Deltaz/Deltar) / (2*sigmas[2]*sigmas[2]))
    factor = q/Deltar/(sigmas[0]*sigmas[1]*sigmas[2]*sqrt(8*pi*pi*pi))
    sqrt_a_2 = 2*sqrt(a)

    b = _b(x, y, z, start, sigmas, segment, Deltar)

    delta = (x-start[0])*(x-start[0])/(2*sigmas[0]*sigmas[0]) + \
            (y-start[1])*(y-start[1])/(2*sigmas[1]*sigmas[1]) + \
            (z-start[2])*(z-start[2])/(2*sigmas[2]*sigmas[2])

    integral = sqrt(pi) * \
               (-erf(b/sqrt_a_2) + erf((b + 2*a*Deltar)/sqrt_a_2)) / \
               sqrt_a_2
    
    expo = 0

    if factor and integral:
        expo = exp(b*b/(4*a) - delta + log(factor) + log(integral))

    return expo 

@cuda.jit(device=True)
def truncexpon(x, loc=0, scale=1):
    """
    A truncated exponential distribution.
    To shift and/or scale the distribution use the `loc` and `scale` parameters.
    """
    y = (x-loc)/scale
    if y > 0:
        return exp(-y)/scale
    else:
        return 0

@cuda.jit(device=True)
def current_model(t, t0, x, y):
    """
    Parametrization of the induced current on the pixel, which depends
    on the of arrival at the anode (:math:`t_0`) and on the position
    on the pixel pad.

    Args:
        t (float): time where we evaluate the current
        t0 (float): time of arrival at the anode
        x (float): distance between the point on the pixel and the pixel center
            on the :math:`x` axis
        y (float): distance between the point on the pixel and the pixel center
            on the :math:`y` axis

    Returns:
        float: the induced current at time :math:`t`
    """
    B_params = (1.060, -0.909, -0.909, 5.856, 0.207, 0.207)
    C_params = (0.679, -1.083, -1.083, 8.772, -5.521, -5.521)
    D_params = (2.644, -9.174, -9.174, 13.483, 45.887, 45.887)
    t0_params = (2.948, -2.705, -2.705, 4.825, 20.814, 20.814)

    a = B_params[0] + B_params[1]*x+B_params[2]*y+B_params[3]*x*y+B_params[4]*x*x+B_params[5]*y*y
    b = C_params[0] + C_params[1]*x+C_params[2]*y+C_params[3]*x*y+C_params[4]*x*x+C_params[5]*y*y
    c = D_params[0] + D_params[1]*x+D_params[2]*y+D_params[3]*x*y+D_params[4]*x*x+D_params[5]*y*y
    shifted_t0 = t0 + t0_params[0] + t0_params[1]*x + t0_params[2]*y + \
                      t0_params[3]*x*y + t0_params[4]*x*x + t0_params[5]*y*y

    a = min(a, 1)

    return a * truncexpon(-t, -shifted_t0, b) + (1-a) * truncexpon(-t, -shifted_t0, c)
 
@cuda.jit(device=True)
def track_point(start, direction, z):
    """
    This function returns the segment coordinates for a point along the `z` coordinate

    Args:
        start (tuple): start coordinates
        direction (tuple): direction coordinates
        z (float): `z` coordinate corresponding to the `x`, `y` coordinates

    Returns:
        tuple: the (x,y) pair of coordinates for the segment at `z`
    """
    l = (z - start[2]) / direction[2]
    xl = start[0] + l * direction[0]
    yl = start[1] + l * direction[1]

    return xl, yl

@nb.njit
def get_pixel_coordinates(pixel_id):
    """
    Returns the coordinates of the pixel center given the pixel ID
    """
    plane_id = pixel_id[0] // n_pixels[0]
    this_border = module_borders[plane_id]

    pix_x = (pixel_id[0] - n_pixels[0] * plane_id) * pixel_size[0] + this_border[0][0] + pixel_size[0]/2
    pix_y = pixel_id[1] * pixel_size[0] + this_border[1][0] + pixel_size[1]/2

    return pix_x, pix_y

@cuda.jit
def tracks_current(signals, pixels, tracks):
    """
    This CUDA kernel calculates the charge induced on the pixels by the input tracks.

    Args:
        signals (:obj:`numpy.ndarray`): empty 3D array with dimensions S x P x T,
            where S is the number of track segments, P is the number of pixels, and T is
            the number of time ticks. The output is stored here.
        pixels (:obj:`numpy.ndarray`): 3D array with dimensions S x P x 2, where S is
            the number of track segments, P is the number of pixels and the third dimension
            contains the two pixel ID numbers.
        tracks (:obj:`numpy.ndarray`): 2D array containing the detector segments.
    """
    itrk, ipix, it = cuda.grid(3)
    
    if itrk < signals.shape[0] and ipix < signals.shape[1] and it < signals.shape[2]:
        t = tracks[itrk]
        pID = pixels[itrk][ipix]

        if pID[0] >= 0 and pID[1] >= 0:

            # Pixel coordinates
            x_p, y_p = get_pixel_coordinates(pID)
            this_pixel_point = (x_p, y_p)

            if t[i.z_start] < t[i.z_end]:
                start = (t[i.x_start], t[i.y_start], t[i.z_start]-0.2)
                end = (t[i.x_end], t[i.y_end], t[i.z_end]+0.2)
            else:
                end = (t[i.x_start], t[i.y_start], t[i.z_start]+0.2)
                start = (t[i.x_end], t[i.y_end], t[i.z_end]-0.2)
                

            segment = (end[0]-start[0], end[1]-start[1], end[2]-start[2])
            length = sqrt(segment[0]**2 + segment[1]**2 + segment[2]**2)

            direction = (segment[0]/length, segment[1]/length, segment[2]/length)
            sigmas = (t[i.tran_diff], t[i.tran_diff], t[i.long_diff])
            impact_factor = max(sqrt((5*sigmas[0])**2+(5*sigmas[1])**2), sqrt(pixel_size[0]**2 + pixel_size[1]**2)/2)*2

            z_poca, z_start, z_end = z_interval(start, end, x_p, y_p, impact_factor)

            if z_start != 0 and z_end != 0:
                z_start = max(start[2]-3*sigmas[2], z_start)
                z_end = min(end[2]+3*sigmas[2], z_end)

                x_start, y_start = track_point(start, direction, z_start)
                x_end, y_end = track_point(start, direction, z_end)

                y_step = (abs(y_end-y_start) + 6*sigmas[1]) / (consts.sampled_points - 1)
                x_step = (abs(x_end-x_start) + 6*sigmas[0]) / (consts.sampled_points - 1)

                z_sampling = consts.t_sampling / 2.
                z_steps = max(consts.sampled_points, ceil(abs(z_end-z_start) / z_sampling))

                z_step = (z_end-z_start) / (z_steps-1)
                t_start =  max(time_interval[0],(t[i.t_start] - consts.time_padding) // consts.t_sampling * consts.t_sampling)
                
                for iz in range(z_steps):
                    z = z_start + iz*z_step

                    time_tick = t_start + it*consts.t_sampling
                    t0 = (z - module_borders[int(t[i.pixel_plane])][2][0]) / consts.vdrift

                    # FIXME: this sampling is far from ideal, we should sample around the track 
                    # and not in a cube containing the track
                    for ix in range(consts.sampled_points):
                        
                        x = x_start + ix*sign(direction[0])*x_step - 3*sign(direction[0])*sigmas[0]
                        x_dist = abs(x_p - x)
                        
                        for iy in range(consts.sampled_points):
                            
                            y = y_start + iy*sign(direction[1])*y_step - 3*sign(direction[1])*sigmas[1]
                            y_dist = abs(y_p - y)
                            
                            if x_dist < pixel_size[0]/2. and y_dist < pixel_size[1]/2.:
                                charge = rho((x,y,z), t[i.n_electrons], start, sigmas, segment) * abs(x_step) * abs(y_step) * abs(z_step)
                                signals[itrk][ipix][it] += current_model(time_tick, t0, x_dist, y_dist) * charge * consts.e_charge
                
@cuda.jit(device=True)
def sign(x):
    """
    Sign function
    """
    return 1 if x >= 0 else -1

@cuda.jit
def sum_pixel_signals(pixels_signals, signals, track_starts, index_map):
    """
    This function sums the induced current signals on the same pixel.

    Args:
        pixels_signals (:obj:`numpy.ndarray`): 2D array that will contain the
            summed signal for each pixel. First dimension is the pixel ID, second
            dimension is the time tick
        signals (:obj:`numpy.ndarray`): 3D array with dimensions S x P x T,
            where S is the number of track segments, P is the number of pixels, and T is
            the number of time ticks.
        track_starts (:obj:`numpy.ndarray`): 1D array containing the starting time of
            each track
        index_map (:obj:`numpy.ndarray`): 2D array containing the correspondence between
            the track index and the pixel ID index.
    """
    it, ipix, itick = cuda.grid(3)

    if it < signals.shape[0] and ipix < signals.shape[1]:
        index = index_map[it][ipix]
        start_tick = track_starts[it] // consts.t_sampling
        if itick < signals.shape[2] and index >= 0:
            itime = int(start_tick+itick)
            cuda.atomic.add(pixels_signals, (index, itime), signals[it][ipix][itick])
