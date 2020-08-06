"""
Module that calculates the current induced by edep-sim track segments
on the pixels
"""

import numba as nb
import numpy as np
import skimage.draw

from math import pi, ceil, sqrt, erf, exp
from .consts import *


@nb.njit(fastmath=True)
def current_response(t, A=1, B=5, t0=0):
    """Current response parametrization"""
    result = A * np.exp((t - t0) / B)
    result[t > t0] = 0
    return result

@nb.njit(fastmath=True)
def slice_coordinates(point, padding, slice_size):
    xl = point[0]
    yl = point[1]
    xx = np.linspace(xl - padding, xl + padding, slice_size)
    yy = np.linspace(yl - padding, yl + padding, slice_size)

    return xx, yy, np.array([point[2]])

@nb.njit(fastmath=True)
def track_point(start, direction, z):
    l = (z - start[2]) / direction[2]
    xl = start[0] + l * direction[0]
    yl = start[1] + l * direction[1]

    return  np.array([xl, yl, z])

@nb.njit(fastmath=True)
def z_interval(start_point, end_point, x_p, y_p, tolerance):
    """Here we calculate the interval in the drift direction for the pixel pID
    using the impact factor"""

    if start_point[0] > end_point[0]:
        start = end_point
        end = start_point
    elif start_point[0] < end_point[0]:
        start = start_point
        end = end_point
    else: # Limit case that we should probably manage better
        return 0, 0

    xs, ys = start[0], start[1]
    xe, ye = end[0], end[1]

    m = (ye - ys) / (xe - xs)
    q = (xe * ys - xs * ye) / (xe - xs)

    a, b, c = m, -1, q

    x_poca = (b*(b*x_p-a*y_p) - a*c)/(a*a+b*b)

    segment = end - start
    length = np.linalg.norm(segment)
    dir3D = segment/length

    if x_poca < start[0]:
        doca = np.sqrt((x_p - start[0])**2 + (y_p - start[1])**2)
        x_poca = start[0]
    elif x_poca > end[0]:
        doca = np.sqrt((x_p - end[0])**2 + (y_p - end[1])**2)
        x_poca = end[0]
    else:
        doca = np.abs(a*x_p+b*y_p+c)/np.sqrt(a*a+b*b)

    plusDeltaZ, minusDeltaZ = 0, 0

    if tolerance > doca:
        length2D = np.sqrt((xe-xs)**2 + (ye-ys)**2)
        dir2D = (end[0]-start[0])/length2D, (end[1]-start[1])/length2D
        deltaL2D = np.sqrt(tolerance**2 - doca**2) # length along the track in 2D

        x_plusDeltaL = x_poca + deltaL2D*dir2D[0] # x coordinates of the tolerance range
        x_minusDeltaL = x_poca - deltaL2D*dir2D[0]
        plusDeltaL = (x_plusDeltaL - start[0])/dir3D[0] # length along the track in 3D
        minusDeltaL = (x_minusDeltaL - start[0])/dir3D[0] # of the tolerance range

        plusDeltaZ = start[2] + dir3D[2] * plusDeltaL # z coordinates of the
        minusDeltaZ = start[2] + dir3D[2] * minusDeltaL # tolerance range

    return min(minusDeltaZ, plusDeltaZ), max(minusDeltaZ, plusDeltaZ)

@nb.jit
def get_pixels(track, cols, pixel_size):
    s = (track[cols["x_start"]], track[cols["y_start"]])
    e = (track[cols["x_end"]], track[cols["y_end"]])

    start_pixel = (int(round((s[0]-tpc_borders[0][0]) // pixel_size[0])),
                   int(round((s[1]-tpc_borders[1][0]) // pixel_size[1])))

    end_pixel = (int(round((e[0]-tpc_borders[0][0]) // pixel_size[0])),
                 int(round((e[1]-tpc_borders[1][0]) // pixel_size[1])))

    active_pixels = skimage.draw.line(start_pixel[0], start_pixel[1],
                                      end_pixel[0], end_pixel[1])

    xx, yy = active_pixels
    involved_pixels = []

    for x, y in zip(xx, yy):
        neighbors = ((x, y),
                     (x, y + 1), (x + 1, y),
                     (x, y - 1), (x - 1, y),
                     (x + 1, y + 1), (x - 1, y - 1),
                     (x + 1, y - 1), (x - 1, y + 1))
        nneighbors = ((x + 2, y), (x + 2, y + 1), (x + 2, y + 2), (x + 2, y - 1), (x + 2, y - 2),
                      (x - 2, y), (x - 2, y + 1), (x - 2, y + 2), (x - 2, y - 1), (x + 2, y - 2),
                      (x, y + 2), (x - 1, y + 2), (x + 1, y + 2),
                      (x, y - 2), (x - 1, y - 2), (x + 1, y - 2))

        for ne in (neighbors+nneighbors):
            if ne not in involved_pixels:
                involved_pixels.append(ne)

    return involved_pixels

@nb.jit
def list2array(pixelTrackIDs):
    lens = [len(pIDs) for pIDs in pixelTrackIDs]
    pIDs_array = np.full((len(pixelTrackIDs), max(lens), 2), np.inf)
    for i, pIDs in enumerate(pixelTrackIDs):
        for j, pID in enumerate(pIDs):
            pIDs_array[i][j] = pID
    
    return pIDs_array

@nb.jit
def pixelID_track(tracks, cols, pixel_size):
    dictPixelTrackID = []

    for track in tracks:
        dictPixelTrackID.append(get_pixels(track, cols, pixel_size))

    return dictPixelTrackID

@nb.njit(fastmath=True)
def slice_signal(x_p, y_p, weights, xv, yv, this_current_response):
    distances = np.empty(len(xv)*len(yv))
    i = 0
    for x in xv:
        for y in yv:
            distances[i] = exp(-10*sqrt((x - x_p)*(x - x_p) + (y - y_p)*(y - y_p)))
            i += 1

    weights_attenuated = weights * distances
    signals = np.outer(weights_attenuated, this_current_response)

    return signals

@nb.njit(fastmath=True)
def _b(x, y, z, start, sigmas, segment, Deltar):
    return -((x-start[0]) / (sigmas[0]*sigmas[0]) * (segment[0]/Deltar) + \
             (y-start[1]) / (sigmas[1]*sigmas[1]) * (segment[1]/Deltar) + \
             (z-start[2]) / (sigmas[2]*sigmas[2]) * (segment[2]/Deltar))

@nb.njit(fastmath=True)
def rho(x, y, z, a, start, sigmas, segment, Deltar, factor):
    """Charge distribution in space"""
    b = _b(x, y, z, start, sigmas, segment, Deltar)
    sqrt_a_2 = 2*sqrt(a)

    delta = (x-start[0])*(x-start[0])/(2*sigmas[0]*sigmas[0]) + \
            (y-start[1])*(y-start[1])/(2*sigmas[1]*sigmas[1]) + \
            (z-start[2])*(z-start[2])/(2*sigmas[2]*sigmas[2])

    expo = exp(b*b/(4*a) - delta)

    integral = sqrt(pi) * \
               (-erf(b/sqrt_a_2) + erf((b + 2*a*Deltar)/sqrt_a_2)) / \
               sqrt_a_2

    return expo * factor * integral

@nb.njit(fastmath=True)
def diffusion_weights(n_electrons, point, start, end, sigmas, slice_size):
    segment = end - start

    Deltar = np.linalg.norm(segment)

    factor = n_electrons/Deltar/(sigmas.prod()*sqrt(8*pi*pi*pi))
    a = ((segment/Deltar)**2 / (2*sigmas**2)).sum()


    xx = np.linspace(point[0] - sigmas[0] * 5,
                     point[0] + sigmas[0] * 5,
                     slice_size)
    yy = np.linspace(point[1] - sigmas[1] * 5,
                     point[1] + sigmas[1] * 5,
                     slice_size)
    zz = np.array([point[2]])

    weights = np.empty(len(xx)*len(yy)*len(zz))
    i = 0
    for x in xx:
        for y in yy:
            for z in zz:
                weights[i] = rho(x, y, z, a, start, sigmas, segment, Deltar, factor)
                i += 1

    return weights.ravel() * (xx[1]-xx[0]) * (yy[1]-yy[0])


@nb.njit(fastmath=True)
def track_current(track, pixels, cols, slice_size, t_sampling, active_pixels, pixel_size, time_padding=20):

    start = np.array([track[cols["x_start"]], track[cols["y_start"]], track[cols["z_start"]]])
    end = np.array([track[cols["x_end"]], track[cols["y_end"]], track[cols["z_end"]]])
    mid_point = (start+end)/2

    segment = end - start
    length = np.linalg.norm(segment)
    direction = segment/length

    sigmas = np.array([track[cols["tranDiff"]],
                       track[cols["tranDiff"]],
                       track[cols["longDiff"]]])
    endcap_size = 5 * track[cols["longDiff"]]

    weights_bulk = diffusion_weights(track[cols["NElectrons"]], mid_point, start, end, sigmas, slice_size)

    t_start = (track[cols["t_start"]] - time_padding) // t_sampling * t_sampling
    t_end = (track[cols["t_end"]] + time_padding) // t_sampling * t_sampling
    t_length = t_end - t_start
    time_interval = np.linspace(t_start, t_end, int(round(t_length / t_sampling)))

    z_sampling = t_sampling * vdrift

    for i in nb.prange(pixels.shape[0]):
        pID = pixels[i]
        if pID[0] == np.inf:
            break

        x_p = pID[0] * pixel_size[0] + tpc_borders[0][0] + pixel_size[0] / 2
        y_p = pID[1] * pixel_size[1] + tpc_borders[1][0] + pixel_size[1] / 2

        z_start, z_end = z_interval(start, end, x_p, y_p, 3*np.sqrt(pixel_size[0]**2 + pixel_size[1]**2))
        z_range = np.linspace(z_start, z_end, ceil(abs(z_end-z_start)/z_sampling)+1)

        if z_range.size <= 1:
            continue

        signal = np.zeros_like(time_interval)

        for z in z_range:
            point = track_point(start, direction, z)
            xv, yv, zv = slice_coordinates(point, track[cols["tranDiff"]] * 5, slice_size)
            if track[cols["z_end"]] - endcap_size <= z <= track[cols["z_end"]] + endcap_size or \
               track[cols["z_start"]] - endcap_size <= z <= track[cols["z_start"]] + endcap_size:
                weights = diffusion_weights(track[cols["NElectrons"]], point, start, end, sigmas, slice_size)
            else:
                weights = weights_bulk

            t0 = (z - tpc_borders[2][0]) / vdrift

            current_response_z = current_response(time_interval, t0=t0)
            signals = slice_signal(x_p, y_p, weights, xv, yv, current_response_z)
            signal += np.sum(signals, axis=0) * (z_range[1]-z_range[0])

        if not signal.any():
            continue

        t = (pID[0], pID[1])
        if t not in active_pixels:
            pixel_signal = nb.typed.List()
            pixel_signal.append((t_start, t_end, signal))
            active_pixels[t] = pixel_signal
        else:
            active_pixels[t].append((t_start, t_end, signal))


@nb.jit
def pixel_response(pixel_signals, anode_t):
    current = np.zeros_like(anode_t)

    for signal in pixel_signals:
        current[(anode_t >= signal[0]) & (anode_t <= signal[1])] += signal[2]

    return current


float_array = nb.types.float64[::1]
pixelID_type = nb.types.Tuple((nb.int64, nb.int64))
signal_type = nb.types.ListType(nb.types.Tuple((nb.float64, nb.float64, float_array)))

@nb.njit(fastmath=True)
def tracks_current(tracks, pIDs_array, cols, pixel_size, t_sampling=1, slice_size=20):
    active_pixels = nb.typed.Dict.empty(key_type=pixelID_type,
                                        value_type=signal_type)

    for i in nb.prange(tracks.shape[0]):
        track = tracks[i]
        pID = pIDs_array[i]
        track_current(track, pID, cols, slice_size, t_sampling, active_pixels, pixel_size)

    return active_pixels

@nb.jit
def pixel_from_coordinates(x, y, n_pixels):
    x_pixel = np.linspace(tpc_borders[0][0], tpc_borders[0][1], n_pixels)
    y_pixel = np.linspace(tpc_borders[1][0], tpc_borders[1][1], n_pixels)
    return np.digitize(x, x_pixel), np.digitize(y, y_pixel)
