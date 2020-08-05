#!/usr/bin/env python
"""
Detector simulation module
"""

import numpy as np
import scipy.stats
import numba as nb

from math import pi, sqrt, ceil, exp
from spycial import erf
import skimage.draw

from . import consts
from . import drifting
from . import quenching


@nb.njit
def nb_linspace(start, stop, n):
    return np.linspace(start, stop, n)


spec = [
    ('start', nb.float64[:]),
    ('end', nb.float64[:]),
    ('segment', nb.float64[:]),
    ('Deltar', nb.float64),
    ('sigmas', nb.float64[:]),
    ('Q', nb.float64),
    ('factor', nb.float64),
    ('a', nb.float64),
]

@nb.experimental.jitclass(spec)
class TrackCharge:
    """Track charge deposition"""

    def __init__(self, Q, start, end, sigmas):
        segment = end - start
        self.start = start
        self.end = end
        self.segment = segment

        self.Deltar = np.linalg.norm(segment)
        self.sigmas = sigmas
        self.Q = Q
        self.factor = self.Q/self.Deltar/(self.sigmas.prod()*sqrt(8*pi*pi*pi))
        self.a = ((segment/self.Deltar)**2 / (2*self.sigmas**2)).sum()

    def __repr__(self):
        instanceDescription = "<%s instance at %s>\nQ %f\nxyz (%f, %f, %f), (%f, %f, %f)\nsigmas (%f, %f, %f)" % \
                               (self.__class__.__name__, id(self),
                                self.Q, *self.start, *self.end, *self.sigmas)

        return instanceDescription

    def _b(self, position):
        b = -((position.T - self.start) / self.sigmas**2 * self.segment / self.Deltar).T
        return b.sum(axis=0)

    def rho(self, position):
        """Charge distribution in space"""
        b = self._b(position)
        sqrt_a_2 = 2*np.sqrt(self.a)

        delta = ((position.T - self.start)**2 / (2*self.sigmas**2)).T

        expo = np.exp(b*b/(4*self.a) - delta.sum(axis=0))
        integral = (sqrt(pi)
                    * (-erf(b/sqrt_a_2) + erf((b + 2*self.a*self.Deltar)/sqrt_a_2))
                    / sqrt_a_2)

        return expo * integral * self.factor


class PixelSignal:
    """Signal induced on pixel at a given time interval"""
    def __init__(self, pID, trackID, current, time_interval):
        self.id = pID
        self.trackID = trackID
        self.current = current
        self.time_interval = time_interval

    def __repr__(self):
        instanceDescription = "<%s instance at %s>\n\tPixel ID (%i, %i)\n\tTrack ID %i\n\tTime interval (%g, %g)\n\tCurrent integral %g" % \
                              (self.__class__.__name__, id(self),
                               *self.id, self.trackID, *self.time_interval, self.current.sum())

        return instanceDescription


class TPC:
    """This class implements the detector simulation of a pixelated LArTPC.
    It calculates the charge deposited on each pixel.

    Args:
        col: dictionary containing the tensor indeces
        n_pixels (int): number of pixels that tile the anode
        t_sampling (float): time sampling

    Attributes:
        n_pixels (int): number of pixels per axis that tile the anode
        t_sampling (float): time sampling
        anode_t (float array): time window
        x_pixel_size (float): x dimension of the pixel
        y_pixel_size (float): y dimension of the pixel
        _slice_size (int): number of points of the slice for each axis
        active_pixels (dict): dictionary of active pixels
    """
    def __init__(self, col, n_pixels=50, t_sampling=0.1):

        self.n_pixels = n_pixels
        self.col = col

        t_length = consts.timeInterval[1] - consts.timeInterval[0]

        self.t_sampling = t_sampling
        self.anode_t = nb_linspace(consts.timeInterval[0], consts.timeInterval[1], int(round(t_length/self.t_sampling)))

        self.x_pixel_size = (consts.tpcBorders[0][1]-consts.tpcBorders[0][0]) / n_pixels
        self.y_pixel_size = (consts.tpcBorders[1][1]-consts.tpcBorders[1][0]) / n_pixels
        self.active_pixels = {}

        self._slice_size = 20
        self._time_padding = 20

    @staticmethod
    def current_response(t, A=1, B=5, t0=0):
        """Current response parametrization"""
        result = A * np.exp((t - t0) / B)
        result[t > t0] = 0
        return result

    @staticmethod
    @nb.njit(fastmath=True)
    def sigmoid(t, t0, t_rise=1):
        """Sigmoid function for FEE response"""
        result = 1 / (1 + np.exp(-(t-t0)/t_rise))
        return result

    @staticmethod
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

    def calculate_current(self, track):
        pixelsIDs = self.get_pixels(track)

        xs, xe = track[self.col["x_start"]], track[self.col["x_end"]]
        ys, ye = track[self.col["y_start"]], track[self.col["y_end"]]
        zs, ze = track[self.col["z_start"]], track[self.col["z_end"]]

        start = np.array([xs, ys, zs])
        end = np.array([xe, ye, ze])
        segment = end - start
        length = np.linalg.norm(segment)
        direction = segment/length

        sigmas = np.array([track[self.col["tranDiff"]],
                           track[self.col["tranDiff"]],
                           track[self.col["longDiff"]]])

        trackCharge = TrackCharge(track[self.col["NElectrons"]],
                                  start,
                                  end,
                                  sigmas=sigmas)

        endcap_size = 5 * sigmas[2]
        x = nb_linspace((xe + xs) / 2 - track[self.col["tranDiff"]] * 5,
                        (xe + xs) / 2 + track[self.col["tranDiff"]] * 5,
                        self._slice_size)
        y = nb_linspace((ye + ys) / 2 - track[self.col["tranDiff"]] * 5,
                        (ye + ys) / 2 + track[self.col["tranDiff"]] * 5,
                        self._slice_size)
        z = (ze + zs) / 2
        z_sampling = self.t_sampling * consts.vdrift
        weights = trackCharge.rho(np.array(np.meshgrid(x, y, z)))
        weights_bulk = weights.ravel() * (x[1]-x[0]) * (y[1]-y[0])

        t_start = (track[self.col["t_start"]] - self._time_padding) // self.t_sampling * self.t_sampling
        t_end = (track[self.col["t_end"]] + self._time_padding) // self.t_sampling * self.t_sampling
        t_length = t_end - t_start
        time_interval = nb_linspace(t_start, t_end, int(round(t_length / self.t_sampling)))

        for pixelID in pixelsIDs:
            pID = (pixelID[0], pixelID[1])

            x_p = pID[0] * self.x_pixel_size+consts.tpcBorders[0][0] + self.x_pixel_size / 2
            y_p = pID[1] * self.y_pixel_size+consts.tpcBorders[1][0] + self.y_pixel_size / 2

            z_start, z_end = self.z_interval(start, end, x_p, y_p,
                                             3*np.sqrt(self.x_pixel_size**2 + self.y_pixel_size**2))

            z_range = nb_linspace(z_start, z_end, ceil(abs(z_end-z_start)/z_sampling)+1)

            if z_range.size <= 1:
                continue

            signal = np.zeros_like(time_interval)

            for z in z_range:

                xv, yv, zv = self.slice_coordinates(start, direction, z, track[self.col["tranDiff"]] * 5, self._slice_size)

                if ze - endcap_size <= z <= ze + endcap_size or zs - endcap_size <= z <= zs + endcap_size:
                    position = np.array([xv, yv, zv])
                    weights = trackCharge.rho(position).ravel() * (xv[0][1][0]-xv[0][0][0]) * (yv[1][0][0]-yv[0][0][0])
                else:
                    weights = weights_bulk


                t0 = (z - consts.tpcBorders[2][0]) / consts.vdrift

                current_response = self.current_response(time_interval, t0=t0)
                signals = self.slice_signal(x_p, y_p, weights, xv, yv, current_response)
                signal += np.sum(signals, axis=0) * (z_range[1]-z_range[0])


            if not signal.any():
                continue

            pixel_signal = PixelSignal(pID, int(track[self.col["trackID"]]), signal, (t_start, t_end))

            if pID in self.active_pixels:
                self.active_pixels[pID].append(pixel_signal)
            else:
                self.active_pixels[pID] = [pixel_signal]

    @staticmethod
    @nb.jit(fastmath=True)
    def slice_coordinates(start, direction, z, padding, slice_size):
        l = (z - start[2]) / direction[2]
        xl = start[0] + l * direction[0]
        yl = start[1] + l * direction[1]
        xx = np.linspace(xl - padding, xl + padding, slice_size)
        yy = np.linspace(yl - padding, yl + padding, slice_size)
        xv, yv, zv = np.meshgrid(xx, yy, np.array([z]))

        return xv, yv, zv

    @staticmethod
    @nb.njit(fastmath=True)
    def slice_signal(x_p, y_p, weights, xv, yv, current_response):
        distances = np.exp(-10*np.sqrt((xv - x_p)*(xv - x_p) + (yv - y_p)*(yv - y_p)))
        weights_attenuated = weights * distances.ravel()
        signals = np.outer(weights_attenuated, current_response)

        return signals

    def get_pixels(self, track):
        s = (track[self.col["x_start"]], track[self.col["y_start"]])
        e = (track[self.col["x_end"]], track[self.col["y_end"]])

        start_pixel = (int(round((s[0]-consts.tpcBorders[0][0]) // self.x_pixel_size)),
                       int(round((s[1]-consts.tpcBorders[1][0]) // self.y_pixel_size)))

        end_pixel = (int(round((e[0]-consts.tpcBorders[0][0]) // self.x_pixel_size)),
                     int(round((e[1]-consts.tpcBorders[1][0]) // self.y_pixel_size)))

        active_pixels = skimage.draw.line(start_pixel[0], start_pixel[1],
                                          end_pixel[0], end_pixel[1])

        xx, yy = active_pixels
        involvedPixels = []

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
            for ne in neighbors:
                if ne not in involvedPixels:
                    involvedPixels.append(ne)

            for ne in nneighbors:
                if ne not in involvedPixels:
                    involvedPixels.append(ne)

        return np.array(involvedPixels)

    def pixel_response(self, pixelID):
        pixelSignals = self.active_pixels[pixelID]
        current = np.zeros_like(self.anode_t)

        for signal in pixelSignals:
            current[(self.anode_t >= signal.time_interval[0]) & (self.anode_t <= signal.time_interval[1])] += signal.current

        return current

    def pixel_from_coordinates(self, x, y):
        x_pixel = nb_linspace(consts.tpcBorders[0][0], consts.tpcBorders[0][1], self.n_pixels)
        y_pixel = nb_linspace(consts.tpcBorders[1][0], consts.tpcBorders[1][1], self.n_pixels)
        return np.digitize(x, x_pixel), np.digitize(y, y_pixel)
