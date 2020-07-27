#!/usr/bin/env python
"""
Detector simulation module
"""

import numpy as np
import scipy.stats
import numba as nb

from math import pi, sqrt, ceil
from spycial import erf
from tqdm import tqdm_notebook as progress_bar
import skimage.draw

from . import consts
from . import drifting
from . import quenching

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
        sliceSize (int): number of points of the slice for each axis
        activePixels (dict): dictionary of active pixels
    """
    def __init__(self, col, n_pixels=50, t_sampling=0.1):

        self.n_pixels = n_pixels
        self.col = col

        t_length = consts.timeInterval[1] - consts.timeInterval[0]

        self.t_sampling = t_sampling
        self.anode_t = np.linspace(consts.timeInterval[0], consts.timeInterval[1], round(t_length/self.t_sampling))

        self.x_pixel_size = (consts.tpcBorders[0][1]-consts.tpcBorders[0][0]) / n_pixels
        self.y_pixel_size = (consts.tpcBorders[1][1]-consts.tpcBorders[1][0]) / n_pixels
        self.sliceSize = 10
        self.activePixels = {}

    @staticmethod
    @nb.njit(fastmath=True)
    def currentResponse(t, A=1, B=5, t0=0):
        """Current response parametrization"""
        result = A * np.exp((t.T - t0).T / B)
        result[t > t0] = 0
        return result

    @staticmethod
    def sigmoid(t, t0, t_rise=1):
        """Sigmoid function for FEE response"""
        result = 1 / (1 + np.exp(-(t-t0)/t_rise))
        return result

    @staticmethod
    @nb.njit(fastmath=True)
    def distanceAttenuation(distances, t, B=5, t0=0):
        """Attenuation of the signal"""
        return np.exp(np.outer(distances, ((t.T-t0).T) / B))

    def getZInterval(self, track, pID):
        """Here we calculate the interval in Z for the pixel pID
        using the impact factor"""
        xs, xe = track[self.col["x_start"]], track[self.col["x_end"]]
        ys, ye = track[self.col["y_start"]], track[self.col["y_end"]]
        zs, ze = track[self.col["z_start"]], track[self.col["z_end"]]
        length = np.sqrt((xe - xs)*(xe - xs) + (ye - ys)*(ye - ys) + (ze - zs)*(ze - zs))
        trackDir = (xe-xs)/length, (ye-ys)/length, (ze-zs)/length

        x_p = pID[0]*self.x_pixel_size+consts.tpcBorders[0][0] + self.x_pixel_size/2
        y_p = pID[1]*self.y_pixel_size+consts.tpcBorders[1][0] + self.y_pixel_size/2

        m = (ye - ys) / (xe - xs)
        q = (xe * ys - xs * ye) / (xe - xs)

        a, b, c = m, -1, q

        x_poca = (b*(b*x_p-a*y_p) - a*c)/(a*a+b*b)

        doca = np.abs(a*x_p+b*y_p+c)/np.sqrt(a*a+b*b)
        tolerance = 1.5*np.sqrt(self.x_pixel_size**2 + self.y_pixel_size**2)
        plusDeltaZ, minusDeltaZ = 0, 0

        if tolerance > doca:
            length2D = np.sqrt((xe-xs)**2 + (ye-ys)**2)
            dir2D = (xe-xs)/length2D, (ye-ys)/length2D
            deltaL2D = np.sqrt(tolerance**2 - doca**2) # length along the track in 2D

            x_plusDeltaL = x_poca + deltaL2D*dir2D[0] # x coordinates of the tolerance range
            x_minusDeltaL = x_poca - deltaL2D*dir2D[0]

            plusDeltaL = (x_plusDeltaL - xs)/trackDir[0] # length along the track in 3D
            minusDeltaL = (x_minusDeltaL - xs)/trackDir[0] # of the tolerance range

            plusDeltaZ = zs + trackDir[2] * plusDeltaL # z coordinates of the
            minusDeltaZ = zs + trackDir[2] * minusDeltaL # tolerance range

        return min(minusDeltaZ, plusDeltaZ), max(minusDeltaZ, plusDeltaZ)

    def calculateCurrent(self, track):
        pixelsIDs = self.getPixels(track)

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

        endcap_size = 3 * sigmas[2]
        x = np.linspace((xe + xs) / 2 - track[self.col["tranDiff"]] * 5,
                        (xe + xs) / 2 + track[self.col["tranDiff"]] * 5,
                        self.sliceSize)
        y = np.linspace((ye + ys) / 2 - track[self.col["tranDiff"]] * 5,
                        (ye + ys) / 2 + track[self.col["tranDiff"]] * 5,
                        self.sliceSize)
        z = (ze + zs) / 2

        z_sampling = self.t_sampling * consts.vdrift
        xv, yv, zv = np.meshgrid(x, y, z)
        weights = trackCharge.rho(np.array([xv, yv, zv]))
        weights_bulk = weights.ravel() * (x[1]-x[0]) * (y[1]-y[0])
        t_start = (track[self.col["t_start"]]-20) // self.t_sampling * self.t_sampling
        t_end = (track[self.col["t_end"]]+20) // self.t_sampling * self.t_sampling
        t_length = t_end-t_start
        time_interval = np.linspace(t_start, t_end, round(t_length/self.t_sampling))

        weights_endcap = {}
        positions_endcap = {}
        zRanges = {}
        for pixelID in pixelsIDs:
            pID = (pixelID[0], pixelID[1])
            z_start, z_end = self.getZInterval(track, pID)

            z_range = np.linspace(z_start, z_end, ceil((z_end-z_start)/z_sampling)+1)
            zRanges[pID] = z_range
            z_endcaps_range = z_range[(z_range >= ze - endcap_size) | (z_range <= zs + endcap_size)]
            for z in z_endcaps_range:
                xv, yv, zv = self._getSliceCoordinates(start, direction, z, track[self.col["tranDiff"]] * 5)
                positions_endcap[z] = np.array([xv, yv, zv])
                weights_endcap[z] = trackCharge.rho(positions_endcap[z]).ravel() * (x[1]-x[0]) * (y[1]-y[0])


        for pixelID in progress_bar(pixelsIDs, desc="Calculating pixel response..."):
            pID = (pixelID[0], pixelID[1])
            signal = np.zeros_like(time_interval)

            x_p = pID[0] * self.x_pixel_size+consts.tpcBorders[0][0] + self.x_pixel_size / 2
            y_p = pID[1] * self.y_pixel_size+consts.tpcBorders[1][0] + self.y_pixel_size / 2
            z_range = zRanges[pID]

            if z_range.size <= 1:
                continue

            for z in z_range:
                if z >= ze - endcap_size or z <= zs + endcap_size:
                    weights = weights_endcap[z]
                    positions = positions_endcap[z]
                else:
                    positions = self._getSliceCoordinates(start, direction, z, track[self.col["tranDiff"]] * 5)
                    weights = weights_bulk

                xv, yv, zv = positions

                signals = self._getSliceSignal(x_p, y_p, z, weights, xv, yv, time_interval)
                signal += np.sum(signals, axis=0) * (z_range[1]-z_range[0])

            if not signal.any():
                continue

            pixelSignal = PixelSignal(pID, int(track[self.col["trackID"]]), signal, (t_start, t_end))

            if pID in self.activePixels:
                self.activePixels[pID].append(pixelSignal)
            else:
                self.activePixels[pID] = [pixelSignal]

    def _getSliceCoordinates(self, startVector, direction, z, padding):
        l = (z - startVector[2]) / direction[2]
        xl = startVector[0] + l * direction[0]
        yl = startVector[1] + l * direction[1]
        xx = np.linspace(xl - padding, xl + padding, self.sliceSize)
        yy = np.linspace(yl - padding, yl + padding, self.sliceSize)
        xv, yv, zv = np.meshgrid(xx, yy, z)

        return xv, yv, zv

    def _getSliceSignal(self, x_p, y_p, z, weights, xv, yv, time_interval):
        t0 = (z - consts.tpcBorders[2][0]) / consts.vdrift
        signals = np.outer(weights, self.currentResponse(time_interval, t0=t0))
        # distances = np.sqrt((xv - x_p)*(xv - x_p) + (yv - y_p)*(yv - y_p))
        # signals *= self.distanceAttenuation(distances.ravel(), time_interval, t0=t0)

        return signals

    def getPixels(self, track):
        s = (track[self.col["x_start"]], track[self.col["y_start"]])
        e = (track[self.col["x_end"]], track[self.col["y_end"]])

        start_pixel = (round((s[0]-consts.tpcBorders[0][0]) // self.x_pixel_size),
                       round((s[1]-consts.tpcBorders[1][0]) // self.y_pixel_size))

        end_pixel = (round((e[0]-consts.tpcBorders[0][0]) // self.x_pixel_size),
                     round((e[1]-consts.tpcBorders[1][0]) // self.y_pixel_size))

        activePixels = skimage.draw.line(start_pixel[0], start_pixel[1],
                                         end_pixel[0], end_pixel[1])

        xx, yy = activePixels
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

    def getPixelResponse(self, pixelID):
        pixelSignals = self.activePixels[pixelID]
        current = np.zeros_like(self.anode_t)

        for signal in pixelSignals:
            current[(self.anode_t >= signal.time_interval[0]) & (self.anode_t <= signal.time_interval[1])] += signal.current

        return current

    def getPixelFromCoordinates(self, x, y):
        x_pixel = np.linspace(consts.tpcBorders[0][0], consts.tpcBorders[0][1], self.n_pixels)
        y_pixel = np.linspace(consts.tpcBorders[1][0], consts.tpcBorders[1][1], self.n_pixels)
        return np.digitize(x, x_pixel), np.digitize(y, y_pixel)
