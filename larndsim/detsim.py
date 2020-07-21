#!/usr/bin/env python
"""
Detector simulation module
"""

import numpy as np
import scipy.stats

from math import pi, sqrt, ceil
from scipy.special import erf
from tqdm import tqdm_notebook as progress_bar
import skimage.draw

from . import consts
from . import drifting
from . import quenching


class TrackCharge:
    """Track charge deposition"""

    def __init__(self, Q, start, end, sigmas):
        segment = end - start
        self.start = start
        self.end = end
        self.segment = segment

        self.Deltar = np.linalg.norm(segment)
        self.sigmas = np.array(sigmas)
        self.Q = Q
        self.factor = Q/self.Deltar/(self.sigmas.prod()*sqrt(8*pi*pi*pi))
        self.a = ((segment/self.Deltar)**2 / (2*self.sigmas**2)).sum()

    def __repr__(self):
        instanceDescription = "<%s instance at %s>\nQ %f\nxyz (%f, %f, %f), (%f, %f, %f)\nsigmas (%f, %f, %f)" % \
                               (self.__class__.__name__, id(self),
                                self.Q, *self.start, *self.end, *self.sigmas)

        return instanceDescription

    def _b(self, position):
        b = -((position.T - self.start) / self.sigmas**2 * self.segment / self.Deltar).T

        return b.sum(axis=0)

    def rho(self, x, y, z):
        """Charge distribution in space"""
        position = np.array([x,y,z])
        b = self._b(position)
        sqrt_a_2 = 2*np.sqrt(self.a)

        delta = ((position.T - self.start)**2 / (2*self.sigmas**2)).T

        expo = np.exp(b*b/(4*self.a) - delta.sum(axis=0))

        integral = (sqrt(pi)
                    * (-erf(b/sqrt_a_2) + erf((b + 2*self.a*self.Deltar)/sqrt_a_2))
                    / sqrt_a_2)

        return integral*expo*self.factor


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
        n_pixels (int): number of pixels that tile the anode
        t_sampling (float): time sampling
        **kwargs: dictionary containing the tensor indeces

    Attributes:
        n_pixels (int): number of pixels per axis that tile the anode
        x_start (float): starting x coordinate of the TPC
        x_end (float): ending x coordinate of the TPC
    """
    def __init__(self, n_pixels=50, t_sampling=0.1, **kwargs):

        self.n_pixels = n_pixels

        self.x_start = consts.tpcBorders[0][0]
        self.x_end = consts.tpcBorders[0][1]
        x_length = self.x_end - self.x_start

        self.y_start = consts.tpcBorders[1][0]
        self.y_end = consts.tpcBorders[1][1]
        y_length = self.y_end - self.y_start

        self.t_start = consts.timeInterval[0]
        self.t_end = consts.timeInterval[1]
        t_length = self.t_end - self.t_start

        self.x_sampling = x_length/n_pixels/4
        self.y_sampling = y_length/n_pixels/4
        self.t_sampling = t_sampling

        self.anode_x = np.linspace(self.x_start, self.x_end, int(x_length/self.x_sampling))
        self.anode_y = np.linspace(self.y_start, self.y_end, int(y_length/self.y_sampling))
        self.anode_t = np.linspace(self.t_start, self.t_end, int(t_length/self.t_sampling))

        self.tpc = np.zeros((int(x_length / self.x_sampling),
                             int(y_length / self.y_sampling),
                             int(t_length / t_sampling)))

        self.ixStart = kwargs['x_start']
        self.ixEnd = kwargs['x_end']
        self.iyStart = kwargs['y_start']
        self.iyEnd = kwargs['y_end']
        self.izStart = kwargs['z_start']
        self.izEnd = kwargs['z_end']
        self.itStart = kwargs['t_start']
        self.itEnd = kwargs['t_end']
        self.iNElectrons = kwargs['NElectrons']
        self.iLongDiff = kwargs['longDiff']
        self.iTranDiff = kwargs['tranDiff']
        self.iTrackID = kwargs['trackID']

        self.x_pixel_size = x_length / n_pixels
        self.y_pixel_size = y_length / n_pixels

        self.x_pixel_range = np.linspace(0, self.x_pixel_size, int(self.x_pixel_size/self.x_sampling))
        self.y_pixel_range = np.linspace(0, self.y_pixel_size, int(self.y_pixel_size/self.y_sampling))

        self.activePixels = {}

    @staticmethod
    def currentResponse(t, A=1, B=5, t0=0):
        """Current response parametrization"""
        result = np.heaviside((-t.T + t0).T, 0.5) * A * np.exp((t.T - t0).T / B)
        result = np.nan_to_num(result)

        return result

    @staticmethod
    def sigmoid(t, t0, t_rise=1):
        """Sigmoid function for FEE response"""
        result = 1 / (1 + np.exp(-(t-t0)/t_rise))
        return result

    @staticmethod
    def distanceAttenuation(distances, t, B=5, t0=0):
        """Attenuation of the signal"""
        return np.exp(np.outer(distances, ((t.T-t0).T) / B))

    def getZInterval(self, track, pID):
        """Here we calculate the interval in Z for the pixel pID
        using the impact factor"""
        xs, xe = track[self.ixStart], track[self.ixEnd]
        ys, ye = track[self.iyStart], track[self.iyEnd]
        zs, ze = track[self.izStart], track[self.izEnd]
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

            plusDeltaZ = min(zs + trackDir[2] * plusDeltaL, ze) # z coordinates of the
            minusDeltaZ = max(zs, zs + trackDir[2] * minusDeltaL) # tolerance range

        return minusDeltaZ, plusDeltaZ

    def calculateCurrent(self, track):
        pixelsIDs = self.getPixels(track)

        xs, xe = track[self.ixStart], track[self.ixEnd]
        ys, ye = track[self.iyStart], track[self.iyEnd]
        zs, ze = track[self.izStart], track[self.izEnd]

        start = np.array([xs, ys, zs])
        end = np.array([xe, ye, ze])
        segment = end - start

        length = np.linalg.norm(segment)

        direction = segment/length

        sigmas = np.array([track[self.iTranDiff].item()*100,
                           track[self.iTranDiff].item()*100,
                           track[self.iLongDiff].item()*100])

        trackCharge = TrackCharge(track[self.iNElectrons],
                                  start,
                                  end,
                                  sigmas=sigmas)

        endcap_size = 3 * track[self.iLongDiff].item() * 100
        x = np.linspace((xe + xs) / 2 - self.x_pixel_size * 2,
                        (xe + xs) / 2 + self.x_pixel_size * 2,
                        10)
        y = np.linspace((ye + ys) / 2 - self.y_pixel_size * 2,
                        (ye + ys) / 2 + self.y_pixel_size * 2,
                        10)
        z = (ze + zs) / 2

        z_sampling = self.t_sampling * consts.vdrift
        xv, yv, zv = np.meshgrid(x, y, z)
        weights = trackCharge.rho(xv, yv, zv)
        weights_bulk = weights.ravel()

        t_start = (track[self.itStart]-20) // self.t_sampling * self.t_sampling
        t_end = (track[self.itEnd]+20) // self.t_sampling * self.t_sampling
        t_length = t_end-t_start
        time_interval = np.linspace(t_start, t_end, round(t_length/self.t_sampling))

        weights_endcap = {}
        zIntervals = {}
        for pixelID in pixelsIDs:
            pID = (pixelID[0], pixelID[1])
            z_start, z_end = self.getZInterval(track, pID)
            zIntervals[pID] = (z_start, z_end)

            z_range = np.linspace(z_start, z_end, ceil((z_end-z_start)/z_sampling))
            z_endcaps_range = z_range[(z_range >= ze - endcap_size) | (z_range <= zs + endcap_size)]
            for z in z_endcaps_range:
                xv, yv, zv = self._getSliceCoordinates(start, direction, z)
                weights_endcap[z] = trackCharge.rho(xv, yv, zv)


        for pixelID in progress_bar(pixelsIDs, desc="Calculating pixel response..."):
            pID = (pixelID[0], pixelID[1])
            z_start, z_end = zIntervals[pID]

            signal = np.zeros_like(time_interval)

            x_p = pID[0] * self.x_pixel_size+consts.tpcBorders[0][0] + self.x_pixel_size / 2
            y_p = pID[1] * self.y_pixel_size+consts.tpcBorders[1][0] + self.y_pixel_size / 2

            z_range = np.linspace(z_start, z_end, ceil((z_end-z_start)/z_sampling))

            if z_range.size <= 1:
                continue

            for z in z_range:
                xv, yv, zv = self._getSliceCoordinates(start, direction, z)

                weights = weights_bulk
                if z >= ze - endcap_size or z <= zs + endcap_size:
                    weights = weights_endcap[z]

                signals = self._getSliceSignal(x_p, y_p, z, weights, xv, yv, time_interval)
                signal += np.sum(signals, axis=0) \
                          * (x[1]-x[0]) * (y[1]-y[0]) * (z_range[1]-z_range[0])

            pixelSignal = PixelSignal(pID, int(track[self.iTrackID]), signal, (t_start, t_end))
            if pID in self.activePixels:
                self.activePixels[pID].append(pixelSignal)
            else:
                self.activePixels[pID] = [pixelSignal]

    def _getSliceCoordinates(self, startVector, direction, z):
        l = (z - startVector[2]) / direction[2]
        xl = startVector[0] + l * direction[0]
        yl = startVector[1] + l * direction[1]
        xx = np.linspace(xl - self.x_pixel_size * 2, xl + self.x_pixel_size * 2, 10)
        yy = np.linspace(yl - self.y_pixel_size * 2, yl + self.y_pixel_size * 2, 10)
        xv, yv, zv = np.meshgrid(xx, yy, z)

        return xv, yv, zv

    def _getSliceSignal(self, x_p, y_p, z, weights, xv, yv, time_interval):
        t0 = (z - consts.tpcBorders[2][0]) / consts.vdrift
        signals = np.outer(weights, self.currentResponse(time_interval, t0=t0))
        distances = np.sqrt((xv - x_p)*(xv - x_p) + (yv - y_p)*(yv - y_p))

        signals *= self.distanceAttenuation(distances.ravel(), time_interval, t0=t0)

        return signals

    def getPixels(self, track):
        s = (track[self.ixStart], track[self.iyStart])
        e = (track[self.ixEnd], track[self.iyEnd])

        start_pixel = (int((s[0]-consts.tpcBorders[0][0]) // self.x_pixel_size),
                       int((s[1]-consts.tpcBorders[1][0]) // self.y_pixel_size))

        end_pixel = (int((e[0]-consts.tpcBorders[0][0]) // self.x_pixel_size),
                     int((e[1]-consts.tpcBorders[1][0]) // self.y_pixel_size))

        activePixels = skimage.draw.line(start_pixel[0], start_pixel[1],
                                         end_pixel[0], end_pixel[1])

        xx, yy = activePixels
        involvedPixels = []

        for x, y in zip(xx, yy):
            neighbors = (x, y), \
                        (x, y + 1), (x + 1, y), \
                        (x, y - 1), (x - 1, y), \
                        (x + 1, y + 1), (x - 1, y - 1), \
                        (x + 1, y - 1), (x - 1, y + 1)
            for ne in neighbors:
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
        x_pixel = np.linspace(self.x_start, self.x_end, self.n_pixels)
        y_pixel = np.linspace(self.y_start, self.y_end, self.n_pixels)
        return np.digitize(x, x_pixel), np.digitize(y, y_pixel)
