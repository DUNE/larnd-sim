#!/usr/bin/env python
"""
Detector simulation module
"""

import numpy as np
import scipy.stats
import numba as nb

from math import pi, sqrt, ceil
from scipy.special import erf
from tqdm import tqdm_notebook as progress_bar
from skimage.draw import line

from . import consts
from . import drifting
from . import quenching
from . import TPC



def getPixels(track, col):
    """
    Array of Impacted Pixel ID
    """
    s = (track[col["x_start"]], track[col["y_start"]])
    e = (track[col["x_end"]], track[col["y_end"]])
    
    start_pixel = (int((s[0]-consts.tpcBorders[0][0]) // TPC.x_pixel_size),
                   int((s[1]-consts.tpcBorders[1][0]) // TPC.y_pixel_size))
        
    end_pixel = (int((e[0]-consts.tpcBorders[0][0]) // TPC.x_pixel_size),
                 int((e[1]-consts.tpcBorders[1][0]) // TPC.y_pixel_size))
    
    activePixels = line(start_pixel[0], start_pixel[1],
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


def getZInterval(track, cols, pixelIDs):
    """Here we calculate the interval in Z for the pixel pID
        using the impact factor"""

    zIntervals = np.zeros_like(pixelIDs, dtype = float)
    
    xs, xe = track[cols['x_start']], track[cols['x_end']]
    ys, ye = track[cols['y_start']], track[cols['y_end']]
    zs, ze = track[cols['z_start']], track[cols['z_end']]

    m = (ye - ys) / (xe - xs)
    q = (xe * ys - xs * ye) / (xe - xs)
    a, b, c = m, -1, q
    
    length = np.sqrt((xe - xs)*(xe - xs) + (ye - ys)*(ye - ys) + (ze - zs)*(ze - zs))
    trackDir = (xe-xs)/length, (ye-ys)/length, (ze-zs)/length

    for ind, pID in enumerate(pixelIDs):
    
        x_p = pID[0]*TPC.x_pixel_size+consts.tpcBorders[0][0] + TPC.x_pixel_size/2
        y_p = pID[1]*TPC.y_pixel_size+consts.tpcBorders[1][0] + TPC.y_pixel_size/2        
    
        x_poca = (b*(b*x_p-a*y_p) - a*c)/(a*a+b*b)

        doca = np.abs(a*x_p+b*y_p+c)/np.sqrt(a*a+b*b)
        tolerance = 1.5*np.sqrt(TPC.x_pixel_size**2 + TPC.y_pixel_size**2)
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


        zIntervals[ind] = (minusDeltaZ, plusDeltaZ)

    return zIntervals


def getPixelSignal(track, cols, pixelIDs, zIntervals, weights):

    start = np.array([track[cols['x_start']], track[cols['y_start']], track[cols['z_start']]])
    end = np.array([track[cols['x_end']], track[cols['y_end']], track[cols['z_end']]])
    direction = (end-start)/ np.linalg.norm(end-start)
    zs = start[2]
    ze = end[2]
    
    t_start = (track[cols['t_start']]-20) // TPC.t_sampling * TPC.t_sampling
    t_end = (track[cols['t_end']]+20) // TPC.t_sampling * TPC.t_sampling
    time_interval = np.linspace(t_start, t_end, round((t_end-t_start)/TPC.t_sampling))


    endcap_size = 3*track[cols['longDiff']]*100
    
    pixel_signal = []
    
    for ind, pixelID in enumerate(pixelIDs):

        signal = np.zeros_like(time_interval)    
        x_p = pixelID[0] * TPC.x_pixel_size+consts.tpcBorders[0][0] + TPC.x_pixel_size / 2
        y_p = pixelID[1] * TPC.y_pixel_size+consts.tpcBorders[1][0] + TPC.y_pixel_size / 2

        z_start, z_end = zIntervals[ind]
        z_sampling = TPC.t_sampling * consts.vdrift
        z_range = np.linspace(z_start, z_end, ceil((z_end-z_start)/z_sampling))

        if z_range.size <= 1:
            pixel_signal.append(signal)
            continue

        for z in z_range:
            weight = weights[(ze+zs)/2] #TODO: bulk
            if z >= ze - endcap_size or z <= zs + endcap_size:
                weight = weights[z]

            signals = _getSliceSignal(x_p, y_p, z, weight, time_interval, start, direction)
            signal += np.sum(signals, axis=0) \
                * (TPC.x_pixel_size*4/9.) * (TPC.y_pixel_size*4/9.) * (z_range[1]-z_range[0]) #TODO TPC Nums


        pixel_signal.append(signal)


    return pixel_signal


def _getSliceSignal(x_p, y_p, z, weights, time_interval, start, direction):
    t0 = (z - consts.tpcBorders[2][0]) / consts.vdrift
    signals = np.outer(weights, currentResponse(time_interval, t0=t0))

    l = (z - start[2]) / direction[2]
    xl = start[0] + l * direction[0]
    yl = start[1] + l * direction[1]
    xx = np.linspace(xl - TPC.x_pixel_size * 2, xl + TPC.x_pixel_size * 2, 10)
    yy = np.linspace(yl - TPC.y_pixel_size * 2, yl + TPC.y_pixel_size * 2, 10)
    xv, yv, zv = np.meshgrid(xx, yy, z)

    distances = np.sqrt((xv - x_p)*(xv - x_p) + (yv - y_p)*(yv - y_p))
    
    signals *= distanceAttenuation(distances.ravel(), time_interval, t0=t0)
    
    return signals

def currentResponse(t, A=1, B=5, t0=0):
    """Current response parametrization"""
    result = np.heaviside((-t.T + t0).T, 0.5) * A * np.exp((t.T - t0).T / B)
    result = np.nan_to_num(result)
    
    return result

@nb.njit
def sigmoid(t, t0, t_rise=1):
    """Sigmoid function for FEE response"""
    result = 1 / (1 + np.exp(-(t-t0)/t_rise))
    return result

@nb.njit
def distanceAttenuation(distances, t, B=5, t0=0):
    """Attenuation of the signal"""
    return np.exp(np.outer(distances, ((t.T-t0).T) / B))


def getPixelFromCoordinates(x, y):
        x_pixel = np.linspace(consts.tpcBorders[0][0], consts.tpcBorders[0][1], TPC.n_pixels)
        y_pixel = np.linspace(consts.tpcBorders[1][0], consts.tpcBorders[1][1], TPC.n_pixels)
        return np.digitize(x, x_pixel), np.digitize(y, y_pixel)
