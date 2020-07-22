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
import skimage.draw

from . import consts
from . import drifting
from . import quenching
from . import TPC


def TrackCharge(track, cols, zIntervals):
    
    start = np.array([track[cols['x_start']], track[cols['y_start']], track[cols['z_start']]])
    end = np.array([track[cols['x_end']], track[cols['y_end']], track[cols['z_end']]])
    segment = end - start
    Deltar = np.linalg.norm(segment)
    direction = segment/Deltar
    Q = track[cols['NElectrons']]
    sigmas=np.array([track[cols['tranDiff']]*100,
            track[cols['tranDiff']]*100,
            track[cols['longDiff']]*100])

    a = ((segment/Deltar)**2 / (2*sigmas**2)).sum()
    factor = Q/Deltar/(sigmas.prod()*sqrt(8*pi*pi*pi))
    
    weights_bulk = getChargeWeights((start[0]+end[0])/2.,
                                    (start[1]+end[1])/2.,
                                    (start[2]+end[2])/2.,
                                    a, factor,
                                    start, sigmas, direction, Deltar)

    endcap_size = 3*track[cols['longDiff']]*100
    z_endcaps = getZEndcaps(zIntervals, start[2], end[2],  endcap_size)
    weights_atz = {}
    weights_atz[(start[2]+end[2])/2] = weights_bulk
    
    
    for z in z_endcaps:
        l = (z - start[2]) / direction[2]
        xl = start[0] + l * direction[0]
        yl = start[1] + l * direction[1]
        weights_atz[z]=getChargeWeights(xl, yl, z, a, factor,
                                        start, sigmas,direction, Deltar)

    return weights_atz
    
def getChargeWeights(x,y,z,a,factor, start, sigmas, direction, Deltar):
    
    xx = np.linspace(x - TPC.x_pixel_size * 2,
                     x + TPC.x_pixel_size * 2,
                    10)
    yy = np.linspace(y - TPC.y_pixel_size * 2,
                     y + TPC.y_pixel_size * 2,
                    10)
    xv, yv, zv = np.meshgrid(xx, yy, z)
    weights = rho(xv, yv, zv, a, start,sigmas, direction, Deltar).ravel()

    return weights


def rho(x, y, z, a, start, sigmas, direction, Deltar):

    position = np.array([x,y,z])
    _b = -((position.T - start) / sigmas**2 * direction).T

    b=  _b.sum(axis=0)
    sqrt_a_2 = 2*np.sqrt(a)    
    deltaVector = ((position.T - start)**2 / (2*sigmas**2)).T
    
    expo = np.exp(b*b/(4*a) - deltaVector.sum(axis=0))
    
    integral = (sqrt(pi)
                * (-erf(b/sqrt_a_2) + erf((b + 2*a*Deltar)/sqrt_a_2))
                / sqrt_a_2)
    

    return integral*expo


def getZEndcaps(zIntervals, zs, ze, endcap_size):
    endcaps = []
    for key in zIntervals.keys():
        z_start, z_end = zIntervals[key]
        z_range = np.linspace(z_start, z_end, ceil((z_end-z_start)/TPC.z_sampling))
        z_endcaps_range = z_range[(z_range >= ze - endcap_size) | (z_range <= zs + endcap_size)]
        endcaps = endcaps + list(z_endcaps_range)

    endcaps = np.array(list(set(endcaps)))

    return endcaps
