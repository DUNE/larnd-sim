#!/usr/bin/env python
"""
Detector simulation module
"""

import math
import torch
import numpy as np
import skimage.draw
import scipy.ndimage
import scipy.stats
import scipy.signal
from tqdm import tqdm_notebook as progress_bar

from shapely.geometry import MultiLineString, LineString


PHYSICAL_PARAMS = {
    'MeVToElectrons': 4.237e+04,
    'alpha': 0.847,
    'beta': 0.2061
}

TPC_PARAMS = {
    'vdrift': 0.153812, # u.cm / u.us,
    'lifetime': 10e3, # u.us,
    'tpcBorders': ((-150, 150), (-150, 150), (-150, 150)), # u.cm,
    'timeInterval': (0, 3000),
    'longDiff': 6.2e-6, # u.cm * u.cm / u.us,
    'tranDiff': 16.3e-6 # u.cm
}


def sigmoid(t, t0, t_rise=1):
    result = 1 / (1 + np.exp(-(t-t0)/t_rise))
    return result


class Quenching(torch.nn.Module):
    """
    PyTorch module which implements the quenching of the electrons
    in the TPC.
    """
    def __init__(self, **kwargs):
        super(Quenching, self).__init__()

        self.idEdx = kwargs['dEdx']
        self.idE = kwargs['dE']
        self.iNElectrons = kwargs['NElectrons']

    def forward(self, x):
        """The number of electrons ionized by the track segment is calculated
        taking into account the recombination.

        Returns":
            x: a new tensor with an additional column for the number of ionized
            electrons
        """

        add_columns = torch.nn.ZeroPad2d((0, 1, 0, 0))
        x = add_columns(x)

        recomb = torch.log(PHYSICAL_PARAMS['alpha'] + PHYSICAL_PARAMS['beta'] * x[:, self.idEdx]) / (PHYSICAL_PARAMS['beta'] * x[:, self.idEdx])
        recomb = torch.where(recomb <= 0, torch.zeros_like(recomb), recomb)
        recomb = torch.where(torch.isnan(recomb), torch.zeros_like(recomb), recomb)
        x[:, self.iNElectrons] = PHYSICAL_PARAMS['MeVToElectrons'] * x[:, self.idE] * recomb
        return x


class Drifting(torch.nn.Module):
    """
    PyTorch module which implements the propagation of the
    electrons towards the anode.
    """

    def __init__(self, **kwargs):
        super(Drifting, self).__init__()

        self.iz = kwargs['z']
        self.izStart = kwargs['z_start']
        self.izEnd = kwargs['z_end']
        self.it = kwargs['t']
        self.itStart = kwargs['t_start']
        self.itEnd = kwargs['t_end']
        self.iNElectrons = kwargs['NElectrons']
        self.iLongDiff = kwargs['longDiff']
        self.iTranDiff = kwargs['tranDiff']


    def forward(self, x):
        """The z coordinate of the track segment is set to
        the z coordinate of the anode.
        The number of electrons is corrected by the electron lifetime.
        The longitudinal and transverse diffusion factors are calculated.
        The time is set to the drift time, calculated taking into account the transverse
        diffusion factor

        Returns:
            x: a new tensor with 2 additional column, for the longitudinal
            diffusion and transverse diffusion coefficients
        """

        add_columns = torch.nn.ZeroPad2d((0, 2, 0, 0))
        x = add_columns(x)

        zStart = TPC_PARAMS['tpcBorders'][2][0]

        driftDistance = torch.abs(x[:, self.iz] - zStart)
        driftStart = torch.abs(x[:, self.izStart] - zStart)
        driftEnd = torch.abs(x[:, self.izEnd] - zStart)

        driftTime = driftDistance / TPC_PARAMS['vdrift']
        x[:, self.iz] = zStart

        lifetime = torch.exp(-driftTime / TPC_PARAMS['lifetime'])
        x[:, self.iNElectrons] = x[:, self.iNElectrons] * lifetime

        x[:, self.iLongDiff] = torch.sqrt(driftTime) * TPC_PARAMS['longDiff']
        x[:, self.iTranDiff] = torch.sqrt(driftTime) * TPC_PARAMS['tranDiff']
        x[:, self.it] += driftTime + x[:, self.iTranDiff] / TPC_PARAMS['vdrift']
        x[:, self.itStart] += driftStart + x[:, self.iTranDiff] / TPC_PARAMS['vdrift']
        x[:, self.itEnd] += driftEnd + x[:, self.iTranDiff] / TPC_PARAMS['vdrift']

        return x

class PixelSignal:
    def __init__(self, id, charge, current, time_interval):
        self.id = id
        self.charge = charge
        self.current = current
        self.time_interval = time_interval


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

        self.x_start = TPC_PARAMS['tpcBorders'][0][0]
        self.x_end = TPC_PARAMS['tpcBorders'][0][1]
        x_length = self.x_end - self.x_start

        self.y_start = TPC_PARAMS['tpcBorders'][1][0]
        self.y_end = TPC_PARAMS['tpcBorders'][1][1]
        y_length = self.y_end - self.y_start

        self.t_start = TPC_PARAMS['timeInterval'][0]
        self.t_end = TPC_PARAMS['timeInterval'][1]
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

        self.x_pixel_size = x_length / n_pixels
        self.y_pixel_size = y_length / n_pixels

        self.x_pixel_range = np.linspace(0, self.x_pixel_size, int(self.x_pixel_size/self.x_sampling))
        self.y_pixel_range = np.linspace(0, self.y_pixel_size, int(self.y_pixel_size/self.y_sampling))

        self.activePixels = {}

    @staticmethod
    def currentResponse(x, y, t, A=1, B=5, t0=0):
        distance = 1#np.sqrt((x-pixel_size/2)**2+(y-pixel_size/2)**2)
        result = np.heaviside(-t + t0, 0.5) * A * np.exp((1 + distance) * (t - t0) / B)
        result = np.nan_to_num(result)

        return result


    def depositTrackCharge(self, track):
        t = track

        # Ugly rounding procedure to avoid floating-point error FIXME
        x1, y1, t1 = round(t[self.ixStart].numpy() // self.x_sampling * self.x_sampling, 1), \
                     round(t[self.iyStart].numpy() // self.x_sampling * self.x_sampling, 1), \
                     round(t[self.itStart].numpy() // self.t_sampling * self.t_sampling, 1)
        x2, y2, t2 = round(t[self.ixEnd].numpy() // self.x_sampling * self.x_sampling, 1), \
                     round(t[self.iyEnd].numpy() // self.x_sampling * self.x_sampling, 1), \
                     round(t[self.itEnd].numpy() // self.t_sampling * self.t_sampling, 1)

        x_size = math.ceil((x2-x1)/self.x_sampling)
        y_size = math.ceil((y2-y1)/self.y_sampling)
        t_size = math.ceil((t2-t1)/self.t_sampling)

        pixelsIDs = self.getPixels(track)
        pixel_sampling = (self.x_end-self.x_start) / self.n_pixels

        firstPixel = np.min(pixelsIDs,axis=0)
        lastPixel = np.max(pixelsIDs,axis=0)

        x1_pixel, y1_pixel = firstPixel * pixel_sampling + self.x_start
        x2_pixel, y2_pixel = lastPixel * pixel_sampling + self.x_start

        padding_left = int(round((x1-x1_pixel)/self.x_sampling))
        padding_right = int(round((x2_pixel-x2)/self.x_sampling))
        padding_bottom = int(round((y1-y1_pixel)/self.y_sampling))
        padding_top = int(round((y2_pixel-y2)/self.y_sampling))
        padding_before = padding_after = math.ceil(2/self.t_sampling)

        t_start = np.digitize(t1, self.anode_t)
        t_end = np.digitize(t2, self.anode_t)

        line = skimage.draw.line_nd((0, 0, 0),
                                    ((x2 - x1) / self.x_sampling,
                                     (y2 - y1) / self.y_sampling,
                                     t_start - t_end),
                                    endpoint=True)

        img = np.zeros((x_size + 1, y_size + 1, t_size + 1),
                       dtype=np.float32)

        img[line] = t[self.iNElectrons]/len(line[0])

        img = np.pad(img,
                     ((padding_left, padding_right),
                      (padding_bottom, padding_top),
                      (padding_before, padding_after)),
                     mode='constant')

        img = scipy.ndimage.gaussian_filter(img, sigma=(t[self.iTranDiff].item()*2000,
                                                        t[self.iTranDiff].item()*2000,
                                                        t[self.iLongDiff].item()*1000))

        t_start -= padding_before
        t_end += padding_after
        pixel_size_sampling = self.x_pixel_size/self.x_sampling

        for p in progress_bar(pixelsIDs, desc="Calculating pixel response..."):
            ix1 = int((p[0]-firstPixel[0])*pixel_size_sampling)
            ix2 = int((p[0]+1-firstPixel[0])*pixel_size_sampling)
            iy1 = int((p[1]-firstPixel[1])*pixel_size_sampling)
            iy2 = int((p[1]+1-firstPixel[1])*pixel_size_sampling)

            img_slice = img[ix1:ix2, iy1:iy2]
            pixelID = (p[0], p[1])

            if img_slice.any():
                dep_charge = img_slice.sum(axis=0).sum(axis=0)

                t_range = np.linspace(0, img.shape[2], img.shape[2])
                x_r, y_r, t_r = np.meshgrid(self.x_pixel_range, self.y_pixel_range, t_range)

                response = self.currentResponse(x_r, y_r, t_r, t0=img.shape[2]/2)
                conv3d = scipy.signal.fftconvolve(response, img_slice, mode='same') * self.x_sampling * self.y_sampling * self.t_sampling

                ind_current = conv3d.sum(axis=0).sum(axis=0)
                if pixelID in self.activePixels:
                    self.activePixels[pixelID].append(PixelSignal(pixelID, dep_charge, ind_current, (t_start, t_end)))
                else:
                    self.activePixels[pixelID] = [PixelSignal(pixelID, dep_charge, ind_current, (t_start, t_end))]

        return img

    def getPixels(self, track):
        n = self.n_pixels+1

        lines = []
        binx = np.linspace(self.x_start, self.x_end, n)
        biny = np.linspace(self.y_start, self.y_end, n)

        for x in binx:
            lines.append(((x, self.y_start), (x, self.y_end)))

        for y in biny:
            lines.append(((self.x_start, y), (self.x_end, y)))

        grid = MultiLineString(lines)

        xx = np.linspace(track[self.ixStart], track[self.ixEnd], int(track[self.ixEnd]-track[self.ixStart])*1000)
        m = (track[self.iyEnd] - track[self.iyStart]) / (track[self.ixEnd] - track[self.ixStart])
        q = (track[self.ixEnd] * track[self.iyStart] - track[self.ixStart] * track[self.iyEnd]) / (track[self.ixEnd] - track[self.ixStart])
        yy = m * xx + q

        line = LineString(np.c_[xx, yy])
        means_x = []
        means_y = []
        for segment in line.difference(grid):
            x, y = segment.xy
            means_x.append(np.mean(x))
            means_y.append(np.mean(y))

        binned = scipy.stats.binned_statistic_2d(means_x, means_y, means_x, 'count', bins=[binx, biny])
        activePixels = np.nonzero(binned[0])

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
        charge = np.zeros_like(self.anode_t)
        current = np.zeros_like(self.anode_t)

        for signal in pixelSignals:
            charge[signal.time_interval[0]:signal.time_interval[1]+1] += signal.charge
            current[signal.time_interval[0]:signal.time_interval[1]+1] += signal.current

        return charge, current

    def getPixelFromCoordinates(self, x, y):
        x_pixel = np.linspace(self.x_start, self.x_end, self.n_pixels)
        y_pixel = np.linspace(self.y_start, self.y_end, self.n_pixels)
        return np.digitize(x, x_pixel), np.digitize(y, y_pixel)