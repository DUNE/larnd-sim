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
    'timeInterval': (0, 300),
    'longDiff': 6.2e-6, # u.cm * u.cm / u.us,
    'tranDiff': 16.3e-6 # u.cm
}

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



class TPC:

    def __init__(self, x_sampling, y_sampling, t_sampling, n_pixels=20, **kwargs):

        self.x_sampling = x_sampling
        self.y_sampling = y_sampling
        self.t_sampling = t_sampling
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

        self.anode_x = np.linspace(self.x_start, self.x_end, int(x_length/x_sampling))
        self.anode_y = np.linspace(self.y_start, self.y_end, int(y_length/y_sampling))
        self.anode_t = np.linspace(self.t_start, self.t_end, int(t_length/t_sampling))

        self.tpc = np.zeros((int(x_length / x_sampling),
                             int(y_length / y_sampling),
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

        x_range = np.linspace(0, self.x_pixel_size, int(self.x_pixel_size/x_sampling))
        y_range = np.linspace(0, self.y_pixel_size, int(self.y_pixel_size/y_sampling))

        x_r, y_r, t_r = np.meshgrid(x_range, y_range, self.anode_t)
        self.inducedCurrent = self.currentResponse(x_r, y_r, t_r, t0=t_length/2)

    @staticmethod
    def currentResponse(x, y, t, A=1, B=5, t0=0):
        distance = 1#np.sqrt((x-pixel_size/2)**2+(y-pixel_size/2)**2)
        result = np.heaviside(-t + t0, 0.5) * A * np.exp((1 + distance) * (t - t0) / B)
        result = np.nan_to_num(result)

        return result

    def drawTrack(self, track):
        t = track

        x1, y1, t1 = t[self.ixStart], t[self.iyStart], t[self.itStart]
        x2, y2, t2 = t[self.ixEnd], t[self.iyEnd], t[self.itEnd]

        x_size = math.ceil((x2-x1)/self.x_sampling)
        y_size = math.ceil((y2-y1)/self.y_sampling)
        t_size = math.ceil((t2-t1)/self.t_sampling)

        line = skimage.draw.line_nd((0, 0, 0),
                                    ((x2 - x1) / self.x_sampling,
                                     (y2 - y1) / self.y_sampling,
                                     (t2 - t1) / self.t_sampling),
                                    endpoint=True)

        img = np.zeros((x_size + 1, y_size + 1, t_size + 1),
                       dtype=np.float32)

        img[line] = t[self.iNElectrons]/len(line[0])
        img = np.pad(img,
                     ((math.ceil(1/self.x_sampling), math.ceil(1/self.x_sampling)),
                      (math.ceil(1/self.y_sampling), math.ceil(1/self.y_sampling)),
                      (math.ceil(1/self.t_sampling), math.ceil(1/self.t_sampling))),
                     mode='constant')

        img = scipy.ndimage.gaussian_filter(img, sigma=(t[self.iTranDiff].item()*2000,
                                                        t[self.iTranDiff].item()*2000,
                                                        t[self.iLongDiff].item()*1000))

        x_bin = np.digitize(x1, self.anode_x)
        y_bin = np.digitize(y1, self.anode_y)
        t_bin = np.digitize(t1, self.anode_t)

        self.tpc[x_bin-1:x_bin+x_size+2*math.ceil(1/self.x_sampling),
                 y_bin-1:y_bin+y_size+2*math.ceil(1/self.y_sampling),
                 t_bin-1:t_bin+t_size+2*math.ceil(1/self.t_sampling)] = img

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

        xx = np.linspace(track[self.ixStart], track[self.ixEnd], int(track[self.ixEnd]-track[self.ixStart])*100)
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
                        (x + 1, y + 1), (x - 1, y - 1)

            for ne in neighbors:
                if ne not in involvedPixels:
                    involvedPixels.append(ne)

        return involvedPixels

    def pixelResponse(self, pixel):
        x, y = pixel
        px_bin = int(self.x_pixel_size / self.x_sampling)
        py_bin = int(self.y_pixel_size / self.y_sampling)
        tpcSlice = self.tpc[x * px_bin:(x + 1) * px_bin, y * py_bin:(y + 1) * py_bin]

        conv3d = scipy.signal.fftconvolve(self.inducedCurrent, tpcSlice, mode='same')
        depCharge = tpcSlice.sum(axis=0).sum(axis=0)
        indCurrent = conv3d.sum(axis=0).sum(axis=0)

        return depCharge, indCurrent


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pickle

    tracks = pickle.load(open('tracks.p', 'rb'))
    tracks['t'] = 0
    tracks['t_start'] = 0
    tracks['t_end'] = 0

    tracks['dx'] = np.sqrt(pow(tracks['x_end']-tracks['x_start'], 2) +
                        pow(tracks['y_end']-tracks['y_start'], 2) +
                        pow(tracks['z_end']-tracks['z_start'], 2))
    tracks['x'] = (tracks['x_end']+tracks['x_start'])/2
    tracks['y'] = (tracks['y_end']+tracks['y_start'])/2
    tracks['z'] = (tracks['z_end']+tracks['z_start'])/2
    tracks['dE'] = np.abs(tracks['dE'])*1e3
    tracks['dEdx'] = tracks['dE']/tracks['dx']
    tracks['NElectrons'] = 0
    tracks['longDiff'] = 0
    tracks['tranDiff'] = 0

    indeces = {c:i for i, c, in enumerate(tracks.columns)}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    track_tensor = torch.tensor(tracks.values).to(device)

    quenching = Quenching(**indeces)
    quenchedTracks = quenching(track_tensor)

    drifting = Drifting(**indeces)
    driftedTracks = drifting(quenchedTracks)

    selectedTracks = driftedTracks[driftedTracks[:, indeces['trackID']] < 100]

    tpc = TPC(0.5, 0.5, 0.05, **indeces)
    tpc.drawTrack(selectedTracks[0])

    pixels = tpc.getPixels(driftedTracks[0])
    for p in pixels:
        depCharge, indCurrent = tpc.pixelResponse(p)

        t_sampling = 0.05
        anode_t = np.linspace(0, 300, int(300 / t_sampling))
        if depCharge.any():
            fig, ax = plt.subplots(1, 1)
            ax.plot(anode_t, depCharge*1e4)
            ax.plot(anode_t, indCurrent)
            ax.set_xlim(200, 300)