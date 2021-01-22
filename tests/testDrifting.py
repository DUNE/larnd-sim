#!/usr/bin/env python

import random
import numpy as np
import numba as nb
import pytest

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from larndsim import detsim
from larndsim import drifting
from larndsim import consts
from larndsim import indeces as i

from math import ceil

class TestDrifting:
    tracks = np.zeros((10, 29))
    tracks[:, i.z] = np.random.uniform(-15, 15, 10)
    tracks[:, i.n_electrons] = np.random.uniform(1e6, 1e7, 10)

    def test_lifetime(self):

        pixel_plane = -1
        for itrk in range(self.tracks.shape[0]):
            track = self.tracks[itrk]

            for ip,plane in enumerate(consts.module_borders):
                if plane[0][0] < track[i.x] < plane[0][1] and plane[1][0] < track[i.y] < plane[1][1] and plane[2][0] < track[i.z] < plane[2][1]:
                    pixel_plane = ip
                    break

            track[i.pixel_plane] = pixel_plane

        zAnode = consts.module_borders[pixel_plane][2][0]

        driftDistance = np.abs(self.tracks[:, i.z] - zAnode)
        driftTime = driftDistance / consts.vdrift

        lifetime = np.exp(-driftTime / consts.lifetime)

        tracks = self.tracks
        electronsAtAnode = tracks[:, i.n_electrons] * lifetime

        TPB = 128
        BPG = ceil(tracks.shape[0] / TPB)
        drifting.drift[BPG,TPB](tracks)

        assert tracks[:, i.n_electrons] == pytest.approx(electronsAtAnode)
