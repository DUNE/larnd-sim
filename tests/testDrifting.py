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
    
    tracks = np.zeros((1, 29))
    tracks[:, i.z] = np.random.uniform(consts.module_borders[0][2][0], consts.module_borders[0][2][1], 1)
    tracks[:, i.x] = np.random.uniform(consts.module_borders[0][0][0], consts.module_borders[0][0][1], 1)
    tracks[:, i.y] = np.random.uniform(consts.module_borders[0][1][0], consts.module_borders[0][1][1], 1)
    tracks[:, i.n_electrons] = np.random.uniform(1e6, 1e7, 1)

    def test_lifetime(self):

        zAnode = consts.module_borders[0][2][0]

        driftDistance = np.abs(self.tracks[:, i.z] - zAnode)
        driftTime = driftDistance / consts.vdrift

        lifetime = np.exp(-driftTime / consts.lifetime)

        tracks = self.tracks
        electronsAtAnode = tracks[:, i.n_electrons] * lifetime

        TPB = 128
        BPG = ceil(tracks.shape[0] / TPB)
        drifting.drift[BPG,TPB](tracks)

        assert tracks[:, i.n_electrons] == pytest.approx(electronsAtAnode)
