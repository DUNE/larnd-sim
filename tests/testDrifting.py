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
    tracks = np.zeros((100, 29))
    tracks[:, i.z] = np.random.uniform(-150, 150, 100)
    tracks[:, i.n_electrons] = np.random.uniform(1e6, 1e7, 100)

    def test_lifetime(self):
        zAnode = consts.tpc_borders[2][0]
        driftDistance = np.abs(self.tracks[:, i.z] - zAnode)
        driftTime = driftDistance / consts.vdrift

        lifetime = np.exp(-driftTime / consts.lifetime)

        tracks = self.tracks
        electronsAtAnode = tracks[:, i.n_electrons] * lifetime

        TPB = 128
        BPG = ceil(tracks.shape[0] / TPB)
        drifting.drift[TPB,BPG](tracks)

        assert tracks[:, i.n_electrons] == pytest.approx(electronsAtAnode)
