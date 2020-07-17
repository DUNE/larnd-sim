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

class TestDrifting:
    tracks = np.zeros((100, 9))
    tracks[:, 0] = np.random.uniform(-150, 150, 100)
    tracks[:, 1] = np.random.uniform(1e6, 1e7, 100)

    col = nb.typed.Dict()
    col["z"] = 0
    col["NElectrons"] = 1
    col["z_start"] = 2
    col["z_end"] = 3
    col["t_start"] = 4
    col["t"] = 5
    col["t_end"] = 6
    col["tranDiff"] = 7
    col["longDiff"] = 8

    def test_lifetime(self):
        zAnode = consts.tpcZStart
        driftDistance = np.abs(self.tracks[:, self.col["z"]] - zAnode)
        driftTime = driftDistance / consts.vdrift

        lifetime = np.exp(-driftTime / consts.lifetime)

        tracks = self.tracks
        electronsAtAnode = tracks[:, self.col["NElectrons"]] * lifetime

        drifting.Drift(tracks, self.col)

        assert tracks[:, self.col["NElectrons"]] == pytest.approx(electronsAtAnode)
