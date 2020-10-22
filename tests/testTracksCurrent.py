#!/usr/bin/env python

import random
import numpy as np
import numba as nb
import pytest

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from math import pi, exp
from larndsim import detsim
from larndsim import drifting
from larndsim import consts
from larndsim import indeces as i

from math import ceil

class TestDrifting:
    tracks = np.zeros((1, 29))
    tracks[:, i.z_start] = np.random.uniform(consts.tpc_borders[2][0], consts.tpc_borders[2][1], 1)
    tracks[:, i.z_end] = np.random.uniform(consts.tpc_borders[2][0], consts.tpc_borders[2][1], 1)
    tracks[:, i.y_start] = np.random.uniform(consts.tpc_borders[1][0], consts.tpc_borders[1][0]+consts.y_pixel_size, 1)
    tracks[:, i.y_end] = np.random.uniform(consts.tpc_borders[1][0], consts.tpc_borders[1][0]+consts.y_pixel_size, 1)
    tracks[:, i.x_start] = np.random.uniform(consts.tpc_borders[0][0], consts.tpc_borders[0][0]+consts.x_pixel_size, 1)
    tracks[:, i.x_end] = np.random.uniform(consts.tpc_borders[0][0], consts.tpc_borders[0][0]+consts.x_pixel_size, 1)
    tracks[:, i.n_electrons] = np.random.uniform(1e6, 1e7, 1)
    tracks[:, i.tran_diff] = 1e-6
    tracks[:, i.long_diff] = 1e-6
    tracks[:, i.t_start] = tracks[:, i.z_start] - consts.tpc_borders[2][0]
    t0 = np.random.uniform(consts.time_interval[0], consts.time_interval[1], 1)

    x_pos = np.random.uniform(0, consts.x_pixel_size/2, 1)
    y_pos = np.random.uniform(0, consts.y_pixel_size/2, 1)

    def test_current_model(self):
        integral = 0

        x = self.x_pos
        y = self.y_pos

        for t in consts.time_ticks:
            integral += detsim.current_model(t, self.t0, x, y)

        assert pytest.approx(integral * consts.t_sampling, rel=0.05) == consts.e_charge

a = TestDrifting()
a.test_current_model()