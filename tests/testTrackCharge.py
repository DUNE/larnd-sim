#!/usr/bin/env python

import random
import numpy as np
import pytest

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from larndsim import detsim

class TestTrackCharge:
    charge = random.randint(100,1000)
    start = np.array([random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(-5, 5)], dtype=np.float32)
    end =  np.array([random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(-5, 5)], dtype=np.float32)
    sigmas = np.array([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)], dtype=np.float32)
    trackCharge = detsim.TrackCharge(charge, start, end, sigmas)

    def test_rho(self):
        x = np.linspace(-10, 10, 100, dtype=np.float32)
        y = np.linspace(-10, 10, 100, dtype=np.float32)
        z = np.linspace(-10, 10, 100, dtype=np.float32)
        xv, yv, zv = np.meshgrid(x, y, z)
        rho = self.trackCharge.rho(np.array((xv, yv, zv))).sum() * (x[1]-x[0]) * (y[1]-y[0]) * (z[1]-z[0])

        assert rho == pytest.approx(self.charge, rel=0.05)
