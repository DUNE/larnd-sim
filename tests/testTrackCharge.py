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
    trackCharge = detsim.TrackCharge(charge,
                                     random.uniform(-5,5), random.uniform(-5,5),
                                     random.uniform(-5,5), random.uniform(-5,5),
                                     random.uniform(-5,5), random.uniform(-5,5),
                                     [random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)])

    def test_rho(self):
        x = np.linspace(-10,10,100)
        y = np.linspace(-10,10,100)
        z = np.linspace(-10,10,100)
        xv, yv, zv = np.meshgrid(x,y,z)
        rho = self.trackCharge.rho(xv, yv, zv).sum() * (x[1]-x[0]) * (y[1]-y[0]) * (z[1]-z[0])
        print(self.trackCharge)

        assert rho == pytest.approx(self.charge, rel=0.05)