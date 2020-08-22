#!/usr/bin/env python

import random
import numpy as np
import pytest
from math import sqrt, pi
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from larndsim import detsim

class TestTrackCharge:
    charge = random.randint(100,1000)
    start = np.array([random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(-5, 5)])
    end =  np.array([random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(-5, 5)])
    sigmas = np.array([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)])

    def test_rho(self):
        xx = np.linspace(-10, 10, 100)
        yy = np.linspace(-10, 10, 100)
        zz = np.linspace(-10, 10, 100)

        weights = np.empty(len(xx)*len(yy)*len(zz))
        segment = self.end-self.start
        i = 0
        for x in xx:
            for y in yy:
                for z in zz:
                    weights[i] = detsim.rho((x, y, z), self.charge, self.start, self.sigmas, segment) * (xx[1]-xx[0]) * (yy[1]-yy[0]) * (zz[1] - zz[0])
                    i += 1

        assert np.sum(weights) == pytest.approx(self.charge, rel=0.05)
