#!/usr/bin/env python

import random
import numpy as np
import pytest

from larndsim import detsim

class TestTrackCharge:
    charge = random.randint(100,1000)
    start = np.array([random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(-5, 5)])
    end =  np.array([random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(-5, 5)])
    sigmas = np.array([random.uniform(0, 0.5), random.uniform(0, 0.5), random.uniform(0, 0.5)])

    def test_rho(self):
        xx = np.linspace(-5, 5, 50)
        yy = np.linspace(-5, 5, 50)
        zz = np.linspace(-5, 5, 50)

        segment = self.end-self.start

        calculated_charge = 0
        for x in xx:
            for y in yy:
                for z in zz:
                    calculated_charge += detsim.rho((x, y, z), self.charge, self.start, self.sigmas, segment) * (xx[1]-xx[0]) * (yy[1]-yy[0]) * (zz[1] - zz[0])

        assert calculated_charge == pytest.approx(self.charge, rel=0.05)