#!/usr/bin/env python

import random
import numpy as np
import pytest
import numba as nb

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from larndsim import quenching
from larndsim import consts


class TestQuenching:
    tracks = np.zeros((100, 3))
    tracks[:, 0] = np.random.uniform(40, 90, 100)
    tracks[:, 1] = np.random.uniform(1, 4, 100)
    col = nb.typed.Dict()
    col['dE'] = 0
    col['dEdx'] = 1
    col['NElectrons'] = 2

    def test_birksModel(self):
        tracks_birks = np.copy(self.tracks)

        dedx = self.tracks[:, self.col['dEdx']]
        de = self.tracks[:, self.col['dE']]

        quenching.Quench(tracks_birks, self.col, mode="birks")

        nelectrons = tracks_birks[:, self.col['NElectrons']]

        recomb = consts.Ab / (1 + consts.kb * dedx / (consts.eField * consts.lArDensity))

        assert nelectrons == pytest.approx(recomb * de * consts.MeVToElectrons)

    def test_boxModel(self):
        tracks_box = np.copy(self.tracks)

        dedx = self.tracks[:, self.col['dEdx']]
        de = self.tracks[:, self.col['dE']]

        quenching.Quench(tracks_box, self.col, mode="box")

        csi = consts.beta * dedx / (consts.eField * consts.lArDensity)
        recomb = np.log(consts.alpha + csi)/csi

        nelectrons = tracks_box[:, self.col['NElectrons']]

        assert nelectrons == pytest.approx(recomb * de * consts.MeVToElectrons)
