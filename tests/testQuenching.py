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
from larndsim import indeces as i

from math import ceil

class TestQuenching:
    tracks = np.zeros((100, 20))
    tracks[:, i.dE] = np.random.uniform(40, 90, 100)
    tracks[:, i.dEdx] = np.random.uniform(1, 4, 100)

    def test_birksModel(self):
        tracks_birks = np.copy(self.tracks)

        dedx = self.tracks[:, i.dEdx]
        de = self.tracks[:, i.dE]
        TPB = 128
        BPG = ceil(tracks_birks.shape[0] / TPB)
        quenching.quench[TPB,BPG](tracks_birks, consts.birks)

        nelectrons = tracks_birks[:, i.n_electrons]

        recomb = consts.Ab / (1 + consts.kb * dedx / (consts.eField * consts.lArDensity))

        assert nelectrons == pytest.approx(recomb * de * consts.MeVToElectrons)

    def test_boxModel(self):
        tracks_box = np.copy(self.tracks)

        dedx = self.tracks[:, i.dEdx]
        de = self.tracks[:, i.dE]
        TPB = 128
        BPG = ceil(tracks_box.shape[0] / TPB)
        quenching.quench[TPB,BPG](tracks_box, consts.box)

        csi = consts.beta * dedx / (consts.eField * consts.lArDensity)
        recomb = np.log(consts.alpha + csi)/csi

        nelectrons = tracks_box[:, i.n_electrons]

        assert nelectrons == pytest.approx(recomb * de * consts.MeVToElectrons)
