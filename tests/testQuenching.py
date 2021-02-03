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
    #normal valid values
    tracks = np.zeros((100, 22))
    tracks[:, i.dE] = np.random.uniform(0.1, 100, 100)
    tracks[:, i.dEdx] = np.random.uniform(1, 100, 100)
    
    #extreme valid values 
    #a track with dEdx = 0, dE was set to 1 (any non-zero value) to test the recombination factor calculation
    track_zero = np.zeros((1,22)) 
    track_zero[:,i.dE] = 1

    #a track with extremely high dEdx
    track_inf = np.zeros((1,22)) 
    track_inf[:,i.dE] = 1e10 
    track_inf[:,i.dEdx] = 1e10

    def test_birksModel(self):

        tracks_birks = np.copy(self.tracks)
        dedx = self.tracks[:, i.dEdx]
        de = self.tracks[:, i.dE]

        TPB = 128
        BPG = ceil(tracks_birks.shape[0] / TPB)
        quenching.quench[BPG,TPB](tracks_birks, consts.birks)
        nelectrons = tracks_birks[:, i.n_electrons]

        recomb = consts.Ab / (1 + consts.kb * dedx / (consts.eField * consts.lArDensity))

        assert nelectrons == pytest.approx(recomb * de * consts.MeVToElectrons)

    def test_boxModel(self):

        tracks_box = np.copy(self.tracks)
        dedx = self.tracks[:, i.dEdx]
        de = self.tracks[:, i.dE]

        TPB = 128
        BPG = ceil(tracks_box.shape[0] / TPB)
        quenching.quench[BPG,TPB](tracks_box, consts.box)
        nelectrons = tracks_box[:, i.n_electrons]

        csi = consts.beta * dedx / (consts.eField * consts.lArDensity)
        recomb = np.log(consts.alpha + csi)/csi

        assert nelectrons == pytest.approx(recomb * de * consts.MeVToElectrons)

    def test_birksModel_zero(self):

        track_birks_zero = np.copy(self.track_zero)
        dedx = self.track_zero[:, i.dEdx]
        de = self.track_zero[:, i.dE]

        TPB = 1
        BPG = 1
        quenching.quench[BPG,TPB](track_birks_zero, consts.birks)
        nelectrons = track_birks_zero[:, i.n_electrons]

        recomb = consts.Ab

        assert nelectrons == pytest.approx(recomb * de * consts.MeVToElectrons)

    def test_boxModel_zero(self):
        
        tracks_box_zero = np.copy(self.track_zero)
        dedx = self.track_zero[:, i.dEdx]
        de = self.track_zero[:, i.dE]

        TPB = 1
        BPG = 1
        quenching.quench[BPG,TPB](tracks_box_zero, consts.box)
        nelectrons = tracks_box_zero[:, i.n_electrons]

        recomb = 0.0

        assert nelectrons == pytest.approx(recomb * de * consts.MeVToElectrons)

    def test_birksModel_inf(self):

        track_birks_inf = np.copy(self.track_inf)
        dedx = self.track_inf[:, i.dEdx]
        de = self.track_inf[:, i.dE]

        TPB = 1
        BPG = 1
        quenching.quench[BPG,TPB](track_birks_inf, consts.birks)
        nelectrons = track_birks_inf[:, i.n_electrons]

        recomb =  nelectrons/(de * consts.MeVToElectrons)
        
        assert recomb > 0 and recomb < 1e-6

    def test_boxModel_inf(self):

        tracks_box_inf = np.copy(self.track_inf)
        dedx = self.track_inf[:, i.dEdx]
        de = self.track_inf[:, i.dE]

        TPB = 1
        BPG = 1
        quenching.quench[BPG,TPB](tracks_box_inf, consts.box)
        nelectrons = tracks_box_inf[:, i.n_electrons]

        recomb =  nelectrons/(de * consts.MeVToElectrons)   
        
        assert recomb > 0 and recomb < 1e-6
        