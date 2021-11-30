#!/usr/bin/env python

import numpy as np
import pytest

from larndsim import quenching
from larndsim import consts

from math import ceil

class TestQuenching:
    """
    Quenching module testing
    """
    #normal valid values
    tracks = np.zeros((100, 22))
    tracks = np.core.records.fromarrays(tracks.transpose(), 
                                        names="eventID, dEdx, x_start, dE, t_start, z_end, trackID, x_end, y_end, n_electrons, n_photons, t, dx, pdgId, y, x, long_diff, z, z_start, y_start, tran_diff, t_end, pixel_plane",
                                        formats = "i8, f8, f8, f8, f8, f8, i8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, i8")
    tracks["dE"] = np.random.uniform(0.1, 100, 100)
    tracks["dEdx"] = np.random.uniform(1, 100, 100)

    #extreme valid values
    #a track with dEdx = 0, dE was set to 1 (any non-zero value) to test the recombination factor calculation
    track_zero = np.zeros((1,22))
    track_zero = np.core.records.fromarrays(track_zero.transpose(), 
                                            names="eventID, dEdx, x_start, dE, t_start, z_end, trackID, x_end, y_end, n_electrons, n_photons, t, dx, pdgId, y, x, long_diff, z, z_start, y_start, tran_diff, t_end, pixel_plane",
                                           formats = "i8, f8, f8, f8, f8, f8, i8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, i8")
    track_zero["dE"] = 1

    #a track with extremely high dEdx
    track_inf = np.zeros((1,22))
    track_inf = np.core.records.fromarrays(track_inf.transpose(), 
                                           names="eventID, dEdx, x_start, dE, t_start, z_end, trackID, x_end, y_end, n_electrons, n_photons, t, dx, pdgId, y, x, long_diff, z, z_start, y_start, tran_diff, t_end, pixel_plane",
                                           formats = "i8, f8, f8, f8, f8, f8, i8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, i8")
    track_inf["dE"] = 1e10
    track_inf["dEdx"] = 1e10

    def test_birksModel(self):

        tracks_birks = np.copy(self.tracks)
        dedx = self.tracks["dEdx"]
        de = self.tracks["dE"]

        TPB = 128
        BPG = ceil(tracks_birks.shape[0] / TPB)
        quenching.quench[BPG,TPB](tracks_birks, consts.birks)
        nelectrons = tracks_birks["n_electrons"]

        recomb = consts.Ab / (1 + consts.kb * dedx / (consts.eField * consts.lArDensity))

        assert nelectrons == pytest.approx(recomb * de * consts.MeVToElectrons)

    def test_boxModel(self):

        tracks_box = np.copy(self.tracks)
        dedx = self.tracks["dEdx"]
        de = self.tracks["dE"]

        TPB = 128
        BPG = ceil(tracks_box.shape[0] / TPB)
        quenching.quench[BPG,TPB](tracks_box, consts.box)
        nelectrons = tracks_box["n_electrons"]

        csi = consts.beta * dedx / (consts.eField * consts.lArDensity)
        recomb = np.log(consts.alpha + csi)/csi

        assert nelectrons == pytest.approx(recomb * de * consts.MeVToElectrons)

    def test_birksModel_zero(self):

        track_birks_zero = np.copy(self.track_zero)
        de = self.track_zero["dE"]

        TPB = 1
        BPG = 1
        quenching.quench[BPG,TPB](track_birks_zero, consts.birks)
        nelectrons = track_birks_zero["n_electrons"]

        recomb = consts.Ab

        assert nelectrons == pytest.approx(recomb * de * consts.MeVToElectrons)

    def test_boxModel_zero(self):

        tracks_box_zero = np.copy(self.track_zero)
        de = self.track_zero["dE"]

        TPB = 1
        BPG = 1
        quenching.quench[BPG,TPB](tracks_box_zero, consts.box)
        nelectrons = tracks_box_zero["n_electrons"]

        recomb = 0.0

        assert nelectrons == pytest.approx(recomb * de * consts.MeVToElectrons)

    def test_birksModel_inf(self):

        track_birks_inf = np.copy(self.track_inf)
        de = self.track_inf["dE"]

        TPB = 1
        BPG = 1
        quenching.quench[BPG,TPB](track_birks_inf, consts.birks)
        nelectrons = track_birks_inf["n_electrons"]

        recomb = nelectrons/(de * consts.MeVToElectrons)

        assert recomb > 0 and recomb < 1e-6

    def test_boxModel_inf(self):

        tracks_box_inf = np.copy(self.track_inf)
        de = self.track_inf["dE"]

        TPB = 1
        BPG = 1
        quenching.quench[BPG,TPB](tracks_box_inf, consts.box)
        nelectrons = tracks_box_inf["n_electrons"]

        recomb = nelectrons/(de * consts.MeVToElectrons)

        assert recomb > 0 and recomb < 1e-6
