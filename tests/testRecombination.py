#!/usr/bin/env python

import numpy as np
import pytest

from larndsim import quenching
from larndsim.consts import detector, physics

from math import ceil

class TestRecombination:
    """
    Quenching module testing
    """
    segments_dtype = np.dtype([("event_id","u4"),("vertex_id", "u8"),
                               ("segment_id", "u4"), ("z_end", "f4"),
                               ("traj_id", "i4"), ("file_traj_id", "u4"),
                               ("tran_diff", "f4"),
                               ("z_start", "f4"), ("x_end", "f4"),
                               ("y_end", "f4"), ("n_electrons", "u4"),
                               ("pdg_id", "i4"), ("x_start", "f4"),
                               ("y_start", "f4"), ("t_start", "f8"),
                               ("t0_start", "f8"), ("t0_end", "f8"), ("t0", "f8"),
                               ("dx", "f4"), ("long_diff", "f4"),
                               ("pixel_plane", "i4"), ("t", "f8"), ("t_end", "f8"),
                               ("dEdx", "f4"), ("dE", "f4"), ("dE_secondary", "f4"),
                               ("y", "f4"), ("x", "f4"), ("z", "f4"),
                               ("n_photons","f4")], align=True)
    N_test_rows = 100
    
    # normal valid values
    tracks = np.zeros(N_test_rows, dtype = segments_dtype)
    
    tracks["dE"] = np.random.uniform(0.1, 100, N_test_rows)
    tracks["dEdx"] = np.random.uniform(1, 100, N_test_rows)

    # extreme valid values
    # a track with dEdx = 0, dE was set to 1 (any non-zero value)
    # to test the recombination factor calculation
    track_zero = np.zeros(N_test_rows, dtype = segments_dtype)
    track_zero["dE"] = 1

    # a track with extremely high dEdx
    track_inf = np.zeros(N_test_rows, dtype = segments_dtype)
    track_inf["dE"] = 1e10
    track_inf["dEdx"] = 1e10

    def test_birksModel(self):

        tracks_birks = np.copy(self.tracks)
        dedx = self.tracks["dEdx"]
        de = self.tracks["dE"]

        TPB = 128
        BPG = ceil(tracks_birks.shape[0] / TPB)
        quenching.quench[BPG,TPB](tracks_birks, physics.BIRKS)
        nelectrons = tracks_birks["n_electrons"]

        recomb = physics.BIRKS_Ab / (1 + physics.BIRKS_kb * dedx / (detector.E_FIELD * detector.LAR_DENSITY))

        assert nelectrons == pytest.approx(recomb * de / physics.W_ION, abs = 2)

    def test_boxModel(self):

        tracks_box = np.copy(self.tracks)
        dedx = self.tracks["dEdx"]
        de = self.tracks["dE"]

        TPB = 128
        BPG = ceil(tracks_box.shape[0] / TPB)
        quenching.quench[BPG,TPB](tracks_box, physics.BOX)
        nelectrons = tracks_box["n_electrons"]

        csi = physics.BOX_BETA * dedx / (detector.E_FIELD * detector.LAR_DENSITY)
        recomb = np.log(physics.BOX_ALPHA + csi)/csi

        assert nelectrons == pytest.approx(recomb * de / physics.W_ION, abs = 2)

    def test_birksModel_zero(self):

        track_birks_zero = np.copy(self.track_zero)
        de = self.track_zero["dE"]

        TPB = 128
        BPG = ceil(track_birks_zero.shape[0] / TPB)
        quenching.quench[BPG,TPB](track_birks_zero, physics.BIRKS)
        nelectrons = track_birks_zero["n_electrons"]

        recomb = physics.BIRKS_Ab

        assert nelectrons == pytest.approx(recomb * de / physics.W_ION, abs = 2)

    def test_boxModel_zero(self):

        tracks_box_zero = np.copy(self.track_zero)
        de = self.track_zero["dE"]

        TPB = 128
        BPG = ceil(tracks_box_zero.shape[0] / TPB)
        quenching.quench[BPG,TPB](tracks_box_zero, physics.BOX)
        nelectrons = tracks_box_zero["n_electrons"]

        recomb = 0.0

        assert nelectrons == pytest.approx(recomb * de / physics.W_ION)

    def test_birksModel_inf(self):

        track_birks_inf = np.copy(self.track_inf)
        de = self.track_inf["dE"]

        TPB = 128
        BPG = ceil(track_birks_inf.shape[0] / TPB)
        quenching.quench[BPG,TPB](track_birks_inf, physics.BIRKS)
        nelectrons = track_birks_inf["n_electrons"]

        recomb = nelectrons/(de / physics.W_ION)

        assert 0 == pytest.approx(recomb, abs = 1e-6)

    def test_boxModel_inf(self):

        tracks_box_inf = np.copy(self.track_inf)
        de = self.track_inf["dE"]

        TPB = 128
        BPG = ceil(tracks_box_inf.shape[0] / TPB)
        quenching.quench[BPG,TPB](tracks_box_inf, physics.BOX)
        nelectrons = tracks_box_inf["n_electrons"]

        recomb = nelectrons/(de / physics.W_ION)

        assert 0 == pytest.approx(recomb, abs = 1e-6)
