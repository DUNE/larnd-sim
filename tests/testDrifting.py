#!/usr/bin/env python

import numpy as np
import pytest

from math import ceil

from larndsim import consts

consts.load_properties("larndsim/detector_properties/module0.yaml",
                       "larndsim/pixel_layouts/multi_tile_layout-2.3.16.yaml")
from larndsim.consts import detector, physics

from larndsim import drifting

class TestDrifting:
    """
    Drifting module testing
    """
    tracks = np.zeros((1, 22))
    tracks = np.core.records.fromarrays(tracks.transpose(), 
                                        names="eventID, z_end, trackID, tran_diff, z_start, x_end, y_end, n_electrons, pdgId, x_start, y_start, t_start, dx, long_diff, pixel_plane, t_end, dEdx, dE, t, y, x, z",
                                        formats = "i8, f8, i8, f8, f8, f8, f8, i8, i8, f8, f8, f8, f8, f8, i8, f8, f8, f8, f8, f8, f8, f8")
    tracks["z"] = np.random.uniform(detector.TPC_BORDERS[0][2][0], detector.TPC_BORDERS[0][2][1], 1)
    tracks["x"] = np.random.uniform(detector.TPC_BORDERS[0][0][0], detector.TPC_BORDERS[0][0][1], 1)
    tracks["y"] = np.random.uniform(detector.TPC_BORDERS[0][1][0], detector.TPC_BORDERS[0][1][1], 1)
    tracks["n_electrons"] = np.random.uniform(1e6, 1e7, 1)

    def test_lifetime(self):
        """
        Testing the lifetime correction
        """
        z_anode = detector.TPC_BORDERS[0][2][0]

        drift_distance = np.abs(self.tracks["z"] - z_anode)
        drift_time = drift_distance / detector.V_DRIFT

        lifetime = np.exp(-drift_time / detector.ELECTRON_LIFETIME)

        tracks = self.tracks
        electrons_anode = tracks["n_electrons"] * lifetime

        TPB = 128
        BPG = ceil(tracks.shape[0] / TPB)
        drifting.drift[BPG,TPB](tracks)
        
        assert tracks["n_electrons"] == pytest.approx(electrons_anode)
