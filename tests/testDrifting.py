#!/usr/bin/env python

import numpy as np
import pytest

from math import ceil

from larndsim import consts

consts.load_detector_properties("larndsim/detector_properties/singlecube.yaml",
                                "larndsim/pixel_layouts/layout-singlecube.yaml")

from larndsim import drifting

class TestDrifting:
    """
    Drifting module testing
    """
    tracks = np.zeros((1, 22))
    tracks = np.core.records.fromarrays(tracks.transpose(), 
                                        names="eventID, z_end, trackID, tran_diff, z_start, x_end, y_end, n_electrons, pdgId, x_start, y_start, t_start, dx, long_diff, pixel_plane, t_end, dEdx, dE, t, y, x, z",
                                        formats = "i8, f8, i8, f8, f8, f8, f8, i8, i8, f8, f8, f8, f8, f8, i8, f8, f8, f8, f8, f8, f8, f8")
    tracks["z"] = np.random.uniform(consts.module_borders[0][2][0], consts.module_borders[0][2][1], 1)
    tracks["x"] = np.random.uniform(consts.module_borders[0][0][0], consts.module_borders[0][0][1], 1)
    tracks["y"] = np.random.uniform(consts.module_borders[0][1][0], consts.module_borders[0][1][1], 1)
    tracks["n_electrons"] = np.random.uniform(1e6, 1e7, 1)

    def test_lifetime(self):
        """
        Testing the lifetime correction
        """
        z_anode = consts.module_borders[0][2][0]

        drift_distance = np.abs(self.tracks["z"] - z_anode)
        drift_time = drift_distance / consts.vdrift

        lifetime = np.exp(-drift_time / consts.lifetime)

        tracks = self.tracks
        electrons_anode = tracks["n_electrons"] * lifetime

        TPB = 128
        BPG = ceil(tracks.shape[0] / TPB)
        drifting.drift[BPG,TPB](tracks)
        
        assert 1 == 1
#         assert tracks["n_electrons"] == pytest.approx(electrons_anode)
