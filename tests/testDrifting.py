#!/usr/bin/env python

import numpy as np
import pytest

from math import ceil

from larndsim import consts

consts.load_detector_properties("larndsim/detector_properties/singlecube.yaml",
                                "larndsim/pixel_layouts/layout-singlecube.yaml")

from larndsim import indeces as i
from larndsim import drifting

class TestDrifting:
    """
    Drifting module testing
    """
    tracks = np.zeros((10, 29))
    tracks[:, i.z] = np.random.uniform(consts.module_borders[0][2][0],
                                       consts.module_borders[0][2][1], 10)
    tracks[:, i.x] = np.random.uniform(consts.module_borders[0][0][0],
                                       consts.module_borders[0][0][1], 10)
    tracks[:, i.y] = np.random.uniform(consts.module_borders[0][1][0],
                                       consts.module_borders[0][1][1], 10)
    tracks[:, i.n_electrons] = np.random.uniform(1e6, 1e7, 10)

    def test_lifetime(self):
        """
        Testing the lifetime correction
        """
        z_anode = consts.module_borders[0][2][0]
        drift_distance = np.abs(self.tracks[:, i.z] - z_anode)
        drift_time = drift_distance / consts.vdrift

        lifetime = np.exp(-drift_time / consts.lifetime)

        tracks = self.tracks
        electrons_anode = tracks[:, i.n_electrons] * lifetime

        TPB = 128
        BPG = ceil(tracks.shape[0] / TPB)
        drifting.drift[BPG,TPB](tracks)

        assert tracks[:, i.n_electrons] == pytest.approx(electrons_anode)
