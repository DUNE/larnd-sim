#!/usr/bin/env python

import numpy as np
import pytest

from larndsim import consts

consts.load_detector_properties("larndsim/detector_properties/module0.yaml",
                                "larndsim/pixel_layouts/multi_tile_layout-2.3.16.yaml")

from larndsim import detsim
from larndsim import drifting, quenching, pixels_from_track

from math import ceil

class TestTrackCurrent:
    tracks = np.zeros((10, 23))
    tracks = np.core.records.fromarrays(tracks.transpose(), 
                                        names="eventID, dEdx, x_start, dE, t_start, z_end, trackID, x_end, y_end, n_electrons, n_photons, t, dx, pdgId, y, x, long_diff, z, z_start, y_start, tran_diff, t_end, pixel_plane",
                                        formats = "i8, f8, f8, f8, f8, f8, i8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, i8")
    tracks["z_start"] = np.random.uniform(consts.tpc_borders[0][2][0], consts.tpc_borders[0][2][1], 10)
    tracks["z_end"] = np.random.uniform(consts.tpc_borders[0][2][0], consts.tpc_borders[0][2][0]+2, 10)
    tracks["z"] = (tracks["z_end"]+tracks["z_start"])/2.
    tracks["y_start"] = np.random.uniform(consts.tpc_borders[0][1][0], consts.tpc_borders[0][1][0]+2, 10)
    tracks["y_end"] = np.random.uniform(consts.tpc_borders[0][1][0], consts.tpc_borders[0][1][0]+2, 10)
    tracks["x_start"] = np.random.uniform(consts.tpc_borders[0][0][0], consts.tpc_borders[0][0][0]+2, 10)
    tracks["x_end"] = np.random.uniform(consts.tpc_borders[0][0][0], consts.tpc_borders[0][0][0]+2, 10)
    tracks["x"] = (tracks["x_end"]+tracks["x_start"])/2.
    tracks["y"] = (tracks["y_end"]+tracks["y_start"])/2.
    tracks["dx"] = np.sqrt((tracks["x_end"]-tracks["x_start"])**2+(tracks["y_end"]-tracks["y_start"])**2+(tracks["z_end"]-tracks["z_start"])**2)
    tracks["dEdx"] = [2]*10
    tracks["dE"] = tracks["dEdx"]*tracks["dx"]
    tracks["tran_diff"] = [1e-1]*10
    tracks["long_diff"] = [1e-1]*10

    def test_current_model(self):

        tracks = np.copy(self.tracks)
        threadsperblock = 128
        blockspergrid = ceil(tracks.shape[0] / threadsperblock)
        quenching.quench[blockspergrid,threadsperblock](tracks, consts.box)
        drifting.drift[blockspergrid,threadsperblock](tracks)

        MAX_PIXELS = 110
        MAX_ACTIVE_PIXELS = 50

        active_pixels = np.full((tracks.shape[0], MAX_ACTIVE_PIXELS), -1, dtype=np.int32)
        neighboring_pixels = np.full((tracks.shape[0], MAX_PIXELS), -1, dtype=np.int32)

        n_pixels_list = np.zeros(shape=(tracks.shape[0]))
        blockspergrid = ceil(tracks.shape[0] / threadsperblock)
        pixels_from_track.get_pixels[blockspergrid,threadsperblock](tracks,
                                                                    active_pixels,
                                                                    neighboring_pixels,
                                                                    n_pixels_list,
                                                                    2)
        signals = np.zeros((tracks.shape[0],
                            neighboring_pixels.shape[1],
                            consts.time_ticks.shape[0]), dtype=np.float32)
        threadsperblock = (1,1,64)
        blockspergrid_x = ceil(signals.shape[0] / threadsperblock[0])
        blockspergrid_y = ceil(signals.shape[1] / threadsperblock[1])
        blockspergrid_z = ceil(signals.shape[2] / threadsperblock[2])
        blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
        response = np.load('larndsim/response_44.npy')
        detsim.tracks_current[blockspergrid,threadsperblock](signals,
                                                             neighboring_pixels,
                                                             tracks,
                                                             response)
        print(np.sum(signals)*consts.t_sampling/consts.e_charge)

        assert np.sum(signals)*consts.t_sampling/consts.e_charge == pytest.approx(np.sum(tracks['n_electrons']), rel=0.05)
