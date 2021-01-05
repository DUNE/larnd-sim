#!/usr/bin/env python

import random
import numpy as np
import numba as nb
import pytest

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from math import pi, exp
from larndsim import detsim
from larndsim import drifting, quenching, pixels_from_track
from larndsim import consts
from larndsim import indeces as i

from math import ceil

class TestTrackCurrent:
    tracks = np.zeros((10, 29))
    tracks[:, i.z_start] = np.random.uniform(consts.tpc_borders[2][0], consts.tpc_borders[2][0]+2, 10)
    tracks[:, i.z_end] = np.random.uniform(consts.tpc_borders[2][0], consts.tpc_borders[2][0]+2, 10)
    tracks[:, i.z] = (tracks[:, i.z_end]+tracks[:, i.z_start])/2.
    tracks[:, i.y_start] = np.random.uniform(consts.tpc_borders[1][0], consts.tpc_borders[1][0]+2, 10)
    tracks[:, i.y_end] = np.random.uniform(consts.tpc_borders[1][0], consts.tpc_borders[1][0]+2, 10)
    tracks[:, i.x_start] = np.random.uniform(consts.tpc_borders[0][0], consts.tpc_borders[0][0]+2, 10)
    tracks[:, i.x_end] = np.random.uniform(consts.tpc_borders[0][0], consts.tpc_borders[0][0]+2, 10)
    tracks[:, i.x] = (tracks[:, i.x_end]+tracks[:, i.x_start])/2.
    tracks[:, i.y] = (tracks[:, i.y_end]+tracks[:, i.y_start])/2.
    tracks[:, i.dx] = np.sqrt((tracks[:, i.x_end]-tracks[:, i.x_start])**2+(tracks[:, i.y_end]-tracks[:, i.y_start])**2+(tracks[:, i.z_end]-tracks[:, i.z_start])**2)
    tracks[:, i.dEdx] = [2]*10
    tracks[:, i.dE] = tracks[:, i.dEdx]*tracks[:, i.dx]
    tracks[:, i.tran_diff] = [1e-1]*10
    tracks[:, i.long_diff] = [1e-1]*10

    def test_current_model(self):
         
        tracks = np.copy(self.tracks)
        threadsperblock = 128
        blockspergrid = ceil(tracks.shape[0] / threadsperblock)
        quenching.quench[blockspergrid,threadsperblock](tracks, consts.box)
        drifting.drift[blockspergrid,threadsperblock](tracks)

        MAX_PIXELS = 110
        MAX_ACTIVE_PIXELS = 50

        active_pixels = np.full((tracks.shape[0], MAX_ACTIVE_PIXELS, 2), -1, dtype=np.int32)
        neighboring_pixels = np.full((tracks.shape[0], MAX_PIXELS, 2), -1, dtype=np.int32)
        
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
        threadsperblock = (4,4,4)
        blockspergrid_x = ceil(signals.shape[0] / threadsperblock[0])
        blockspergrid_y = ceil(signals.shape[1] / threadsperblock[1])
        blockspergrid_z = ceil(signals.shape[2] / threadsperblock[2])
        blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
        detsim.tracks_current[blockspergrid,threadsperblock](signals, 
                                                             neighboring_pixels, 
                                                             tracks)

        assert np.sum(signals)*consts.t_sampling/consts.e_charge == pytest.approx(np.sum(tracks[:,i.n_electrons]), rel=0.05)
