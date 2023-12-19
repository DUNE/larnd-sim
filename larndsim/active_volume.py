import numpy as np
import cupy as cp

def select_active_volume(track_seg, tpc_borders, i_module=-1):
    """
    Extracts track segments that are within the described TPC borders

    Args:
        track_seg (:obj:`numpy.ndarray`): edep-sim track segment array
        tpc_borders (:obj:`numpy.ndarray`): bounding box of each tpc, shape `(ntpc, 3, 2)`
        i_module (int): module id, default value i_module = -1. 
                        When i_module < 0, select track in all active TPC volumes.
                        When i_module > 0, it indicates a specific LAr module.
                        The first module has i_module = 1.
                        Then the selected tracks (segments) will be in this module.

    Returns:
        indices of track segments that are at least partially contained
    """
    xp = cp.get_array_module(track_seg)
    tpc_mask = xp.zeros(track_seg.shape, dtype=bool)
    tpc_start_mask = xp.zeros(track_seg.shape, dtype=bool)
    tpc_end_mask = xp.zeros(track_seg.shape, dtype=bool)
    tpc_borders = xp.sort(tpc_borders, axis=-1)
    if i_module < 0:
        tpc_list = range(tpc_borders.shape[0])
    else:
        tpc_list = range((i_module-1)*2, i_module*2)  # 2 tpcs per module
    for i_tpc in tpc_list:
        tpc_bound = tpc_borders[i_tpc]
        # keep a segment if any part of it is in the tpc_bound
        tpc_mask = tpc_mask | (
                ((track_seg['x_end'] > tpc_bound[0,0])
                 & (track_seg['x_end'] < tpc_bound[0,1])
                 & (track_seg['y_end'] > tpc_bound[1,0])
                 & (track_seg['y_end'] < tpc_bound[1,1])
                 & (track_seg['z_end'] > tpc_bound[2,0])
                 & (track_seg['z_end'] < tpc_bound[2,1]))
                | ((track_seg['x_start'] > tpc_bound[0,0])
                   & (track_seg['x_start'] < tpc_bound[0,1])
                   & (track_seg['y_start'] > tpc_bound[1,0])
                   & (track_seg['y_start'] < tpc_bound[1,1])
                   & (track_seg['z_start'] > tpc_bound[2,0])
                   & (track_seg['z_start'] < tpc_bound[2,1])))

    return xp.nonzero(tpc_mask)[0]

