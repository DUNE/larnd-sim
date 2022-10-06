import numpy as np
import cupy as cp

def select_active_volume(track_seg, tpc_borders):
    """
    Extracts track segments that are within the described TPC borders

    Args:
        track_seg (:obj:`numpy.ndarray`): edep-sim track segment array
        tpc_borders (:obj:`numpy.ndarray`): bounding box of each tpc, shape `(ntpc, 3, 2)`

    Returns:
        indices of track segments that are at least partially contained
    """
    xp = cp.get_array_module(track_seg)
    tpc_mask = xp.zeros(track_seg.shape, dtype=bool)
    tpb_borders = xp.sort(tpc_borders, axis=-1)
    for i_tpc in range(tpc_borders.shape[0]):
        tpc_bound = tpc_borders[i_tpc]
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
