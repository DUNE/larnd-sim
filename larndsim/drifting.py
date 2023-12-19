"""
Module to implement the propagation of the
electrons towards the anode.
"""

from math import exp, sqrt
from numba import cuda
from .consts import detector
#from .consts.detector import TPC_BORDERS

@cuda.jit
def drift(tracks):
    """
    This function takes as input an array of track segments and calculates
    the properties of the segments at the anode:

      * z coordinate at the anode
      * number of electrons taking into account electron lifetime
      * longitudinal diffusion
      * transverse diffusion
      * time of arrival at the anode

    Args:
        tracks (:obj:`numpy.ndarray`): array containing the tracks segment information
    """
    itrk = cuda.grid(1)

    if itrk < tracks.shape[0]:

        track = tracks[itrk]

        pixel_plane = detector.DEFAULT_PLANE_INDEX

        for ip, plane in enumerate(detector.TPC_BORDERS):
            if plane[0][0]-2e-2 <= track["x"] <= plane[0][1]+2e-2 and \
               plane[1][0]-2e-2 <= track["y"] <= plane[1][1]+2e-2 and \
               min(plane[2][1]-2e-2,plane[2][0]-2e-2) <= track["z"] <= max(plane[2][1]+2e-2,plane[2][0]+2e-2):
                pixel_plane = ip
                break

        track["pixel_plane"] = pixel_plane

        if pixel_plane != detector.DEFAULT_PLANE_INDEX:
            z_anode = detector.TPC_BORDERS[pixel_plane][2][0]
            drift_distance = abs(track["z"] - z_anode)
            drift_start = abs(min(track["z_start"],track["z_end"]) - z_anode)
            drift_end = abs(max(track["z_start"],track["z_end"]) - z_anode)
            drift_time = drift_distance / detector.V_DRIFT
            lifetime_red = exp(-drift_time / detector.ELECTRON_LIFETIME)

            track["n_electrons"] *= lifetime_red

            track["long_diff"] = sqrt(drift_time * 2 * detector.LONG_DIFF)
            track["tran_diff"] = sqrt(drift_time * 2 * detector.TRAN_DIFF)

            track["t"] += drift_time + track["t0"]
            track["t_start"] += min(drift_start, drift_end) / detector.V_DRIFT + track["t0"]
            track["t_end"] += max(drift_start, drift_end) / detector.V_DRIFT + track["t0"]
