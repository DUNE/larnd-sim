"""
Module that calculates the current induced by edep-sim track segments
on the pixels
"""

from math import pi, ceil, sqrt, erf, exp, log
import cupy as cp
import numba as nb

from numba import cuda
from numba.cuda.random import xoroshiro128p_normal_float32

from .consts import detector
from .consts import sim
from .pixels_from_track import id2pixel

@nb.njit
def get_pixel_coordinates(pixel_id):
    """
    Returns the coordinates of the pixel center given the pixel ID
    """
    i_x, i_y, plane_id = id2pixel(pixel_id)

    this_border = detector.TPC_BORDERS[int(plane_id)]
    pix_x = i_x * detector.PIXEL_PITCH + this_border[0][0]
    pix_y = i_y * detector.PIXEL_PITCH + this_border[1][0]

    return pix_x,pix_y

@nb.njit
def get_closest_waveform(x, y, t, response):
    """
    This function, given a point on the pixel pad and a time, returns the
    closest tabulated waveformm from the response array.

    Args:
        x (float): x coordinate of the point
        y (float): y coordinate of the point
        t (float): time of the waveform
        response (:obj:`numpy.ndarray`): array containing the tabulated waveforms

    Returns:
        float: the value of the induced current at time `t` for a charge at `(x,y)`
    """
    dt = detector.RESPONSE_SAMPLING
    bin_width = detector.RESPONSE_BIN_SIZE

    i = round((x/bin_width) - 0.5)
    j = round((y/bin_width) - 0.5)
    k = round(t/dt)

    if 0 <= i < response.shape[0] and 0 <= j < response.shape[1] and 0 <= k < response.shape[2]:
        return response[i][j][k]

    return 0

@nb.njit
def overlapping_segment(x, y, start, end, radius):
    """
    Computes the relevant segment part that's around the pixel center (x, y).
    The projected positions of the new start and end on the pixel plane has distances of one "radius" away from the pixel center.
    The new start and end of the segment is along the original segment.
    """
    skip = False
    dxy = x - start[0], y - start[1]
    v = end[0] - start[0], end[1] - start[1]
    l = sqrt(v[0]**2 + v[1]**2)
    if l == 0: # vertical to the anode
        dist = sqrt((x-start[0])**2 + (y-start[1])**2)
        if dist > radius:
            skip = True
            return start, end, skip
        else:
            return start, end, skip
    v = v[0]/l, v[1]/l
    s = (dxy[0] * v[0] + dxy[1] * v[1])/l # position of point of closest approach

    r = sqrt((dxy[0] - v[0] * s * l)**2 + (dxy[1] - v[1] * s * l)**2)
    if r > radius:
        skip = True
        return start, start, skip # no overlap

    s_plus = s + sqrt(radius**2 - r**2) / l
    s_minus = s - sqrt(radius**2 - r**2) / l

    if s_plus > 1:
        s_plus = 1
    elif s_plus < 0:
        s_plus = 0
    if s_minus > 1:
        s_minus = 1
    elif s_minus < 0:
        s_minus = 0

    new_start = (start[0] * (1 - s_minus) + end[0] * s_minus,
                 start[1] * (1 - s_minus) + end[1] * s_minus,
                 start[2] * (1 - s_minus) + end[2] * s_minus)
    new_end = (start[0] * (1 - s_plus) + end[0] * s_plus,
               start[1] * (1 - s_plus) + end[1] * s_plus,
               start[2] * (1 - s_plus) + end[2] * s_plus)

    return new_start, new_end, skip

# @cuda.jit
@cuda.jit(max_registers=128,  fastmath=True)
def tracks_current_mc(signals, pixels, tracks, response, rng_states):
    """
    This CUDA kernel calculates the charge induced on the pixels by the input tracks using a
    MC method

    Args:
        signals (:obj:`numpy.ndarray`): empty 3D array with dimensions S x P x T,
            where S is the number of track segments, P is the number of pixels, and T is
            the number of time ticks. The output is stored here.
        pixels (:obj:`numpy.ndarray`): 2D array with dimensions S x P , where S is
            the number of track segments, P is the number of pixels and contains the pixel ID number.
        tracks (:obj:`numpy.ndarray`): 2D array containing the detector segments.
        response (:obj:`numpy.ndarray`): 3D array containing the tabulated response.
        rng_states (:obj:`numpy.ndarray`): array of random states for noise
            generation
    """
    itrk, ipix, it = cuda.grid(3)
    ntrk, npix, nt = cuda.gridsize(3)

    if itrk < signals.shape[0] and ipix < signals.shape[1] and it < signals.shape[2]:
        t = tracks[itrk]
        pID = pixels[itrk][ipix]
        pID_x, pID_y, pID_plane = id2pixel(pID)

        if pID_x >= 0 and pID_y >= 0:

            # Pixel coordinates
            x_p, y_p = get_pixel_coordinates(pID)
            x_p += detector.PIXEL_PITCH / 2
            y_p += detector.PIXEL_PITCH / 2

            if t["z_start"] < t["z_end"]:
                start = (t["x_start"], t["y_start"], t["z_start"])
                end = (t["x_end"], t["y_end"], t["z_end"])
            else:
                end = (t["x_start"], t["y_start"], t["z_start"])
                start = (t["x_end"], t["y_end"], t["z_end"])

            # if the time tick is before the segment start time, pass
            this_time = it * detector.TIME_SAMPLING

            # detector.TPC_BORDERS[t["pixel_plane"]][2][1]) is the corresponding cathode
            dist_cathode = min(abs(t["z_end"] - detector.TPC_BORDERS[t["pixel_plane"]][2][1]), abs(t["z_start"] - detector.TPC_BORDERS[t["pixel_plane"]][2][1])) # closest distance to the cathode
            # The valid time for integrating the charge signal is the response with the shifted collection position
            # In order to conservatively include more time ticks
            # we use the longest response time, and shortest distance to the cathode from the segments
            # the distance is converted to time using nominal drift velocity
            # pad with 5 times of longitudinal diffusion
            if this_time > (detector.RESPONSE_MAX_TIME - dist_cathode / detector.V_DRIFT) + t['long_diff'] / detector.V_DRIFT * detector.DIFF_N_SIGMAS:
                return

            segment = (end[0]-start[0], end[1]-start[1], end[2]-start[2])
            length = sqrt(segment[0]**2 + segment[1]**2 + segment[2]**2)

            direction = (segment[0]/length, segment[1]/length, segment[2]/length)
            sigmas = (t["tran_diff"], t["tran_diff"], t["long_diff"])

            # full response range and 5 sigmas of transverse diffusion
            impact_factor = sqrt(response.shape[0]**2 +
                                     response.shape[1]**2) * detector.RESPONSE_BIN_SIZE + t['tran_diff'] * detector.DIFF_N_SIGMAS

            subsegment_start, subsegment_end, skip = overlapping_segment(x_p, y_p, start, end, impact_factor)
            if skip:
                return
            subsegment = (subsegment_end[0]-subsegment_start[0],
                          subsegment_end[1]-subsegment_start[1],
                          subsegment_end[2]-subsegment_start[2])
            subsegment_length = sqrt(subsegment[0]**2 + subsegment[1]**2 + subsegment[2]**2)
            if subsegment_length == 0:
                return

            nstep = max(round(subsegment_length / detector.MIN_STEP_SIZE), 1)
            step = subsegment_length / nstep # refine step size

            charge = t["n_electrons"] * (subsegment_length/length) / nstep
            total_current = 0
            for istep in range(nstep):
                x = subsegment_start[0] + step * (istep + 0.5) * direction[0]
                y = subsegment_start[1] + step * (istep + 0.5) * direction[1]
                z = subsegment_start[2] + step * (istep + 0.5) * direction[2]

                z += xoroshiro128p_normal_float32(rng_states, itrk * npix * nt + ipix * nt + it ) * sigmas[2]

                # find how much to shift the time for anode (collection time)
                # detector.TPC_BORDERS[t["pixel_plane"]][2][0] is anode
                # detector.TPC_BORDERS[t["pixel_plane"]][2][1] is cathode
                # equivalent to detector.DRIFT_LENGTH - abs(z - detector.TPC_BORDERS[t["pixel_plane"]][2][1])
                shift_t_collect = abs(z - detector.TPC_BORDERS[t["pixel_plane"]][2][1]) / detector.V_DRIFT

                x += xoroshiro128p_normal_float32(rng_states, itrk * npix * nt + ipix * nt + it ) * sigmas[0]
                y += xoroshiro128p_normal_float32(rng_states, itrk * npix * nt + ipix * nt + it ) * sigmas[1]
                x_dist = abs(x_p - x)
                y_dist = abs(y_p - y)

                if x_dist > detector.RESPONSE_BIN_SIZE * response.shape[0]:
                    continue
                if y_dist > detector.RESPONSE_BIN_SIZE * response.shape[1]:
                    continue
                if (this_time + shift_t_collect) < 0 or (this_time + shift_t_collect) > detector.RESPONSE_MAX_TIME:
                    continue

                # this_time is the drift/readout time
                # t0 is considered in a later stage
                # (shift_t_collect) shifts the readout to the corresponding position 
                total_current += charge * get_closest_waveform(x_dist, y_dist, this_time + shift_t_collect, response)

            signals[itrk,ipix,it] = total_current

@cuda.jit
def sum_pixel_signals(pixels_signals, signals, track_t0, pixel_index_map, track_pixel_map, pixels_tracks_signals,
                      num_backtrack, offset_backtrack, overflow_flag):
    """
    This function sums the induced current signals on the same pixel.
    Converting "signals" from per segment to per pixel ("pixel_signals" and "pixels_tracks_signals")

    Args:
        pixels_signals (:obj:`numpy.ndarray`): 2D array that will contain the
            summed signal for each pixel. First dimension is the pixel ID, second
            dimension is the time tick
        signals (:obj:`numpy.ndarray`): 3D array with dimensions S x P x T,
            where S is the total number of track segments, P is the max number of pixels for any segment, and T is
            the number of time ticks.
        track_starts (:obj:`numpy.ndarray`): 1D array containing the starting time of
            each track
        pixel_index_map (:obj:`numpy.ndarray`): 2D array containing the correspondence between
            the track index and the pixel ID index.
        track_pixel_map (:obj:`numpy.ndarray`): 2D array containing the association between
            the unique pixels array and the array containing the pixels for each track.
        pixels_tracks_signals (:obj:`numpy.ndarray`): 1D jagged array that collapse the information of (#unique_pix, #ticks, backtracked_segments) for backtracking info per pixel per time tick.
            for each pixel and each track that induced current on the pixel.
        overflow_flag (:obj:`cp.array`): Single-element output array to indicate whether
            MAX_TRACKS_PER_PIXEL is insufficient
    """
    # itrk: segment index in signals collection, goes up to the total number of the segments in this batch
    # ipix: pixel index within signals collection, goes up to the max number of pixel for any segment
    # itick: time ticks along drift

    # pixel_index goes up to the total number of pixels in this batch
    # track_index (counter) goes up to "MAX_TRACKS_PER_PIXEL"
    # counter = segment index in pixels_tracks_signals collection, same as track_index ordering

    # size of signals
    itrk, ipix, itick = cuda.grid(3)

    # equivalent to num_backtrack.sum()
    total_backtracks = offset_backtrack[-1] + num_backtrack[-1]

    if itrk < signals.shape[0] and ipix < signals.shape[1]:

        pixel_index = pixel_index_map[itrk][ipix]
        # index into the jagged pixels_tracks_signals array for this pixel and tick
        # account track t0 in the backtracking
        base_idx = total_backtracks * (itick + track_t0[itrk]) + offset_backtrack[pixel_index]

        if pixel_index >= 0:
            counter = -99
            for track_idx in range(track_pixel_map[pixel_index].shape[0]):
                if int(track_pixel_map[pixel_index][track_idx]) == -1:
                    break
                if itrk == int(track_pixel_map[pixel_index][track_idx]):
                    counter = track_idx
                    if counter >= 0 and itick < signals.shape[2]:
                        if itick < pixels_signals.shape[1] and itick > -1:
                            # account track t0 here
                            cuda.atomic.add(pixels_signals,
                                            (pixel_index, itick + track_t0[itrk]),
                                            signals[itrk][ipix][itick])
                            cuda.atomic.add(pixels_tracks_signals,
                                            base_idx + counter,
                                            signals[itrk][ipix][itick])
                    break

            if counter < 0:
                # The overflow_flag is for both overflow (too many segments for backtracking) and underflow (no backtracking the pixel is considered too far from the segments)
                overflow_flag[pixel_index] = 1

@cuda.jit
def get_track_pixel_map(track_pixel_map, unique_pix, pixels):
    """
    This kernel fills a 2D array which contains, for each unique pixel,
    an array with the track indeces associated to that pixel.

    Args:
        track_pixel_map (:obj:`numpy.ndarray`): 2D array that will contain the
            association between the unique pixels array and the track indeces
        unique_pix (:obj:`numpy.ndarray`): 1D array containing the unique pixels
        pixels (:obj:`numpy.ndarray`): 2D array containing the pixels for each
            track.
    """
    # index of unique_pix array
    # although this function could get rid of some segments depending on MAX_TRACKS_PER_PIXEL
    # the index is with respect to the total number segments in the batch
    # so when it is translated to "segment_id", it is correct
    index = cuda.grid(1)
    if index >= unique_pix.shape[0]:
        return
    upix = unique_pix[index]

    for itrk in range(pixels.shape[0]):

        for ipix in range(pixels.shape[1]):
            pID = pixels[itrk][ipix]

            if upix == pID:

                imap = 0
                while imap < track_pixel_map.shape[1] and track_pixel_map[index][imap] != -1 and track_pixel_map[index][imap] != itrk:
                    imap += 1

                if imap < track_pixel_map.shape[1]:
                    track_pixel_map[index][imap] = itrk

@cuda.jit
def get_track_pixel_map2(track_pixel_map, unique_pix, pixels, distances):
    """
    This kernel fills a 2D array which contains, for each unique pixel,
    an array with the track indeces associated to that pixel.
    Summary of the different get_track_pixel_map
    get_track_pixel_map, fills track_pixel_map without distance ranking
    get_track_pixel_map3, fills track_pixel_map ranked by distances of unit pixel pitch
    """
    # index of unique_pix array
    index = cuda.grid(1)
    if index >= unique_pix.shape[0]:
        return
    upix = unique_pix[index]

    for target_dist in detector.NEIGHBORING_PIX_DIST:

        for itrk in range(pixels.shape[0]):

            for ipix in range(pixels.shape[1]):
                pID  = pixels[itrk][ipix]
                dist = distances[itrk][ipix]

                if (upix == pID):
                    if abs(dist - target_dist) < 1E-6:
                        imap = 0
                        #while imap < track_pixel_map.shape[1] and track_pixel_map[index][imap] != -1 and track_pixel_map[index][imap] != itrk:
                        while imap < track_pixel_map.shape[1]:
                            if track_pixel_map[index][imap] == itrk:
                                imap = -1
                                break
                            if track_pixel_map[index][imap] == -1:
                                break
                            else:
                                imap += 1

                        if (imap >= 0) and (imap < track_pixel_map.shape[1]):
                            track_pixel_map[index][imap] = itrk

                    break
