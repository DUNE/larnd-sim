# larnd-sim [![Build Status](https://travis-ci.com/soleti/larnd-sim.svg?token=ySJqJBz9YiZ99571Y3et&branch=master)](https://travis-ci.com/soleti/larnd-sim)

<img alt="larnd-sim" src="docs/logo.png" height="160">

This software aims to simulate a pixelated Liquid Argon Time Projection Chamber. It consists of a set of highly-parallelized algorithms implemented on the CUDA architecture.

## Overview

The framework takes as input a 2D array containing the necessary information for each simulated segment in the detector (e.g. starting point, energy deposition) and produces a simulated electronics signal for each affected pixel.
It is divided into two main parts: the first one simulates the drifting of the tracks in the detector and the quenching of the deposited charge, and the second one simulates the electronics response of the pixels placed at the anode.
A full example is available in `examples/Pixel induced current.ipynb`.

## How to run a simulation

### Input dataset

The simulation of the pixel response takes as input a bi-dimensional `numpy` array containing the information for each track segment that deposited energy in the TPC. The indeces that correspond to each track segment attribute are specified in `larndsim/indeces.py`.

### Quenching and drifting stage

The particles that interact in the TPC ionize the argon atoms. Some of the resulting electrons will immediately recombine with the atoms. This effect is simulated in the `quenching` module.

The remaining electrons travel towards the anode and their spatial distribution is affected by longitudinal and transverse diffusion. The presence of impurities reduces the amount of electrons that reach the anode. These effects are simulated in the `drifting` module.

These two modules modify in place the input array.

```python
from larndsim import quenching, drifting

threadsperblock = 256
blockspergrid = ceil(tracks.shape[0] / threadsperblock)
quenching.quench[blockspergrid,threadsperblock](tracks, consts.box)
drifting.drift[blockspergrid,threadsperblock](tracks)
```

### Pixel simulation stage

Once we have calculated the number and the position of the electrons reaching the anode, we can calculate the current induced on each pixel.
First, we find the pixels interesected by the projection of each track segment on the anode plane using the [Bresenham's line algorithm](https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm) implented in the `pixels_from_track` module. Due to diffusion, we consider also the neighboring pixels.

```python
from larndsim import pixels_from_track
...
pixels_from_track.get_pixels[blockspergrid,threadsperblock](segments,
                                                            active_pixels,
                                                            neighboring_pixels,
                                                            n_pixels_list)
```

Finally, we calculate the current induced on each pixel using the `tracks_current` function in the `detsim` module. The induced current is stored in the `signals` array, which is a three-dimensional array, where the dimensions correspond to the track segment, the pixel, and the time tick, respectively: `signals[0][1][2]` will contain the current induced by the track `0`, for the pixel `1`, at the time tick `2`.

```python
from larndsim import detsim
...
detsim.tracks_current[blockspergrid,threadsperblock](signals,
                                                      neighboring_pixels,
                                                      segments)
```

### Accessing the signals

The three-dimensional array can contain more than one signal for each pixel at different times. If we want to plot the full induced signal on the pixel, we need to join the signals corresponding to the same pixel. First, we find the start time of each signal with `time_intervals`:

```python
from larndsim import detsim
...
max_length = np.array([0])
track_starts = np.empty(segments.shape[0])
detsim.time_intervals[blockspergrid,threadsperblock](track_starts,
                                                      max_length,
                                                      segments)
```

Thus, we join them using `join_pixel_signals`:

```python
from larndsim import detsim
...
joined_pixels = detsim.join_pixel_signals(signals,
                                          neighboring_pixels,
                                          track_starts,
                                          max_length[0])
```
