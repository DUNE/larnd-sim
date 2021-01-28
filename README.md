# larnd-sim ![CI status](https://github.com/DUNE/larnd-sim/workflows/CI/badge.svg)


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
quenching.quench[blockspergrid,threadsperblock](segments, consts.box)
drifting.drift[blockspergrid,threadsperblock](segments)
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
                                                            n_pixels_list,
                                                            radius)
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
detsim.time_intervals[blockspergrid,threadsperblock](track_starts,
                                                     max_length,
                                                     event_id_map,
                                                     segments)
```

Thus, we join them using `sum_pixel_signals`:

```python
from larndsim import detsim
...
detsim.sum_pixel_signals[blockspergrid,threadsperblock](pixels_signals, 
                                                        signals, 
                                                        track_starts, 
                                                        pixel_index_map)
```

### Electronics simulation

Once we have the induced current for each active pixel in the detector we can apply our electronics simulation, which will calculate the ADC values for each pixel:

```python
from larndsim import fee
from numba.cuda.random import create_xoroshiro128p_states
...

rng_states = create_xoroshiro128p_states(TPB * BPG, seed=0)
fee.get_adc_values[BPG,TPB](pixels_signals, 
                            time_ticks, 
                            integral_list, 
                            adc_ticks_list,
                            0,
                            rng_states)
```

where the random states `rng_states` are neede for the noise simulation.
The final output can be exported to the [LArPix HDF5 format](https://larpix-control.readthedocs.io/en/stable/api/format/hdf5format.html):
```python
fee.export_to_hdf5(adc_list, adc_ticks_list, unique_pix, "example.h5")
```
