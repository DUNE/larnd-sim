# larnd-sim [![Build Status](https://travis-ci.com/soleti/larnd-sim.svg?token=ySJqJBz9YiZ99571Y3et&branch=master)](https://travis-ci.com/soleti/larnd-sim)
<img alt="larnd-sim" src="https://i.imgur.com/DnIuxpM.png" height="140">

This software aims to simulate a pixelated Liquid Argon Time Projection Chamber. It consists of a set of highly-parallelized algorithms implemented on the CUDA architecture.

### Overview

The framework takes as input a 2D array containing the necessary information for each simulated segment in the detector (e.g. starting point, energy deposition) and produces a simulated electronics signal for each affected pixel.
It is divided into two main parts: the first one simulates the drifting of the tracks in the detector and the quenching of the deposited charge, and the second one simulated the electronics response of the pixels placed at the anode.


### Example

#### Quenching and drifting stage

```python
from larndsim import quenching, drifting
...
threadsperblock = 128
blockspergrid = ceil(segments.shape[0] / threadsperblock)
quenching.GPU_Quench[threadsperblock,blockspergrid](segments, consts.box)
drifting.GPU_Drift[threadsperblock,blockspergrid](segments)
```

#### Pixel simulation stage
```python
from larndsim import pixels_from_track, detsim

pixels_from_track.get_pixels[threadsperblock,blockspergrid](segments, 
                                                            active_pixels, 
                                                            neighboring_pixels, 
                                                            n_pixels_list)
detsim.tracks_current[threadsperblock, blockspergrid](signals, 
                                                      neighboring_pixels, 
                                                      segments, 
                                                      slice_size)
```
