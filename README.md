# larnd-sim

![CI status](https://github.com/DUNE/larnd-sim/workflows/CI/badge.svg)
[![Documentation](https://img.shields.io/badge/docs-online-success)](https://dune.github.io/larnd-sim)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4582721.svg)](https://doi.org/10.5281/zenodo.4582721)

<img alt="larnd-sim" src="docs/logo.png" height="160" />

This software aims to simulate a pixelated Liquid Argon Time Projection Chamber using [LArPix](https://arxiv.org/abs/1808.02969). It consists of a set of highly-parallelized algorithms implemented on the [CUDA architecture](https://developer.nvidia.com/cuda-toolkit).

## Overview

The software takes as input an array containing the necessary truth information for each simulated segment of deposited energy in the detector (e.g. starting point, amount of energy) and produces a list of packets with an ADC count and timestamp in the [LArPix HDF5 format](https://larpix-control.readthedocs.io/en/stable/api/format/hdf5format.html).

## Installation

The package can be installed in this way:

```bash
git clone https://github.com/DUNE/larnd-sim.git
cd larnd-sim
pip install .
```

which should take care of installing the required dependencies. `cupy` installation might take a long time. You can considerably speed up the process by pre-installing `cupy` precompiled binaries, available [here](https://docs.cupy.dev/en/stable/install.html#installing-cupy). The version will depend on the version of CUDA installed on your system.

If you are a developer you might want to install it in _editable mode_, so you can make changes to the code without having to re-install the package every time:

```bash
pip install -e .
```

`larnd-sim` requires a GPU to function properly. To check if the GPU is setup properly and can talk to `larnd-sim` you can run:

```python
>>> from numba.cuda import is_available
>>> is_available()
```

## How to run a simulation

### Input dataset

The input array can be created by converting [edep-sim](https://github.com/ClarkMcGrew/edep-sim) ROOT output files using the `cli/dumpTree.py` script (which is independent from the rest of the software and requires ROOT and Geant4 to be installed).

This script produces a bi-dimensional structured array saved in the HDF5 format, which is used as input for the simulation of the pixel response.

### Command line interface

We provide a command-line interface available at `cli/simulate_pixels.py`, which can run as:

```bash
simulate_pixels.py \
--input_filename=examples/lbnfSpillLAr.edep.h5 \
--detector_properties=larndsim/detector_properties/ndlar-module.yaml \
--pixel_layout=larndsim/pixel_layouts/multi_tile_layout-3.0.40.yaml \
--output_filename=lbnfSpillLAr.larndsim.h5 \
--response_file=larndsim/bin/response_38.npy
```

The `response_38.npy` is a file containing an array of induced currents for several $(x,y)$ positions on a pixel with a 38 mm pitch. It is calculated externally to `larnd-sim`. Two versions, one with 44 mm pitch and one with 38 mm pitch, are available in the `larndsim/bin` directory.

The output file will contain the datasets described in the [LArPix HDF5 documentation](https://larpix-control.readthedocs.io/en/stable/api/format/hdf5format.html), plus a dataset `tracks` containing the _true_ energy depositions in the detector, and a dataset `mc_packets_assn`, which has a list of indeces corresponding to the true energy deposition associated to each packet.
