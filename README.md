# larnd-sim

![CI status](https://github.com/DUNE/larnd-sim/workflows/CI/badge.svg)
[![Documentation](https://img.shields.io/badge/docs-online-success)](https://dune.github.io/larnd-sim)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4582721.svg)](https://doi.org/10.5281/zenodo.4582721)

<img alt="larnd-sim" src="docs/logo.png" height="160" />

This software aims to simulate the light readout and the pixelated charge readout of a Liquid Argon Time Projection Chamber. It consists of a set of highly-parallelized algorithms implemented on the [CUDA architecture](https://developer.nvidia.com/cuda-toolkit) using [Numba](https://numba.pydata.org).

Software documentation is available [here](https://dune.github.io/larnd-sim/index.html). In addition, a paper about larnd-sim performance can be found here: [here](https://iopscience.iop.org/article/10.1088/1748-0221/18/04/P04034).

## Overview

The software takes as input a dataset containing segments of deposited energy in the detector, generated with a [Geant4](https://geant4.web.cern.ch) wrapper called [`edep-sim`](https://github.com/ClarkMcGrew/edep-sim). The output of `edep-sim` is in the [ROOT](https://root.cern) format and must be converted into [HDF5](https://www.hdfgroup.org/solutions/hdf5/) to be used by `larnd-sim`. For this purpose, we provide [`cli/dumpTree.py`](https://github.com/DUNE/larnd-sim/blob/develop/cli/dumpTree.py) for non-beam events and [`2x2_sim/run-convert2h5
/convert_edepsim_roottoh5.py`](https://github.com/DUNE/2x2_sim/blob/develop/run-convert2h5/convert_edepsim_roottoh5.py) (Note that this is in another repository, namely [`2x2_sim`](https://github.com/DUNE/2x2_sim/tree/develop)) for beam events. The two versions will be merged in future.

`larnd-sim` simulates both the scintillation light acquired by the light sensors and the charge induced on the pixels.

## Installation

The package can be installed in this way:

```bash
git clone https://github.com/DUNE/larnd-sim.git
cd larnd-sim
pip install .
```

which should take care of installing the required dependencies. `cupy` installation might take a long time. You can considerably speed up the process by pre-installing `cupy` precompiled binaries, available [here](https://docs.cupy.dev/en/stable/install.html#installing-cupy). The version will depend on the version of CUDA installed on your system. If you already have `cupy` installed in your environment which meets `larnd-sim`'s requirements, you can execute `export SKIP_CUPY_INSTALL=1` to skip cupy installation before running `pip install .`.

`larnd-sim` requires a CUDA-compatible GPU to function properly. To check if the GPU is setup properly and can talk to `larnd-sim` you can run:

```python
>>> from numba.cuda import is_available
>>> is_available()
```

## How to run a simulation

### Command line interface
To run the simulation, simply execute the following:

```bash
simulate_pixels.py (--config CONFIG_KEYWORD) --input_filename INPUT_FILENAME --output_filename OUTPUT_FILENAME
```

Note that this is a new feature developed during the 2x2 MiniRun5 era (2023-12). It may not be available or functional before [this commit](https://github.com/DUNE/larnd-sim/commit/da701c12e0feadc7c2f8d356b41d4dfcab193da0) in the `develop` branch.

Currently, CONFIG_KEYWORD supports `module0`, `2x2`, `2x2_mod2mod_variation`, `2x2_non_beam` and `ndlar`, which cover the common use of `larnd-sim`. They are set in `larndsim/config/config.yaml`. Other configurations can be developed per requestion, and we welcome your PR.

- `module0` is for simulating non-beam events in a single 2x2 style module setup (tuned for module0 cosmic data taking). Note that to apply this on other 2x2 single modules, small changes in charge or light detector configuration may be required. See **Configuration details** for further information. The detector position corresponds to `module0.gdml`.

- `2x2` is for simulating NuMI beam events in the 2x2 detector, four modules arranged on a 2x2 grid. In this configuration, all four modules are assumed to be identical which are module1- or module3-like. The detector position corresponds to [Merged2x2MINERvA_v4_withRock.gdml](https://github.com/DUNE/2x2_sim/blob/develop/geometry/Merged2x2MINERvA_v4/Merged2x2MINERvA_v4_withRock.gdml).

- `2x2_mod2od_variation` is for simulating NuMI beam events in the 2x2 detector with a more realistic charge and light arrangement which accounts for the hardware and setup differences in the four modules. The detector position corresponds to the same gdml for `2x2`.

- `2x2_non_beam` is for simulating non-beam events in the 2x2 detector, and all four modules are assumed to be the same as in `2x2`. The detector position corresponds to (???a 2x2 only gdml???), [Merged2x2MINERvA_v4_noRock.gdml](https://github.com/DUNE/2x2_sim/blob/develop/geometry/Merged2x2MINERvA_v4/Merged2x2MINERvA_v4_noRock.gdml) (2x2 + MINERvA) and [Merged2x2MINERvA_v4_withRock.gdml](https://github.com/DUNE/2x2_sim/blob/develop/geometry/Merged2x2MINERvA_v4/Merged2x2MINERvA_v4_withRock.gdml) (2x2 + MINERvA + MINOS Hall). The 2x2 location is consisten in these gdml's.

- `ndlar` is for simulating beam events in the DUNE ND-LAr detector. The beam properties are taken from the NuMI beam simulation for the moment. In the configuration, all the modules are considered to have same configuration. Note that the "light visibility look-up table" is missing and "light detector noise" still needs to be extracted into the appropriate format. (The detector position corresponds to `nd_hall_only_lar_TRUE_1.gdml`. To be confirmed.)

If no argument given for `--config`, the simulation will use the defualt configuration `2x2_mod2od_variation`.

Alternatively, the simulation can be run with explicit configurations listed in the command line such as (assuming it is run at the top level of the directory `larnd-sim/.`):

```bash
simulate_pixels.py \
--input_filename=INPUT_FOR_A_2x2_BEAM_EXAMPLE.h5 \
--output_filename=OUTPUT_FOR_A_2x2_BEAM_EXAMPLE.h5 \
--mod2mod_variation=False \
--pixel_layout=larndsim/pixel_layouts/multi_tile_layout-2.4.16.yaml\
--detector_properties=larndsim/detector_properties/2x2.yaml \
--response_file=larndsim/bin/response_44.npy \
--light_simulated=True \
--light_lut_filename=larndsim/bin/lightLUT.npz \
--light_det_noise_filename=larndsim/bin/light_noise_2x2_4mod_July2023.npy
```

Note that the default configuration is always active, so in order to control the configuration thoroughly and properly, please pass at least the list of arguments above.

You can also use a combination of the above two methods to configure the simulation. Taking the default configuration list and substitute some using the command line configuration. For example, here we simulate with four identical modules all with Module2-like LArPix geometry (3.8 mm pixel pitch etc.).

```bash
simulate_pixels.py --config 2x2 --pixel_layout=larndsim/pixel_layouts/multi_tile_layout-2.5.16.yaml\
--input_filename INPUT_FILENAME --output_filename OUTPUT_FILENAME
```
### Input dataset

The input array can be created by converting [edep-sim](https://github.com/ClarkMcGrew/edep-sim) ROOT output files using the `cli/dumpTree.py` script (which is independent from the rest of the software and requires ROOT and Geant4 to be installed).

This script produces a bi-dimensional structured array saved in the HDF5 format, which is used as input for the simulation of the pixel response.

`examples/lbnfSpillLAr.edep.h5` is an example of the input dataset. Otherwise you can find some example of 2x2 NuMI simulation input files on [the 2x2_sim wiki](https://github.com/DUNE/2x2_sim/wiki).

### Output
Detailed file data definition can be found in the [2x2_sim wiki](https://github.com/DUNE/2x2_sim/wiki/File-data-definitions).

Briefly, the larnd-sim output includes generator truth information, edep-sim/geant4 truth information, simulated charge detector output, charge backtracking, detector propagated light truth information, simulated light detector output and light backtracking.

Generator truth information includes `mc_hdr` and `mc_stack` which are reserved for neutrino interaction records. `mc_hdr` is the log for neutrino interactions, and `mc_stack` is the log for final-state particles from the neutrino interactions. This part is copied directly from the root converted h5 input file. If the upstream simulation does not run neutrino generator, rather cosmic or particle bomb simulation instead, then this information will not be presented in the output.

edep-sim/geant4 truth information contains `vertices` (interaction-level or equivalent), `trajectories` (particle-level) and `segments` (segment of energy depositions). Primary trajectories should overlap `mc_stack` if it originates from neutrino generators. `segments` is the essential input to the simulation. This part should exist in most of the output, and it is a direct copy of the corresponding part in the input with possible minor extensions such as event time (vertices['t_event']).

The charged readout output is stored in the datasets described in the [LArPix HDF5 documentation](https://larpix-control.readthedocs.io/en/stable/api/format/hdf5format.html), plus a dataset `tracks` containing the _true_ energy depositions in the detector, and a dataset `mc_packets_assn`, which has a list of indeces corresponding to the true energy deposition associated to each packet.

The light output is stored in the `light_dat`, `light_trig`, and `light_wvfm` datasets, containing the number of photons, the optical sensors triggers and the optical sensors waveforms, respectively.
