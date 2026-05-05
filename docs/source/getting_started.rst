.. _getting_started:

Getting Started
===============

.. contents:: On this page
   :local:
   :depth: 2

larnd-sim simulates both the **light readout** and the **pixelated charge
readout** of a Liquid Argon Time Projection Chamber (LArTPC).  It consists
of a set of highly-parallelised algorithms implemented on the
`CUDA architecture <https://developer.nvidia.com/cuda-toolkit>`_ using
`Numba <https://numba.pydata.org>`_ and `CuPy <https://cupy.dev>`_, with
a driver script written in Python.

A paper describing the architecture and performance of larnd-sim is available
`here <https://iopscience.iop.org/article/10.1088/1748-0221/18/04/P04034>`_.

Overview
--------

larnd-sim takes as input a dataset of energy-deposition segments in the
detector — typically generated with
`edep-sim <https://github.com/ClarkMcGrew/edep-sim>`_, a Geant4 wrapper.
The driver script pre-processes the data, runs each kernel in the pipeline
in sequence, and saves the results.  The output contains simulated charge
data packets from the front-end electronics, simulated light waveforms, and
Monte Carlo truth information.

Simulation pipeline
~~~~~~~~~~~~~~~~~~~

.. code-block::

   Geant4/edep-sim (.root)
        │
        ▼  cli/dumpTree.py
   HDF5 segments file (.hdf5)
        │
        ▼  simulate_pixels.py
        │
        ├─► Quenching       (recombination model: Box or Birks)
        ├─► Drifting        (electron transport, diffusion, lifetime)
        ├─► Charge simulation
        └───► Induced current (pixel response using the Shockley-Ramo theorem)
        └───► Front-end elec. (electronics readout, e.g. ADC, noise, etc.)
        ├─► Light simulation
        └───► Photocurrent  (visibility look-up table, scintillation)
        └───► Optical elec. (electronics readout, e.g. SiPM response, noise, etc.)
        │
        ▼
   HDF5 output (charge data packets + light waveforms + Monte Carlo truth)

Installation
------------

Clone the repository and install via ``pip``:

.. code-block:: bash

   git clone https://github.com/DUNE/larnd-sim.git
   cd larnd-sim
   pip install .

This installs all required dependencies.  The GPU libraries — in particular
``cupy`` — may take a while to compile from source.  Installation can be
considerably sped up by pre-installing a pre-compiled ``cupy`` binary that
matches your CUDA version; see the
`CuPy installation guide <https://docs.cupy.dev/en/stable/install.html>`_
for the full list of available packages.

If you already have a compatible ``cupy`` in your environment, you can skip
its reinstallation:

.. code-block:: bash

   export SKIP_CUPY_INSTALL=1
   pip install .

.. note::
   larnd-sim requires a **CUDA-compatible NVIDIA GPU**.  To verify that
   your GPU is accessible:

   .. code-block:: python

      >>> from numba.cuda import is_available
      >>> is_available()   # should return True

Running a Simulation
--------------------

Command-line interface
~~~~~~~~~~~~~~~~~~~~~~

The simulation is run using ``cli/simulate_pixels.py``:

.. code-block:: bash

   simulate_pixels.py (--config CONFIG_KEYWORD) \
       --input_filename INPUT_FILENAME \
       --output_filename OUTPUT_FILENAME

If ``--config`` is not provided, the default configuration is ``2x2``.
Valid keywords are defined in ``larndsim/config/config.yaml``; the most
common ones are described below.

.. list-table:: Built-in configuration keywords
   :header-rows: 1
   :widths: 20 80

   * - Keyword
     - Description
   * - ``module0``
     - Non-beam events in a single 2×2-style module, tuned for Module 0
       cosmic data-taking.  Small adjustments to charge or light detector
       configuration may be needed for other single-module setups.
       Detector geometry: ``module0.gdml``.
   * - ``2x2``
     - NuMI beam events in the four-module 2×2 detector.  Module-by-module
       hardware variation is enabled, including a different pixel pitch for
       Module 2.  Detector geometry:
       ``Merged2x2MINERvA_v4_withRock.gdml``.
   * - ``2x2_no_modvar``
     - NuMI beam events in the 2×2 detector with all modules using an
       identical hardware/layout configuration.  Same detector position as
       ``2x2``.
   * - ``fsd``
     - Cosmic-ray events in the Full Scale Demonstrator (FSD) prototype —
       a single module with full ND-LAr dimensions and a later pixel
       electronics version than the 2×2.  Detector geometry:
       ``fsd_with_cryostat.gdml``.
   * - ``ndlar``
     - Beam events in the full DUNE ND-LAr detector (NuMI beam properties).
       All modules are assumed to have the same configuration.

Explicit configuration
~~~~~~~~~~~~~~~~~~~~~~

Individual options can be passed directly on the command line, either
instead of or in combination with a ``--config`` keyword.  Options not
specified fall back to their defaults, so pass at least the full set below
for a well-controlled 2×2 beam simulation:

.. code-block:: bash

   simulate_pixels.py \
       --input_filename=INPUT_FOR_A_2x2_BEAM_EXAMPLE.h5 \
       --output_filename=OUTPUT_FOR_A_2x2_BEAM_EXAMPLE.h5 \
       --mod2mod_variation=False \
       --pixel_layout=larndsim/pixel_layouts/multi_tile_layout-2.4.16.yaml \
       --detector_properties=larndsim/detector_properties/2x2.yaml \
       --response_file=larndsim/bin/response_44.npy \
       --light_simulated=True \
       --light_lut_filename=larndsim/bin/lightLUT.npz \
       --light_det_noise_filename=larndsim/bin/light_noise_2x2_4mod_July2023.npy

To use a keyword as a base and override only one option — for example, to
swap in a different pixel layout:

.. code-block:: bash

   simulate_pixels.py \
       --config 2x2 \
       --pixel_layout=larndsim/pixel_layouts/multi_tile_layout-2.5.16.yaml \
       --input_filename INPUT_FILENAME \
       --output_filename OUTPUT_FILENAME

Input Dataset
-------------

The input array is typically created by converting
`edep-sim <https://github.com/ClarkMcGrew/edep-sim>`_ ROOT output with the
``cli/dumpTree.py`` script.  This script parses the ROOT output and stores a
subset of the data as NumPy structured arrays in an HDF5 file.  It is
independent from the rest of larnd-sim and requires ROOT and Geant4 to be
installed.

Other sources of energy depositions can be used provided they match the
expected input format, but this repository only supplies a conversion script
for edep-sim ROOT files.

Output Dataset
--------------

A detailed file data definition is maintained in the
`2x2_sim wiki <https://github.com/DUNE/2x2_sim/wiki/File-data-definitions>`_.
The larnd-sim output includes generator truth, edep-sim/Geant4 truth,
simulated charge detector output, charge backtracking, detector-propagated
light truth, simulated light detector output, and light backtracking.

Generator truth (``mc_hdr``, ``mc_stack``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These datasets are reserved for neutrino interaction records and are
copied directly from the converted HDF5 input.  ``mc_hdr`` logs neutrino
interactions; ``mc_stack`` logs the initial- and final-state particles.
They are absent when the upstream simulation uses cosmic or particle-gun
input rather than a neutrino generator.

edep-sim / Geant4 truth (``vertices``, ``trajectories``, ``segments``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``vertices`` holds interaction-level truth, ``trajectories`` holds
particle-level truth, and ``segments`` holds energy-deposition segments.
Primary trajectories overlap with the final-state particles in ``mc_stack``
for neutrino-generator input.  ``segments`` is the essential input to
larnd-sim and is always present in the output; it is a direct copy of the
corresponding input data, possibly extended with quantities such as event
time (``vertices['t_event']``).

Charge output (``packets``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simulated charge packets compatible with converted raw LArPix data.  Each
data packet encodes a point-like charge readout in time and position on the
anode plane, including readout charge, position, and time.  Trigger,
timestamp, and sync packets are also present.  See the
`LArPix HDF5 documentation <https://larpix-control.readthedocs.io/en/stable/api/format/hdf5format.html>`_
for the full packet format.

Charge backtracking (``mc_packets_assn``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Aligned with ``packets`` (same length, shared indices).  Each entry records
the contributing ``segments`` for that packet via ``event_ids``,
``segment_ids``, and ``fraction``.  Only data packets carry meaningful
backtracking; other packet types have the fields present but unfilled.

``event_ids`` has length one and contains the true event ID.
``segment_ids`` and ``fraction`` have length ``fee.ASSOCIATION_COUNT_TO_STORE``
(contributions sorted by fractional contribution, descending).  The total
number of contributions considered in the fraction calculation is set by
``fee.MAX_ADC_VALUES``; it is recommended to keep
``fee.MAX_ADC_VALUES >= fee.ASSOCIATION_COUNT_TO_STORE``.

Light truth (``light_dat``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detector-propagated light truth per edep-sim segment.  When module-by-module
variation is enabled the data are split per module as
``light_dat/light_dat_module{Z}``; otherwise they are stored under
``light_dat/light_dat_allmodules``.  Shape:
``(#segments, #light readout channels)``.  Each entry provides
``n_photons_det`` (photons reaching the SiPM) and ``t0_det`` (first-photon
arrival time).

Light output (``light_trig``, ``light_wvfm``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two trigger modes are supported, selected by ``light.LIGHT_TRIG_MODE``:

- ``0`` — threshold trigger
- ``1`` — beam trigger (activates all light readout channels)

``light_trig`` (shape: ``(#triggers,)``) records ``op_channel``,
``ts_s`` (trigger time in seconds), and ``ts_sync`` (trigger time in
LArPix clock ticks, accounting for PPS synchronisation and clock
roll-over).  In beam-trigger mode the number of light triggers equals the
number of simulated beam spills, so not every trigger necessarily
corresponds to a meaningful charge or light signal.

``light_wvfm`` (shape: ``(#triggers, #channels, #samples)``) contains
digitised ADC waveforms.  The number of samples is determined by
``light.LIGHT_TRIG_WINDOW`` and ``light.LIGHT_DIGIT_SAMPLE_SPACING``.

.. note::
   For 2×2 Module 1/2/3 cosmic data-taking configurations, divide waveform
   values by 4 to obtain ADC counts proportional to the light signal.

Light backtracking (``light_wvfm_mc``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Present when ``light.MAX_MC_TRUTH_IDS > 0``.  Same shape as
``light_wvfm``; each time sample stores ``segment_ids`` and ``pe_current``
arrays of length ``light.MAX_MC_TRUTH_IDS``.

.. note::
   Light readout channel ordering follows channel ID in geometrical order,
   sequentially from bottom to top, TPC by TPC.

Configuration Files
-------------------

See :ref:`configuration` for full details on each file type.  A brief
summary:

- **Pixel layout** (``larndsim/pixel_layouts/``) — channel mapping, pixel
  pitch, tile positions and orientations, TPC geometry.
- **Detector properties** (``larndsim/detector_properties/``) — electric
  field, electron lifetime, response sampling, per-module variation
  settings.  Loaded into :mod:`larndsim.consts.detector` and
  :mod:`larndsim.consts.light`.
- **Simulation properties** (``larndsim/simulation_properties/``) — batch
  sizes and other algorithmic parameters.  Loaded into
  :mod:`larndsim.consts.sim`.
- **Pixel response file** — near-field charge induction look-up table
  (FEM-based).  ``response_44.npy`` for 4.4 mm pitch (LArPix v2a);
  ``response_38.npy`` for 3.8 mm pitch (LArPix v2b/v3).
- **Light LUT** — photon visibility from TPC volumes to SiPM channels.
  An example is bundled at ``larndsim/bin/lightLUT.npz``; high-resolution
  production LUTs are on the
  `NERSC web portal <https://portal.nersc.gov/project/dune/data/2x2/simulation/larndsim_data/light_LUT/>`_.
- **Light detector noise** — per-channel noise profile extracted from
  single-module cosmic runs (e.g.
  ``larndsim/bin/light_noise_2x2_4mod_July2023.npy``).
- **Bad channels** *(optional)* — deactivates flagged LArPix channels.
- **Pixel thresholds** *(optional)* — per-channel discriminator threshold;
  defaults to ``detector.DISCRIMINATION_THRESHOLD``.
- **Pixel pedestals** *(optional)* — per-channel pedestal value; defaults
  to ``detector.V_PEDESTAL``.
- **Pixel gains** *(optional)* — per-channel gain factor; defaults to
  ``detector.GAIN``.
