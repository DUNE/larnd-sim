.. _configuration:

Configuration Reference
=======================

.. contents:: On this page
   :local:
   :depth: 2

larnd-sim is controlled by a set of YAML configuration files and optional
look-up-table (LUT) binary files.  The sections below describe each file
type, its location inside the repository, and the key fields it contains.

Pixel Layout
------------

**Directory:** ``larndsim/pixel_layouts/``

A pixel layout file (YAML) describes the physical arrangement of LArPix
channels across all tiles and anodes.  It provides:

- Channel mapping (chip → pixel ID)
- Pixel pitch
- Tile positions and orientations
- TPC geometry

The files are produced by
`larpix-geometry <https://github.com/larpix/larpix-geometry>`_.

+---------------------------------------+----------------------------------------+
| File                                  | Description                            |
+=======================================+========================================+
| ``multi_tile_layout-2.3.16.yaml``     | Module 0 (v2a, realistic dead channels)|
+---------------------------------------+----------------------------------------+
| ``multi_tile_layout-2.4.16.yaml``     | Generic LArPix v2a, 2×2 module         |
+---------------------------------------+----------------------------------------+
| ``multi_tile_layout-2.5.16.yaml``     | Generic LArPix v2b, 2×2 module         |
+---------------------------------------+----------------------------------------+
| ``multi_tile_layout-3.0.40.yaml``     | LArPix v2b/v3, ND-LAr module           |
+---------------------------------------+----------------------------------------+

.. note::
   The suffix ``.16`` / ``.40`` indicates the number of tiles per anode.
   For ``2.4.16`` and ``2.5.16`` all channels are activated; ``2.3.16``
   includes realistic dead-channel masking from Module 0 data taking.

Detector Properties
-------------------

**Directory:** ``larndsim/detector_properties/``

A detector properties YAML file sets physical detector constants that are
loaded into :mod:`larndsim.consts.detector` and :mod:`larndsim.consts.light`.
Key fields include:

- Electric field (``e_field``)
- Electron lifetime
- Pixel response sampling / bin sizes
- Light-related constants (trigger windows, SiPM response model, etc.)

Per-module variations (for ``2x2_mod2mod_variation``) are also expressed
here.

Simulation Properties
---------------------

**Directory:** ``larndsim/simulation_properties/``

Controls algorithmic parameters loaded into :mod:`larndsim.consts.sim`:

- Batch sizes (``batch_size``, ``event_batch_size``, ``write_batch_size``)
- Number of sampled points per segment
- MC truth tracking settings (``MAX_MC_TRUTH_IDS``)

Response File
-------------

A NumPy ``.npy`` binary containing the **near-field charge induction
look-up table**.  The table is produced from finite-element-method (FEM)
electric-field simulations.

**Directory:** ``larndsim/bin/``

+---------------------+-----------------------------------------------+
| File                | Pixel pitch / ASIC version                    |
+=====================+===============================================+
| ``response_44.npy`` | 4.4 mm pitch (LArPix v2a)                     |
+---------------------+-----------------------------------------------+
| ``response_38.npy`` | 3.8 mm pitch (LArPix v2b / v3)                |
+---------------------+-----------------------------------------------+

.. warning::
   The ``RESPONSE_BIN_SIZE`` and ``RESPONSE_SAMPLING`` constants in the
   detector properties YAML must match the response file selected.

Light Look-Up Table (LUT)
--------------------------

The light LUT encodes the **photon visibility** from each voxel in the
active TPC volume to each SiPM channel.  It is produced with a GEANT4
optical simulation using
`ArgonCubeLUTSim <https://github.com/DUNE/ArgonCubeLUTSim>`_.

A 2×2 example LUT is shipped at ``larndsim/bin/lightLUT.npz``.  Higher
resolution LUTs (required for production-quality simulations) are hosted on
`NERSC <https://portal.nersc.gov/project/dune/data/2x2/simulation/larndsim_data/light_LUT/>`_.

Light Detector Noise
---------------------

An ``.npy`` file containing the per-channel FFT noise spectrum extracted
from single-module cosmic-ray runs.

**Example:** ``larndsim/bin/light_noise_2x2_4mod_July2023.npy``

Optional Inputs
---------------

Bad channels
~~~~~~~~~~~~

A JSON or YAML file listing LArPix chip keys whose channels should be
deactivated in the simulation.  If omitted, all channels are treated as
active.

Pixel thresholds
~~~~~~~~~~~~~~~~

Provides a channel-by-channel discriminator threshold.  If omitted, every
channel uses the global ``fee.DISCRIMINATION_THRESHOLD`` value.

Pixel gains
~~~~~~~~~~~

Provides a channel-by-channel front-end gain factor.  If omitted, every
channel uses the global ``fee.GAIN`` value.

Configuration YAML (keyword presets)
--------------------------------------

The named keyword configurations (``module0``, ``2x2``, …) are defined in
``larndsim/config/config.yaml``.  Each entry maps a keyword to a complete
set of the file paths described above.  You can add custom entries to this
file or override individual fields via the command-line interface.
