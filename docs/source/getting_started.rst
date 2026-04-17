.. _getting_started:

Getting Started
===============

.. contents:: On this page
   :local:
   :depth: 2

Prerequisites
-------------

larnd-sim requires a **CUDA-compatible NVIDIA GPU**.  Before installing,
verify that your GPU is accessible:

.. code-block:: python

   from numba.cuda import is_available
   is_available()  # should return True

Installation
------------

Clone the repository and install via ``pip``:

.. code-block:: bash

   git clone https://github.com/DUNE/larnd-sim.git
   cd larnd-sim
   pip install .

.. note::

   ``cupy`` installation can be slow when compiled from source.  For a
   faster setup, install a pre-compiled binary that matches your CUDA
   version *before* running ``pip install .``:

   .. code-block:: bash

      # Example for CUDA 12.x
      pip install cupy-cuda12x
      # Then install larnd-sim (skip cupy re-install)
      SKIP_CUPY_INSTALL=1 pip install .

   See the `CuPy installation guide <https://docs.cupy.dev/en/stable/install.html>`_
   for the full list of pre-built packages.

Quick-Start: Running a Simulation
-----------------------------------

Command-line interface
~~~~~~~~~~~~~~~~~~~~~~

The primary entry point is ``simulate_pixels.py``.  A named
*configuration keyword* selects a pre-bundled detector setup:

.. code-block:: bash

   simulate_pixels.py \
       --config 2x2 \
       --input_filename  INPUT.h5 \
       --output_filename OUTPUT.h5

Available configuration keywords
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+---------------------------+-----------------------------------------------+
| Keyword                   | Description                                   |
+===========================+===============================================+
| ``module0``               | Single 2×2-style module, non-beam cosmics     |
+---------------------------+-----------------------------------------------+
| ``2x2``                   | Four-module 2×2 array, NuMI beam events       |
+---------------------------+-----------------------------------------------+
| ``2x2_mod2mod_variation`` | 2×2 with realistic per-module hardware        |
|                           | differences                                   |
+---------------------------+-----------------------------------------------+
| ``2x2_non_beam``          | Four-module 2×2, non-beam events              |
+---------------------------+-----------------------------------------------+
| ``ndlar``                 | Full DUNE ND-LAr detector, beam events        |
+---------------------------+-----------------------------------------------+

Explicit configuration
~~~~~~~~~~~~~~~~~~~~~~

All configuration options can also be provided directly on the command line,
overriding (or replacing) the keyword defaults:

.. code-block:: bash

   simulate_pixels.py \
       --input_filename  INPUT.h5 \
       --output_filename OUTPUT.h5 \
       --mod2mod_variation False \
       --pixel_layout    larndsim/pixel_layouts/multi_tile_layout-2.4.16.yaml \
       --detector_properties larndsim/detector_properties/2x2.yaml \
       --response_file   larndsim/bin/response_44.npy \
       --light_simulated True \
       --light_lut_filename larndsim/bin/lightLUT.npz \
       --light_det_noise_filename larndsim/bin/light_noise_2x2_4mod_July2023.npy

You can also *combine* a keyword with explicit overrides.  The following
runs the ``2x2`` defaults but swaps in a v2b pixel layout:

.. code-block:: bash

   simulate_pixels.py \
       --config 2x2 \
       --pixel_layout larndsim/pixel_layouts/multi_tile_layout-2.5.16.yaml \
       --input_filename INPUT.h5 --output_filename OUTPUT.h5

Preparing input data
--------------------

larnd-sim consumes HDF5 files produced by
`edep-sim <https://github.com/ClarkMcGrew/edep-sim>`_.
The ROOT output of edep-sim must first be converted:

* **Non-beam / single-module events** — use the bundled script:

  .. code-block:: bash

     python cli/dumpTree.py INPUT.root OUTPUT.h5

* **NuMI beam events** (2×2) — use
  `convert_edepsim_roottoh5.py <https://github.com/DUNE/2x2_sim>`_
  from the ``2x2_sim`` repository.

An example input file is provided at ``examples/lbnfSpillLAr.edep.h5``.

Simulation pipeline overview
-----------------------------

.. code-block::

   edep-sim (.root)
        │
        ▼  cli/dumpTree.py
   HDF5 segments file
        │
        ▼  simulate_pixels.py
        │
        ├─► Quenching       (recombination model: Box or Birks)
        ├─► Drifting        (electron transport, diffusion, lifetime)
        ├─► Induced charge  (near-field pixel response look-up table)
        ├─► Front-end elec. (discriminator, ADC, LArPix packet format)
        └─► Light sim       (LUT visibility, scintillation, SiPM response)
        │
        ▼
   HDF5 output (packets + light waveforms + MC truth)
