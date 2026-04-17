.. _api-cli:

Command-line utilities
======================

.. contents:: Scripts
   :local:
   :depth: 1

----

simulate\_pixels.py
--------------------

The main simulation entry point.  Run with ``--help`` for the full option
list:

.. code-block:: bash

   simulate_pixels.py --help

Key options:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``--config``
     - Named configuration keyword (``module0``, ``2x2``, ``ndlar``, …)
   * - ``--input_filename``
     - Path to HDF5 input file (edep-sim output)
   * - ``--output_filename``
     - Path for the HDF5 output file
   * - ``--pixel_layout``
     - Path to pixel layout YAML
   * - ``--detector_properties``
     - Path to detector properties YAML
   * - ``--response_file``
     - Path to charge response ``.npy`` look-up table
   * - ``--light_simulated``
     - ``True`` / ``False`` — enable light simulation
   * - ``--light_lut_filename``
     - Path to light LUT ``.npz`` file
   * - ``--light_det_noise_filename``
     - Path to SiPM noise spectrum ``.npy`` file
   * - ``--mod2mod_variation``
     - ``True`` / ``False`` — enable per-module hardware variation

----

dumpTree.py
-----------

Converts the ROOT output of ``edep-sim`` into the HDF5 format expected by
``simulate_pixels.py``.  Requires ROOT and Geant4 to be installed; it is
independent from the rest of the larnd-sim package.

.. automodule:: cli.dumpTree
   :members:
   :undoc-members:
   :show-inheritance:

----

list\_config\_keys.py
----------------------

Prints all configuration keywords defined in
``larndsim/config/config.yaml``:

.. code-block:: bash

   list_config_keys.py

----

diff\_files.py
---------------

Utility for comparing two larnd-sim HDF5 output files field-by-field.
Useful for regression testing after code changes.

----

sort\_packets.py
-----------------

Sorts the ``packets`` dataset of a larnd-sim output file by timestamp.
Some downstream tools assume time-ordered packets.
