.. _api-consts:

larndsim.consts subpackage
===========================

The ``larndsim.consts`` subpackage holds all global simulation constants.
Constants are populated at run-time by calling
:func:`larndsim.consts.load_properties` (or the individual
``set_*_properties`` helpers) with YAML configuration files.

.. contents:: Submodules
   :local:
   :depth: 1

.. toctree::
   :maxdepth: 1
   :caption: Groups of constants

   consts/detector
   consts/physics
   consts/light
   consts/sim

----

Top-level loader
----------------

.. automodule:: larndsim.consts
   :members:
   :undoc-members:
   :show-inheritance:
