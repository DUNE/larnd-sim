.. larnd-sim documentation master file, created by
   sphinx-quickstart on Thu Aug  6 15:33:22 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to larnd-sim's documentation!
=====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
This software aims to simulate a pixelated Liquid Argon Time Projection Chamber. It consists of a set of highly-parallelized algorithms implemented on the CUDA architecture.

The software takes as input an array containing the necessary truth iformation for each simulated segment of deposited energy in the detector (e.g. starting point, amount of energy) and produces a list of packets with an ADC count and timestamp in the `LArPix HDF5 format <https://larpix-control.readthedocs.io/en/stable/api/format/hdf5format.html>`_.

Quenching stage
###############

.. automodule:: larndsim.quenching
    :members:

Drifting stage
###############

.. automodule:: larndsim.drifting
   :members:

Detector simulation stage
#########################

.. automodule:: larndsim.detsim
    :members:

Pixel finder utility
####################

.. automodule:: larndsim.pixels_from_track
    :members:

Electronics module
##################

.. automodule:: larndsim.fee
    :members:

Constants
####################

.. automodule:: larndsim.consts
    :members:
    
Detector constants
**********************
    
.. automodule:: larndsim.consts.detector
    :members:

Physics constants
**********************

.. automodule:: larndsim.consts.physics
    :members:

Light constants
**********************

.. automodule:: larndsim.consts.light
    :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
