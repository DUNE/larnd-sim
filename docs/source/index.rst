.. larnd-sim documentation master file, created by
   sphinx-quickstart on Thu Aug  6 15:33:22 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to larnd-sim's documentation!
=====================================

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   larndsim
   cli
   output

This software aims to simulate a pixelated Liquid Argon Time Projection Chamber using `LArPix <https://arxiv.org/abs/1808.02969>`_. It consists of a set of highly-parallelized algorithms implemented on the `CUDA architecture <https://developer.nvidia.com/cuda-toolkit>`_.

The software takes as input a dataset containing segments of deposited energy in the detector, generated with a Geant4 wrapper called `edep-sim <https://github.com/ClarkMcGrew/edep-sim>`_.
The output of :code:`edep-sim` is in the ROOT format and must be converted into HDF5 to be used by :code:`larnd-sim`. To this purpose we provide the :code:`cli/dumpTree.py` script.

:code:`larnd-sim` simulates both the scintillation light acquired by the light sensors and the charge induced on the pixels.
The output consists of a list of packets in the `LArPix HDF5 format <https://larpix-control.readthedocs.io/en/stable/api/format/hdf5format.html>`_ for the charge readout and of a list of waveforms for the light readout.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
