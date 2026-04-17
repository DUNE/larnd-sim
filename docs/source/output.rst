.. _output:

Output Format
=============

.. contents:: On this page
   :local:
   :depth: 2

larnd-sim writes all output to an **HDF5** file.
`HDF5 <https://www.hdfgroup.org/solutions/hdf5>`_ is a portable,
self-describing binary format well-suited to large numerical arrays.
The Python library `h5py <https://www.h5py.org>`_ is recommended for reading
the output.

.. code-block:: python

   import h5py
   f = h5py.File("output.h5", "r")
   print(list(f.keys()))

Charge Data
-----------

The charge output uses the same dataset schema as
`larpix-control <https://larpix-control.readthedocs.io/en/stable/api/format/hdf5format.html>`_,
ensuring compatibility with downstream reconstruction code that processes
real detector data.

``packets`` dataset
~~~~~~~~~~~~~~~~~~~

Shape ``(n_packets,)``.  Each entry is a structured array element that
encodes a LArPix data packet.  The most important fields are:

- ``packet_type`` — ``0`` = data, ``4`` = timestamp, ``6`` = trigger,
  ``7`` = external trigger
- ``chip_key`` — identifies the LArPix ASIC
- ``channel_id`` — pixel channel within the chip
- ``dataword`` — raw ADC value
- ``timestamp`` — LArPix clock tick of the hit

``mc_packets_assn`` dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Shape ``(n_packets,)``.  Aligned with ``packets``; provides MC-truth
back-tracking for each packet.

- ``segment_ids`` — indices into the ``segments`` dataset for the
  edep-sim track segments that contributed charge
- ``fraction`` — fractional contribution of each segment to the ADC value
  (sorted descending; padded with ``-1`` when fewer than
  ``fee.ASSOCIATION_COUNT_TO_STORE`` segments contribute)
- ``event_ids`` — event ID of the primary contributing segment

The maximum number of contributing segments stored per packet is set by
``fee.ASSOCIATION_COUNT_TO_STORE``.  Increasing
``fee.MAX_ADC_VALUES`` (which must be ``≥ ASSOCIATION_COUNT_TO_STORE``)
improves the accuracy of the fraction calculation.

Truth Datasets
~~~~~~~~~~~~~~

Copied verbatim from the edep-sim input:

- ``vertices`` — interaction-level truth (neutrino vertex, reaction mode, …)
- ``trajectories`` — particle-level truth (PDG code, momentum, …)
- ``segments`` — energy-deposition segments (the primary input to the simulation)
- ``mc_hdr``, ``mc_stack`` — neutrino generator records (present only for
  beam/neutrino inputs)

Light Data
----------

``light_trig`` dataset
~~~~~~~~~~~~~~~~~~~~~~

Shape ``(n_triggers,)``.  Metadata for each light-system self-trigger:

- ``op_channel`` shape ``(n_op_channels_per_trig,)`` — optical channel IDs
  included in this trigger
- ``ts_s`` — global trigger timestamp in seconds
- ``ts_sync`` — trigger timestamp in LArPix clock ticks (accounts for PPS
  synchronisation and clock roll-over)

``light_wvfm`` dataset
~~~~~~~~~~~~~~~~~~~~~~

Shape ``(n_triggers, n_op_channels_per_trig, n_adc_samples)``.
Digitized ADC waveforms for each SiPM channel at each trigger.

.. note::
   For 2×2 Module 1/2/3 cosmic data-taking configurations, divide waveform
   values by 4 to obtain ADC counts proportional to the light signal.

``light_dat`` dataset
~~~~~~~~~~~~~~~~~~~~~~

Shape ``(n_segments, n_op_channels)``.  True light truth per edep-sim
segment.

- ``n_photons_det`` — number of photons that reach each SiPM
- ``t0_det`` — arrival time of the first photon on each SiPM

When module-by-module variation is enabled, this dataset is split into
``light_dat/light_dat_module{Z}`` per module.

``light_wvfm_mc`` dataset *(optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Shape ``(n_triggers, n_op_channels_per_trig, n_adc_samples)``.
Generated when ``light.MAX_MC_TRUTH_IDS > 0``.

- ``segment_ids`` shape ``(n_max_tracks,)`` — true segment IDs per sample
- ``pe_current``  shape ``(n_max_tracks,)`` — equivalent photocurrent per
  segment per sample

Charge / Light Matching
-----------------------

The charge and light readouts have independent clocks; association is done
via the shared LArPix timestamp embedded in the ``packet_type == 7``
(external trigger) packets and in ``light_trig.ts_sync``.

.. code-block:: python

   import h5py
   import numpy as np

   f = h5py.File("output.h5", "r")

   packets = f["packets"][:]
   ext_trig_mask = packets["packet_type"] == 7
   charge_ts = packets[ext_trig_mask]["timestamp"]

   light_ts = f["light_trig"]["ts_sync"][:]

   # shared timestamps, and their indices in each array
   shared_ts, pkt_idx, lt_idx = np.intersect1d(
       charge_ts, light_ts, return_indices=True
   )

   # packets belonging to the first matched event
   event0_packets   = packets[pkt_idx[0]:pkt_idx[1]]
   event0_waveforms = f["light_wvfm"][lt_idx[0]:lt_idx[1]]
