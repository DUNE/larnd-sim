#!/usr/bin/env python

from collections import defaultdict

import h5py
import math
import numpy as np
import yaml
import fire

from tqdm import tqdm
from larcv import larcv
from larndsim import consts, fee
from larndsim.consts import detector, physics

#VERBOSE = 0

def charge_to_MeV(charge_at_anode, drift_distance):
    """ Estimate the energy deposit that corresponds to the observed charge at anode.

        In simulation, this correction is redundant (we know the true energy deposited anyway).
        But for real data, and for the initial tests of the reconstruction
        without pass-through truth info through larnd-sim,
        it's necessary.

        There are three corrections:
           * electron lifetime/drift  (electrons at anode -> electrons at interaction point)
           * recombination            (account for electrons lost at interaction point)
           * Argon work function      (energy per atomic electron liberated)

        Each is the inverse of a calculation in larnd-sim.
    """

    drift_time = drift_distance / detector.V_DRIFT
    charge = charge_at_anode * math.exp(drift_time / detector.ELECTRON_LIFETIME)

    # for the recombination correction we need to know dE/dx,
    # since the recombination function depends on it.
    # unfortunately at this stage we don't.
    # instead we use the most probable dE/dx (minimum-ionizing peak, about 1.9 MeV/cm)
    # and the recombination factor that arises from it, 0.7056.
    # (the value of the most probable dE/dx was determined by examining the true dE/dxs
    #  in "voxelized edep-sim".  the corresponding recombination factor comes from the formula in larndsim/quenching.py.)
    charge /= 0.7056

    return charge * consts.physics.W_ION


def get_x_coordinate(io_group_io_channel_to_tile,
                     io_group, io_channel,
                     tile_positions,
                     tile_orientations,
                     time):
    """
    Get drift coordinate for a charge packet
    Essentially copied from 'get_z_coordinate' in larnd-display module,
    but now with x as the drift coordinate. Add some unit conversions
    so that the output value is in cm
    """
    tile_id = io_group_io_channel_to_tile[(io_group, io_channel)]
    # Factor of 10 converts mm -> cm
    x_anode = tile_positions[tile_id][0]/10
    drift_direction = tile_orientations[tile_id][0]
    # Convert ticks to microseconds (based on ND-LAr sampling rate, there are 10 ticks per microsecond)
    time = time/10
    return x_anode + time * detector.V_DRIFT * drift_direction

def run_conversion(input_file, output_file, pixel_file, detector_file, verbosity=0, calibration=True):
    """Converts larnd-sim file into larcv format"""

    VERBOSE = verbosity

    # TODO Make these load from some config file
    # Currently set for ND-LAr values
    BBoxSize   = [ 740,  320, 530]
    BBoxTop    = [ 370,  160, 930]
    BBoxBottom = [-370, -160, 400]
    VoxelSize  = [0.4, 0.4, 0.4]

    # Set up bounding box and 3DMeta parameters
    xmin, ymin, zmin = BBoxBottom[0], BBoxBottom[1], BBoxBottom[2]
    xmax, ymax, zmax = BBoxTop   [0], BBoxTop   [1], BBoxTop   [2]
    xlen, ylen, zlen = BBoxSize  [0], BBoxSize  [1], BBoxSize  [2]
    xvox, yvox, zvox = VoxelSize [0], VoxelSize [1], VoxelSize [2]

    xnum, ynum, znum = int(np.floor(xlen/xvox)), int(np.floor(ylen/yvox)), int(np.floor(zlen/zvox))

    bbox_args = BBoxBottom + BBoxTop
    bbox = larcv.BBox3D(*bbox_args)
    voxel_meta = larcv.Voxel3DMeta()
    voxel_meta.set(xmin, ymin, zmin, xmax, ymax, zmax, xnum, ynum, znum)
    assert voxel_meta.valid(), "Invalid voxel metadata!  Grid (xmin, ymin, zmin, xmax, ymax, zmax, xnum, ynum, znum) = %s" % str((xmin, ymin, zmin, xmax, ymax, zmax, xnum, ynum, znum))

    print('Try to open ', input_file)
    with h5py.File(input_file, 'r') as fin:
        if VERBOSE:
            print('Printing input file keys:\n', fin.keys())
        # configs         = fin['configs'][:]
        # event_truth     = fin['eventTruth']
        # messages        = fin['messages'][:]
        mc_packets_assn = fin['mc_packets_assn'][:]
        packets         = fin['packets'][:]
        tracks          = fin['tracks'][:]
        #trajectories    = fin['trajectories']

    if VERBOSE:
        print('{} unique event IDs in this file'.format(len(np.unique(tracks['eventID']))))

    consts.load_properties(detector_file, pixel_file)

    # Make dictionary for chip yz-values
    # Mostly copied from larn-sim, except that we flip x to be the 
    # drift coordinate and z to the beam direction
    geometry_yaml = yaml.load(open(pixel_file), Loader=yaml.FullLoader)
    det_yaml = yaml.load(open(detector_file), Loader=yaml.FullLoader)

    pixel_pitch              = geometry_yaml['pixel_pitch']
    chip_channel_to_position = geometry_yaml['chip_channel_to_position']
    tile_orientations        = geometry_yaml['tile_orientations']
    tile_positions           = geometry_yaml['tile_positions']
    tpc_centers              = det_yaml     ['tpc_offsets']

    ys = np.array(list(chip_channel_to_position.values()))[:, 1] * pixel_pitch
    zs = np.array(list(chip_channel_to_position.values()))[:, 0] * pixel_pitch
    y_size = max(ys)-min(ys)+pixel_pitch
    z_size = max(zs)-min(zs)+pixel_pitch

    # Swap indices here relative to default larnd-sim, since pixel_pos is
    # now (y, z) instead of (x, y)
    def rotate_pixel(pixel_pos, tile_orientation):
        return pixel_pos[0]*tile_orientation[1], pixel_pos[1]*tile_orientation[2]

    tile_geometry = defaultdict(int)
    geometry = {}
    io_group_io_channel_to_tile = {}

    for tile in geometry_yaml['tile_chip_to_io']: # io_group loop?
        tile_orientation = tile_orientations[tile]
        tile_geometry[tile] = tile_positions[tile], tile_orientations[tile]

        for chip in geometry_yaml['tile_chip_to_io'][tile]:
            io_group_io_channel = geometry_yaml['tile_chip_to_io'][tile][chip]
            io_group = io_group_io_channel//1000
            io_channel = io_group_io_channel % 1000
            io_group_io_channel_to_tile[(io_group, io_channel)] = tile

        for chip_channel in geometry_yaml['chip_channel_to_position']:
            chip = chip_channel // 1000
            channel = chip_channel % 1000
            io_group_io_channel = geometry_yaml['tile_chip_to_io'][tile][chip]

            io_group = io_group_io_channel // 1000
            io_channel = io_group_io_channel % 1000

            y = chip_channel_to_position[chip_channel][1] * pixel_pitch - y_size / 2
            z = chip_channel_to_position[chip_channel][0] * pixel_pitch - z_size / 2

            y, z = rotate_pixel((y, z), tile_orientation)
            y += tile_positions[tile][1]
            z += tile_positions[tile][2]

            geometry[(io_group, io_channel, chip, channel)] = y, z

    # Divide events based on triggers
    # Find packets with trigger type (packet_type==7), then use
    # "gaps" in packet_type to find where events start/end
    tr = packets['packet_type'] == 7
    trigger_packets = np.argwhere(tr).T[0]
    event_dividers = trigger_packets[:-1][np.diff(trigger_packets)!=1]
    event_dividers = np.append(event_dividers,[trigger_packets[-1],len(packets)])
    num_events = len(event_dividers) - 1
    if VERBOSE:
        print('{} events'.format(num_events))
        print('{} event dividers'.format(len(event_dividers)))
        print('Event dividers:\n', event_dividers)

    module_ids = []
    print('Processing', num_events, 'events')
    event_packets = np.empty((num_events, 4))

    io = larcv.IOManager(larcv.IOManager.kWRITE)
    io.set_out_file(output_file)
    io.initialize()

    for event in tqdm(range(num_events), desc='Converting events...'):
        start_packet = event_dividers[event]
        end_packet   = event_dividers[event+1]
        event_packets = packets[start_packet:end_packet]
        last_trigger = packets[start_packet]['timestamp']
        if VERBOSE:
            print('-------Processing event', event, '---------')
            print('{} packets in this event'.format(len(event_packets)))
            print('start_packet = {} and end_packet = {}'.format(start_packet, end_packet))
            print('Timestamp from most recent trigger: {}'.format(last_trigger))

        # Calculate charge scale for this event
        event_adcs = event_packets['dataword'][event_packets['packet_type']==0]
        event_charges = np.array( (event_adcs/fee.ADC_COUNTS*(fee.V_REF - fee.V_CM) + (fee.V_CM - fee.V_PEDESTAL))/fee.GAIN )
        if VERBOSE:
            print('Total charge in this event: {}'.format(event_charges.sum()))

        # Collect packet charge and xyz information
        event_voxel_set = larcv.VoxelSet()

        eventids = None
        if len(mc_packets_assn) > 0 and end_packet > start_packet:
            packet_indices = np.where(packets[start_packet:end_packet]["packet_type"] == 0)[0] + start_packet
            track_ids = np.unique(mc_packets_assn[packet_indices]["track_ids"])
            track_ids = track_ids[track_ids >= 0]  # will have lots of -1s from unfilled slots in the track_ids array
            eventids = np.unique(tracks[track_ids]["eventID"])
            assert len(eventids) == 1, "Packets correspond to more than a single true event!  Event numbers: %s" % eventids

        for packet in packets[start_packet:end_packet]:
            # Only look at data packets (type 0)
            if packet['packet_type'] == 0:
                io_group, io_channel, chip, channel = packet['io_group'], packet['io_channel'], packet['chip_id'], packet['channel_id']
                packet_adc = packet['dataword']
                # Convert ADC to number of electrons
                packet_charge = (packet_adc / fee.ADC_COUNTS * (fee.V_REF-fee.V_CM) + (fee.V_CM-fee.V_PEDESTAL)) / fee.GAIN
                module_id = (io_group-1)//4
                if module_id not in module_ids:
                    module_ids.append(module_id)
                io_group = io_group - (io_group-1)//4*4

                # Offsets in cm
                x_offset = tpc_centers[module_id][0]
                y_offset = tpc_centers[module_id][1]
                z_offset = tpc_centers[module_id][2]

                ### Find packet xyz coordinates ###
                # For y and z, use the previously defined dictionary
                # For x, use the timestamp for the event trigger and
                # feed into get_x_coordinate
                y,z = geometry[(io_group, io_channel, chip, channel)]
                y = y/10 # Factor of 10 to convert mm -> cm
                z = z/10 # Factor of 10 to convert mm -> cm
                y = y + y_offset
                z = z + z_offset

                x = get_x_coordinate(io_group_io_channel_to_tile,
                                    io_group, io_channel,
                                    tile_positions,
                                    tile_orientations,
                                    packet['timestamp'] - last_trigger) + x_offset

                point = larcv.Point3D(x, y, z)
                if not bbox.contains(point):
                    print("rejecting point outside bbox: {:.2f}, {:.2f}, {:.2f}".format(x, y, z))
                    continue

                voxel_id = voxel_meta.id(point)
                voxel = larcv.Voxel(voxel_id, charge_to_MeV(packet_charge, x) if calibration else packet_charge)
                event_voxel_set.add(voxel)

        event_num = event if eventids is None else int(eventids[0])
        io.set_id(0, 0, event_num)  # run numbers
        prod = io.get_data("sparse3d", "larndsim")
        prod.set(event_voxel_set, voxel_meta)
        io.save_entry()

    io.finalize()

if __name__ == '__main__':
    fire.Fire(run_conversion)
