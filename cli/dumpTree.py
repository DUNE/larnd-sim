#! /usr/bin/env python
#
# Read almost every field in the event tree.
#

from math import sqrt

import numpy as np
import fire
import h5py

from ROOT import TG4Event, TFile

# Print the fields in a TG4PrimaryParticle object
def printPrimaryParticle(depth, primaryParticle):
    print(depth,"Class: ", primaryParticle.ClassName())
    print(depth,"Track Id:", primaryParticle.GetTrackId())
    print(depth,"Name:", primaryParticle.GetName())
    print(depth,"PDG Code:",primaryParticle.GetPDGCode())
    print(depth,"Momentum:",primaryParticle.GetMomentum().X(),
          primaryParticle.GetMomentum().Y(),
          primaryParticle.GetMomentum().Z(),
          primaryParticle.GetMomentum().E(),
          primaryParticle.GetMomentum().P(),
          primaryParticle.GetMomentum().M())

# Print the fields in an TG4PrimaryVertex object
def printPrimaryVertex(depth, primaryVertex):
    print(depth,"Class: ", primaryVertex.ClassName())
    print(depth,"Position:", primaryVertex.GetPosition().X(),
          primaryVertex.GetPosition().Y(),
          primaryVertex.GetPosition().Z(),
          primaryVertex.GetPosition().T())
    print(depth,"Generator:",primaryVertex.GetGeneratorName())
    print(depth,"Reaction:",primaryVertex.GetReaction())
    print(depth,"Filename:",primaryVertex.GetFilename())
    print(depth,"InteractionNumber:",primaryVertex.GetInteractionNumber())
    depth = depth + ".."
    for infoVertex in primaryVertex.Informational:
        printPrimaryVertex(depth,infoVertex)
    for primaryParticle in primaryVertex.Particles:
        printPrimaryParticle(depth,primaryParticle)

# Print the fields in a TG4TrajectoryPoint object
def printTrajectoryPoint(depth, trajectoryPoint):
    print(depth,"Class: ", trajectoryPoint.ClassName())
    print(depth,"Position:", trajectoryPoint.GetPosition().X(),
          trajectoryPoint.GetPosition().Y(),
          trajectoryPoint.GetPosition().Z(),
          trajectoryPoint.GetPosition().T())
    print(depth,"Momentum:", trajectoryPoint.GetMomentum().X(),
          trajectoryPoint.GetMomentum().Y(),
          trajectoryPoint.GetMomentum().Z(),
          trajectoryPoint.GetMomentum().Mag())
    print(depth,"Process",trajectoryPoint.GetProcess())
    print(depth,"Subprocess",trajectoryPoint.GetSubprocess())

# Print the fields in a TG4Trajectory object
def printTrajectory(depth, trajectory):
    print(depth,"Class: ", trajectory.ClassName())
    depth = depth + ".."
    print(depth,"Track Id/Parent Id:",
          trajectory.GetTrackId(),
          trajectory.GetParentId())
    print(depth,"Name:",trajectory.GetName())
    print(depth,"PDG Code",trajectory.GetPDGCode())
    print(depth,"Initial Momentum:",trajectory.GetInitialMomentum().X(),
          trajectory.GetInitialMomentum().Y(),
          trajectory.GetInitialMomentum().Z(),
          trajectory.GetInitialMomentum().E(),
          trajectory.GetInitialMomentum().P(),
          trajectory.GetInitialMomentum().M())
    for trajectoryPoint in trajectory.Points:
        printTrajectoryPoint(depth,trajectoryPoint)

# Print the fields in a TG4HitSegment object
def printHitSegment(depth, hitSegment):
    print(depth,"Class: ", hitSegment.ClassName())
    print(depth,"Primary Id:", hitSegment.GetPrimaryId())
    print(depth,"Energy Deposit:",hitSegment.GetEnergyDeposit())
    print(depth,"Secondary Deposit:", hitSegment.GetSecondaryDeposit())
    print(depth,"Track Length:",hitSegment.GetTrackLength())
    print(depth,"Start:", hitSegment.GetStart().X(),
          hitSegment.GetStart().Y(),
          hitSegment.GetStart().Z(),
          hitSegment.GetStart().T())
    print(depth,"Stop:", hitSegment.GetStop().X(),
          hitSegment.GetStop().Y(),
          hitSegment.GetStop().Z(),
          hitSegment.GetStop().T())
    print(depth,"Contributor:", [contributor for contributor in hitSegment.Contrib])

# Print the fields in a single element of the SegmentDetectors map.
# The container name is the key, and the hitSegments is the value (a
# vector of TG4HitSegment objects).
def printSegmentContainer(depth, containerName, hitSegments):
    print(depth,"Detector: ", containerName, hitSegments.size())
    depth = depth + ".."
    for hitSegment in hitSegments: printHitSegment(depth, hitSegment)

# Read a file and dump it.
def dump(input_file, output_file):

    # The input file is generated in a previous test (100TestTree.sh).
    inputFile = TFile(input_file)

    # Get the input tree out of the file.
    inputTree = inputFile.Get("EDepSimEvents")
    print("Class:", inputTree.ClassName())

    # Attach a brach to the events.
    event = TG4Event()
    inputTree.SetBranchAddress("Event",event)

    # Read all of the events.
    entries = inputTree.GetEntriesFast()

    segments_dtype = np.dtype([('eventID', 'u4'), ('z_end', 'f4'),
                               ('trackID', 'u4'), ('tran_diff', 'f4'),
                               ('z_start', 'f4'), ('x_end', 'f4'),
                               ('y_end', 'f4'), ('n_electrons', 'u4'),
                               ('pdgId', 'i4'), ('x_start', 'f4'),
                               ('y_start', 'f4'), ('t_start', 'f4'),
                               ('dx', 'f4'), ('long_diff', 'f4'),
                               ('pixel_plane', 'u4'), ('t_end', 'f4'),
                               ('dEdx', 'f4'), ('dE', 'f4'), ('t', 'f4'),
                               ('y', 'f4'), ('x', 'f4'), ('z', 'f4')])

    trajectories_dtype = np.dtype([('eventID', 'u4'), ('trackID', 'u4'),
                                   ('parentID', 'i4'),
                                   ('pxyz_start', 'f4', (3,)),
                                   ('xyz_start', 'f4', (3,)), ('t_start', 'f4'),
                                   ('pxyz_end', 'f4', (3,)),
                                   ('xyz_end', 'f4', (3,)), ('t_end', 'f4'),
                                   ('pdgId', 'i4'), ('process', 'u4'),
                                   ('subprocess', 'u4')])

    segments_list = []
    trajectories_list = []

    for jentry in range(entries):
        print(jentry)
        nb = inputTree.GetEntry(jentry)
        if nb <= 0:
            continue

        print("Class: ", event.ClassName())
        print("Event number:", event.EventId)

        # Dump the primary vertices
        # for primaryVertex in event.Primaries:
        #    printPrimaryVertex("PP", primaryVertex)

        # Dump the trajectories
        trajectories = {'pdgId': [], 'trackId': []}
        print("Number of trajectories ", len(event.Trajectories))

        trajectories = np.empty(len(event.Trajectories), dtype=trajectories_dtype)
        for iTraj, trajectory in enumerate(event.Trajectories):
            start_pt, end_pt = trajectory.Points[0], trajectory.Points[-1]
            trajectories[iTraj]['eventID'] = jentry
            trajectories[iTraj]['trackID'] = trajectory.GetTrackId()
            trajectories[iTraj]['parentID'] = trajectory.GetParentId()
            trajectories[iTraj]['pxyz_start'] = (start_pt.GetMomentum().X(), start_pt.GetMomentum().Y(), start_pt.GetMomentum().Z())
            trajectories[iTraj]['pxyz_end'] = (end_pt.GetMomentum().X(), end_pt.GetMomentum().Y(), end_pt.GetMomentum().Z())
            trajectories[iTraj]['xyz_start'] = (start_pt.GetPosition().X(), start_pt.GetPosition().Y(), start_pt.GetPosition().Z())
            trajectories[iTraj]['xyz_end'] = (end_pt.GetPosition().X(), end_pt.GetPosition().Y(), end_pt.GetPosition().Z())
            trajectories[iTraj]['t_start'] = start_pt.GetPosition().T()
            trajectories[iTraj]['t_end'] = end_pt.GetPosition().T()
            trajectories[iTraj]['process'] = start_pt.GetProcess()
            trajectories[iTraj]['subprocess'] = start_pt.GetSubprocess()
            trajectories[iTraj]['pdgId'] = trajectory.GetPDGCode()
        trajectories_list.append(trajectories)

        # Dump the segment containers
        print("Number of segment containers:", event.SegmentDetectors.size())

        for containerName, hitSegments in event.SegmentDetectors:

            segment = np.empty(len(hitSegments), dtype=segments_dtype)
            for iHit, hitSegment in enumerate(hitSegments):
                segment[iHit]['eventID'] = jentry
                segment[iHit]['trackID'] = trajectories[hitSegment.Contrib[0]]['trackId']
                segment[iHit]['x_start'] = hitSegment.GetStart().X() / 10
                segment[iHit]['y_start'] = hitSegment.GetStart().Y() / 10
                segment[iHit]['z_start'] = hitSegment.GetStart().Z() / 10
                segment[iHit]['x_end'] = hitSegment.GetStop().X() / 10
                segment[iHit]['y_end'] = hitSegment.GetStop().Y() / 10
                segment[iHit]['z_end'] = hitSegment.GetStop().Z() / 10
                segment[iHit]['dE'] = hitSegment.GetEnergyDeposit()
                segment[iHit]['t'] = 0
                segment[iHit]['t_start'] = 0
                segment[iHit]['t_end'] = 0
                xd = segment[iHit]['x_end'] - segment[iHit]['x_start']
                yd = segment[iHit]['y_end'] - segment[iHit]['y_start']
                zd = segment[iHit]['z_end'] - segment[iHit]['z_start']
                dx = sqrt(xd**2 + yd**2 + zd**2)
                segment[iHit]['dx'] = dx
                segment[iHit]['x'] = (segment[iHit]['x_start'] + segment[iHit]['x_end']) / 2.
                segment[iHit]['y'] = (segment[iHit]['y_start'] + segment[iHit]['y_end']) / 2.
                segment[iHit]['z'] = (segment[iHit]['z_start'] + segment[iHit]['z_end']) / 2.
                segment[iHit]['dEdx'] = hitSegment.GetEnergyDeposit() / dx if dx > 0 else 0
                segment[iHit]['pdgId'] = trajectories[hitSegment.Contrib[0]]['pdgId']
                segment[iHit]['n_electrons'] = 0
                segment[iHit]['long_diff'] = 0
                segment[iHit]['tran_diff'] = 0
                segment[iHit]['pixel_plane'] = 0
            segments_list.append(segment)
    trajectories_list = np.concatenate(trajectories_list, axis=0)
    segments_list = np.concatenate(segments_list, axis=0)

    with h5py.File(output_file, 'w') as f:
        f.create_dataset("trajectories", data=trajectories_list)
        f.create_dataset("segments", data=segments_list)


if __name__ == "__main__":
    fire.Fire(dump)
