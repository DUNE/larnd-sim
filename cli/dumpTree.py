#! /usr/bin/env python3
"""
Converts ROOT file created by edep-sim into HDF5 format
"""

from math import sqrt
import os
import numpy as np
import fire
import h5py
from tqdm import tqdm
import glob

from ROOT import TG4Event, TFile, TMap

# Output array datatypes
segments_dtype = np.dtype([("event_id","u4"), ("vertex_id", "u8"), ("file_vertex_id", "u8"), ("segment_id", "u4"),
                           ("z_end", "f4"),("traj_id", "u4"), ("file_traj_id", "u4"), ("tran_diff", "f4"),
                           ("z_start", "f4"), ("x_end", "f4"),
                           ("y_end", "f4"), ("n_electrons", "u4"),
                           ("pdg_id", "i4"), ("x_start", "f4"),
                           ("y_start", "f4"), ("t_start", "f4"),
                           ("t0_start", "f8"), ("t0_end", "f8"), ("t0", "f8"),
                           ("dx", "f4"), ("long_diff", "f4"),
                           ("pixel_plane", "i4"), ("t_end", "f4"),
                           ("dEdx", "f4"), ("dE", "f4"), ("t", "f4"),
                           ("y", "f4"), ("x", "f4"), ("z", "f4"),
                           ("n_photons","f4")], align=True)

trajectories_dtype = np.dtype([("event_id","u4"), ("vertex_id", "u8"), ("file_vertex_id", "u8"),
                               ("traj_id", "u4"), ("file_traj_id", "u4"), ("parent_id", "i4"), ("primary", "?"),
                               ("E_start", "f4"), ("pxyz_start", "f4", (3,)),
                               ("xyz_start", "f4", (3,)), ("t_start", "f8"),
                               ("E_end", "f4"), ("pxyz_end", "f4", (3,)),
                               ("xyz_end", "f4", (3,)), ("t_end", "f8"),
                               ("pdg_id", "i4"), ("start_process", "u4"),
                               ("start_subprocess", "u4"), ("end_process", "u4"),
                               ("end_subprocess", "u4"),("dist_travel", "f4")], align=True)

vertices_dtype = np.dtype([("event_id","u4"), ("vertex_id","u8"), ("file_vertex_id", "u8"),
                           ("x_vert","f4"), ("y_vert","f4"), ("z_vert","f4"),
                           ("t_vert","f4"), ("t_event","f4")], align=True)

# Convert from EDepSim default units (mm, ns)
edep2cm = 0.1   # convert to cm
edep2us = 0.001 # convert to microseconds

# Convert GENIE to common units
gev2mev  = 1000 # convert to MeV
meter2cm = 100  # convert to cm

# Needed for event kinematics calculation
nucleon_mass = 938.272 # MeV
beam_dir  = np.asarray([0.0, -0.05836, 1.0]) # -3.34 degrees in the y-direction
beam_norm = np.linalg.norm(beam_dir)

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

# Prep HDF5 file for writing
def initHDF5File(output_file):
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('trajectories', (0,), dtype=trajectories_dtype, maxshape=(None,))
        f.create_dataset('segments', (0,), dtype=segments_dtype, maxshape=(None,))
        f.create_dataset('vertices', (0,), dtype=vertices_dtype, maxshape=(None,))

# Resize HDF5 file and save output arrays
def updateHDF5File(output_file, trajectories, segments, vertices):
    if any([len(trajectories), len(segments), len(vertices)]):
        with h5py.File(output_file, 'a') as f:
            if len(trajectories):
                ntraj = len(f['trajectories'])
                f['trajectories'].resize((ntraj+len(trajectories),))
                f['trajectories'][ntraj:] = trajectories

            if len(segments):
                nseg = len(f['segments'])
                f['segments'].resize((nseg+len(segments),))
                f['segments'][nseg:] = segments

            if len(vertices):
                nvert = len(f['vertices'])
                f['vertices'].resize((nvert+len(vertices),))
                f['vertices'][nvert:] = vertices

# Read a file and dump it.
def dump(input_file, output_file, keep_all_dets=False):

    """
    Script to convert edep-sim root output to an h5 file formatted in a way
    that larnd-sim expects for consumption.
    Args:
        input_file (str): path to an input ROOT file containing spills.
        output_file (str): name of the h5 output file to which the information should
            be written
    """

    # Prep output file
    initHDF5File(output_file)

    segment_id = 0
    file_vertex_id = 0

    # Get the input tree out of the file.
    inputFile = TFile(input_file)
    inputTree = inputFile.Get("EDepSimEvents")
    # print("Class: ", inputTree.ClassName())

    # IF CRASH: Uncomment this section (also see IF CRASH below)
    # Attach a brach to the events.
    # event = TG4Event()
    # inputTree.SetBranchAddress("Event",event)

    # map that gives which spill each event lives in
    event_spill_map = inputFile.Get("event_spill_map")

    if not event_spill_map:
        spillPeriod_s = 0.
    else:
        spillPeriod_s = inputFile.Get("spillPeriod_s").GetVal()
        # for setting t_spill
        spillCounter = -1
        lastSpill = None        # Most-recent global spill ID

    # Read all of the events.
    entries = inputTree.GetEntriesFast()

    segments_list = list()
    trajectories_list = list()
    vertices_list = list()

    # For assigning unique-in-file track IDs:
    trackCounter = 0

    for jentry in tqdm(range(entries)):
        #print(jentry,"/",entries)
        nb = inputTree.GetEntry(jentry)

        # IF CRASH: Comment this line (also see IF CRASH above)
        event = inputTree.Event

        globalVertexID = (event.RunId * 1E6) + event.EventId

        if not event_spill_map:
            spill_it = globalVertexID
            t_spill = 0.
        else:
            spill_it_tobj = event_spill_map.GetValue(f"{event.RunId} {event.EventId}")
            spill_it = int(spill_it_tobj.GetName())
            if spill_it != lastSpill: # New spill?
                spillCounter += 1
                lastSpill = spill_it
            t_spill = spillCounter * spillPeriod_s * 1E6 # convert to us

        # write to file
        if len(trajectories_list) >= 1000 or nb <= 0:
            updateHDF5File(
                output_file,
                np.concatenate(trajectories_list, axis=0) if trajectories_list else np.empty((0,)),
                np.concatenate(segments_list, axis=0) if segments_list else np.empty((0,)),
                np.concatenate(vertices_list, axis=0) if vertices_list else np.empty((0,)))

            trajectories_list = list()
            segments_list = list()
            vertices_list = list()

        if nb <= 0:
            continue

        if keep_all_dets:
            if len(event.SegmentDetectors) == 0:
                continue
        else:
            # If ARCUBE_ACTIVE_VOLUME is not set, default to previously hard
            # coded containerName.
            if not any(containerName == os.environ.get("ARCUBE_ACTIVE_VOLUME", "volTPCActive")
                       for containerName, _hits in event.SegmentDetectors):
                continue

        # Count total number of vertices and trajectories
        n_traj = 0

        # TrackId is unique within each edep "event"
        # When there are multiple interactions/vertices passing edepsim as an event, TrackId's for trajectories are unique
        # For beam simulation, an edep "event" means an interaction/vertex. In that case TrackId's are only unique within the interaction
        vertexMap = {}
        file_vertexMap = {}

        # Dump the primary vertices
        vertices = np.empty(len(event.Primaries), dtype=vertices_dtype)
        for iVtx, primaryVertex in enumerate(event.Primaries):
            #printPrimaryVertex("PP", primaryVertex)
            vertices[iVtx]["event_id"] = event.EventId
            vertices[iVtx]["vertex_id"] = iVtx
            vertices[iVtx]["file_vertex_id"] = file_vertex_id
            vertices[iVtx]["x_vert"] = primaryVertex.GetPosition().X() * edep2cm
            vertices[iVtx]["y_vert"] = primaryVertex.GetPosition().Y() * edep2cm
            vertices[iVtx]["z_vert"] = primaryVertex.GetPosition().Z() * edep2cm
            vertices[iVtx]["t_vert"] = primaryVertex.GetPosition().T() * edep2us
            vertices[iVtx]["t_event"] = t_spill

            for primaryPar in primaryVertex.Particles:
                vertexMap[primaryPar.GetTrackId()] = iVtx
                file_vertexMap[primaryPar.GetTrackId()] = file_vertex_id

            file_vertex_id += 1

        vertices_list.append(vertices)

        trackMap = {}
        daughters = [] # In the end, "daughters" should have n primaries numbers of lists. Each list contains a single family line.

        # Dump the trajectories
        trajectories = np.full(len(event.Trajectories), np.iinfo(trajectories_dtype['traj_id']).max, dtype=trajectories_dtype)
        for iTraj, trajectory in enumerate(event.Trajectories):
            fileTrackID = trackCounter
            trackCounter += 1
            trackMap[trajectory.GetTrackId()] = fileTrackID

            if trajectory.GetParentId() == -1:
                start_pt, end_pt = trajectory.Points[0], trajectory.Points[-1]
                trajectories[n_traj]["event_id"] = event.EventId
                trajectories[n_traj]["vertex_id"] = vertexMap[trajectory.GetTrackId()]
                trajectories[n_traj]["file_vertex_id"] = file_vertexMap[trajectory.GetTrackId()]
                trajectories[n_traj]["traj_id"] = trajectory.GetTrackId()
                trajectories[n_traj]["file_traj_id"] = trackMap[trajectory.GetTrackId()]
                trajectories[n_traj]["parent_id"] = trajectory.GetParentId()
                trajectories[n_traj]["primary"] = True if trajectory.GetParentId() == -1 else False # primary particle parents trajectory id are -1

                mass = trajectory.GetInitialMomentum().M()
                p_start = (start_pt.GetMomentum().X(), start_pt.GetMomentum().Y(), start_pt.GetMomentum().Z())
                p_end = (end_pt.GetMomentum().X(), end_pt.GetMomentum().Y(), end_pt.GetMomentum().Z())

                trajectories[n_traj]["pxyz_start"] = p_start #(start_pt.GetMomentum().X(), start_pt.GetMomentum().Y(), start_pt.GetMomentum().Z())
                trajectories[n_traj]["pxyz_end"] = p_end #(end_pt.GetMomentum().X(), end_pt.GetMomentum().Y(), end_pt.GetMomentum().Z())
                trajectories[n_traj]["xyz_start"] = (start_pt.GetPosition().X() * edep2cm, start_pt.GetPosition().Y() * edep2cm, start_pt.GetPosition().Z() * edep2cm)
                trajectories[n_traj]["xyz_end"] = (end_pt.GetPosition().X() * edep2cm, end_pt.GetPosition().Y() * edep2cm, end_pt.GetPosition().Z() * edep2cm)
                trajectories[n_traj]["E_start"] = np.sqrt(np.sum(np.square(p_start)) + mass**2)
                trajectories[n_traj]["E_end"] = np.sqrt(np.sum(np.square(p_end)) + mass**2)
                trajectories[n_traj]["t_start"] = start_pt.GetPosition().T() * edep2us
                trajectories[n_traj]["t_end"] = end_pt.GetPosition().T() * edep2us
                trajectories[n_traj]["start_process"] = start_pt.GetProcess()
                trajectories[n_traj]["start_subprocess"] = start_pt.GetSubprocess()
                trajectories[n_traj]["end_process"] = end_pt.GetProcess()
                trajectories[n_traj]["end_subprocess"] = end_pt.GetSubprocess()
                trajectories[n_traj]["pdg_id"] = trajectory.GetPDGCode()
                trajectories[n_traj]["dist_travel"] = 0
                for i in range(len(trajectory.Points)-1):
                    trajectories[n_traj]["dist_travel"] += (trajectory.Points[i].GetPosition()-trajectory.Points[i+1].GetPosition()).Vect().Mag()* edep2cm

                n_traj += 1

            this_daughters = []
            i_primary = -1
            while trajectory.GetParentId() >= -1:
                if len(daughters) > 0 and trajectory.GetTrackId() in np.concatenate(daughters):
                    for i_primary in range(len(daughters)):
                        if trajectory.GetTrackId() in daughters[i_primary]:
                            break
                    this_daughters = this_daughters + daughters[i_primary]
                    break # once it's in "daughters", its all ancestors should be covered
                else:
                    this_daughters.append(trajectory.GetTrackId())
                    if trajectory.GetParentId() == -1:
                        break
                    else:
                        parent_traj_id = trajectory.GetParentId()
                        trajectory = event.Trajectories[parent_traj_id]
                    continue
            if i_primary >= 0:
                daughters[i_primary] = this_daughters
            else:
                daughters.append(this_daughters)

        # Dump the segment containers
        for containerName, hitSegments in event.SegmentDetectors:
            # If ARCUBE_ACTIVE_VOLUME is not set, default to previously hard
            # coded containerName.
            if (not keep_all_dets) and containerName != os.environ.get("ARCUBE_ACTIVE_VOLUME", "volTPCActive"):
                continue
            segment = np.empty(len(hitSegments), dtype=segments_dtype)
            for iHit, hitSegment in enumerate(hitSegments):
                segment[iHit]["event_id"] = event.EventId
                segment[iHit]["segment_id"] = segment_id
                segment_id += 1
                try:
                    segment[iHit]["traj_id"] = hitSegment.Contrib[0]
                    segment[iHit]["file_traj_id"] = trackMap[hitSegment.Contrib[0]]
                    seg_traj_id = hitSegment.Contrib[0]
                    if seg_traj_id in vertexMap.keys():
                        primary_traj_id = seg_traj_id
                    if seg_traj_id not in trajectories["traj_id"]: # trajectories is reinitialized for each edep event
                        for i_primary in range(len(daughters)):
                            if seg_traj_id in daughters[i_primary]:
                                this_line = daughters[i_primary]
                                this_line.reverse()
                                break
                        # Find the primary trackID
                        for traj_id in this_line:
                            if traj_id in vertexMap.keys():
                                primary_traj_id = traj_id
                                break
                        for traj_id in this_line:
                            if traj_id not in trajectories["traj_id"]:
                                # Given event.Trajectories is ordered by traj_id (trajectory.GetTrackId())
                                trajectory = event.Trajectories[traj_id]

                                start_pt, end_pt = trajectory.Points[0], trajectory.Points[-1]
                                trajectories[n_traj]["event_id"] = event.EventId
                                trajectories[n_traj]["vertex_id"] = vertexMap[primary_traj_id]
                                trajectories[n_traj]["file_vertex_id"] = file_vertexMap[primary_traj_id]

                                trajectories[n_traj]["traj_id"] = trajectory.GetTrackId()
                                trajectories[n_traj]["file_traj_id"] = trackMap[trajectory.GetTrackId()]
                                trajectories[n_traj]["parent_id"] = trajectory.GetParentId()
                                trajectories[n_traj]["primary"] = True if trajectory.GetParentId() == -1 else False # primary particle parents trajectory id are -1

                                mass = trajectory.GetInitialMomentum().M()
                                p_start = (start_pt.GetMomentum().X(), start_pt.GetMomentum().Y(), start_pt.GetMomentum().Z())
                                p_end = (end_pt.GetMomentum().X(), end_pt.GetMomentum().Y(), end_pt.GetMomentum().Z())

                                trajectories[n_traj]["pxyz_start"] = p_start #(start_pt.GetMomentum().X(), start_pt.GetMomentum().Y(), start_pt.GetMomentum().Z())
                                trajectories[n_traj]["pxyz_end"] = p_end #(end_pt.GetMomentum().X(), end_pt.GetMomentum().Y(), end_pt.GetMomentum().Z())
                                trajectories[n_traj]["xyz_start"] = (start_pt.GetPosition().X() * edep2cm, start_pt.GetPosition().Y() * edep2cm, start_pt.GetPosition().Z() * edep2cm)
                                trajectories[n_traj]["xyz_end"] = (end_pt.GetPosition().X() * edep2cm, end_pt.GetPosition().Y() * edep2cm, end_pt.GetPosition().Z() * edep2cm)
                                trajectories[n_traj]["E_start"] = np.sqrt(np.sum(np.square(p_start)) + mass**2)
                                trajectories[n_traj]["E_end"] = np.sqrt(np.sum(np.square(p_end)) + mass**2)
                                trajectories[n_traj]["t_start"] = start_pt.GetPosition().T() * edep2us
                                trajectories[n_traj]["t_end"] = end_pt.GetPosition().T() * edep2us
                                trajectories[n_traj]["start_process"] = start_pt.GetProcess()
                                trajectories[n_traj]["start_subprocess"] = start_pt.GetSubprocess()
                                trajectories[n_traj]["end_process"] = end_pt.GetProcess()
                                trajectories[n_traj]["end_subprocess"] = end_pt.GetSubprocess()
                                trajectories[n_traj]["pdg_id"] = trajectory.GetPDGCode()
                                trajectories[n_traj]["dist_travel"] = 0
                                for i in range(len(trajectory.Points)-1):
                                    trajectories[n_traj]["dist_travel"] += (trajectory.Points[i].GetPosition()-trajectory.Points[i+1].GetPosition()).Vect().Mag()* edep2cm
                                n_traj += 1
                    segment[iHit]["vertex_id"] = vertexMap[primary_traj_id]
                    segment[iHit]["file_vertex_id"] = file_vertexMap[primary_traj_id]

                except IndexError as e:
                    print(e)
                    print("iHit:",iHit)
                    print("len(segment):",len(segment))
                    print("hitSegment.Contrib[0]:",hitSegment.Contrib[0])
                    print("len(trajectories):",len(trajectories))

                segment[iHit]["x_start"] = hitSegment.GetStart().X() * edep2cm
                segment[iHit]["y_start"] = hitSegment.GetStart().Y() * edep2cm
                segment[iHit]["z_start"] = hitSegment.GetStart().Z() * edep2cm
                segment[iHit]["t0_start"] = hitSegment.GetStart().T() * edep2us
                segment[iHit]["t_start"] = 0
                segment[iHit]["x_end"] = hitSegment.GetStop().X() * edep2cm
                segment[iHit]["y_end"] = hitSegment.GetStop().Y() * edep2cm
                segment[iHit]["z_end"] = hitSegment.GetStop().Z() * edep2cm
                segment[iHit]["t0_end"] = hitSegment.GetStop().T() * edep2us
                segment[iHit]["t_end"] = 0
                segment[iHit]["dE"] = hitSegment.GetEnergyDeposit()
                xd = segment[iHit]["x_end"] - segment[iHit]["x_start"]
                yd = segment[iHit]["y_end"] - segment[iHit]["y_start"]
                zd = segment[iHit]["z_end"] - segment[iHit]["z_start"]
                dx = sqrt(xd**2 + yd**2 + zd**2)
                segment[iHit]["dx"] = dx
                segment[iHit]["x"] = (segment[iHit]["x_start"] + segment[iHit]["x_end"]) / 2.
                segment[iHit]["y"] = (segment[iHit]["y_start"] + segment[iHit]["y_end"]) / 2.
                segment[iHit]["z"] = (segment[iHit]["z_start"] + segment[iHit]["z_end"]) / 2.
                segment[iHit]["t0"] = (segment[iHit]["t0_start"] + segment[iHit]["t0_end"]) / 2.
                segment[iHit]["t"] = 0
                segment[iHit]["dEdx"] = hitSegment.GetEnergyDeposit() / dx if dx > 0 else 0
                segment[iHit]["pdg_id"] = trajectories[trajectories["traj_id"]==hitSegment.Contrib[0]]["pdg_id"]
                segment[iHit]["n_electrons"] = 0
                segment[iHit]["long_diff"] = 0
                segment[iHit]["tran_diff"] = 0
                segment[iHit]["pixel_plane"] = 0
                segment[iHit]["n_photons"] = 0

            segments_list.append(segment)
        trajectories_list.append(trajectories[:n_traj])

    # save any lingering data not written to file
    updateHDF5File(
        output_file,
        np.concatenate(trajectories_list, axis=0) if trajectories_list else np.empty((0,)),
        np.concatenate(segments_list, axis=0) if segments_list else np.empty((0,)),
        np.concatenate(vertices_list, axis=0) if vertices_list else np.empty((0,)))

if __name__ == "__main__":
    fire.Fire(dump)

