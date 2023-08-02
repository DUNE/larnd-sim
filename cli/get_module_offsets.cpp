// This is a script to get the TGeoManager from the edep-sim ROOT file and 
// extract the module offsets for use in larnd-sim. To get the module offsets, 
// we need to recursively traverse the geometry tree until we encounter the LAr volumes, 
// taking account of the translations and rotations along the way.

#include <TGeoManager.h>
#include <TFile.h>
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <functional>

std::pair<bool, std::vector<std::vector<double>>> get_module_offsets(const char* fileName) {
    
    TFile file(fileName);

    // If the file does not have a TGeoManager, return false
    if (!file.Get("EDepSimGeometry")) {
        return {false, {}};
    }

    // Import the TGeoManager
    TGeoManager* geoManager = TGeoManager::Import(fileName);

    // Get the top node of the geometry
    TGeoNode* topVolume = geoManager->GetTopNode();

    // The global origins of the target volumes, to be determined below
    std::vector<std::vector<double>> globalOrigins;

    // Define a recursive function to traverse the geometry tree
    std::function<void(TGeoNode*, std::unique_ptr<TGeoHMatrix>)> traverse;

    traverse = [&traverse, &globalOrigins](TGeoNode* node, std::unique_ptr<TGeoHMatrix> globalMatrix) {
        if (globalMatrix == nullptr) {
            globalMatrix = std::make_unique<TGeoHMatrix>();  // identity matrix
        } else {
            // Combine the current node's transformation with the accumulated transformation
            TGeoHMatrix localMatrix = *globalMatrix;
            localMatrix.Multiply(node->GetMatrix());
            globalMatrix = std::make_unique<TGeoHMatrix>(localMatrix);
        }

        // The volume names we're looking for. Note that you may need to add
        // new names that correspond to new geometries (e.g. updated 2x2, FSD, ndlar)
        std::vector<std::string> targetVolumes = {"volTPCActive_PV"}; // single module or SingleCube
        // for 2x2, get the module wall volumes for the offsets
        for (int i = 0; i < 2; ++i) { 
            for (int j = 0; j < 2; ++j) {
                targetVolumes.push_back("volLArActiveModWall" + std::to_string(i) + std::to_string(j) + "_PV");
            }
        }

        // Check if the node's volume is one of the volumes we're looking for
        for (const auto& volumeName : targetVolumes) {
            if (volumeName == node->GetVolume()->GetName()) { 
                Double_t localOrigin[3] = {0., 0., 0.};
                Double_t globalOrigin[3];
                // Convert local coordinates to global coordinates
                globalMatrix->LocalToMaster(localOrigin, globalOrigin);
                // Add the global origin to the list
                globalOrigins.push_back({globalOrigin[0], globalOrigin[1], globalOrigin[2]});
            }
        }

        // Recursively traverse the node's daughters
        for (int i = 0; i < node->GetNdaughters(); ++i) {
            TGeoNode* daughter = node->GetDaughter(i);
            traverse(daughter, std::make_unique<TGeoHMatrix>(*globalMatrix));
        }
    };

    // Start the traversal from the top volume
    traverse(topVolume, nullptr);
    
    file.Close();
    
    return {true, globalOrigins};
}
