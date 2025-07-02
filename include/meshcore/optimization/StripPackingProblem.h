//
// Created by Jonas Tollenaere on 02/07/2025.
//

#ifndef STRIPPACKINGPROBLEM_H
#define STRIPPACKINGPROBLEM_H

#include <unordered_map>
#include <nlohmann/json.hpp>

#include "meshcore/core/ModelSpaceMesh.h"
#include "meshcore/core/Color.h"

enum class ObjectOrigin {
    Original,   // Keep original reference point, vertices as defined in the files
    AlignToMinimum, // Align to AABB's minimum
    AlignToCenter, // Align to AABB's center
    AlignToCentroid // Align to centroid of the mesh
};

class StripPackingProblem {

    // Actual problem data
    const AABB container;
    std::vector<std::shared_ptr<ModelSpaceMesh>> requiredItems;
    std::vector<size_t> requiredItemCounts;

    // Data computed from the problem data
    std::unordered_map<std::shared_ptr<ModelSpaceMesh>, size_t> requiredItemsMap;
    float totalItemVolume;
    size_t totalNumberOfItems;

    // Some bookkeeping
    const std::string name; // Readable name of the problem
    const std::string instancePath; // Path to the instance file, relative to the data folder

    // Where the origin of the item was placed upon loading from a file, required to correctly export solutions
    const ObjectOrigin itemOrigin;

public:
    StripPackingProblem(std::string instancePath, std::string name, const AABB &container,
                         const std::vector<std::shared_ptr<ModelSpaceMesh>> &requiredItems,
                         const std::vector<size_t>& requiredItemCounts, const ObjectOrigin& itemOrigin);

    static std::shared_ptr<StripPackingProblem> fromInstancePath(const std::string& instancePath, ObjectOrigin itemOrigin=ObjectOrigin::AlignToCenter);
    static Color getItemColor(const std::string& itemName);

    [[nodiscard]] const AABB &getContainer() const;
    [[nodiscard]] const std::vector<std::shared_ptr<ModelSpaceMesh>>& getRequiredItems() const;
    [[nodiscard]] const std::vector<size_t>& getRequiredItemCounts() const;

    [[nodiscard]] std::vector<std::shared_ptr<ModelSpaceMesh>> listRequiredItems() const;
    [[nodiscard]] size_t getTotalNumberOfItems() const;
    [[nodiscard]] const std::unordered_map<std::shared_ptr<ModelSpaceMesh>, size_t>& getRequiredItemsMap() const;
    [[nodiscard]] float getTotalItemVolume() const;

    [[nodiscard]] const std::string& getName() const;
    [[nodiscard]] const std::string& getInstancePath() const;
    [[nodiscard]] ObjectOrigin getItemOrigin() const;
};

#endif //STRIPPACKINGPROBLEM_H
