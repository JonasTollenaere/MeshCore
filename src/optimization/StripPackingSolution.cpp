//
// Created by Jonas Tollenaere on 02/07/2025.
//

#include "meshcore/optimization/StripPackingSolution.h"

#include "meshcore/geometric/Intersection.h"
#include <fstream>
#include <iostream>

StripPackingSolution::StripPackingSolution(const std::shared_ptr<StripPackingProblem> &problem):
    problem(problem), cachedAABBs(problem->getTotalNumberOfItems(), std::nullopt) {
    // Initialize items and item names
    items.reserve(problem->getTotalNumberOfItems());
    itemNames.reserve(problem->getTotalNumberOfItems());
    for (const auto& mesh : problem->listRequiredItems()) {
        items.emplace_back(mesh); // Emplace back constructs a WorldSpaceMesh with identity transformation
        itemNames.push_back(mesh->getName());
    }
}

StripPackingSolution::StripPackingSolution(const StripPackingSolution &other): problem(other.problem), itemNames(other.itemNames), cachedAABBs(other.cachedAABBs) {
    items.reserve(other.items.size());
    for (const auto& item : other.items) {
        items.emplace_back(item);
    }
}

const std::vector<WorldSpaceMesh>& StripPackingSolution::getItems() const {
    return items;
}

const WorldSpaceMesh& StripPackingSolution::getItem(size_t itemIndex) const {
    return items[itemIndex];
}

const std::string & StripPackingSolution::getItemName(size_t itemIndex) const {
    return itemNames[itemIndex];
}

const AABB & StripPackingSolution::getItemAABB(size_t itemIndex) const {

    // Compute the AABB if not cached
    if (!cachedAABBs[itemIndex]){
        const auto& item = items[itemIndex];

        // Considering the vertices on the convex hull suffices for computing the AABB
        const auto& modelSpaceVertices = item.getModelSpaceMesh()->getConvexHull()->getVertices();

        // Compute the AABB from these vertices and the item transformation
        cachedAABBs[itemIndex] = AABBFactory::createAABB(modelSpaceVertices, item.getModelTransformation());
    }

    // Return the reference to the cached AABB
    return *cachedAABBs[itemIndex];
}

const std::shared_ptr<StripPackingProblem> & StripPackingSolution::getProblem() const {
    return problem;
}

const Transformation & StripPackingSolution::getItemTransformation(size_t itemIndex) const {
    return items[itemIndex].getModelTransformation();
}

void StripPackingSolution::setItemTransformation(size_t itemIndex, const Transformation &transformation) {
    // Reset cached AABB when the transformation is updated
    cachedAABBs[itemIndex].reset();
    items[itemIndex].setModelTransformation(transformation);
}

float StripPackingSolution::computeTotalHeight() const {
    float maximumHeight = 0.0f;
    for (size_t itemIndex = 0; itemIndex < items.size(); ++itemIndex){
        // Update the maximum height based on the AABB of each item
        maximumHeight = std::max(maximumHeight, getItemAABB(itemIndex).getMaximum().z);
    }
    return maximumHeight;
}

bool StripPackingSolution::isFeasible() const {

    // Check if the items are embedded within the container
    for (size_t itemIndex = 0; itemIndex < itemNames.size(); ++itemIndex) {
        bool contained = problem->getContainer().containsAABB(getItemAABB(itemIndex));
        if (!contained){
            return false;
        }
    }

    // Check if the items collide with each other
    for (size_t firstItemIndex = 0; firstItemIndex < itemNames.size(); ++firstItemIndex) {

        const auto& firstItem = items[firstItemIndex];
        const auto& firstItemAABB = getItemAABB(firstItemIndex);

        for (size_t secondItemIndex = firstItemIndex+1; secondItemIndex < itemNames.size(); ++secondItemIndex) {

            const auto& secondItem = items[secondItemIndex];
            const auto& secondItemAABB = getItemAABB(secondItemIndex);

            // AABB separation check
            if (!Intersection::intersect(firstItemAABB, secondItemAABB)) {
                continue; // No intersection if AABBs do not intersect
            }

            // Mesh intersection check
            if (Intersection::intersect(firstItem, secondItem)) {
                return false;
            }
        }
    }
    return true;
}

std::shared_ptr<AbstractSolution> StripPackingSolution::clone() const {
    return std::make_shared<StripPackingSolution>(*this);
}

std::shared_ptr<StripPackingSolution> StripPackingSolution::fromJson(nlohmann::ordered_json &json) {

    auto itemOrigin = ObjectOrigin::Original;
    if(json.contains("itemOrigin")){
        if(json["itemOrigin"] == "AlignToCenter"){
            itemOrigin = ObjectOrigin::AlignToCenter;
        }
        else if(json["itemOrigin"] == "AlignToMinimum"){
            itemOrigin = ObjectOrigin::AlignToMinimum;
        }
        else if(json["itemOrigin"] == "AlignToCentroid"){
            itemOrigin = ObjectOrigin::AlignToCentroid;
        }
        // If not defined, we assume the original reference coordinates were kept
    }

    auto problem = StripPackingProblem::fromInstancePath(json["instancePath"], itemOrigin);
    auto solution = std::make_shared<StripPackingSolution>(problem);

    std::map<std::string, std::vector<size_t>> itemIndexMap;

    for (size_t itemIndex = 0; itemIndex < solution->getItems().size(); ++itemIndex) {
        const auto& item = solution->getItems()[itemIndex];
        auto name = item.getModelSpaceMesh()->getName();
        if(itemIndexMap.find(name) == itemIndexMap.end()){
            itemIndexMap[name] = std::vector<size_t>();
        }
        itemIndexMap[name].push_back(itemIndex);
    }

    for (const auto &jsonItem: json["items"]){

        // Get the index of an item with this name that has not been assigned a transformation yet
        auto& entry = itemIndexMap[jsonItem["name"]];
        size_t itemIndex = entry.back();
        entry.pop_back();

        Transformation transformation;
        transformation.setRotation(Quaternion(jsonItem["transformation"]["yaw"], jsonItem["transformation"]["pitch"], jsonItem["transformation"]["roll"]));
        transformation.setPosition(glm::vec3(jsonItem["transformation"]["position"][0],
                                             jsonItem["transformation"]["position"][1],
                                             jsonItem["transformation"]["position"][2]));

        solution->setItemTransformation(itemIndex, transformation);

    }

    if (!glm::epsilonEqual(static_cast<float>(json["totalHeight"]), solution->computeTotalHeight(), 1e-3f)) {
        std::cout << "Warning: The total height in the JSON does not match the computed maximum height." << std::endl;
    }

    return solution;
}

std::shared_ptr<StripPackingSolution> StripPackingSolution::fromJson(std::string &path) {

    if (!std::filesystem::exists(path)) {
        return nullptr;
    }

    std::ifstream file(path);
    nlohmann::ordered_json json;
    file >> json;
    return fromJson(json);
}

nlohmann::ordered_json StripPackingSolution::toJson() const {
    nlohmann::ordered_json result;
    for (size_t itemIndex = 0; itemIndex < this->getItems().size(); ++itemIndex) {

        nlohmann::ordered_json itemJson;
        const auto& transformation = getItemTransformation(itemIndex);

        nlohmann::ordered_json transformationJson;
        transformationJson["position"] = {transformation.getPosition().x, transformation.getPosition().y, transformation.getPosition().z};
        transformationJson["yaw"] = transformation.getRotation().getYaw();
        transformationJson["pitch"] = transformation.getRotation().getPitch();
        transformationJson["roll"] = transformation.getRotation().getRoll();
        transformationJson["matrix"] = {{transformation.getMatrix()[0][0], transformation.getMatrix()[0][1], transformation.getMatrix()[0][2], transformation.getMatrix()[0][3]},
                    {transformation.getMatrix()[1][0], transformation.getMatrix()[1][1], transformation.getMatrix()[1][2], transformation.getMatrix()[1][3]},
                    {transformation.getMatrix()[2][0], transformation.getMatrix()[2][1], transformation.getMatrix()[2][2], transformation.getMatrix()[2][3]},
                    {transformation.getMatrix()[3][0], transformation.getMatrix()[3][1], transformation.getMatrix()[3][2], transformation.getMatrix()[3][3]}};

        itemJson["name"] = this->getItemName(itemIndex);
        itemJson["transformation"] = transformationJson;

        result["items"].emplace_back(itemJson);
    }
    result["totalHeight"] = this->computeTotalHeight();
    result["instancePath"] = this->getProblem()->getInstancePath();

    switch (this->getProblem()->getItemOrigin()) {
        case ObjectOrigin::AlignToCenter:
            result["itemOrigin"] = "AlignToCenter";
            break;
        case ObjectOrigin::AlignToMinimum:
            result["itemOrigin"] = "AlignToMinimum";
            break;
        case ObjectOrigin::AlignToCentroid:
            result["itemOrigin"] = "AlignToCentroid";
            break;
        case ObjectOrigin::Original:
        default:
            result["itemOrigin"] = "Original";
            break;
    }

    return result;
}
