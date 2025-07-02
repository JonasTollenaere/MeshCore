//
// Created by Jonas on 5/03/2023.
//

#include "meshcore/optimization/StripPackingProblem.h"

#include <fstream>
#include <iostream>

#include "meshcore/utility/FileParser.h"
#include "meshcore/geometric/Intersection.h"
#include <boost/functional/hash.hpp>

#ifndef MESHCORE_DATA_DIR
#define MESHCORE_DATA_DIR ""
#endif

float StripPackingProblem::getTotalItemVolume() const {
    return totalItemVolume;
}

size_t StripPackingProblem::getTotalNumberOfItems() const {
    return totalNumberOfItems;
}

const std::unordered_map<std::shared_ptr<ModelSpaceMesh>, size_t> &StripPackingProblem::getRequiredItemsMap() const {
    return requiredItemsMap;
}

const std::vector<std::shared_ptr<ModelSpaceMesh>> &StripPackingProblem::getRequiredItems() const {
    return requiredItems;
}

const std::vector<size_t> &StripPackingProblem::getRequiredItemCounts() const {
    return requiredItemCounts;
}

Color StripPackingProblem::getItemColor(const std::string& itemName) {

    // Manually defined colors for specific items
    static std::unordered_map<std::string, Color> colorMap = {{"tetrahedron.obj", Color::Yellow()},
                                                              {"star.obj", Color::LightBlue()},
                                                              {"cube.obj", Color::Blue()},
                                                              {"ring.obj", Color::Red()},
                                                              {"arrow.obj", Color::Green()},
                                                              {"polytope1.obj", Color(129.0f/255.0f, 88.0f/255.0f, 63.0f/255.0f, 1.0f)},
                                                              {"polytope2.obj", Color(222.0f/255.0f, 12.0f/255.0f, 82.0f/255.0f, 1.0f)},
                                                              {"polytope3.obj", Color(73.0f/255.0f, 169.0f/255.0f, 213.0f/255.0f, 1.0f)},
                                                              {"polytope4.obj", Color(89.0f/255.0f, 96.0f/255.0f, 144.0f/255.0f, 1.0f)},
                                                              {"polytope5.obj", Color(174.0f/255.0f, 98.0f/255.0f, 251.0f/255.0f, 1.0f)},
                                                              {"polytope6.obj", Color(152.0f/255.0f, 249.0f/255.0f, 78.0f/255.0f, 1.0f)},
                                                              {"polytope7.obj", Color::Locust()},
                                                              {"Convex1", Color::Red()},
                                                              {"Convex2", Color::Yellow()},
                                                              {"Convex3", Color::Green()},
                                                              {"Convex4", Color::Orange()},
                                                              {"Convex5", Color::Blue()},
                                                              {"Convex6", Color::Red()},
                                                              {"Convex7", Color::Magenta()}};

    // Test if custom color is defined
    if(colorMap.find(itemName) == colorMap.end()){

        // Boost for portable hash function
        auto hash = boost::hash<std::string>{}(itemName);

        auto R = (hash & 0x0000FF);
        auto G = (hash & 0x00FF00) >> 8;
        auto B = (hash & 0xFF0000) >> 16;

        float r = static_cast<float>(R) / 255.0f;
        float g = static_cast<float>(G) / 255.0f;
        float b = static_cast<float>(B) / 255.0f;

        colorMap[itemName] = Color(r,g,b, 1.0f);
    }

    return colorMap[itemName];
}

ObjectOrigin StripPackingProblem::getItemOrigin() const {
    return itemOrigin;
}

std::shared_ptr<StripPackingProblem> StripPackingProblem::fromInstancePath(const std::string &instancePath, ObjectOrigin itemOrigin) {

    std::vector<std::shared_ptr<ModelSpaceMesh>> itemTypes;
    std::vector<size_t> itemDemand;
    std::string name;
    float containerSizeX;
    float containerSizeY;

    // Test if the problem file exists
    if (auto completePath = MESHCORE_DATA_DIR + instancePath; std::filesystem::exists(completePath)) {
        // Parse the JSON file
        std::ifstream stream(completePath);
        std::string problemJsonString;
        std::stringstream buffer;
        buffer << stream.rdbuf();
        problemJsonString = buffer.str();
        auto json = nlohmann::ordered_json::parse(problemJsonString);

        // Parse item types in json array
        auto itemTypesArray = json["item-types"];
        for (const auto& itemType : itemTypesArray) {
            std::string path = itemType["path"];
            size_t demands = itemType["demand"];
            std::string completeItemPath = MESHCORE_DATA_DIR + path;

            FileParser::loadMeshFile(completeItemPath);
            itemTypes.emplace_back(FileParser::loadMeshFile(completeItemPath));
            itemDemand.emplace_back(demands);
        }

        name = json["name"];

        const auto& containerJson = json["container"];
        containerSizeX = containerJson["size-x"];
        containerSizeY = containerJson["size-y"];
    }
    else {
        // Return a fake problem if the instance file doesn't exist
        auto absolutePath = std::filesystem::absolute(completePath);
        std::cout << "Warning: File " << absolutePath << " does not exist, returning dummy!" << std::endl;

        std::vector<Vertex> dummyVertices = {glm::vec3(0,0,0), glm::vec3(1,0,0), glm::vec3(0,1,0), glm::vec3(1,1,0),
                                             glm::vec3(0,0,1), glm::vec3(1,0,1), glm::vec3(0,1,1), glm::vec3(1,1,1)};
        itemTypes = {ModelSpaceMesh(dummyVertices).getConvexHull()};
        itemTypes[0]->setName("Unit Cube");
        itemDemand = {6};

        name = "DUMMY_PROBLEM";

        containerSizeX = 2;
        containerSizeY = 2;
    }

    if(itemOrigin == ObjectOrigin::AlignToCenter){
        // Center the items in their own coordinate space
        std::vector<std::shared_ptr<ModelSpaceMesh>> centeredItems;
        for (const auto& item : itemTypes){
            std::vector<Vertex> centeredVertices;
            centeredVertices.reserve(item->getVertices().size());
            auto center = item->getBounds().getCenter();
            for (const auto &vertex: item->getVertices()){
                centeredVertices.emplace_back(vertex - center);
            }
            auto centeredModelSpaceMesh = std::make_shared<ModelSpaceMesh>(centeredVertices, item->getTriangles());
            centeredModelSpaceMesh->setName(item->getName());
            centeredItems.emplace_back(centeredModelSpaceMesh);
        }
        assert(centeredItems.size() == itemTypes.size());
        itemTypes = centeredItems;
    }
    else if(itemOrigin == ObjectOrigin::AlignToMinimum){
        std::vector<std::shared_ptr<ModelSpaceMesh>> translatedItems;
        for (const auto& item : itemTypes){
            std::vector<Vertex> translatedVertices;
            translatedVertices.reserve(item->getVertices().size());
            auto minimum = item->getBounds().getMinimum();
            for (const auto &vertex: item->getVertices()){
                translatedVertices.emplace_back(vertex - minimum);
            }
            auto translatedModelSpaceMesh = std::make_shared<ModelSpaceMesh>(translatedVertices, item->getTriangles());
            translatedModelSpaceMesh->setName(item->getName());
            translatedItems.emplace_back(translatedModelSpaceMesh);
        }
        assert(translatedItems.size() == itemTypes.size());
        itemTypes = translatedItems;
    }
    else if(itemOrigin == ObjectOrigin::AlignToCentroid){
        std::vector<std::shared_ptr<ModelSpaceMesh>> translatedItems;
        for (const auto& item : itemTypes){
            std::vector<Vertex> translatedVertices;
            translatedVertices.reserve(item->getVertices().size());
            auto centroid = item->getVolumeCentroid();
            for (const auto &vertex: item->getVertices()){
                translatedVertices.emplace_back(vertex - centroid);
            }
            auto translatedModelSpaceMesh = std::make_shared<ModelSpaceMesh>(translatedVertices, item->getTriangles());
            translatedModelSpaceMesh->setName(item->getName());
            translatedItems.emplace_back(translatedModelSpaceMesh);
        }
        assert(translatedItems.size() == itemTypes.size());
        itemTypes = translatedItems;
    }

    // Set a sensible maximum container height
    float maximumContainerHeight = 0;
    for (int i = 0; i < itemTypes.size(); ++i){
        auto& item = itemTypes[i];
        auto itemHeight = item->getBounds().getMaximum().z - item->getBounds().getMinimum().z;
        maximumContainerHeight += itemHeight * static_cast<float>(itemDemand[i]);
    }

    AABB container = AABB(glm::vec3(0,0,0), glm::vec3(containerSizeX, containerSizeY, maximumContainerHeight));
    return std::make_shared<StripPackingProblem>(instancePath, name, container, itemTypes, itemDemand, itemOrigin);
}

const std::string &StripPackingProblem::getName() const {
    return name;
}

std::vector<std::shared_ptr<ModelSpaceMesh>> StripPackingProblem::listRequiredItems() const {
    std::vector<std::shared_ptr<ModelSpaceMesh>> result;
    for (int i = 0; i < requiredItems.size(); ++i){
        const auto& item = requiredItems[i];
        const auto& count = requiredItemCounts[i];
        for (int j = 0; j < count; ++j){
            result.push_back(item);
        }
    }
    return result;
}

StripPackingProblem::StripPackingProblem(std::string instancePath, std::string name, const AABB &container,
                                           const std::vector<std::shared_ptr<ModelSpaceMesh>> &requiredItems,
                                           const std::vector<size_t> &requiredItemCounts,
                                           const ObjectOrigin &itemOrigin) :
        container(container), requiredItems(requiredItems), requiredItemCounts(requiredItemCounts),
        name(std::move(name)), instancePath(std::move(instancePath)), itemOrigin(itemOrigin) {

    assert(requiredItems.size() == requiredItemCounts.size());
    totalItemVolume = 0.0f;
    totalNumberOfItems = 0;
    for (int i = 0; i < requiredItems.size(); ++i){
        const auto& item = requiredItems[i];
        const auto& count = requiredItemCounts[i];

        totalItemVolume += item->getVolume() * static_cast<float>(count);
        totalNumberOfItems += count;
        requiredItemsMap[item] = count;
    }
}

const std::string &StripPackingProblem::getInstancePath() const {
    return instancePath;
}

const AABB &StripPackingProblem::getContainer() const {
    return container;
}