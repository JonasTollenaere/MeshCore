//
// Created by tolle on 20/09/2023.
//

#include <filesystem>
#include <iostream>
#include <glm/gtc/epsilon.hpp>
#include <gtest/gtest.h>

#include "meshcore/utility/FileParser.h"

TEST(ConvexHullTest, TestConvexHulls) {

    std::vector<std::string> objectFolders;
    objectFolders.emplace_back(MESHCORE_DATA_DIR + std::string("Liu et al. 2015/"));
    objectFolders.emplace_back(MESHCORE_DATA_DIR + std::string("Tollenaere, J. et al/Items"));
    objectFolders.emplace_back(MESHCORE_DATA_DIR + std::string("Tollenaere, J. et al/Containers"));

    for (const auto &folder: objectFolders){
        for (const auto &entry : std::filesystem::directory_iterator(folder)) {

            auto start = std::chrono::high_resolution_clock::now();
            auto modelSpaceMesh = FileParser::loadMeshFile(entry.path().string());

            if(!modelSpaceMesh){
                continue;
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto fileLoadingTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            start = std::chrono::high_resolution_clock::now();
            auto convexHull = modelSpaceMesh->getConvexHull();
            end = std::chrono::high_resolution_clock::now();
            auto convexHullTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            std::cout << "Computed convex hull for " << modelSpaceMesh->getName() << " in " << convexHullTime << " ms (File loading took " << fileLoadingTime << " ms)" << std::endl;

            // The result should be convex
            EXPECT_TRUE(convexHull->isConvex());

            // If the original mesh is convex, the resulting volume should be the same
            if(modelSpaceMesh->isConvex()){
                EXPECT_TRUE(glm::epsilonEqual(modelSpaceMesh->getVolume()/convexHull->getVolume(), 1.0f, 1e-5f));
//                printf("%s: Original convex mesh volume: %.6f, convex hull volume: %.6f\n", modelSpaceMesh->getName().c_str(), modelSpaceMesh->getVolume(), convexHull->getVolume());
            }

            // The volume can never be smaller than the original mesh
            EXPECT_FALSE(convexHull->getVolume()/modelSpaceMesh->getVolume() < 1.0f - 1e-5f);
//            printf("%s: Original mesh volume: %.6f, convex hull volume: %.6f\n", modelSpaceMesh->getName().c_str(), modelSpaceMesh->getVolume(), convexHull->getVolume());
        }
    }
}