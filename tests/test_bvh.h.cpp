//
// Created by Jonas Tollenaere on 05/09/2025.
//

#include <filesystem>
#include <iostream>
#include <gtest/gtest.h>

#include "meshcore/acceleration/BoundingVolumeHierarchy.h"
#include "meshcore/acceleration/CachingBoundsTreeFactory.h"
#include "meshcore/core/WorldSpaceMesh.h"
#include "meshcore/utility/FileParser.h"
#include "meshcore/utility/random.h"

TEST(BVH, Construction) {

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
            auto bvh = CachingBoundsTreeFactory<BoundingVolumeHierarchy>::getBoundsTree(modelSpaceMesh);
            end = std::chrono::high_resolution_clock::now();
            auto convexHullTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            std::cout << "Computed bvh for " << modelSpaceMesh->getName() << " in " << convexHullTime << " ms (File loading took " << fileLoadingTime << " ms)" << std::endl;

            // The result should be convex
            EXPECT_FALSE(bvh->getNodes().empty());
            EXPECT_FALSE(bvh->getTriangles().empty());
        }
    }
}

TEST(BVH, RandomWalk) {

    // Simple random walk of an item in a container to run the collision detection pipeline

    // Load the item and container
    std::shared_ptr<WorldSpaceMesh> itemWorldSpaceMesh = std::make_shared<WorldSpaceMesh>(FileParser::loadMeshFile(MESHCORE_DATA_DIR + std::string("E. F. Silva et al. 2021/banana.stl")));
    std::shared_ptr<WorldSpaceMesh> containerWorldSpaceMesh = std::make_shared<WorldSpaceMesh>(FileParser::loadMeshFile(MESHCORE_DATA_DIR + std::string("E. F. Silva et al. 2021/stone_1.obj")));

    containerWorldSpaceMesh->getModelTransformation().factorScale(2.0f);

    containerWorldSpaceMesh->getModelTransformation().deltaPosition(-2.0f*containerWorldSpaceMesh->getModelSpaceMesh()->getBounds().getCenter()); // Put the container in the center of the world

    // Make the item small enough to fit inside the container
    itemWorldSpaceMesh->getModelTransformation().setScale(0.5f);

    // Create the Bounding Volume Hierarchy which will speed up intersection tests with the container
    auto bvh = CachingBoundsTreeFactory<BoundingVolumeHierarchy>::getBoundsTree(containerWorldSpaceMesh->getModelSpaceMesh());

    // Test new random transformations for the item
    Random random;
    for(int i=0; i<1e3; i++){

        // 0. Sample a random change to position and orientation
        const Transformation originalTransformation = itemWorldSpaceMesh->getModelTransformation();
        auto translation = containerWorldSpaceMesh->getModelSpaceMesh()->getBounds().getHalf() * glm::vec3(random.nextFloat(-0.1f, 0.1f), random.nextFloat(-0.1f, 0.1f), random.nextFloat(-0.1f, 0.1f));
        itemWorldSpaceMesh->getModelTransformation().deltaPosition(translation);
        Quaternion randomRotation(random.nextFloat(-0.1f, 0.1f), random.nextFloat(-0.1f, 0.1f), random.nextFloat(-0.1f, 0.1f));
        itemWorldSpaceMesh->getModelTransformation().factorRotation(randomRotation);

        // 1. Check if at least one of the vertices is inside the container
        {
            // Transform the vertex to the right coordinate space, a good explanation about transformation matrices and coordinate spaces can be found here:
            // https://www.tapatalk.com/groups/ultimate3d_community/understanding-transformation-matrices-t5483.html
            const auto itemModelSpaceVertex = itemWorldSpaceMesh->getModelSpaceMesh()->getVertices().at(0); // This is the first vertex in the coordinate system of the item
            const auto worldSpaceVertex = itemWorldSpaceMesh->getModelTransformation().transformVertex(itemModelSpaceVertex); // This is the coordinate of the vertex where it is actually located in the world
            const auto containerModelSpaceVertex = containerWorldSpaceMesh->getModelTransformation().inverseTransformVertex(worldSpaceVertex); // This is the coordinate of the vertex when you bring it into the coordinate system of the container

            // Check if this vertex lies inside the container
            if(!bvh->containsPoint(containerModelSpaceVertex)){
                // This position is not feasible, revert to the previous transformation and go to the next iteration
                itemWorldSpaceMesh->getModelTransformation() = originalTransformation;
                continue;
            }

        }

        // 2. Check if none of the triangles of the item intersect the container
        {
            if(Intersection::intersect(*itemWorldSpaceMesh, *containerWorldSpaceMesh)){
                // This position is not feasible, revert to the previous transformation and go to the next iteration
                itemWorldSpaceMesh->getModelTransformation() = originalTransformation;
            }
        }
    }
}