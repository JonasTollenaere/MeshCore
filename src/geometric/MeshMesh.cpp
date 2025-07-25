//
// Created by Jonas on 5/11/2021.
//

#include "meshcore/geometric/Intersection.h"
#include "meshcore/geometric/GJK.h"
#include "meshcore/acceleration/CachingBoundsTreeFactory.h"
#include "meshcore/acceleration/AABBVolumeHierarchy.h"
#include <tbb/parallel_for.h>

#include "meshcore/acceleration/BoundingVolumeHierarchy.h"

namespace Intersection {

    /**
     * @brief Tests whether two meshes intersect.
     *
     * This function performs an intersection test between two WorldSpaceMeshes on a triangular level.
     * These queries are accelerated using bounding volume hierarchies.
     * This function does not implement any quick rejection tests like an AABB intersection,
     * the best option depends on the use case and should therefore be implemented by the user.
     *
     * @param worldSpaceMeshA The first worldSpaceMesh
     * @param worldSpaceMeshB The second worldSpaceMesh
     * @return True if the worldSpaceMeshes intersect, false otherwise.
     */
    bool intersect(const WorldSpaceMesh& worldSpaceMeshA, const WorldSpaceMesh& worldSpaceMeshB){

        // Actual intersection test
        const auto triangleCountA = worldSpaceMeshA.getModelSpaceMesh()->getTriangles().size();
        const auto triangleCountB = worldSpaceMeshB.getModelSpaceMesh()->getTriangles().size();
        const auto& simplerObject = triangleCountA < triangleCountB ? worldSpaceMeshA : worldSpaceMeshB;
        const auto& complexObject = triangleCountA < triangleCountB ? worldSpaceMeshB : worldSpaceMeshA;

        const auto& simpleTransformation = simplerObject.getModelTransformation();
        const auto& complexTransformation = complexObject.getModelTransformation();

        bool equalRotation = simpleTransformation.getRotation() == complexTransformation.getRotation();
        bool equalScaling = simpleTransformation.getScale() == complexTransformation.getScale();

        if (equalRotation && equalScaling) {

            // The specific case were the scaling and rotation of the items are equal
            const auto simpleToComplexTranslation = - complexObject.getModelTransformation().getPosition() + simplerObject.getModelTransformation().getPosition();
            const auto& complexObjectTree = CachingBoundsTreeFactory<BoundingVolumeHierarchy>::getBoundsTree(complexObject.getModelSpaceMesh());
            const auto& simplerObjectTree = CachingBoundsTreeFactory<BoundingVolumeHierarchy>::getBoundsTree(simplerObject.getModelSpaceMesh());

            for (const auto & node : simplerObjectTree->getNodes()) {
                if (complexObjectTree->intersectsAABB(node.bounds.getTranslated(simpleToComplexTranslation))) {
                    for (int i = 0; i < node.triangleCount; ++i) {
                        const VertexTriangle& triangle = simplerObjectTree->getTriangles()[node.firstChildOrTriangleIndex + i];
                        if (complexObjectTree->intersectsTriangle(triangle.getTranslated(simpleToComplexTranslation))) {
                            return true;
                        }
                    }
                }
            }
            return false;
        }

        // The general case where the triangles have to be transformed
        const auto simpleToComplexTransformation = complexObject.getModelTransformation().getInverse() * simplerObject.getModelTransformation();
        const auto& complexObjectTree = CachingBoundsTreeFactory<BoundingVolumeHierarchy>::getBoundsTree(complexObject.getModelSpaceMesh());
        const auto& simplerObjectMesh = simplerObject.getModelSpaceMesh();

        for (const auto & indexTriangle : simplerObjectMesh->getTriangles()) {
            VertexTriangle triangle(simplerObjectMesh->getVertices()[indexTriangle.vertexIndex0],
                                    simplerObjectMesh->getVertices()[indexTriangle.vertexIndex1],
                                    simplerObjectMesh->getVertices()[indexTriangle.vertexIndex2]);
            if (complexObjectTree->intersectsTriangle(triangle.getTransformed(simpleToComplexTransformation))) {
                return true;
            }
        }
        return false;
    }

    bool debugIntersects(const WorldSpaceMesh& worldSpaceMeshA, const WorldSpaceMesh& worldSpaceMeshB){
#if NDEBUG
        std::cout << "[MESHCORE] Using a naive triangleTriangleIntersects implementation -- use for debugging only" << std::endl;
#endif

        auto numOuterLoop =  worldSpaceMeshA.getModelSpaceMesh()->getTriangles().size();

        tbb::parallel_for(tbb::blocked_range<int>(0,numOuterLoop), [&](tbb::blocked_range<int> r) {
            for (int i = r.begin(); i < r.end(); ++i) {
                IndexTriangle thisTriangle = worldSpaceMeshA.getModelSpaceMesh()->getTriangles()[i];

                Transformation thisModelTransformation = worldSpaceMeshA.getModelTransformation();

                Vertex worldSpaceVertex0 = thisModelTransformation.transformVertex(worldSpaceMeshA.getModelSpaceMesh()->getVertices()[thisTriangle.vertexIndex0]);
                Vertex worldSpaceVertex1 = thisModelTransformation.transformVertex(worldSpaceMeshA.getModelSpaceMesh()->getVertices()[thisTriangle.vertexIndex1]);
                Vertex worldSpaceVertex2 = thisModelTransformation.transformVertex(worldSpaceMeshA.getModelSpaceMesh()->getVertices()[thisTriangle.vertexIndex2]);

                for (IndexTriangle otherTriangle: worldSpaceMeshB.getModelSpaceMesh()->getTriangles()) {

                    Transformation otherModelTransformation = worldSpaceMeshB.getModelTransformation();

                    Vertex otherWorldSpaceVertex0 = otherModelTransformation.transformVertex(worldSpaceMeshB.getModelSpaceMesh()->getVertices()[otherTriangle.vertexIndex0]);
                    Vertex otherWorldSpaceVertex1 = otherModelTransformation.transformVertex(worldSpaceMeshB.getModelSpaceMesh()->getVertices()[otherTriangle.vertexIndex1]);
                    Vertex otherWorldSpaceVertex2 = otherModelTransformation.transformVertex(worldSpaceMeshB.getModelSpaceMesh()->getVertices()[otherTriangle.vertexIndex2]);
                    bool intersects = Intersection::intersect(VertexTriangle(otherWorldSpaceVertex0, otherWorldSpaceVertex1, otherWorldSpaceVertex2), VertexTriangle(worldSpaceVertex0, worldSpaceVertex1, worldSpaceVertex2));
                    if(intersects){
                        return true;
                    }
                }
            }
        });
        return false;
    }

    bool inside(const WorldSpaceMesh& worldSpaceMeshA, const WorldSpaceMesh& worldSpaceMeshB){
#if NDEBUG
        std::cout << "[MESHCORE] Using a naive rayTriangleInside implementation -- use for debugging only" << std::endl;
#endif
        for(Vertex modelSpaceOrigin: worldSpaceMeshA.getModelSpaceMesh()->getVertices()){
            Vertex worldSpaceOrigin = worldSpaceMeshA.getModelTransformation().transformVertex(modelSpaceOrigin);
            Vertex otherModelSpaceOrigin = worldSpaceMeshB.getModelTransformation().inverseTransformVertex(worldSpaceOrigin);
            Ray otherModelSpaceRay(otherModelSpaceOrigin, glm::vec3(1, 0, 0)); // Direction doesn't matter

            int numberOfIntersections = 0;
            for(IndexTriangle triangle: worldSpaceMeshB.getModelSpaceMesh()->getTriangles()){
                Vertex otherModelSpaceVertex0 = worldSpaceMeshB.getModelSpaceMesh()->getVertices()[triangle.vertexIndex0];
                Vertex otherModelSpaceVertex1 = worldSpaceMeshB.getModelSpaceMesh()->getVertices()[triangle.vertexIndex1];
                Vertex otherModelSpaceVertex2 = worldSpaceMeshB.getModelSpaceMesh()->getVertices()[triangle.vertexIndex2];
                if(Intersection::intersect(otherModelSpaceRay, VertexTriangle(otherModelSpaceVertex0, otherModelSpaceVertex1, otherModelSpaceVertex2))){
                    numberOfIntersections++;
                }
            }
            if(numberOfIntersections%2 == 0){
                return false;
            }
        }
        return true;
    }
}

namespace Distance{

    float distance(const WorldSpaceMesh& worldSpaceMeshA, const WorldSpaceMesh& worldSpaceMeshB){

        const auto& modelSpaceMeshA = worldSpaceMeshA.getModelSpaceMesh();
        const auto& modelSpaceMeshB = worldSpaceMeshB.getModelSpaceMesh();
        const auto& modelSpaceTransformationA = worldSpaceMeshA.getModelTransformation();
        const auto& modelSpaceTransformationB = worldSpaceMeshB.getModelTransformation();

        // Use the GJK algorithm if both meshes are convex
        if(modelSpaceMeshA->isConvex() && modelSpaceMeshB->isConvex()){
            auto gjkResult = GJK::computeDistanceSqr(worldSpaceMeshA, worldSpaceMeshB);
            if(gjkResult.has_value()){
                return glm::sqrt(gjkResult.value());
            }
            else{
                return 0.0f;
            }
        }

        // If not convex, use methods based on AABBVolumeHierarchy
        if(modelSpaceMeshA->getTriangles().size() >= modelSpaceMeshB->getTriangles().size()){
            const auto modelBToModelASpaceTransformation = modelSpaceTransformationA.getInverseMatrix() * modelSpaceTransformationB.getMatrix();

            const auto &modelBVertices = modelSpaceMeshB->getVertices();
            const auto &treeA = CachingBoundsTreeFactory<AABBVolumeHierarchy>::getBoundsTree(modelSpaceMeshA);
            AABBVolumeHierarchy::ClosestTriangleQueryResult closestTriangleQueryResultA; // By keeping the minimum distance found so far, the queries can build upon each other, which is faster than querying the whole tree each time.

            for (const auto &triangle: modelSpaceMeshB->getTriangles()){
                VertexTriangle vertexTriangle(modelBVertices[triangle.vertexIndex0], modelBVertices[triangle.vertexIndex1], modelBVertices[triangle.vertexIndex2]);
                auto transformedTriangle = vertexTriangle.getTransformed(modelBToModelASpaceTransformation);

                treeA->queryClosestTriangle(transformedTriangle, &closestTriangleQueryResultA);

                if(closestTriangleQueryResultA.lowerDistanceBoundSquared <= 0.0){
                    return 0.0f;
                }

            }
            return glm::sqrt(closestTriangleQueryResultA.lowerDistanceBoundSquared)*modelSpaceTransformationA.getScale();
        }
        else {
            const auto modelAToModelBSpaceTransformation = modelSpaceTransformationB.getInverseMatrix() * modelSpaceTransformationA.getMatrix();
            const auto &modelAVertices = modelSpaceMeshA->getVertices();
            const auto &treeB = CachingBoundsTreeFactory<AABBVolumeHierarchy>::getBoundsTree(modelSpaceMeshB);
            AABBVolumeHierarchy::ClosestTriangleQueryResult closestTriangleQueryResultB;  // By keeping the minimum distance found so far, the queries can build upon each other, which is faster than querying the whole tree each time.


            for (const auto &triangle: modelSpaceMeshA->getTriangles()){
                VertexTriangle vertexTriangle(modelAVertices[triangle.vertexIndex0], modelAVertices[triangle.vertexIndex1], modelAVertices[triangle.vertexIndex2]);
                auto transformedTriangle = vertexTriangle.getTransformed(modelAToModelBSpaceTransformation);

                treeB->queryClosestTriangle(transformedTriangle, &closestTriangleQueryResultB);

                if(closestTriangleQueryResultB.lowerDistanceBoundSquared <= 0.0){
                    return 0.0f;
                }
            }
            return glm::sqrt(closestTriangleQueryResultB.lowerDistanceBoundSquared)*modelSpaceTransformationB.getScale();
        }
    }

    float distance(const WorldSpaceMesh& worldSpaceMeshA, const WorldSpaceMesh& worldSpaceMeshB, Vertex* closestVertexA, Vertex* closestVertexB){

        const auto& modelSpaceMeshA = worldSpaceMeshA.getModelSpaceMesh();
        const auto& modelSpaceMeshB = worldSpaceMeshB.getModelSpaceMesh();

        // Use the GJK algorithm if both meshes are convex
        if(modelSpaceMeshA->isConvex() && modelSpaceMeshB->isConvex()){
            auto gjkResult = GJK::computeClosestPoints(worldSpaceMeshA, worldSpaceMeshB);
            if(gjkResult.has_value()){
                *closestVertexA = gjkResult.value().first;
                *closestVertexB = gjkResult.value().second;
                return glm::distance(*closestVertexA, *closestVertexB);
            }
            else{
                return 0.0f;
            }
        }

        // If not convex, use methods based on AABBVolumeHierarchy
        const auto& modelSpaceTransformationA = worldSpaceMeshA.getModelTransformation();
        const auto& modelSpaceTransformationB = worldSpaceMeshB.getModelTransformation();
        if(modelSpaceMeshA->getTriangles().size() >= modelSpaceMeshB->getTriangles().size()){
            const auto modelBToModelASpaceTransformation = modelSpaceTransformationA.getInverseMatrix() * modelSpaceTransformationB.getMatrix();

            const auto &modelBVertices = modelSpaceMeshB->getVertices();
            const auto &treeA = CachingBoundsTreeFactory<AABBVolumeHierarchy>::getBoundsTree(modelSpaceMeshA);
            AABBVolumeHierarchy::ClosestTriangleQueryResult closestTriangleQueryResultA;  // By keeping the minimum distance found so far, the queries can build upon each other, which is faster than querying the whole tree each time.

            for (const auto &triangle: modelSpaceMeshB->getTriangles()){
                VertexTriangle vertexTriangleB(modelBVertices[triangle.vertexIndex0], modelBVertices[triangle.vertexIndex1], modelBVertices[triangle.vertexIndex2]);
                auto transformedTriangle = vertexTriangleB.getTransformed(modelBToModelASpaceTransformation);

                closestTriangleQueryResultA.closestTriangle = nullptr;
                treeA->queryClosestTriangle(transformedTriangle, &closestTriangleQueryResultA);

                if(closestTriangleQueryResultA.closestTriangle != nullptr){
                    // Closest was updated
                    *closestVertexA = modelSpaceTransformationA.transformVertex(closestTriangleQueryResultA.closestVertex);
                    *closestVertexB = vertexTriangleB.getTransformed(modelSpaceTransformationB).getClosestPoint(*closestVertexA);
                }

                if(closestTriangleQueryResultA.lowerDistanceBoundSquared <= 0.0){
                    break;
                }
            }

            return glm::sqrt(closestTriangleQueryResultA.lowerDistanceBoundSquared)*modelSpaceTransformationA.getScale();
        }
        else {
            const auto modelAToModelBSpaceTransformation = modelSpaceTransformationB.getInverseMatrix() * modelSpaceTransformationA.getMatrix();
            const auto &modelAVertices = modelSpaceMeshA->getVertices();
            const auto &treeB = CachingBoundsTreeFactory<AABBVolumeHierarchy>::getBoundsTree(modelSpaceMeshB);
            AABBVolumeHierarchy::ClosestTriangleQueryResult closestTriangleQueryResultB;  // By keeping the minimum distance found so far, the queries can build upon each other, which is faster than querying the whole tree each time.

            for (const auto &triangle: modelSpaceMeshA->getTriangles()){
                VertexTriangle vertexTriangleA(modelAVertices[triangle.vertexIndex0], modelAVertices[triangle.vertexIndex1], modelAVertices[triangle.vertexIndex2]);
                auto transformedTriangle = vertexTriangleA.getTransformed(modelAToModelBSpaceTransformation);

                closestTriangleQueryResultB.closestTriangle = nullptr;
                treeB->queryClosestTriangle(transformedTriangle, &closestTriangleQueryResultB);

                if(closestTriangleQueryResultB.closestTriangle != nullptr){
                    // Closest was updated
                    *closestVertexB = modelSpaceTransformationB.transformVertex(closestTriangleQueryResultB.closestVertex);
                    *closestVertexA = vertexTriangleA.getTransformed(modelSpaceTransformationA).getClosestPoint(*closestVertexB);
                }

                if(closestTriangleQueryResultB.lowerDistanceBoundSquared <= 0.0){
                    break;
                }
            }

            return glm::sqrt(closestTriangleQueryResultB.lowerDistanceBoundSquared)*modelSpaceTransformationB.getScale();
        }
    }
}