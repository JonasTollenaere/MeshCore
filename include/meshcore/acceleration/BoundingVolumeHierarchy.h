//
// Created by Jonas Tollenaere on 22/05/2025.
//

#ifndef FLATBVH_H
#define FLATBVH_H

#include "meshcore/core/VertexTriangle.h"
#include "meshcore/core/ModelSpaceMesh.h"
#include "meshcore/core/Ray.h"
#include <vector>

class BoundingVolumeHierarchy {
public:
    struct Node {
        AABB bounds;
        unsigned short triangleCount=0;
        bool split = false;
        unsigned int firstChildOrTriangleIndex=0; // The node is either split and contains no triangles, a leaf node and contains triangles

        [[nodiscard]] bool isEmpty() const {
            return triangleCount==0 && !split;
        }
    };

    struct ClosestTriangleQueryResult{
        const VertexTriangle* closestTriangle = nullptr;
        Vertex closestVertex{};
        float lowerDistanceBoundSquared = std::numeric_limits<float>::max();
    };

private:
    std::vector<VertexTriangle> triangles;
    std::vector<Node> nodes;
public:
    explicit BoundingVolumeHierarchy(const std::shared_ptr<ModelSpaceMesh> &mesh);
    [[nodiscard]] bool intersectsTriangle(const VertexTriangle &triangle) const;
    [[nodiscard]] bool intersectsAABB(const AABB &aabb) const;
    [[nodiscard]] bool containsPoint(const glm::vec3& point) const;
    void queryClosestTriangle(const Vertex &vertex, ClosestTriangleQueryResult* result) const;
    void queryClosestTriangle(const VertexTriangle &triangle, ClosestTriangleQueryResult* result) const;



    [[nodiscard]] float getShortestDistanceSquared(const glm::vec3& point) const;
    [[nodiscard]] const std::vector<Node>& getNodes() const;
    [[nodiscard]] const std::vector<VertexTriangle>& getTriangles() const;

private:
    [[nodiscard]] bool hitsBacksideFirst(const Ray &ray) const;

};



#endif //FLATBVH_H
