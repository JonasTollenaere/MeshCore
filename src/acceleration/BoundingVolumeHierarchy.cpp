//
// Created by Jonas Tollenaere on 22/05/2025.
//

#include "meshcore/acceleration/BoundingVolumeHierarchy.h"

#include <numeric>
#include <stack>

#include "meshcore/acceleration/CachingBoundsTreeFactory.h"
#include "meshcore/acceleration/AABBVolumeHierarchy.h"
#include "meshcore/geometric/AABBTriangleData.h"

#define STACK_DEPTH 128

BoundingVolumeHierarchy::BoundingVolumeHierarchy(const std::shared_ptr<ModelSpaceMesh> &mesh) {

    // We use the default MeshCore BVH to construct this performance oriented one
    auto root = CachingBoundsTreeFactory<AABBVolumeHierarchy>::getBoundsTree(mesh);

    std::stack<std::pair<AbstractBoundsTree<AABB,2, true>*, size_t>> stack;
    stack.emplace(root.get(), nodes.size());
    nodes.emplace_back();

    while (!stack.empty()) {
        const auto origNode = stack.top().first;
        const auto flatNodeIndex = stack.top().second;
        auto& flatNode = nodes[flatNodeIndex];
        stack.pop();

        flatNode.bounds = origNode->getBounds();

        if (origNode->isSplit()) {

            // Split node: add children on the stack and set the firstChildIndex in the current node
            flatNode.split = true;
            flatNode.firstChildOrTriangleIndex = nodes.size();
            for (const auto & child : origNode->getChildren()) {
                stack.emplace(child.get(), nodes.size());
                nodes.emplace_back();
            }
        }
        else {
            // Add the triangles for the current node
            flatNode.firstChildOrTriangleIndex = triangles.size();
            flatNode.triangleCount = origNode->getTriangles().size();
            for (const auto& triangle : origNode->getTriangles()) {
                triangles.emplace_back(triangle);
            }
        }
    }
}

bool BoundingVolumeHierarchy::hitsBacksideFirst(const Ray &ray) const {

    unsigned int stack[STACK_DEPTH];
    int stackIndex = 0;
    stack[stackIndex++] = 0; // Start with the root node
    float closest_t = std::numeric_limits<float>::max();
    bool closest_backside = false;
    while (stackIndex > 0) {
        const auto nodeIndex = stack[--stackIndex];
        const auto& node = nodes[nodeIndex];
        if (Intersection::intersect(node.bounds, ray, 0.0f, closest_t)) {
            if (node.split) {
                // Add its children to the stack
                for (int i = 0; i < 2; ++i) {
                    stack[stackIndex++] = node.firstChildOrTriangleIndex + i;
                }
            }
            else {
                for (int i = 0; i < node.triangleCount; ++i) {
                    const auto& triangle = triangles[node.firstChildOrTriangleIndex + i];
                    if (const auto t = Intersection::intersectionDistance(ray, triangle); t > 0) {
                        if (t < closest_t) {
                            closest_t = t;
                            closest_backside = glm::dot(ray.direction, triangle.normal) > 0;
                        }
                    }
                }
            }
        }
    }
    return closest_backside;
}


bool BoundingVolumeHierarchy::containsPoint(const glm::vec3& point) const {
    return hitsBacksideFirst(Ray(point, glm::vec3(0.8255, -0.1687, 0.3645)));;
}


void BoundingVolumeHierarchy::queryClosestTriangle(const Vertex &vertex, ClosestTriangleQueryResult* result) const {

    size_t nodeIndexStack[STACK_DEPTH];
    float nodeDistanceStack[STACK_DEPTH];
    int stackIndex = 0;
    nodeIndexStack[stackIndex] = 0; // Start with the root node
    nodeDistanceStack[stackIndex] = nodes[0].bounds.getDistanceSquaredTo(vertex);
    stackIndex++;

    while (stackIndex > 0) {

        // Take the top node from the stack
        stackIndex--;
        const auto nodeIndex = nodeIndexStack[stackIndex];
        assert(nodeIndex < this->nodes.size());
        const Node node = this->nodes[nodeIndex];
        const auto nodeDistance = nodeDistanceStack[stackIndex];

        // Check if the node can still improve the shortest distance
        // (lowerDistanceBoundSquared could have changes since the node has been put on the stack)
        if(nodeDistance < result->lowerDistanceBoundSquared){

            if (node.split) {
                float squaredChildDistances[2];
                size_t childIndices[2];

                for(size_t i = 0; i < 2; i++) {
                    const auto childIndex = node.firstChildOrTriangleIndex + i;
                    assert(childIndex < this->nodes.size());
                    const auto& childNode = this->nodes[childIndex];

                    squaredChildDistances[i] = childNode.bounds.getDistanceSquaredTo(vertex);
                    childIndices[i] = i; // Sorted later
                }

                // Sort closest distance first: switch the indices if the first child is further away than the second
                if (squaredChildDistances[1] < squaredChildDistances[0]) {
                    childIndices[0] = 1;
                    childIndices[1] = 0;
                }

                // Put the children on the stack in this order, if they could improve the shortest distance
                for (const auto &childIndex : childIndices){

                    // If this node cannot improve the shortest distance, neither will the following ones
                    if(squaredChildDistances[childIndex] >= result->lowerDistanceBoundSquared){
                        break;
                    }

                    assert(node.split);
                    auto childNodeIndex = node.firstChildOrTriangleIndex + childIndex;
                    assert(childNodeIndex < this->nodes.size());
                    assert(stackIndex < STACK_DEPTH);
                    nodeIndexStack[stackIndex] = childNodeIndex;
                    nodeDistanceStack[stackIndex] = squaredChildDistances[childIndex];
                    stackIndex++;
                }
            }
            else {
                for (int i = 0; i < node.triangleCount; ++i) {
                    auto& nodeTriangle = this->triangles[node.firstChildOrTriangleIndex + i];
                    auto closestPoint = nodeTriangle.getClosestPoint(vertex);
                    auto delta = closestPoint - vertex;
                    auto distanceSquared = glm::dot(delta, delta);
                    if(distanceSquared < result->lowerDistanceBoundSquared){ // If multiple triangles are equally close, the first one will be returned
                        result->closestTriangle = &nodeTriangle;
                        result->lowerDistanceBoundSquared = distanceSquared;
                        result->closestVertex = closestPoint;
                        if(distanceSquared <= 0.0){
                            return;
                        }
                    }
                }
            }
        }
    }
}

void BoundingVolumeHierarchy::queryClosestTriangle(const VertexTriangle &triangle, ClosestTriangleQueryResult* result) const {

    size_t nodeIndexStack[STACK_DEPTH];
    float nodeDistanceStack[STACK_DEPTH];
    int stackIndex = 0;
    nodeIndexStack[stackIndex] = 0; // Start with the root node
    nodeDistanceStack[stackIndex] = Distance::distanceSquared(nodes[0].bounds, triangle.bounds);
    stackIndex++;

    while (stackIndex > 0) {

        // Take the top node from the stack
        stackIndex--;
        const auto nodeIndex = nodeIndexStack[stackIndex];
        assert(nodeIndex < this->nodes.size());
        const Node node = this->nodes[nodeIndex];
        const auto nodeDistance = nodeDistanceStack[stackIndex];

        // Check if the node can still improve the shortest distance
        // (lowerDistanceBoundSquared could have changes since the node has been put on the stack)
        if(nodeDistance < result->lowerDistanceBoundSquared){

            if (node.split) {
                float squaredChildDistances[2];
                size_t childIndices[2];

                for(size_t i = 0; i < 2; i++) {
                    const auto childIndex = node.firstChildOrTriangleIndex + i;
                    assert(childIndex < this->nodes.size());
                    const auto& childNode = this->nodes[childIndex];

                    squaredChildDistances[i] = Distance::distanceSquared(childNode.bounds, triangle.bounds);
                    childIndices[i] = i; // Sorted later
                }

                // Sort closest distance first: switch the indices if the first child is further away than the second
                if (squaredChildDistances[1] < squaredChildDistances[0]) {
                    childIndices[0] = 1;
                    childIndices[1] = 0;
                }

                // Put the children on the stack in this order, if they could improve the shortest distance
                for (const auto &childIndex : childIndices){

                    // If this node cannot improve the shortest distance, neither will the following ones
                    if(squaredChildDistances[childIndex] >= result->lowerDistanceBoundSquared){
                        break;
                    }

                    assert(node.split);
                    auto childNodeIndex = node.firstChildOrTriangleIndex + childIndex;
                    assert(childNodeIndex < this->nodes.size());
                    assert(stackIndex < STACK_DEPTH);
                    nodeIndexStack[stackIndex] = childNodeIndex;
                    nodeDistanceStack[stackIndex] = squaredChildDistances[childIndex];
                    stackIndex++;
                }
            }
            else {
                for (int i = 0; i < node.triangleCount; ++i) {
                    auto& nodeTriangle = this->triangles[node.firstChildOrTriangleIndex + i];
                    Vertex closestPointTriangle, closestPointOtherTriangle;
                    auto distanceSquared = Distance::distanceSquared(triangle, nodeTriangle, &closestPointTriangle, &closestPointOtherTriangle);
                    if(distanceSquared < result->lowerDistanceBoundSquared){ // If multiple triangles are equally close, the first one will be returned
                        result->closestTriangle = &nodeTriangle;
                        result->lowerDistanceBoundSquared = distanceSquared;
                        result->closestVertex = closestPointOtherTriangle;
                        if(distanceSquared <= 0.0){
                            return; // No need to continue searching if the triangles intersect
                        }
                    }
                }
            }
        }
    }
}

bool BoundingVolumeHierarchy::intersectsAABB(const AABB &aabb) const {

    unsigned int stack[STACK_DEPTH];
    int stackIndex = 0;
    stack[stackIndex++] = 0; // Start with the root node
    while (stackIndex > 0) {
        const auto nodeIndex = stack[--stackIndex];
        const auto& node = nodes[nodeIndex];

        if (Intersection::intersect(node.bounds,aabb)) {

            if (node.split) {
                // Add its children to the stack
                for (int i = 0; i < 2; ++i) {
                    stack[stackIndex++] = node.firstChildOrTriangleIndex + i;
                }
            }
            else {
                return true;
            }
        }
    }
    return false;
}

bool BoundingVolumeHierarchy::intersectsTriangle(const VertexTriangle &triangle) const {

    // Precompute support data for Triangle-AABB queries that only depends on the triangle
    AABBTriangleData triangleData(triangle);

    unsigned int stack[STACK_DEPTH];
    int stackIndex = 0;
    stack[stackIndex++] = 0; // Start with the root node
    while (stackIndex > 0) {
        const auto nodeIndex = stack[--stackIndex];
        const auto& node = nodes[nodeIndex];

        if (Intersection::intersect(node.bounds,triangle,triangleData)) {
            if (node.split) {
                // Add its children to the stack
                for (int i = 0; i < 2; ++i) {
                    stack[stackIndex++] = node.firstChildOrTriangleIndex + i;
                }
            }
            else {
                for (int i = 0; i < node.triangleCount; ++i) {
                    if (Intersection::intersect(triangle.bounds, triangles[node.firstChildOrTriangleIndex + i].bounds)) {
                        if (Intersection::intersect(triangle, triangles[node.firstChildOrTriangleIndex + i])) {
                            return true;
                        }
                    }
                }
            }
        }
    }
    return false;
}

float BoundingVolumeHierarchy::getShortestDistanceSquared(const glm::vec3 &point) const {
    unsigned int stack[STACK_DEPTH];
    int stackIndex = 0;
    stack[stackIndex++] = 0; // Start with the root node
    float closest_distance_sqr = std::numeric_limits<float>::max();
    while (stackIndex > 0) {
        const auto nodeIndex = stack[--stackIndex];
        const auto& node = nodes[nodeIndex];
        if (node.bounds.getDistanceSquaredTo(point) < closest_distance_sqr) {
            if (node.split) {
                // Add its children to the stack
                for (auto i = 0; i<2; i++) {
                    auto childNodeIndex = node.firstChildOrTriangleIndex + i;
                    stack[stackIndex++] = childNodeIndex;
                }
            }
            else {
                for (int i = 0; i < node.triangleCount; ++i) {
                    const auto& triangle = triangles[node.firstChildOrTriangleIndex + i];
                    if (triangle.bounds.getDistanceSquaredTo(point) < closest_distance_sqr) {
                        const auto delta = point - triangle.getClosestPoint(point);
                        auto distance_sqr = glm::dot(delta,delta);
                        if (distance_sqr < closest_distance_sqr) {
                            closest_distance_sqr = distance_sqr;
                        }
                    }
                }
            }
        }
    }
    return closest_distance_sqr;
}

const std::vector<BoundingVolumeHierarchy::Node> & BoundingVolumeHierarchy::getNodes() const {
    return nodes;
}

const std::vector<VertexTriangle> & BoundingVolumeHierarchy::getTriangles() const {
    return triangles;
}
