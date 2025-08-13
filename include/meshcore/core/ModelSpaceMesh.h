//
// Created by Jonas on 27/01/2021.
//

#ifndef MESHCORE_MODELSPACEMESH_H
#define MESHCORE_MODELSPACEMESH_H

#include <vector>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include "Vertex.h"
#include "IndexTriangle.h"
#include "IndexEdge.h"
#include "IndexFace.h"
#include "AABB.h"
#include "meshcore/geometric/GJK.h"

class ModelSpaceMesh: public GJKConvexShape {

    std::string name;
    std::vector<Vertex> vertices;
    std::vector<IndexTriangle> triangles;

    // Cached data
    mutable std::optional<std::vector<IndexEdge>> triangleEdges;
    mutable std::optional<std::vector<IndexFace>> faces;
    mutable std::optional<std::vector<IndexEdge>> faceEdges;
    mutable std::optional<bool> convex;
    mutable std::optional<float> volume;
    mutable std::optional<float> surfaceArea;
    mutable std::optional<Vertex> volumeCentroid;
    mutable std::optional<Vertex> surfaceCentroid;
    mutable std::optional<AABB> bounds;
    mutable std::shared_ptr<ModelSpaceMesh> convexHull = nullptr;
    mutable std::optional<std::vector<std::vector<size_t>>> connectedVertexIndices;

    void computeVolumeAndCentroid() const;
    void computeSurfaceAreaAndCentroid() const;
    void computeConvexity() const;
    void computeFaces() const;

public:
    ModelSpaceMesh() = default;
    explicit ModelSpaceMesh(std::vector<Vertex> vertices);
    ModelSpaceMesh(std::vector<Vertex> vertices, std::vector<IndexTriangle> triangles);
    ModelSpaceMesh(const ModelSpaceMesh& other) = default;
    ~ModelSpaceMesh() override = default;

    [[nodiscard]] const std::vector<Vertex>& getVertices() const;
    [[nodiscard]] const std::vector<IndexTriangle>& getTriangles() const;
    [[nodiscard]] const std::vector<IndexFace>& getFaces() const;
    [[nodiscard]] const std::vector<IndexEdge>& getEdges() const;
    [[nodiscard]] const std::vector<IndexEdge> &getFaceEdges() const;
    [[nodiscard]] std::vector<IndexEdge> getSufficientIntersectionEdges() const; // Edge faces enough?

    const AABB &getBounds() const;

    [[nodiscard]] const std::vector<std::vector<size_t>>& getConnectedVertexIndices() const;
    const std::shared_ptr<ModelSpaceMesh>& getConvexHull() const;
    bool isConvex() const;
    float getVolume() const;
    Vertex getVolumeCentroid() const;
    Vertex getSurfaceCentroid() const;
    float getSurfaceArea() const; //https://people.eecs.berkeley.edu/~wkahan/Triangle.pdf

    const std::string &getName() const;
    void setName(const std::string &newName);

    // GJKConvexShape interface
    glm::vec3 computeSupport(const glm::vec3 &direction) const override;
    glm::vec3 getCenter() const override;

};


#endif //MESHCORE_MODELSPACEMESH_H
