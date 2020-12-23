//
// Created by Jonas on 9/11/2020.
//

#include <unordered_set>
#include "ModelSpaceMesh.h"

ModelSpaceMesh::ModelSpaceMesh(const std::vector<Vertex>& vertices, const std::vector<Triangle>& triangles):
    vertices(vertices), triangles(triangles)
{

}

ModelSpaceMesh::~ModelSpaceMesh() {

}

ModelSpaceMesh::ModelSpaceMesh(const ModelSpaceMesh &other):
triangles(other.triangles),
vertices(other.vertices)
{}

bool ModelSpaceMesh::includes(const Vertex &modelSpaceVertex) const{
    Ray ray(modelSpaceVertex, glm::vec3(1, 0, 0));
    if(this->calculateNumberOfIntersections(ray)%2 == 0){
        return false;
    }
    return true;
}

unsigned int ModelSpaceMesh::calculateNumberOfIntersections(Ray modelSpaceRay) const{
    unsigned int numberOfIntersections = 0;
    for(Triangle triangle: this->triangles){
        bool intersects = modelSpaceRay.intersectsTriangle(this->vertices[triangle.vertexIndex0],
                                                           this->vertices[triangle.vertexIndex1],
                                                           this->vertices[triangle.vertexIndex2]);
        if(intersects){
            numberOfIntersections ++;
        }
    }
    return numberOfIntersections;
}

std::vector<Edge> ModelSpaceMesh::getEdges() const {

    // Set up the hash and equals in a way that the order of vertexIndex0 and vertexIndex1 doesn't matter
    auto hash = [](const Edge& edge) { return std::hash<int>()(edge.vertexIndex0 + edge.vertexIndex1); };
    auto equal = [](const Edge& edge1, const Edge& edge2) {
        return (edge1.vertexIndex0 == edge2.vertexIndex0 && edge1.vertexIndex1 == edge2.vertexIndex1) ||
               (edge1.vertexIndex1 == edge2.vertexIndex0 && edge1.vertexIndex0 == edge2.vertexIndex1); };
    std::unordered_set<Edge, decltype(hash), decltype(equal)> edgeSet(8, hash, equal);
    for(Triangle triangle: triangles){
        edgeSet.insert(Edge{triangle.vertexIndex0, triangle.vertexIndex1});
        edgeSet.insert(Edge{triangle.vertexIndex1, triangle.vertexIndex2});
        edgeSet.insert(Edge{triangle.vertexIndex2, triangle.vertexIndex0});
    }

    return std::vector<Edge>(edgeSet.begin(), edgeSet.end());
}

ModelSpaceMesh::ModelSpaceMesh() = default;
