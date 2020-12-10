//
// Created by Jonas on 9/11/2020.
//

#include "ModelSpaceMesh.h"

ModelSpaceMesh::ModelSpaceMesh(const std::vector<Vertex>& vertices, const std::vector<Triangle>& triangles):
    vertices(vertices), triangles(triangles)
{

}

ModelSpaceMesh::~ModelSpaceMesh() {

}

ModelSpaceMesh::ModelSpaceMesh(const ModelSpaceMesh &other):
triangles(other.triangles),
vertices(other.vertices){
}

bool ModelSpaceMesh::includes(const Vertex &modelSpaceVertex) const{
    Ray ray(modelSpaceVertex, glm::vec3(1, 0, 0));
    if(this->calculateNumberOfIntersections(ray)%2 == 0){
        return false;
    }
    return true;
}

int ModelSpaceMesh::calculateNumberOfIntersections(Ray modelSpaceRay) const{
    int numberOfIntersections = 0;
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

ModelSpaceMesh::ModelSpaceMesh() = default;
