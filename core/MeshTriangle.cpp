//
// Created by Jonas on 9/11/2020.
//
#include <iostream>
#include "MeshTriangle.h"
#include "ModelSpaceMesh.h"

MeshTriangle::MeshTriangle(const int vertexIndex0, const int vertexIndex1, const int vertexIndex2, const ModelSpaceMesh& mesh):
    vertexIndex0(vertexIndex0),
    vertexIndex1(vertexIndex1),
    vertexIndex2(vertexIndex2),
    mesh(mesh)
{
//    meshPointer.
}

std::ostream &operator<<(std::ostream &os, const MeshTriangle &meshTriangle) {
    return os << meshTriangle.toString();
}

std::string MeshTriangle::toString() const {
//    Vertex v0 = meshPointer->vertices[vertexIndex0]
    
    return "";
}
