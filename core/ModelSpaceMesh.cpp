//
// Created by Jonas on 9/11/2020.
//

#include "ModelSpaceMesh.h"

ModelSpaceMesh::ModelSpaceMesh(const std::vector<Vertex> & vertices, const std::vector<Triangle> & triangles) {
    this->vertices = std::vector<Vertex>(vertices);
    this->triangles = std::vector<Triangle>(triangles);
}
