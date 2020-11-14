//
// Created by Jonas on 9/11/2020.
//

#include "ModelSpaceMesh.h"

ModelSpaceMesh::ModelSpaceMesh(const std::vector<Vector> & vertices, const std::vector<Triangle> & triangles) {
    this->vertices = std::vector<Vector>(vertices);
    this->triangles = std::vector<Triangle>(triangles);
}
