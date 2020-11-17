//
// Created by Jonas on 9/11/2020.
//

#ifndef OPTIX_SAMPLES_MODELSPACEMESH_H
#define OPTIX_SAMPLES_MODELSPACEMESH_H


#include <vector>
#include "Vertex.h"
#include "Triangle.h"


class ModelSpaceMesh {
public:
    ModelSpaceMesh(const std::vector<Vertex> & vertices, const std::vector<Triangle> & triangles);

    std::vector<Vertex> vertices;
    std::vector<Triangle> triangles;
};


#endif //OPTIX_SAMPLES_MODELSPACEMESH_H
