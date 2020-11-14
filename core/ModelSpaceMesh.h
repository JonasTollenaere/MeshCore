//
// Created by Jonas on 9/11/2020.
//

#ifndef OPTIX_SAMPLES_MODELSPACEMESH_H
#define OPTIX_SAMPLES_MODELSPACEMESH_H


#include <vector>
#include <ostream>
#include "Vector.h"
#include "Triangle.h"


class ModelSpaceMesh {
private:
    std::vector<Vector> vertices;
    std::vector<Triangle> triangles;
public:
    ModelSpaceMesh(const std::vector<Vector> & vertices, const std::vector<Triangle> & triangles);
};


#endif //OPTIX_SAMPLES_MODELSPACEMESH_H
