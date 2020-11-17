//
// Created by Jonas on 9/11/2020.
//

#ifndef OPTIX_SAMPLES_WORLDSPACEMESH_H
#define OPTIX_SAMPLES_WORLDSPACEMESH_H


#include "ModelSpaceMesh.h"
#include "Transformation.h"

class WorldSpaceMesh {
private:
    ModelSpaceMesh modelSpaceMesh;
    Transformation transformation;
    std::vector<Vertex> transformedVertices;
    std::vector<Triangle> transformedTriangles;
public:
    explicit WorldSpaceMesh(ModelSpaceMesh & modelSpaceMesh);
    void transform(const Transformation &t);
    void setTransformation(const Transformation &t);
    Transformation getTransformation();
};


#endif //OPTIX_SAMPLES_WORLDSPACEMESH_H
