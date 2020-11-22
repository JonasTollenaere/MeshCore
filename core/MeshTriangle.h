//
// Created by Jonas on 9/11/2020.
//
#ifndef OPTIX_SAMPLES_TRIANGLE_H
#define OPTIX_SAMPLES_TRIANGLE_H


#include <vector>
#include "Vertex.h"
#include "ModelSpaceMesh.h"

class MeshTriangle { // A triangle only stores the index of each vertex, so it doesn't make sense outside a mesh
private:
    const int vertexIndex0;
    const int vertexIndex1;
    const int vertexIndex2;
    const ModelSpaceMesh mesh;
public:
    MeshTriangle(const int vertexIndex0, const int vertexIndex1, const int vertexIndex2, const ModelSpaceMesh& mesh);
};


#endif //OPTIX_SAMPLES_TRIANGLE_H
