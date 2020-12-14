//
// Created by Jonas on 12/12/2020.
//

#ifndef MESHCORE_CUDAWORLDSPACEMESH_H
#define MESHCORE_CUDAWORLDSPACEMESH_H

#include "../core/Triangle.h"
#include "../core/Transformation.h"
#include "../core/ModelSpaceMesh.h"
#include "../core/WorldSpaceMesh.h"
#include "CudaStream.h"
#include <iostream>

class CudaWorldSpaceMesh {
private:

    // Pointers in VRAM
    Triangle* triangles;
    Vertex* modelSpaceVertices;
    Vertex* worldSpaceVertices;
    Transformation* transformation;
    bool* flags;

    const cudaStream_t* cudaStream;
    unsigned int numberOfTriangles;
    unsigned int numberOfVertices;

public:
    CudaWorldSpaceMesh(const WorldSpaceMesh& worldSpaceMesh, const CudaStream& cudaStream);
    ~CudaWorldSpaceMesh();

    void setModelTransformationMatrix(const Transformation& transformation);

    bool triangleTriangleIntersects(const CudaWorldSpaceMesh &other) const;
    bool rayTriangleInside(const CudaWorldSpaceMesh &other) const;
};

#endif //MESHCORE_CUDAWORLDSPACEMESH_H
