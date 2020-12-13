//
// Created by Jonas on 12/12/2020.
//

#ifndef MESHCORE_CUDARAY_H
#define MESHCORE_CUDARAY_H

#include <glm/glm.hpp>
#include <iostream>
#include "../core/Vertex.h"
#include "../core/Transformation.h"

class CudaRay {
private:
    Vertex origin;
    Vertex direction;
public:
    __device__ CudaRay();
    __device__ CudaRay(Vertex origin, Vertex direction);
    __device__ void transform(const Transformation &transformation);
    __device__ bool intersectsTriangle(const Vertex &v0, const Vertex &v1, const Vertex &v2);
};

#endif //MESHCORE_CUDARAY_H