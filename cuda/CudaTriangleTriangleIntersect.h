//
// Created by Jonas on 13/12/2020.
//

#ifndef MESHCORE_CUDATRIANGLETRIANGLEINTERSECT_H
#define MESHCORE_CUDATRIANGLETRIANGLEINTERSECT_H

#include <glm/vec3.hpp>

namespace CudaIntersection
{
    __device__ bool NoDivTriTriIsect(float V0[3], float V1[3], float V2[3],
                          float U0[3], float U1[3], float U2[3]);

    __device__ int coplanar_tri_tri(float N[3], float V0[3], float V1[3], float V2[3],
                         float U0[3], float U1[3], float U2[3]);
}

#endif //MESHCORE_CUDATRIANGLETRIANGLEINTERSECT_H
