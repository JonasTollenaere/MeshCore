//
// Created by Jonas on 16/01/2021.
//

#ifndef MESHCORE_CUDAUTILITIES_H
#define MESHCORE_CUDAUTILITIES_H

static float3 vec3ToFloat3(const glm::vec3& vec3){
    return make_float3(vec3.x, vec3.y, vec3.z);
}

static uint3 triangleToUint3(const Triangle& triangle){
    return make_uint3(triangle.vertexIndex0, triangle.vertexIndex1, triangle.vertexIndex2);
}

#endif //MESHCORE_CUDAUTILITIES_H
