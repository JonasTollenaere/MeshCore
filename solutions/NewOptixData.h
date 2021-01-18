//
// Created by Jonas on 16/12/2020.
//

#ifndef MESHCORE_OPTIXDATA_H
#define MESHCORE_OPTIXDATA_H

#include "vector_types.h"

struct RayGenData
{
    float3* origins;
    float3* directions;
    float* transformations;
    unsigned int* rayTransformIndices;
};

struct EdgeIntersectionTestData{
    bool edgeIntersection;
};

struct VertexInsideTestData{
    bool vertexInside;
};

template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};
typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<EdgeIntersectionTestData> EdgeIntersectionSbtRecord;

struct OptixLaunchParameters
{
    OptixTraversableHandle handle;
};

struct IterationData
{
    OptixLaunchParameters optixLaunchParameters;
    EdgeIntersectionSbtRecord edgeIntersectionSbtRecord;
    OptixStaticTransform modelTransformation;
};

#endif //MESHCORE_OPTIXDATA_H
