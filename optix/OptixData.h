//
// Created by Jonas on 16/12/2020.
//

#ifndef MESHCORE_OPTIXDATA_H
#define MESHCORE_OPTIXDATA_H

struct RayGenData
{
    float3* origins;
    float3* directions;
};

struct EdgeIntersectionTestData{
    bool edgeIntersection;
};

struct VertexInsideTestData{
    bool vertexInside;
};

#endif //MESHCORE_OPTIXDATA_H
