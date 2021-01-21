#include <optix.h>
#include "../../solutions/NewOptixData.h"

#define EPSILON 0.000001f

extern "C" {
__constant__ OptixLaunchParameters optixLaunchParameters;
}

extern "C" __global__ void __miss__ms()
{
    printf("Miss\n");
}

extern "C" __global__ void __raygen__edgeIntersectionTest2__()
{
    const unsigned int idx = optixGetLaunchIndex().x;
    const RayGenData* rayGenData = reinterpret_cast<RayGenData*>(optixGetSbtDataPointer());

    const float3 modelSpaceRayOrigin = rayGenData->origins[idx];
    const float3 modelSpaceRayDirection = rayGenData->directions[idx];
    float* transformations = rayGenData->transformations;
    unsigned int* rayTransformIndices = rayGenData->rayTransformIndices;
    unsigned int transformIndex = rayTransformIndices[idx];
    const unsigned int transformOffset = 12*rayTransformIndices[idx];
    const float* transformation = &transformations[transformOffset];

    float3 worldSpaceRayOrigin;
    worldSpaceRayOrigin.x = modelSpaceRayOrigin.x*transformation[0] + modelSpaceRayOrigin.y*transformation[1] + modelSpaceRayOrigin.z*transformation[2] + transformation[3];
    worldSpaceRayOrigin.y = modelSpaceRayOrigin.x*transformation[4] + modelSpaceRayOrigin.y*transformation[5] + modelSpaceRayOrigin.z*transformation[6] + transformation[7];
    worldSpaceRayOrigin.z = modelSpaceRayOrigin.x*transformation[8] + modelSpaceRayOrigin.y*transformation[9] + modelSpaceRayOrigin.z*transformation[10] + transformation[11];

    float3 worldSpaceRayDirection;
    worldSpaceRayDirection.x = modelSpaceRayDirection.x*transformation[0] + modelSpaceRayDirection.y*transformation[1] + modelSpaceRayDirection.z*transformation[2];
    worldSpaceRayDirection.y = modelSpaceRayDirection.x*transformation[4] + modelSpaceRayDirection.y*transformation[5] + modelSpaceRayDirection.z*transformation[6];
    worldSpaceRayDirection.z = modelSpaceRayDirection.x*transformation[8] + modelSpaceRayDirection.y*transformation[9] + modelSpaceRayDirection.z*transformation[10];

    optixTrace(
            optixLaunchParameters.handle,
            worldSpaceRayOrigin,
            worldSpaceRayDirection,
            0,
            1,
            0.0f,                 // Ignored and removed by the compiler if motion is not enabled
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            0,                   // only used for different ray types
            0,                   // only used for different ray types
            0,
            transformIndex); // Instance transformation id
}

extern "C" __global__ void __closesthit__edgeIntersectionTest2__()
{
    if(optixGetPayload_0() != optixGetInstanceId()){
        reinterpret_cast<EdgeIntersectionTestData*>(optixGetSbtDataPointer())->edgeIntersection = true;
    }
}