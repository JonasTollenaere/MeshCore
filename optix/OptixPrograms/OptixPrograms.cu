#include <optix.h>
#include "../OptixLaunchParameters.h"
#include "../OptixData.h"

#define EPSILON 0.000001f

extern "C" {
__constant__ OptixLaunchParameters optixLaunchParameters;
}

extern "C" __global__ void __raygen__edgeIntersectionTest__()
{
    const unsigned int idx = optixGetLaunchIndex().x;
    const RayGenData* rayGenData = reinterpret_cast<RayGenData*>(optixGetSbtDataPointer());
//    printf("RayGen %d: Origin(%.3f,%.3f,%.3f)\n", idx, rayGenData->origins[idx].x, rayGenData->origins[idx].y, rayGenData->origins[idx].z);
    optixTrace(
            optixLaunchParameters.handle,
            rayGenData->origins[idx],
            rayGenData->directions[idx],
            0,
            1 + EPSILON,
            0.0f,                 // Ignored and removed by the compiler if motion is not enabled
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            0,                   // only used for different ray types
            0,                   // only used for different ray types
            0);
}

extern "C" __global__ void __closesthit__edgeIntersectionTest__()
{
//    printf("EdgeIntersection\n");
    reinterpret_cast<EdgeIntersectionTestData*>(optixGetSbtDataPointer())->edgeIntersection = true;
}

extern "C" __global__ void __miss__ms()
{
//    printf("Miss\n");
}

extern "C" __global__ void __raygen__vertexInsideTest__()
{
    const unsigned int idx = optixGetLaunchIndex().x;
    const RayGenData* rayGenData = reinterpret_cast<RayGenData*>(optixGetSbtDataPointer());
    printf("RayGen %d: Origin(%.3f,%.3f,%.3f)\n", idx, rayGenData->origins[idx].x, rayGenData->origins[idx].y, rayGenData->origins[idx].z);
    optixTrace(
            optixLaunchParameters.handle,
            rayGenData->origins[idx],
            rayGenData->directions[idx],    // The direction doesn't really matter
            0,
            1e38f,
            0.0f,                 // Ignored and removed by the compiler if motion is not enabled
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            0,                   // only used for different ray types
            0,                   // only used for different ray types
            0);
}

extern "C" __global__ void __closesthit__vertexInsideTest__()
{
    if(optixIsFrontFaceHit()){
        reinterpret_cast<VertexInsideTestData*>(optixGetSbtDataPointer())->vertexInside = false;
    }
}

extern "C" __global__ void __miss__vertexInsideTest__()
{
    reinterpret_cast<VertexInsideTestData*>(optixGetSbtDataPointer())->vertexInside = false;
}