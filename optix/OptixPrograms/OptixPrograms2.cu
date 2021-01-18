#include <optix.h>
#include "../../solutions/NewOptixData.h"

#define EPSILON 0.000001f

extern "C" {
__constant__ OptixLaunchParameters optixLaunchParameters;
}

extern "C" __global__ void __miss__ms()
{
//    printf("Miss\n");
}

//extern "C" float3 __device__ transform(const float3 input, const float * transform){
//
//    float inputValue[3] = {input.x, input.y, input.z};
//    float returnValue[3] = {0,0,0};
//    for(int i=0; i<3; i++){
//        for(int j=0; j<3; j++){
//            returnValue[i] += transform[4*i + j] * inputValue[j];
////            returnValue[i] += transform[0] * inputValue[0];
//        }
//        returnValue[i] += transform[4*i + 3] * 1;
////        returnValue[i] += transform[0] * 1;
//    }
//    return make_float3(returnValue[0], returnValue[1], returnValue[2]);
//
//}

extern "C" __global__ void __raygen__edgeIntersectionTest2__()
{
    const unsigned int idx = optixGetLaunchIndex().x;
    const RayGenData* rayGenData = reinterpret_cast<RayGenData*>(optixGetSbtDataPointer());

    const float3 modelSpaceRayOrigin = rayGenData->origins[idx];
    const float3 modelSpaceRayDirection = rayGenData->directions[idx];
    float* transformations = rayGenData->transformations;
    unsigned int* rayTransformIndices = rayGenData->rayTransformIndices;

//    const float* transformation = &transformations[12*rayTransformIndices[idx]];
    if(idx<100){
        printf("TransformOffset: %u\tTransformIndex: %u\tidx: %u\n",12*rayTransformIndices[idx],rayTransformIndices[idx],idx);
    }
    const float* transformation = &transformations[0];
    if(idx==0){
        float* transformation0 = &transformations[0];
        float* transformation1 = &transformations[12];

        printf("Transformation 0: %p\n", transformation0);
        for(int i=0; i<12; i++){
            printf("%.2f ",transformation0[i]);
        }
        printf("\n");
        printf("Transformation 1: %p\n", transformation1);
        for(int i=0; i<12; i++){
            printf("%.2f ",transformation1[i]);
        }
        printf("\n");
        printf("Transformation 2: %p\n", transformation);
        for(int i=0; i<12; i++){
            printf("%.2f ",transformation[i]);
        }
        printf("\n");
    }
//
//    float3 worldSpaceRayOrigin = transform(modelSpaceRayOrigin, transformation);
//    float3 worldSpaceRayDirection = transform(modelSpaceRayDirection, transformation);

    float3 worldSpaceRayOrigin(modelSpaceRayOrigin);

    worldSpaceRayOrigin.x = modelSpaceRayOrigin.x*transformation[0] + modelSpaceRayOrigin.y*transformation[1] + modelSpaceRayOrigin.z*transformation[2] + modelSpaceRayOrigin.z*transformation[3];
    worldSpaceRayOrigin.y = modelSpaceRayOrigin.x*transformation[4] + modelSpaceRayOrigin.y*transformation[5] + modelSpaceRayOrigin.z*transformation[6] + modelSpaceRayOrigin.z*transformation[7];
    worldSpaceRayOrigin.z = modelSpaceRayOrigin.x*transformation[8] + modelSpaceRayOrigin.y*transformation[9] + modelSpaceRayOrigin.z*transformation[10] + modelSpaceRayOrigin.z*transformation[11];
//
    float3 worldSpaceRayDirection(modelSpaceRayDirection);
    worldSpaceRayDirection.x = modelSpaceRayDirection.x*transformation[0] + modelSpaceRayDirection.y*transformation[1] + modelSpaceRayDirection.z*transformation[2] + modelSpaceRayDirection.z*transformation[3];
    worldSpaceRayDirection.y = modelSpaceRayDirection.x*transformation[4] + modelSpaceRayDirection.y*transformation[5] + modelSpaceRayDirection.z*transformation[6] + modelSpaceRayDirection.z*transformation[7];
    worldSpaceRayDirection.z = modelSpaceRayDirection.x*transformation[8] + modelSpaceRayDirection.y*transformation[9] + modelSpaceRayDirection.z*transformation[10] + modelSpaceRayDirection.z*transformation[11];

    if(idx==1){

        printf("Input(%.2f,%.2f,%.2f)\n", modelSpaceRayOrigin.x, modelSpaceRayOrigin.y, modelSpaceRayOrigin.z);
        printf("Output(%.2f,%.2f,%.2f)\n", worldSpaceRayOrigin.x, worldSpaceRayOrigin.y, worldSpaceRayOrigin.z);
    }

    optixTrace(
            optixLaunchParameters.handle,
            worldSpaceRayOrigin,
            worldSpaceRayDirection,
            0,
            1 + EPSILON,
            0.0f,                 // Ignored and removed by the compiler if motion is not enabled
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            0,                   // only used for different ray types
            0,                   // only used for different ray types
            0);
}

extern "C" __global__ void __closesthit__edgeIntersectionTest2__()
{
//    printf("EdgeIntersection\n");
//    unsigned int id = optixGetInstanceId;
    reinterpret_cast<EdgeIntersectionTestData*>(optixGetSbtDataPointer())->edgeIntersection = true;
}