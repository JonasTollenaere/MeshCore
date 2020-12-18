//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

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
    printf("EdgeIntersection");
    reinterpret_cast<EdgeIntersectionTestData*>(optixGetSbtDataPointer())->edgeIntersection = true;
}

extern "C" __global__ void __raygen__vertexInsideTest__()
{
    const unsigned int idx = optixGetLaunchIndex().x;
    const RayGenData* rayGenData = reinterpret_cast<RayGenData*>(optixGetSbtDataPointer());
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