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

extern "C" __global__ void __raygen__rg()
{
    const unsigned int idx = optixGetLaunchIndex().x;

    const RayGenData* rayGenData = reinterpret_cast<RayGenData*>(optixGetSbtDataPointer());
//    printf("RayGen %d\tOrigin: (%.2f,%.2f,%.2f)\tDirection: (%.2f,%.2f,%.2f)\n", idx,
//           rayGenData->origins[idx].x, rayGenData->origins[idx].y, rayGenData->origins[idx].z,
//           rayGenData->directions[idx].x, rayGenData->directions[idx].y, rayGenData->directions[idx].z);
    optixTrace(
            optixLaunchParameters.handle,
            rayGenData->origins[idx],
            rayGenData->directions[idx],
            0,                      // tmin
            1 + EPSILON,            // tmax
            0.0f,                 // Ignored and removed by the compiler if motion is not enabled
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, // No need to disable anyhit if this is done in the acceleration structures
            0,                   // SBT offset, only used for different ray types
            0,                   // SBT stride, only used for different ray types
            0);               // missSBTIndex
}

extern "C" __global__ void __miss__ms()
{

}


extern "C" __global__ void __closesthit__ch()
{
    const unsigned int idx = optixGetLaunchIndex().x;

//    printf("Hit %d: %.5f\n", idx, optixGetRayTmax());
    HitGroupData* hitGroupData = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    hitGroupData->hit = true;
}