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
#include "../OptixParams.h"
#include <iostream>

extern "C" {
__constant__ Params params;
}

static __forceinline__ __device__ void trace(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        float*                 prd
)
{
    uint32_t p0;
    p0 = float_as_int( *prd );
    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.5f,                // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset
            0,                   // SBT stride
            0,                   // missSBTIndex
            p0 );
    *prd = int_as_float( p0 );
}


static __forceinline__ __device__ void setPayload( float p )
{
    optixSetPayload_0( float_as_int( p ) );
}


static __forceinline__ __device__ float getPayload()
{
    return int_as_float(optixGetPayload_0());
}


__forceinline__ __device__ uchar1 make_color( const float&  c )
{
    return make_uchar1(
            static_cast<uint8_t>( c )
    );
}


extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    /*if (idx.x == 0 && idx.y == 0) {
        printf("dim.x = %d, dim.y = %d", dim.x, dim.y);
    }*/

    /*const float3      U      = rtData->camera_u;
    const float3      V      = rtData->camera_v;
    const float3      W      = rtData->camera_w;
    const float2      d = 2.0f * make_float2(
            static_cast<float>( idx.x ) / static_cast<float>( dim.x ),
            static_cast<float>( idx.y ) / static_cast<float>( dim.y )
            ) - 1.0f;*/


    //const float3 direction   = normalize( d.x * U + d.y * V + W );

    float       payload = 0.0f;

    //printf("origin: %d, %d, %d; direction: %d, %d, %d\n", origin.x, origin.y, origin.z, direction.x, direction.y, direction.z);

}

extern "C" __global__ void __miss__ms()
{
    //printf("miss!\n");
    //MissData* rt_data  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    //printf("MISS: %d, %d, %d\n", rt_data->r, rt_data->g, rt_data->b);
    //float    payload = getPayload();
    ///setPayload( rt_data->r);
}


extern "C" __global__ void __closesthit__ch()
{

    //const float2 barycentrics = optixGetTriangleBarycentrics();
    //printf("snijden: ");
    //printf("snijden!\n");
    setPayload( 1.0f );
}

extern "C" __global__ void __anyhit__ah()
{
    //printf("snijden!\n");
    setPayload( getPayload() + 1.0f );
}
