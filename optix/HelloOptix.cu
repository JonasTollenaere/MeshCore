
#include <optix.h>
#include <iostream>

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
    optixTrace(handle, ray_origin, ray_direction, tmin, tmax,0.5f, OptixVisibilityMask( 1 ),
               OPTIX_RAY_FLAG_NONE,0, 0, 0, p0);
    *prd = int_as_float( p0 );
}

int main(){
    std::cout << "Hello Optix!" << std::endl;
}