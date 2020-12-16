

#include <iostream>

#include <chrono>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>
#include <glm/gtc/type_ptr.hpp>

#include <cuda.h>


#include "Exception.h"
#include "../core/WorldSpaceMesh.h"
#include "../core/FileParser.h"

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
    optixTrace(handle, ray_origin, ray_direction, tmin, tmax,0.5f, OptixVisibilityMask(1),
               OPTIX_RAY_FLAG_NONE,0, 0, 0, p0);
    *prd = int_as_float( p0 );
}

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* cbdata)
{
    std::cerr << "[" << level << "][" << tag << "]:\t"
              << message << "\n";
}


int main(){

    const ModelSpaceMesh roughModelSpaceMesh = FileParser::parseFile("../../data/models/DIAMCADrough.obj");
    WorldSpaceMesh roughMesh = WorldSpaceMesh(roughModelSpaceMesh, glm::translate(Transformation(1.0f), glm::vec3(0, -1, 0)));
    roughMesh.transform(glm::scale(Transformation(1.0f), glm::vec3(1.2f)));

    try{
        std::cout << "Hello Optix!" << std::endl;

//    0.    Create OptixDeviceContext
        OptixDeviceContext optixContext = nullptr;
        CUDA_CHECK(cudaFree(nullptr));
        CUcontext cuCtx = nullptr;
        OPTIX_CHECK( optixInit() );
        OptixDeviceContextOptions options;
        options.logCallbackFunction = &context_log_cb;
        options.logCallbackLevel = 4;
        OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &optixContext));
        OPTIX_CHECK(optixDeviceContextSetCacheEnabled(optixContext, 1));
        CUstream cuStream;
        CUDA_CHECK( cudaStreamCreate( &cuStream ) );

//    1.    Create one or more acceleration structures that represent a geometry mesh in the scene and select
//          on or more records in the shader binding table for each mesh.

        // Specify options for the build.
        OptixAccelBuildOptions accelOptions;
        memset(&accelOptions, 0, sizeof(OptixAccelBuildOptions));
        accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
        // accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accelOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;

        // Prepare data to build
        unsigned int numberOfVertices = roughMesh.getModelSpaceMesh().vertices.size();
        std::vector<float3> modelSpaceVertices;
        CUdeviceptr d_modelSpaceVertices;
        for(Vertex vertex: roughMesh.getModelSpaceMesh().vertices){
            modelSpaceVertices.emplace_back(make_float3(vertex.x, vertex.y, vertex.z));
        }
        unsigned int verticesBytes = numberOfVertices * sizeof(float3);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_modelSpaceVertices), verticesBytes));
        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(d_modelSpaceVertices), modelSpaceVertices.data(), verticesBytes, cudaMemcpyHostToDevice, cuStream));

        unsigned int numberOfTriangles = roughMesh.getModelSpaceMesh().triangles.size();
        std::vector<unsigned int> triangleIndices;
        CUdeviceptr d_triangleIndices;
        for(Triangle triangle: roughMesh.getModelSpaceMesh().triangles){
            triangleIndices.emplace_back(triangle.vertexIndex0);
            triangleIndices.emplace_back(triangle.vertexIndex1);
            triangleIndices.emplace_back(triangle.vertexIndex2);
        }
        unsigned int indicesBytes = numberOfTriangles * sizeof(float3);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_triangleIndices), indicesBytes));
        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(d_triangleIndices), triangleIndices.data(), indicesBytes, cudaMemcpyHostToDevice, cuStream));

        // Populate the build input struct with our triangle data as well as
        // information about the sizes and types of our data
        OptixBuildInput buildInput;
        memset(&buildInput, 0,  sizeof(OptixBuildInput));
        buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        buildInput.triangleArray.vertexBuffers = &d_modelSpaceVertices;
        buildInput.triangleArray.numVertices = numberOfVertices;
        buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        buildInput.triangleArray.vertexStrideInBytes = sizeof(float3);
        buildInput.triangleArray.indexBuffer = d_triangleIndices;
        buildInput.triangleArray.numIndexTriplets = numberOfTriangles;
        buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        buildInput.triangleArray.indexStrideInBytes = sizeof(unsigned int)*3;
        buildInput.triangleArray.preTransform = 0;
        buildInput.triangleArray.numSbtRecords = 1;
        const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
        buildInput.triangleArray.flags = triangle_input_flags;

        CUdeviceptr tempBuffer, outputBuffer;
        size_t tempBufferSizeInBytes, outputBufferSizeInBytes;
        accelOptions.motionOptions.numKeys = 0; // A numKeys value of zero specifies no motion blur

        OptixAccelBufferSizes bufferSizes;

        OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext, &accelOptions, &buildInput, 1, &bufferSizes));
        CUdeviceptr d_output;
        CUdeviceptr d_temp;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output), bufferSizes.outputSizeInBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp), bufferSizes.tempSizeInBytes));


        OptixTraversableHandle outputHandle = 0;

        OPTIX_CHECK(optixAccelBuild(optixContext, cuStream,       // This runs asyncrhonously on the GPU
                                              &accelOptions, &buildInput, 1, d_temp,
                                              bufferSizes.tempSizeInBytes, d_output,
                                              bufferSizes.outputSizeInBytes, &outputHandle,
                                              nullptr, 0));



//       CUDA_CHECK(cudaStreamSynchronize(cuStream));

//    2.    Create a pipeline of programs that contains all programs that will be invoked during a ray tracing launch.




//    3.    Create a shader binding table that includes references to these programs and their parameters.

//    4.    Launch a device-side kernel that will invoke a ray generation program with a multitude of threads calling
//          optixTrace to begin traversal and the execution of the other programs.



        // Finalize TODO check if these can be called earlier to save VRAM
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_output)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_modelSpaceVertices)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_triangleIndices)));
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }



    return 0;
}