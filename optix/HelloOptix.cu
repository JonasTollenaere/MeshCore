

#include <iostream>

#include <chrono>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>
#include <glm/gtc/type_ptr.hpp>

#include "Exception.h"
#include "../core/WorldSpaceMesh.h"
#include "../core/FileParser.h"

#include <string>
#include <fstream>

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
        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
//        accelOptions.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE; // TODO use this to refit the worldspace instance quicklyu
//        accelOptions.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_COMPACTION; // TODO see documentation 5.7 (don't do this if this degrades tracing performance)

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
        OptixBuildInput buildInput = {};
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
        unsigned int triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT }; // Setting this flag usuallyimproves performance even if no any-hit program is present in the SBT
        buildInput.triangleArray.flags = triangle_input_flags;

        accelOptions.motionOptions.numKeys = 0; // A numKeys value of zero specifies no motion blur

        OptixAccelBufferSizes bufferSizes;

        OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext, &accelOptions, &buildInput, 1, &bufferSizes));
        CUdeviceptr d_outputGAS;
        CUdeviceptr d_temp;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_outputGAS), bufferSizes.outputSizeInBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp), bufferSizes.tempSizeInBytes));


        OptixTraversableHandle modelSpaceGASHandle = 0;

        OPTIX_CHECK(optixAccelBuild(optixContext, cuStream,       // This runs asyncrhonously on the GPU
                                              &accelOptions, &buildInput, 1, d_temp,
                                    bufferSizes.tempSizeInBytes, d_outputGAS,
                                    bufferSizes.outputSizeInBytes, &modelSpaceGASHandle,
                                    nullptr, 0));

        float transform[12] = {1,0,0,0,
                               0,1,0,0,
                               0,0,1,0}; // TODO set current worldspacemesh transform

        OptixStaticTransform modelTransformation;
        memcpy(modelTransformation.transform, transform, sizeof(float)*12);
        memcpy(modelTransformation.invTransform, transform, sizeof(float)*12);
        modelTransformation.child = modelSpaceGASHandle;



//        OptixInstance instance;
//        memcpy(instance.transform, transform, sizeof(float)*12);
//        instance.instanceId = 0;
//        instance.visibilityMask = 255;
//        instance.sbtOffset = 0;
//        instance.flags = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT;
//        instance.traversableHandle = modelSpaceGASHandle;

//        CUdeviceptr d_instance;
//        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_instance), sizeof(OptixInstance)));
//        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_instance), &instance, sizeof(OptixInstance), cudaMemcpyHostToDevice));

//        OptixBuildInput worldSpaceBuildInput;
//        memset(&worldSpaceBuildInput, 0,  sizeof(OptixBuildInput));
//        OptixBuildInputInstanceArray worldSpaceBuildInputArray = worldSpaceBuildInput.instanceArray;
//        worldSpaceBuildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
//        worldSpaceBuildInputArray.instances = d_instance;
//        worldSpaceBuildInputArray.numInstances = 1;

//        CUdeviceptr d_outputInstance;
//        OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext, &accelOptions, &worldSpaceBuildInput, 1, &bufferSizes));
//        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_outputInstance), bufferSizes.outputSizeInBytes));
//        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp), bufferSizes.tempSizeInBytes));
//
//
//        OptixTraversableHandle worldSpaceInstanceHandle = 0;
//
//        OPTIX_CHECK(optixAccelBuild(optixContext, cuStream,       // This runs asyncrhonously on the GPU
//                                    &accelOptions, &worldSpaceBuildInput, 1, d_temp,
//                                    bufferSizes.tempSizeInBytes, d_outputGAS,
//                                    bufferSizes.outputSizeInBytes, &worldSpaceInstanceHandle,
//                                    nullptr, 0));

//    2.    Create a pipeline of programs that contains all programs that will be invoked during a ray tracing launch.

        // Set the options for module compilation
        OptixModuleCompileOptions moduleCompileOptions = {};
        moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

        // Set the options for pipeline compilation
        OptixPipelineCompileOptions pipelineCompileOptions = {};
        pipelineCompileOptions.usesMotionBlur = false;
        pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
        pipelineCompileOptions.numPayloadValues = 2;
        pipelineCompileOptions.numAttributeValues = 2;
        pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
        pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;  // Improves performance
        pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
        char logString[2048];
        size_t logStringSize = sizeof(logString);

        // Get built-in intersection module
        OptixModule intersectionModule;
        OptixBuiltinISOptions builtinISOptions = {};
        builtinISOptions.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_TRIANGLE;
        OPTIX_CHECK(optixBuiltinISModuleGet(optixContext,
                               &moduleCompileOptions,
                               &pipelineCompileOptions,
                               &builtinISOptions,
                               &intersectionModule));


        // Compile the module based on the OptixProgram.ptx
        OptixModule ptxModule;
        std::ifstream t("../../optix/OptixPrograms/OptixPrograms.ptx");
        std::string ptxString((std::istreambuf_iterator<char>(t)),
                              std::istreambuf_iterator<char>());
        OPTIX_CHECK(optixModuleCreateFromPTX(optixContext,
                                                 &moduleCompileOptions,
                                                 &pipelineCompileOptions,
                                                 ptxString.c_str(),
                                                 ptxString.size(),
                                                 logString,
                                                 &logStringSize,
                                                 &ptxModule));

        // Use the modules to create the necessary programgroups (HITGROUP + RAYGEN)
        OptixProgramGroupDesc programGroupDescriptions[2] = {};
        programGroupDescriptions[0].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        programGroupDescriptions[0].hitgroup.moduleCH = ptxModule;
        programGroupDescriptions[0].hitgroup.entryFunctionNameCH = "__closesthit__ch";
        programGroupDescriptions[0].hitgroup.moduleAH = nullptr;
        programGroupDescriptions[0].hitgroup.entryFunctionNameAH = nullptr;
        programGroupDescriptions[0].hitgroup.moduleIS = intersectionModule;
        programGroupDescriptions[0].hitgroup.entryFunctionNameIS = nullptr;
        programGroupDescriptions[1].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        programGroupDescriptions[1].raygen.module = ptxModule;
        programGroupDescriptions[1].raygen.entryFunctionName = "__raygen__rg";
        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroup programGroups[2];
        OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                            programGroupDescriptions,
                                            2,
                                            &pgOptions,
                                            logString, &logStringSize,
                                            programGroups));

        // Create a pipeline with these program groups
        OptixPipeline pipeline = nullptr;
        OptixPipelineLinkOptions pipelineLinkOptions = {};
        pipelineLinkOptions.maxTraceDepth = 1;              // TODO CHECK WHAT THIS DOES
        pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
        OPTIX_CHECK(optixPipelineCreate(optixContext,
                                        &pipelineCompileOptions,
                                        &pipelineLinkOptions,
                                        programGroups, 2,
                                        logString, &logStringSize,
                                        &pipeline));

//    3.    Create a shader binding table that includes references to these programs and their parameters.

//    4.    Launch a device-side kernel that will invoke a ray generation program with a multitude of threads calling
//          optixTrace to begin traversal and the execution of the other programs.



        // Finalize TODO check if these can be called earlier to save VRAM
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_outputGAS)));
//        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_outputInstance)));
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