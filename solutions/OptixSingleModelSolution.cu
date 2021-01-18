//
// Created by Jonas on 16/01/2021.
//

#include "OptixSingleModelSolution.h"
#include "../optix/CudaUtilities.h"
#include "../optix/Exception.h"
#include <glm/gtc/type_ptr.hpp>
#include <optix_stubs.h>
#include <fstream>

//#include <optix_function_table_definition.h>

OptixSingleModelSolution::OptixSingleModelSolution(const WorldSpaceMesh &outerWorldSpaceMesh,
                                                   const WorldSpaceMesh &innerWorldSpaceMesh, const StreamContext& streamContext) {

    /// 1. Build the GAS for the outer mesh
    std::vector<float3> outerModelSpaceVertices;
    std::vector<uint3> outerTriangleIndices;
    std::vector<float3> outerEdgeOrigins;
    std::vector<float3> outerEdgeDirections;

    for(Vertex vertex: outerWorldSpaceMesh.getModelSpaceMesh().vertices){
        outerModelSpaceVertices.emplace_back(vec3ToFloat3(vertex));
    }

    for(Triangle triangle: outerWorldSpaceMesh.getModelSpaceMesh().triangles){
        outerTriangleIndices.emplace_back(triangleToUint3(triangle));
    }

    for(Edge edge: outerWorldSpaceMesh.getModelSpaceMesh().getEdges()){
        Vertex v0 = outerWorldSpaceMesh.getModelSpaceMesh().vertices[edge.vertexIndex0];
        Vertex v1 = outerWorldSpaceMesh.getModelSpaceMesh().vertices[edge.vertexIndex1];
        outerEdgeDirections.emplace_back(vec3ToFloat3(v1 - v0));
        outerEdgeOrigins.emplace_back(vec3ToFloat3(v0));
    }
    unsigned int outerNumberOfTriangles = outerTriangleIndices.size();
    unsigned int outerNumberOfVertices = outerModelSpaceVertices.size();
    unsigned int outerNumberOfEdges = outerEdgeOrigins.size();

    assert(outerEdgeOrigins.size() == outerEdgeDirections.size());
    assert(outerNumberOfEdges == outerEdgeDirections.size());

    CUDA_CALL(cudaMalloc(reinterpret_cast<void **>(&d_outerTriangleIndices), sizeof(uint3) * outerNumberOfTriangles));
    CUDA_CALL(cudaMalloc(reinterpret_cast<void **>(&d_outerModelSpaceVertices), sizeof(float3) * outerNumberOfVertices));
    CUDA_CALL(cudaMalloc(reinterpret_cast<void **>(&d_outerModelSpaceEdgeOrigins), sizeof(float3) * outerNumberOfEdges));
    CUDA_CALL(cudaMalloc(reinterpret_cast<void **>(&d_outerModelSpaceEdgeDirections), sizeof(float3) * outerNumberOfEdges));

    CUDA_CALL(cudaMemcpyAsync(reinterpret_cast<void *>(d_outerTriangleIndices), outerTriangleIndices.data(), sizeof(uint3) * outerNumberOfTriangles, cudaMemcpyHostToDevice, streamContext.cuStream));
    CUDA_CALL(cudaMemcpyAsync(reinterpret_cast<void *>(d_outerModelSpaceVertices), outerModelSpaceVertices.data(), sizeof(float3) * outerNumberOfVertices, cudaMemcpyHostToDevice, streamContext.cuStream));
    CUDA_CALL(cudaMemcpyAsync(reinterpret_cast<void *>(d_outerModelSpaceEdgeOrigins), outerEdgeOrigins.data(), sizeof(float3) * outerNumberOfEdges, cudaMemcpyHostToDevice, streamContext.cuStream));
    CUDA_CALL(cudaMemcpyAsync(reinterpret_cast<void *>(d_outerModelSpaceEdgeDirections), outerEdgeDirections.data(), sizeof(float3) * outerNumberOfEdges, cudaMemcpyHostToDevice, streamContext.cuStream));


    OptixAccelBuildOptions accelOptions = {};
    accelOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_UPDATE;

    // Populate the build input struct with our triangle data as well as
    // information about the sizes and types of our data
    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    buildInput.triangleArray.vertexBuffers = &d_outerModelSpaceVertices;
    buildInput.triangleArray.numVertices = outerModelSpaceVertices.size();
    buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.vertexStrideInBytes = 0;
    buildInput.triangleArray.indexBuffer = d_outerTriangleIndices;
    buildInput.triangleArray.numIndexTriplets = outerTriangleIndices.size();
    buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    buildInput.triangleArray.indexStrideInBytes = 0;  // Defaults to uint3
    buildInput.triangleArray.numSbtRecords = 1;
    buildInput.triangleArray.preTransform = 0;
    unsigned int triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT }; // Setting this flag usuallyimproves performance even if no any-edgeIntersection program is present in the SBT
    buildInput.triangleArray.flags = triangle_input_flags;


    OptixAccelBufferSizes outerBufferSizes;
    OPTIX_CALL(optixAccelComputeMemoryUsage(streamContext.optixDeviceContext, &accelOptions, &buildInput, 1, &outerBufferSizes));


    CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&d_outerGAS), outerBufferSizes.outputSizeInBytes));
    CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&d_temp), outerBufferSizes.tempSizeInBytes));

    OPTIX_CALL(optixAccelBuild(streamContext.optixDeviceContext, streamContext.cuStream,
                               &accelOptions, &buildInput, 1, d_temp,
                               outerBufferSizes.tempSizeInBytes, d_outerGAS,
                               outerBufferSizes.outputSizeInBytes, &outerModelSpaceHandle,
                               nullptr, 0)); // Last 2 elements used for compacting

    CUDA_CALL(cudaFree(reinterpret_cast<void*>(d_temp)));

    /// 2. Build the instance for the outer mesh
    const float* outerTransform = glm::value_ptr(glm::transpose(outerWorldSpaceMesh.getModelTransformation()));
    memcpy(outerInstance.transform, outerTransform, sizeof(float) * 12);
    outerInstance.sbtOffset = 0;
    outerInstance.visibilityMask = 255;
    outerInstance.flags = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT;
    outerInstance.traversableHandle = outerModelSpaceHandle;
    outerInstance.instanceId = 0;

    CUDA_CALL(cudaMalloc(&d_instances, 2 * sizeof(OptixInstance)));
    CUDA_CALL(cudaMemcpyAsync(&d_instances[0], &outerInstance, sizeof(OptixInstance), cudaMemcpyHostToDevice, streamContext.cuStream));

    /// 3. Build the GAS for the inner mesh
    std::vector<float3> innerModelSpaceVertices;
    std::vector<uint3> innerTriangleIndices;
    std::vector<float3> innerEdgeOrigins;
    std::vector<float3> innerEdgeDirections;

    for(Vertex vertex: innerWorldSpaceMesh.getModelSpaceMesh().vertices){
        innerModelSpaceVertices.emplace_back(vec3ToFloat3(vertex));
    }

    for(Triangle triangle: innerWorldSpaceMesh.getModelSpaceMesh().triangles){
        innerTriangleIndices.emplace_back(triangleToUint3(triangle));
    }

    for(Edge edge: innerWorldSpaceMesh.getModelSpaceMesh().getEdges()){
        Vertex v0 = innerWorldSpaceMesh.getModelSpaceMesh().vertices[edge.vertexIndex0];
        Vertex v1 = innerWorldSpaceMesh.getModelSpaceMesh().vertices[edge.vertexIndex1];
        innerEdgeDirections.emplace_back(vec3ToFloat3(v1 - v0));
        innerEdgeOrigins.emplace_back(vec3ToFloat3(v0));
    }
    unsigned int innerNumberOfTriangles = innerTriangleIndices.size();
    unsigned int innerNumberOfVertices = innerModelSpaceVertices.size();
    unsigned int innerNumberOfEdges = innerEdgeOrigins.size();

    assert(innerEdgeOrigins.size() == innerEdgeDirections.size());
    assert(innerNumberOfEdges == innerEdgeDirections.size());

    CUDA_CALL(cudaMalloc(reinterpret_cast<void **>(&d_innerTriangleIndices), sizeof(uint3) * innerNumberOfTriangles));
    CUDA_CALL(cudaMalloc(reinterpret_cast<void **>(&d_innerModelSpaceVertices), sizeof(float3) * innerNumberOfVertices));
    CUDA_CALL(cudaMalloc(reinterpret_cast<void **>(&d_innerModelSpaceEdgeOrigins), sizeof(float3) * innerNumberOfEdges));
    CUDA_CALL(cudaMalloc(reinterpret_cast<void **>(&d_innerModelSpaceEdgeDirections), sizeof(float3) * innerNumberOfEdges));

    CUDA_CALL(cudaMemcpyAsync(reinterpret_cast<void *>(d_innerTriangleIndices), innerTriangleIndices.data(), sizeof(uint3) * innerNumberOfTriangles, cudaMemcpyHostToDevice, streamContext.cuStream));
    CUDA_CALL(cudaMemcpyAsync(reinterpret_cast<void *>(d_innerModelSpaceVertices), innerModelSpaceVertices.data(), sizeof(float3) * innerNumberOfVertices, cudaMemcpyHostToDevice, streamContext.cuStream));
    CUDA_CALL(cudaMemcpyAsync(reinterpret_cast<void *>(d_innerModelSpaceEdgeOrigins), innerEdgeOrigins.data(), sizeof(float3) * innerNumberOfEdges, cudaMemcpyHostToDevice, streamContext.cuStream));
    CUDA_CALL(cudaMemcpyAsync(reinterpret_cast<void *>(d_innerModelSpaceEdgeDirections), innerEdgeDirections.data(), sizeof(float3) * innerNumberOfEdges, cudaMemcpyHostToDevice, streamContext.cuStream));

    // Populate the build input struct with our triangle data as well as
    // information about the sizes and types of our data
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    buildInput.triangleArray.vertexBuffers = &d_innerModelSpaceVertices;
    buildInput.triangleArray.numVertices = innerModelSpaceVertices.size();
    buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.vertexStrideInBytes = 0;
    buildInput.triangleArray.indexBuffer = d_innerTriangleIndices;
    buildInput.triangleArray.numIndexTriplets = innerTriangleIndices.size();
    buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    buildInput.triangleArray.indexStrideInBytes = 0;  // Defaults to uint3
    buildInput.triangleArray.numSbtRecords = 1;
    buildInput.triangleArray.preTransform = 0;
    buildInput.triangleArray.flags = triangle_input_flags;

    OptixAccelBufferSizes innerBufferSizes;
    OPTIX_CALL(optixAccelComputeMemoryUsage(streamContext.optixDeviceContext, &accelOptions, &buildInput, 1, &innerBufferSizes));

    CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&d_innerGAS), innerBufferSizes.outputSizeInBytes));
    CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&d_temp), innerBufferSizes.tempSizeInBytes));

    OPTIX_CALL(optixAccelBuild(streamContext.optixDeviceContext, streamContext.cuStream,
                               &accelOptions, &buildInput, 1, d_temp,
                               innerBufferSizes.tempSizeInBytes, d_innerGAS,
                               innerBufferSizes.outputSizeInBytes, &innerModelSpaceHandle,
                               nullptr, 0)); // Last 2 elements used for compacting

    CUDA_CALL(cudaFree(reinterpret_cast<void*>(d_temp)));

    /// 4. Build the instance for the inner mesh
    const float* innerTransform = glm::value_ptr(glm::transpose(innerWorldSpaceMesh.getModelTransformation()));
    memcpy(innerInstance.transform, innerTransform, sizeof(float) * 12);
    innerInstance.sbtOffset = 0;
    innerInstance.visibilityMask = 255;
    innerInstance.flags = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT;
    innerInstance.traversableHandle = innerModelSpaceHandle;
    innerInstance.instanceId = 1;

    CUDA_CALL(cudaMemcpyAsync(&d_instances[1], &innerInstance, sizeof(OptixInstance), cudaMemcpyHostToDevice, streamContext.cuStream));


    /// 5. Build the instancearray containing inner and outer meshes
    instanceBuildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instanceBuildInput.instanceArray.instances = reinterpret_cast<CUdeviceptr>(d_instances);
    instanceBuildInput.instanceArray.numInstances = 2;

    OPTIX_CALL(optixAccelComputeMemoryUsage(streamContext.optixDeviceContext, &accelOptions, &instanceBuildInput, 1, &instanceBufferSizes));

    CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&d_outputInstance), instanceBufferSizes.outputSizeInBytes));
    CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&d_temp), instanceBufferSizes.tempSizeInBytes));

    OPTIX_CALL(optixAccelBuild(streamContext.optixDeviceContext, streamContext.cuStream,
                               &accelOptions, &instanceBuildInput, 1, d_temp,
                               instanceBufferSizes.tempSizeInBytes, d_outputInstance,
                               instanceBufferSizes.outputSizeInBytes, &instanceWorldSpaceHandle,
                               nullptr, 0)); // Last 2 elements used for compacting

    CUDA_CALL(cudaFree(reinterpret_cast<void*>(d_temp)));
    CUDA_CALL(cudaMalloc(reinterpret_cast<void **>(&d_temp), instanceBufferSizes.tempUpdateSizeInBytes));

    // Allocate memory for updating instance build
    CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&d_temp), instanceBufferSizes.tempUpdateSizeInBytes));

    /// 6. create the optixPipeline
    // Set the options for module compilation
    OptixModuleCompileOptions moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT; // Default is OPTIX_COMPILE_OPTIMIZATION_LEVEL_3 (run all optimizations)
#if NDEBUG
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#else
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#endif

    // Set the options for edgeIntersectionPipeline compilation
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipelineCompileOptions.numPayloadValues = 1;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParameters";
    pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;  // Improves performance

    // Depending on the scenario and combination of flags, enabling exceptions can lead to severe overhead, so some flags shouldbe mainly used in internal and debug builds.
#if NDEBUG
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#else
    pipelineCompileOptions.exceptionFlags |= OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    pipelineCompileOptions.exceptionFlags |= OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_USER;
    pipelineCompileOptions.exceptionFlags |= OPTIX_EXCEPTION_FLAG_TRACE_DEPTH;
#endif

    // Compile the module based on the OptixProgram.ptx
    OptixModule ptxModule;
    std::ifstream t("../../optix/OptixPrograms/OptixPrograms2.ptx");
    std::string ptxString((std::istreambuf_iterator<char>(t)),
                          std::istreambuf_iterator<char>());

    OPTIX_LOG_CALL(optixModuleCreateFromPTX(streamContext.optixDeviceContext,
                                            &moduleCompileOptions,
                                            &pipelineCompileOptions,
                                            ptxString.c_str(),
                                            ptxString.size(),
                                            LOG,
                                            &LOG_SIZE,
                                            &ptxModule));

    // Use the modules to create the necessary programgroups (RAYGEN + HITGROUP + MISS)
    OptixProgramGroupDesc programGroupDescriptions[3] = {};
    programGroupDescriptions[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    programGroupDescriptions[0].raygen.module = ptxModule;
    programGroupDescriptions[0].raygen.entryFunctionName = "__raygen__edgeIntersectionTest2__";
    programGroupDescriptions[1].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    programGroupDescriptions[1].hitgroup.moduleCH = ptxModule;
    programGroupDescriptions[1].hitgroup.entryFunctionNameCH = "__closesthit__edgeIntersectionTest2__";
    // As a special case, the intersection program is not required – and is ignored – for triangle primitives.
    programGroupDescriptions[2].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    programGroupDescriptions[2].miss.module = nullptr;
    programGroupDescriptions[2].miss.entryFunctionName = nullptr;

    OptixProgramGroup programGroups[3];
    OptixProgramGroupOptions pgOptions = {};
    OPTIX_LOG_CALL(optixProgramGroupCreate(streamContext.optixDeviceContext,
                                           programGroupDescriptions,
                                           3,
                                           &pgOptions,
                                           LOG, &LOG_SIZE,
                                           programGroups));

    // Create a edgeIntersectionPipeline with these program groups
    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth = 1;
    pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    OPTIX_LOG_CALL(optixPipelineCreate(streamContext.optixDeviceContext,
                                       &pipelineCompileOptions,
                                       &pipelineLinkOptions,
                                       programGroups, 3,
                                       LOG, &LOG_SIZE,
                                       &edgeIntersectionPipeline));

    numberOfRays = innerNumberOfEdges + outerNumberOfEdges;
    CUDA_CALL(cudaMalloc(reinterpret_cast<float3 **>(&d_rayOrigins), sizeof(float3) * numberOfRays));
    CUDA_CALL(cudaMalloc(reinterpret_cast<float3 **>(&d_rayDirections), sizeof(float3) * numberOfRays));
    CUDA_CALL(cudaMalloc(reinterpret_cast<unsigned int **>(&d_rayTransformIndices), sizeof(unsigned int) * numberOfRays));
    CUDA_CALL(cudaMemcpyAsync(reinterpret_cast<float3 *>(d_rayOrigins), reinterpret_cast<const void *>(d_outerModelSpaceEdgeOrigins), sizeof(float3) * outerNumberOfEdges, cudaMemcpyDeviceToDevice, streamContext.cuStream));
    CUDA_CALL(cudaMemsetAsync(reinterpret_cast<unsigned int *>(d_rayTransformIndices), 1u, outerNumberOfEdges, streamContext.cuStream));
    CUDA_CALL(cudaMemcpyAsync(reinterpret_cast<float3 *>(d_rayOrigins + sizeof(float3) * outerNumberOfEdges), reinterpret_cast<const float3 *>(d_innerModelSpaceEdgeOrigins), sizeof(float3) * innerNumberOfEdges, cudaMemcpyDeviceToDevice, streamContext.cuStream));
    CUDA_CALL(cudaMemsetAsync(reinterpret_cast<unsigned int *>(d_rayTransformIndices + sizeof(unsigned int) * outerNumberOfEdges), 1, innerNumberOfEdges, streamContext.cuStream));

    //void *devPtr, int value, size_t count, cudaStream_t stream __dv(0)

    CUDA_CALL(cudaMalloc(&d_transformations, 2*12*sizeof(float)));
    CUDA_CALL(cudaMemcpyAsync(&d_transformations[0], outerTransform, 12*sizeof(float), cudaMemcpyHostToDevice, streamContext.cuStream));
    CUDA_CALL(cudaMemcpyAsync(&d_transformations[12], innerTransform, 12*sizeof(float), cudaMemcpyHostToDevice, streamContext.cuStream));

    /// 7.    Create a shader binding table that includes references to these programs and their parameters.

    RayGenSbtRecord rayGenRecord;
    rayGenRecord.data = {};
    rayGenRecord.data.origins = reinterpret_cast<float3 *>(this->d_rayOrigins);
    rayGenRecord.data.directions = reinterpret_cast<float3 *>(this->d_rayDirections);
    rayGenRecord.data.transformations = this->d_transformations;
    rayGenRecord.data.rayTransformIndices = reinterpret_cast<unsigned int *>(this->d_rayTransformIndices);

    const size_t raygen_record_size = sizeof(RayGenSbtRecord);
    OPTIX_CALL(optixSbtRecordPackHeader(programGroups[0], &rayGenRecord));
    CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&d_raygenRecord), raygen_record_size));
    CUDA_CALL(cudaMemcpyAsync(reinterpret_cast<void *>(d_raygenRecord), &rayGenRecord, sizeof(RayGenSbtRecord), cudaMemcpyHostToDevice, streamContext.cuStream));
    sbt.raygenRecord = d_raygenRecord;

    hitGroupRecord.data.edgeIntersection = false;
    OPTIX_CALL(optixSbtRecordPackHeader(programGroups[1], &hitGroupRecord));
    CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&d_structEdgeIntersectionSbtRecord), sizeof(EdgeIntersectionSbtRecord)));
    CUDA_CALL(cudaMemcpyAsync(reinterpret_cast<void *>(d_structEdgeIntersectionSbtRecord), &hitGroupRecord, sizeof(EdgeIntersectionSbtRecord), cudaMemcpyHostToDevice, streamContext.cuStream));
    sbt.hitgroupRecordBase = d_structEdgeIntersectionSbtRecord;
    sbt.hitgroupRecordStrideInBytes = sizeof(EdgeIntersectionSbtRecord);
    sbt.hitgroupRecordCount = 1;
//
    char missHeader[OPTIX_SBT_RECORD_HEADER_SIZE];
    OPTIX_CALL(optixSbtRecordPackHeader(programGroups[2], &missHeader));
    CUDA_CALL(cudaMalloc(reinterpret_cast<void **>(&d_missRecord), OPTIX_SBT_RECORD_HEADER_SIZE));
    CUDA_CALL(cudaMemcpyAsync(reinterpret_cast<void *>(d_missRecord), &missHeader, OPTIX_SBT_RECORD_HEADER_SIZE, cudaMemcpyHostToDevice, streamContext.cuStream));
    sbt.missRecordBase = d_missRecord;
    sbt.missRecordCount = 1;
    sbt.missRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;

    d_EdgeIntersectionPointer = &(reinterpret_cast<EdgeIntersectionSbtRecord*>(d_structEdgeIntersectionSbtRecord))->data.edgeIntersection;

    CUDA_CALL(cudaMallocHost(&h_optixLaunchParameters, sizeof(OptixLaunchParameters)));
    CUDA_CALL(cudaMalloc(reinterpret_cast<void **>(&d_optixLaunchParameters), sizeof(OptixLaunchParameters)));
    CUDA_CALL(cudaMallocHost(&h_hitGroupRecord, sizeof(EdgeIntersectionSbtRecord)));

}

bool OptixSingleModelSolution::isFeasible(const StreamContext& streamContext) {

    const float* transform = glm::value_ptr( glm::transpose(this->innerTransformation));
    memcpy(innerInstance.transform, transform, sizeof(float)*12);
    CUDA_CALL(cudaMemcpyAsync(reinterpret_cast<void*>(&d_instances[1]), &innerInstance, sizeof(OptixInstance), cudaMemcpyHostToDevice, streamContext.cuStream));
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.operation = OPTIX_BUILD_OPERATION_UPDATE;
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_UPDATE;

    OPTIX_CALL(optixAccelBuild(streamContext.optixDeviceContext, streamContext.cuStream,
                               &accelOptions, &instanceBuildInput, 1, d_temp,
                               instanceBufferSizes.tempUpdateSizeInBytes, d_outputInstance,
                               instanceBufferSizes.outputSizeInBytes, &instanceWorldSpaceHandle,
                               nullptr, 0)); // Last 2 elements used for compacting

    *h_optixLaunchParameters = OptixLaunchParameters();
    h_optixLaunchParameters->handle = instanceWorldSpaceHandle;
    CUDA_CALL(cudaMemcpyAsync(reinterpret_cast<void *>(d_optixLaunchParameters), h_optixLaunchParameters, sizeof(OptixLaunchParameters), cudaMemcpyHostToDevice, streamContext.cuStream));

    //    4.    Launch a device-side kernel that will invoke a ray generation program with a multitude of threads calling
    //          optixTrace to begin traversal and the execution of the other programs.

    h_hitGroupRecord->data.edgeIntersection = false;
    CUDA_CALL(cudaMemcpyAsync(d_EdgeIntersectionPointer, &h_hitGroupRecord->data.edgeIntersection, sizeof(bool), cudaMemcpyHostToDevice, streamContext.cuStream));

    OPTIX_CALL(optixLaunch(edgeIntersectionPipeline, streamContext.cuStream, d_optixLaunchParameters, sizeof(OptixLaunchParameters), &sbt, numberOfRays, 1, 1));
    CUDA_CALL(cudaMemcpyAsync(&h_hitGroupRecord->data.edgeIntersection, d_EdgeIntersectionPointer, sizeof(bool), cudaMemcpyDeviceToHost, streamContext.cuStream));
    CUDA_CALL(cudaStreamSynchronize(streamContext.cuStream));
    bool returnValue = !h_hitGroupRecord->data.edgeIntersection;

    return returnValue;
}

void OptixSingleModelSolution::setInnerTransformation(const Transformation &newInnerTransformation) {
    this->innerTransformation = newInnerTransformation;
}

OptixSingleModelSolution::~OptixSingleModelSolution() {
    CUDA_CALL(cudaFree(reinterpret_cast<void *>(d_outerModelSpaceVertices)));
    CUDA_CALL(cudaFree(reinterpret_cast<void *>(d_outerTriangleIndices)));
    CUDA_CALL(cudaFree(reinterpret_cast<void *>(d_outerModelSpaceEdgeOrigins)));
    CUDA_CALL(cudaFree(reinterpret_cast<void *>(d_outerModelSpaceEdgeDirections)));

    CUDA_CALL(cudaFree(reinterpret_cast<void *>(d_innerModelSpaceVertices)));
    CUDA_CALL(cudaFree(reinterpret_cast<void *>(d_innerTriangleIndices)));
    CUDA_CALL(cudaFree(reinterpret_cast<void *>(d_innerModelSpaceEdgeOrigins)));
    CUDA_CALL(cudaFree(reinterpret_cast<void *>(d_innerModelSpaceEdgeDirections)));

    CUDA_CALL(cudaFree(reinterpret_cast<void *>(d_outerGAS)));
    CUDA_CALL(cudaFree(reinterpret_cast<void *>(d_innerGAS)));
    CUDA_CALL(cudaFree(reinterpret_cast<void *>(d_outputInstance)));

    CUDA_CALL(cudaFree(reinterpret_cast<void *>(d_rayOrigins)));
    CUDA_CALL(cudaFree(reinterpret_cast<void *>(d_rayDirections)));

    CUDA_CALL(cudaFree(reinterpret_cast<void*>(d_temp)));

    CUDA_CALL(cudaFree(reinterpret_cast<void *>(d_raygenRecord)));
}
