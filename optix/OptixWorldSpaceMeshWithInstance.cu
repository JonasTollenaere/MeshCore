//
// Created by Jonas on 17/12/2020.
//

#include <optix_stubs.h>
#include "OptixWorldSpaceMeshWithInstance.h"
#include "Exception.h"
#include <glm/gtc/type_ptr.hpp>
#include <fstream>
//#include <optix_function_table_definition.h>

static float3 vec3ToFloat3(const glm::vec3& vec3){
    return make_float3(vec3.x, vec3.y, vec3.z);
}

static uint3 triangleToUint3(const Triangle& triangle){
    return make_uint3(triangle.vertexIndex0, triangle.vertexIndex1, triangle.vertexIndex2);
}

OptixWorldSpaceMeshWithInstance::OptixWorldSpaceMeshWithInstance(const WorldSpaceMesh& worldSpaceMesh, const CUstream& cUStream, const OptixDeviceContext& optixDeviceContext):
modelTransformation(worldSpaceMesh.getModelTransformation()),
d_triangleIndices(0),
d_modelSpaceVertices(0),
d_modelTransformation(0),
d_modelSpaceEdgeOrigins(0),
d_modelSpaceEdgeDirections(0),
d_optixLaunchParameters(0),
cuStream(&cUStream),
optixDeviceContext(optixDeviceContext)
{

//    1.    Create acceleration structures that represent a geometry mesh in the scene and select
//          on or more records in the shader binding table for each mesh.

    // Load the necessary modelspacedata on the device
    std::vector<float3> modelSpaceVertices;
    std::vector<uint3> triangleIndices;
    std::vector<float3> edgeOrigins;
    std::vector<float3> edgeDirections;

    for(Vertex vertex: worldSpaceMesh.getModelSpaceMesh().vertices){
        modelSpaceVertices.emplace_back(vec3ToFloat3(vertex));
    }

    for(Triangle triangle: worldSpaceMesh.getModelSpaceMesh().triangles){
        triangleIndices.emplace_back(triangleToUint3(triangle));
    }

    for(Edge edge: worldSpaceMesh.getModelSpaceMesh().getEdges()){
        Vertex v0 = worldSpaceMesh.getModelSpaceMesh().vertices[edge.vertexIndex0];
        Vertex v1 = worldSpaceMesh.getModelSpaceMesh().vertices[edge.vertexIndex1];
        edgeDirections.emplace_back(vec3ToFloat3(v1 - v0));
        edgeOrigins.emplace_back(vec3ToFloat3(v0));
    }
    numberOfTriangles = triangleIndices.size();
    numberOfVertices = modelSpaceVertices.size();
    numberOfEdges = edgeOrigins.size();

    assert(edgeOrigins.size()==edgeDirections.size());
    assert(numberOfEdges==edgeDirections.size());

    CUDA_CALL(cudaMallocAsync(reinterpret_cast<void **>(&d_modelSpaceVertices), sizeof(float3) * numberOfVertices, this->cuStream[0]));
    CUDA_CALL(cudaMallocAsync(reinterpret_cast<void **>(&d_triangleIndices), sizeof(uint3) * numberOfTriangles, this->cuStream[0]));
    CUDA_CALL(cudaMallocAsync(reinterpret_cast<void **>(&d_modelSpaceEdgeOrigins), sizeof(float3) * numberOfEdges, this->cuStream[0]));
    CUDA_CALL(cudaMallocAsync(reinterpret_cast<void **>(&d_modelSpaceEdgeDirections), sizeof(float3) * numberOfEdges, this->cuStream[0]));
    CUDA_CALL(cudaMallocAsync(reinterpret_cast<void **>(&d_modelTransformation), sizeof(OptixStaticTransform), this->cuStream[0]));
    CUDA_CALL(cudaMallocAsync(reinterpret_cast<void **>(&d_optixLaunchParameters), sizeof(OptixLaunchParameters), this->cuStream[0]));

    CUDA_CALL(cudaMemcpyAsync(reinterpret_cast<void *>(d_modelSpaceVertices), modelSpaceVertices.data(), sizeof(float3) * numberOfVertices, cudaMemcpyHostToDevice, this->cuStream[0]));
    CUDA_CALL(cudaMemcpyAsync(reinterpret_cast<void *>(d_triangleIndices), triangleIndices.data(), sizeof(uint3) * numberOfTriangles, cudaMemcpyHostToDevice, this->cuStream[0]));
    CUDA_CALL(cudaMemcpyAsync(reinterpret_cast<void *>(d_modelSpaceEdgeOrigins), edgeOrigins.data(), sizeof(float3) * numberOfEdges, cudaMemcpyHostToDevice, this->cuStream[0]));
    CUDA_CALL(cudaMemcpyAsync(reinterpret_cast<void *>(d_modelSpaceEdgeDirections), edgeDirections.data(), sizeof(float3) * numberOfEdges, cudaMemcpyHostToDevice, this->cuStream[0]));

    CUDA_CALL(cudaMallocAsync(&d_iterationData, sizeof(IterationData), this->cuStream[0]));

    d_structModelTransformation = reinterpret_cast<CUdeviceptr>(&d_iterationData->modelTransformation);
    d_structOptixLaunchParameters = reinterpret_cast<CUdeviceptr>(&d_iterationData->optixLaunchParameters);
    d_structEdgeIntersectionSbtRecord = reinterpret_cast<CUdeviceptr>(&d_iterationData->edgeIntersectionSbtRecord);

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_UPDATE;

    // Populate the build input struct with our triangle data as well as
    // information about the sizes and types of our data
    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    buildInput.triangleArray.vertexBuffers = &d_modelSpaceVertices;
    buildInput.triangleArray.numVertices = modelSpaceVertices.size();
    buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.vertexStrideInBytes = 0;
    buildInput.triangleArray.indexBuffer = d_triangleIndices;
    buildInput.triangleArray.numIndexTriplets = triangleIndices.size();
    buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    buildInput.triangleArray.indexStrideInBytes = 0;  // Defaults to uint3
    buildInput.triangleArray.numSbtRecords = 1;
    buildInput.triangleArray.preTransform = 0;
    unsigned int triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT }; // Setting this flag usuallyimproves performance even if no any-edgeIntersection program is present in the SBT
    buildInput.triangleArray.flags = triangle_input_flags;


    OptixAccelBufferSizes bufferSizes;
    OPTIX_CALL(optixAccelComputeMemoryUsage(optixDeviceContext, &accelOptions, &buildInput, 1, &bufferSizes));

    CUDA_CALL(cudaMallocAsync(reinterpret_cast<void**>(&d_outputGAS), bufferSizes.outputSizeInBytes, this->cuStream[0]));
    CUDA_CALL(cudaMallocAsync(reinterpret_cast<void**>(&d_temp), bufferSizes.tempSizeInBytes, this->cuStream[0]));

    OPTIX_CALL(optixAccelBuild(this->optixDeviceContext, this->cuStream[0],
                    &accelOptions, &buildInput, 1, d_temp,
                    bufferSizes.tempSizeInBytes, d_outputGAS,
                    bufferSizes.outputSizeInBytes, &modelSpaceHandle,
                    nullptr, 0)); // Last 2 elements used for compacting

    CUDA_CALL(cudaFreeAsync(reinterpret_cast<void*>(d_temp), this->cuStream[0]));

    // Allocate d_temp for updating the instancearray


    // Now build the instance containing the transform
    const float* transform = glm::value_ptr(glm::transpose(worldSpaceMesh.getModelTransformation()));
    memcpy(instance.transform, transform, sizeof(float)*12);  // TODO in practice the model-to-model transformation is placed in this slot (see notes)
    instance.sbtOffset = 0;
    instance.visibilityMask = 255;
    instance.flags = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT;
    instance.traversableHandle = modelSpaceHandle;
    instance.instanceId = 0;

    CUDA_CALL(cudaMallocAsync(reinterpret_cast<void**>(&d_instance), sizeof(OptixInstance), this->cuStream[0]));
    CUDA_CALL(cudaMemcpyAsync(reinterpret_cast<void*>(d_instance), &instance, sizeof(OptixInstance), cudaMemcpyHostToDevice, this->cuStream[0]));

    instanceBuildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instanceBuildInput.instanceArray.instances = d_instance;
    instanceBuildInput.instanceArray.numInstances = 1;

    OptixAccelBufferSizes instanceBufferSizes;
    OPTIX_CALL(optixAccelComputeMemoryUsage(optixDeviceContext, &accelOptions, &instanceBuildInput, 1, &instanceBufferSizes));

    CUDA_CALL(cudaMallocAsync(reinterpret_cast<void**>(&d_outputInstance), instanceBufferSizes.outputSizeInBytes, this->cuStream[0]));
    CUDA_CALL(cudaMallocAsync(reinterpret_cast<void**>(&d_temp), instanceBufferSizes.tempSizeInBytes, this->cuStream[0]));

    OPTIX_CALL(optixAccelBuild(this->optixDeviceContext, this->cuStream[0],
                               &accelOptions, &instanceBuildInput, 1, d_temp,
                               instanceBufferSizes.tempSizeInBytes, d_outputInstance,
                               instanceBufferSizes.outputSizeInBytes, &worldSpaceHandle,
                               nullptr, 0)); // Last 2 elements used for compacting

    CUDA_CALL(cudaFreeAsync(reinterpret_cast<void*>(d_temp), this->cuStream[0]));
    CUDA_CALL(cudaMallocAsync(reinterpret_cast<void **>(&d_temp), instanceBufferSizes.tempUpdateSizeInBytes,
                              this->cuStream[0]));

    ///    2.    Create a edgeIntersectionPipeline of programs that contains all programs that will be invoked during a ray tracing launch.
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
    pipelineCompileOptions.numPayloadValues = 0;
    pipelineCompileOptions.numAttributeValues = 0;
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
    std::ifstream t("../../optix/OptixPrograms/OptixPrograms.ptx");
    std::string ptxString((std::istreambuf_iterator<char>(t)),
                          std::istreambuf_iterator<char>());

    OPTIX_LOG_CALL(optixModuleCreateFromPTX(this->optixDeviceContext,
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
    programGroupDescriptions[0].raygen.entryFunctionName = "__raygen__edgeIntersectionTest__";
    programGroupDescriptions[1].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    programGroupDescriptions[1].hitgroup.moduleCH = ptxModule;
    programGroupDescriptions[1].hitgroup.entryFunctionNameCH = "__closesthit__edgeIntersectionTest__";
    // As a special case, the intersection program is not required – and is ignored – for triangle primitives.
    programGroupDescriptions[2].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    programGroupDescriptions[2].miss.module = nullptr;
    programGroupDescriptions[2].miss.entryFunctionName = nullptr;

    OptixProgramGroup programGroups[3];
    OptixProgramGroupOptions pgOptions = {};
    OPTIX_LOG_CALL(optixProgramGroupCreate(this->optixDeviceContext,
                                           programGroupDescriptions,
                                           3,
                                           &pgOptions,
                                           LOG, &LOG_SIZE,
                                           programGroups));

    // Create a edgeIntersectionPipeline with these program groups
    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth = 1;
    pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    OPTIX_LOG_CALL(optixPipelineCreate(this->optixDeviceContext,
                                       &pipelineCompileOptions,
                                       &pipelineLinkOptions,
                                       programGroups, 3,
                                       LOG, &LOG_SIZE,
                                       &edgeIntersectionPipeline));

    //    3.    Create a shader binding table that includes references to these programs and their parameters.
    RayGenSbtRecord rayGenRecord;
    rayGenRecord.data = {};
    rayGenRecord.data.origins = reinterpret_cast<float3 *>(this->d_modelSpaceEdgeOrigins);
    rayGenRecord.data.directions = reinterpret_cast<float3 *>(this->d_modelSpaceEdgeDirections);
    const size_t raygen_record_size = sizeof(RayGenSbtRecord);
    OPTIX_CALL(optixSbtRecordPackHeader(programGroups[0], &rayGenRecord));
    CUDA_CALL(cudaMallocAsync(reinterpret_cast<void**>(&d_raygenRecord), raygen_record_size, this->cuStream[0]));
    CUDA_CALL(cudaMemcpyAsync(reinterpret_cast<void *>(d_raygenRecord), &rayGenRecord, sizeof(RayGenSbtRecord), cudaMemcpyHostToDevice, this->cuStream[0]));
    sbt.raygenRecord = d_raygenRecord;


    hitGroupRecord.data.edgeIntersection = false;
    OPTIX_CALL(optixSbtRecordPackHeader(programGroups[1], &hitGroupRecord));
    CUDA_CALL(cudaMemcpyAsync(reinterpret_cast<void *>(d_structEdgeIntersectionSbtRecord), &hitGroupRecord, sizeof(EdgeIntersectionSbtRecord), cudaMemcpyHostToDevice, this->cuStream[0]));
    sbt.hitgroupRecordBase = d_structEdgeIntersectionSbtRecord;
    sbt.hitgroupRecordStrideInBytes = sizeof(EdgeIntersectionSbtRecord);
    sbt.hitgroupRecordCount = 1;

    char missHeader[OPTIX_SBT_RECORD_HEADER_SIZE];
    OPTIX_CALL(optixSbtRecordPackHeader(programGroups[2], &missHeader));
    CUDA_CALL(cudaMallocAsync(reinterpret_cast<void **>(&d_missRecord), OPTIX_SBT_RECORD_HEADER_SIZE, this->cuStream[0]));
    CUDA_CALL(cudaMemcpyAsync(reinterpret_cast<void *>(d_missRecord), &missHeader, OPTIX_SBT_RECORD_HEADER_SIZE, cudaMemcpyHostToDevice, this->cuStream[0]));
    sbt.missRecordBase = d_missRecord;
    sbt.missRecordCount = 1;
    sbt.missRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;

    d_EdgeIntersectionPointer = &(reinterpret_cast<EdgeIntersectionSbtRecord*>(d_structEdgeIntersectionSbtRecord))->data.edgeIntersection;

    CUDA_CALL(cudaMallocHost(&h_hitGroupRecord, sizeof(EdgeIntersectionSbtRecord)));
    CUDA_CALL(cudaMallocHost(&h_modelTransformation, sizeof(OptixStaticTransform)));
    CUDA_CALL(cudaMallocHost(&h_optixLaunchParameters, sizeof(OptixLaunchParameters)));
    CUDA_CALL(cudaMallocHost(&h_iterationData, sizeof(IterationData)));

    *h_iterationData = IterationData{};
    h_iterationData->optixLaunchParameters = OptixLaunchParameters{};
    h_iterationData->edgeIntersectionSbtRecord = hitGroupRecord;
        OPTIX_CALL(optixConvertPointerToTraversableHandle(this->optixDeviceContext, d_structModelTransformation, OPTIX_TRAVERSABLE_TYPE_STATIC_TRANSFORM, &h_iterationData->optixLaunchParameters.handle));

}

OptixWorldSpaceMeshWithInstance::~OptixWorldSpaceMeshWithInstance() {

    // TODO memory used during building can be freed earlier
    CUDA_CALL(cudaFreeAsync(reinterpret_cast<void *>(d_modelSpaceVertices), this->cuStream[0]));
    CUDA_CALL(cudaFreeAsync(reinterpret_cast<void *>(d_triangleIndices), this->cuStream[0]));
    CUDA_CALL(cudaFreeAsync(reinterpret_cast<void *>(d_modelSpaceEdgeOrigins), this->cuStream[0]));
    CUDA_CALL(cudaFreeAsync(reinterpret_cast<void *>(d_modelSpaceEdgeDirections), this->cuStream[0]));
    CUDA_CALL(cudaFreeAsync(reinterpret_cast<void *>(d_modelTransformation), this->cuStream[0]));
    CUDA_CALL(cudaFreeAsync(reinterpret_cast<void *>(d_optixLaunchParameters), this->cuStream[0]));

    CUDA_CALL(cudaFreeAsync(reinterpret_cast<void *>(d_missRecord), this->cuStream[0]));
    CUDA_CALL(cudaFreeAsync(reinterpret_cast<void *>(d_raygenRecord), this->cuStream[0]));

    CUDA_CALL(cudaFreeAsync(reinterpret_cast<void*>(d_outputGAS), this->cuStream[0]));

    CUDA_CALL(cudaFreeHost(h_hitGroupRecord));
    CUDA_CALL(cudaFreeHost(h_modelTransformation));
    CUDA_CALL(cudaFreeHost(h_optixLaunchParameters));
    CUDA_CALL(cudaFreeHost(h_iterationData));

    CUDA_CALL(cudaFreeAsync(d_iterationData, this->cuStream[0]));
    //    CUDA_CALL(cudaFree(reinterpret_cast<void *>(d_temp)));
}

bool OptixWorldSpaceMeshWithInstance::intersectsWithInstance(const OptixWorldSpaceMeshWithInstance &other){

    Transformation otherToCurrentModelSpaceTransformation = glm::inverse(this->modelTransformation) * other.modelTransformation;
    Transformation otherToCurrentModelSpaceTransformationTransposed = glm::transpose(otherToCurrentModelSpaceTransformation);
    const float* transform = glm::value_ptr(otherToCurrentModelSpaceTransformationTransposed);
    memcpy(instance.transform, transform, sizeof(float)*12);
    CUDA_CALL(cudaMemcpyAsync(reinterpret_cast<void*>(d_instance), &instance, sizeof(OptixInstance), cudaMemcpyHostToDevice, this->cuStream[0]));

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.operation = OPTIX_BUILD_OPERATION_UPDATE;
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_UPDATE;

//    CUDA_CALL(cudaMallocAsync(reinterpret_cast<void **>(&d_temp), instanceBufferSizes.tempUpdateSizeInBytes,
//                              this->cuStream[0]));
    OPTIX_CALL(optixAccelBuild(this->optixDeviceContext, this->cuStream[0],
                               &accelOptions, &instanceBuildInput, 1, d_temp,
                               instanceBufferSizes.tempUpdateSizeInBytes, d_outputInstance,
                               instanceBufferSizes.outputSizeInBytes, &worldSpaceHandle,
                               nullptr, 0)); // Last 2 elements used for compacting

    // Set the launch parameters and copy them to device memory
    *h_optixLaunchParameters = OptixLaunchParameters();
    h_optixLaunchParameters->handle = worldSpaceHandle;
    CUDA_CALL(cudaMemcpyAsync(reinterpret_cast<void*>(d_optixLaunchParameters), h_optixLaunchParameters, sizeof(OptixLaunchParameters),cudaMemcpyHostToDevice, this->cuStream[0]));

    //    4.    Launch a device-side kernel that will invoke a ray generation program with a multitude of threads calling
    //          optixTrace to begin traversal and the execution of the other programs.
    OPTIX_CALL(optixLaunch(edgeIntersectionPipeline, this->cuStream[0], d_optixLaunchParameters, sizeof(OptixLaunchParameters), &sbt, numberOfEdges, 1, 1));
    CUDA_CALL(cudaMemcpyAsync(&h_hitGroupRecord->data.edgeIntersection, d_EdgeIntersectionPointer, sizeof(bool), cudaMemcpyDeviceToHost, this->cuStream[0]));
    CUDA_CALL(cudaStreamSynchronize(this->cuStream[0]));
    bool returnValue = !h_hitGroupRecord->data.edgeIntersection;

    // We reset the value on d_EdgeIntersectionPointer afterwards, this way it can execute asynchronously
    h_hitGroupRecord->data.edgeIntersection = false;
    CUDA_CALL(cudaMemcpyAsync(d_EdgeIntersectionPointer, &h_hitGroupRecord->data.edgeIntersection, sizeof(bool), cudaMemcpyHostToDevice, this->cuStream[0]));
    return returnValue;
}

bool OptixWorldSpaceMeshWithInstance::intersects(const OptixWorldSpaceMeshWithInstance &other) const {

    h_iterationData->edgeIntersectionSbtRecord.data.edgeIntersection= false;
    h_iterationData->modelTransformation = OptixStaticTransform{};
    Transformation otherToCurrentModelSpaceTransformation = glm::inverse(this->modelTransformation) * other.modelTransformation;
    Transformation otherToCurrentModelSpaceTransformationTransposed = glm::transpose(otherToCurrentModelSpaceTransformation);
    Transformation inverseOtherToCurrentModelSpaceTransformationTransposed = glm::inverse(otherToCurrentModelSpaceTransformationTransposed);
    const float* transform = glm::value_ptr(otherToCurrentModelSpaceTransformationTransposed);
    const float* inverseTransform = glm::value_ptr(inverseOtherToCurrentModelSpaceTransformationTransposed);
    memcpy(h_iterationData->modelTransformation.transform, transform, sizeof(float)*12);
    memcpy(h_iterationData->modelTransformation.invTransform, inverseTransform, sizeof(float)*12);
    h_iterationData->modelTransformation.child = other.modelSpaceHandle;

    CUDA_CALL(cudaMemcpyAsync(reinterpret_cast<void*>(d_iterationData), h_iterationData, sizeof(IterationData),cudaMemcpyHostToDevice, this->cuStream[0]));
    OPTIX_CALL(optixLaunch(edgeIntersectionPipeline, this->cuStream[0], d_structOptixLaunchParameters, sizeof(OptixLaunchParameters), &sbt, numberOfEdges, 1, 1));
    CUDA_CALL(cudaMemcpyAsync(&h_hitGroupRecord->data.edgeIntersection, d_EdgeIntersectionPointer, sizeof(bool), cudaMemcpyDeviceToHost, this->cuStream[0]));
    CUDA_CALL(cudaStreamSynchronize(this->cuStream[0]));
    return !h_hitGroupRecord->data.edgeIntersection;
}

void OptixWorldSpaceMeshWithInstance::setModelTransformation(const Transformation &transformation) {
    this->modelTransformation = transformation;
}