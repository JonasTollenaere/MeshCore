//
// Created by Jonas on 17/12/2020.
//

#include <optix_stubs.h>
#include "OptixWorldSpaceMesh.h"
#include "OptixLaunchParameters.h"
#include "OptixData.h"
#include <glm/gtc/type_ptr.hpp>
#include <fstream>
#include <optix_function_table_definition.h>

static float3 vertexToFloat3(const Vertex& vertex){
    return make_float3(vertex.x, vertex.y, vertex.z);
}

template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};
typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<EdgeIntersectionTestData> EdgeIntersectionSbtRecord;

OptixWorldSpaceMesh::OptixWorldSpaceMesh(const WorldSpaceMesh& worldSpaceMesh, const OptixDeviceContext& optixDeviceContext, const CUstream& cuStream):
optixDeviceContext(&optixDeviceContext),
cuStream(&cuStream),
modelTransformation(worldSpaceMesh.getModelTransformation()),
d_triangleIndices(0),
d_modelSpaceVertices(0),
d_modelTransformation(0),
d_modelSpaceEdgeOrigins(0),
d_modelSpaceEdgeDirections(0)

{

    // Load the necessary modelspacedata on the device
    std::vector<float3> modelSpaceVertices;
    std::vector<uint3> triangleIndices;
    std::vector<float3> edgeOrigins;
    std::vector<float3> edgeDirections;

    for(Vertex vertex: worldSpaceMesh.getModelSpaceMesh().vertices){
        modelSpaceVertices.emplace_back(vertexToFloat3(vertex));
    }
    for(Triangle triangle: worldSpaceMesh.getModelSpaceMesh().triangles){
        triangleIndices.emplace_back(make_uint3(triangle.vertexIndex0, triangle.vertexIndex1, triangle.vertexIndex2));
        Vertex v0 = worldSpaceMesh.getModelSpaceMesh().vertices[triangle.vertexIndex0];
        Vertex v1 = worldSpaceMesh.getModelSpaceMesh().vertices[triangle.vertexIndex1];
        Vertex v2 = worldSpaceMesh.getModelSpaceMesh().vertices[triangle.vertexIndex2];
        edgeDirections.emplace_back(vertexToFloat3(v1-v0));
        edgeDirections.emplace_back(vertexToFloat3(v2-v1));
        edgeDirections.emplace_back(vertexToFloat3(v0-v2));
        edgeOrigins.emplace_back(vertexToFloat3(v0));
        edgeOrigins.emplace_back(vertexToFloat3(v1));
        edgeOrigins.emplace_back(vertexToFloat3(v2));
    }
    numberOfTriangles = triangleIndices.size();
    numberOfVertices = modelSpaceVertices.size();
    numberOfEdges = edgeOrigins.size();

    assert(edgeOrigins.size()==edgeDirections.size());

    cudaMallocAsync(reinterpret_cast<void **>(&d_modelSpaceVertices), sizeof(float3) * numberOfVertices, this->cuStream[0]);
    cudaMallocAsync(reinterpret_cast<void **>(&d_triangleIndices), sizeof(uint3) * numberOfTriangles, this->cuStream[0]);
    cudaMallocAsync(reinterpret_cast<void **>(&d_modelSpaceEdgeOrigins), sizeof(float3) * numberOfEdges, this->cuStream[0]);
    cudaMallocAsync(reinterpret_cast<void **>(&d_modelSpaceEdgeDirections), sizeof(float3) * numberOfEdges, this->cuStream[0]);
    cudaMallocAsync(reinterpret_cast<void **>(&d_modelTransformation), sizeof(OptixStaticTransform), this->cuStream[0]);

    cudaMemcpyAsync(reinterpret_cast<void *>(d_modelSpaceVertices), modelSpaceVertices.data(), sizeof(float3) * numberOfVertices, cudaMemcpyHostToDevice, this->cuStream[0]);
    cudaMemcpyAsync(reinterpret_cast<void *>(d_triangleIndices), triangleIndices.data(), sizeof(uint3) * numberOfTriangles, cudaMemcpyHostToDevice, this->cuStream[0]);
    cudaMemcpyAsync(reinterpret_cast<void *>(d_modelSpaceEdgeOrigins), edgeOrigins.data(), sizeof(float3) * numberOfEdges, cudaMemcpyHostToDevice, this->cuStream[0]);
    cudaMemcpyAsync(reinterpret_cast<void *>(d_modelSpaceEdgeDirections), edgeDirections.data(), sizeof(float3) * numberOfEdges, cudaMemcpyHostToDevice, this->cuStream[0]);

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accelOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;

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
    buildInput.triangleArray.indexStrideInBytes = 0;
    buildInput.triangleArray.numSbtRecords = 1;
    buildInput.triangleArray.preTransform = 0;
    unsigned int triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT }; // Setting this flag usuallyimproves performance even if no any-edgeIntersection program is present in the SBT
    buildInput.triangleArray.flags = triangle_input_flags;


    OptixAccelBufferSizes bufferSizes;
    optixAccelComputeMemoryUsage(this->optixDeviceContext[0], &accelOptions, &buildInput, 1, &bufferSizes);
    CUdeviceptr d_outputGAS;
    CUdeviceptr d_temp;
    cudaMalloc(reinterpret_cast<void**>(&d_outputGAS), bufferSizes.outputSizeInBytes);
    cudaMalloc(reinterpret_cast<void**>(&d_temp), bufferSizes.tempSizeInBytes);

    optixAccelBuild(this->optixDeviceContext[0], this->cuStream[0],       // This runs asyncrhonously on the GPU
                    &accelOptions, &buildInput, 1, d_temp,
                    bufferSizes.tempSizeInBytes, d_outputGAS,
                    bufferSizes.outputSizeInBytes, &modelSpaceHandle,
                    nullptr, 0); // Last 2 elements used when compacting

//    const float* transform = glm::value_ptr(glm::transpose(worldSpaceMesh.getModelTransformation()));
//    OptixStaticTransform modelTransformation;
//    memcpy(modelTransformation.transform, transform, sizeof(float)*12);
//    memcpy(modelTransformation.invTransform, transform, sizeof(float)*12);
//    modelTransformation.child = modelSpaceHandle;
//    cudaMemcpyAsync(reinterpret_cast<void *>(d_modelTransformation), &modelTransformation, sizeof(OptixStaticTransform), cudaMemcpyHostToDevice, this->cuStream[0]);
//    optixConvertPointerToTraversableHandle(this->optixDeviceContext[0], d_modelTransformation, OPTIX_TRAVERSABLE_TYPE_STATIC_TRANSFORM, &worldSpaceHandle);
}

OptixWorldSpaceMesh::~OptixWorldSpaceMesh() {
    cudaFreeAsync(reinterpret_cast<void *>(d_modelSpaceVertices), this->cuStream[0]);
    cudaFreeAsync(reinterpret_cast<void *>(d_triangleIndices), this->cuStream[0]);
    cudaFreeAsync(reinterpret_cast<void *>(d_modelSpaceEdgeOrigins), this->cuStream[0]);
    cudaFreeAsync(reinterpret_cast<void *>(d_modelSpaceEdgeDirections), this->cuStream[0]);
    cudaFreeAsync(reinterpret_cast<void *>(d_modelTransformation), this->cuStream[0]);
}

bool OptixWorldSpaceMesh::isFullyInside(const OptixWorldSpaceMesh &other) const {

//    2.    Create a edgeIntersectionPipeline of programs that contains all programs that will be invoked during a ray tracing launch.
    // TODO The pipelines should be created only once, somewhere else

    // Set the options for module compilation
    OptixModuleCompileOptions moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    // Set the options for edgeIntersectionPipeline compilation
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    pipelineCompileOptions.numPayloadValues = 0;
    pipelineCompileOptions.numAttributeValues = 0;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParameters";
    pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;  // Improves performance

    // Depending on the scenario and combination of flags, enabling exceptions can lead to severe overhead, so some flags shouldbe mainly used in internal and debug builds.
//        pipelineCompileOptions.exceptionFlags = (OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_USER);
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    char logString[4096];
    size_t logStringSize = sizeof(logString);

    // Compile the module based on the OptixProgram.ptx
    OptixModule ptxModule;
    std::ifstream t("../../optix/OptixPrograms/OptixPrograms.ptx");
    std::string ptxString((std::istreambuf_iterator<char>(t)),
                          std::istreambuf_iterator<char>());

    optixModuleCreateFromPTX(this->optixDeviceContext[0],
                                         &moduleCompileOptions,
                                         &pipelineCompileOptions,
                                         ptxString.c_str(),
                                         ptxString.size(),
                                         logString,
                                         &logStringSize,
                                         &ptxModule);

//    std::cout << logString << std::endl;

    // Use the modules to create the necessary programgroups (RAYGEN + HITGROUP + MISS)
    OptixProgramGroupDesc programGroupDescriptions[3] = {};
    programGroupDescriptions[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    programGroupDescriptions[0].raygen.module = ptxModule;
    programGroupDescriptions[0].raygen.entryFunctionName = "__raygen__edgeIntersectionTest__";
    programGroupDescriptions[1].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    programGroupDescriptions[1].hitgroup.moduleCH = ptxModule;
    programGroupDescriptions[1].hitgroup.entryFunctionNameCH = "__closesthit__edgeIntersectionTest__";
//      As a special case, the intersection program is not required – and is ignored – for triangle primitives.
    programGroupDescriptions[2].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    programGroupDescriptions[2].miss.module = nullptr;
    programGroupDescriptions[2].miss.entryFunctionName = nullptr;

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroup programGroups[3];
    optixProgramGroupCreate(this->optixDeviceContext[0],
                                        programGroupDescriptions,
                                        3,
                                        &pgOptions,
                                        logString, &logStringSize,
                                        programGroups);

//    std::cout << logString << std::endl;
    // Create a edgeIntersectionPipeline with these program groups
    OptixPipeline edgeIntersectionPipeline = nullptr;
    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth = 16;              // TODO check what this does exactly
    pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    optixPipelineCreate(this->optixDeviceContext[0],
                                    &pipelineCompileOptions,
                                    &pipelineLinkOptions,
                                    programGroups, 3,
                                    logString, &logStringSize,
                                    &edgeIntersectionPipeline);

    //    3.    Create a shader binding table that includes references to these programs and their parameters.

    OptixShaderBindingTable sbt = {};
    RayGenSbtRecord rayGenRecord;
    rayGenRecord.data = {};
    rayGenRecord.data.origins = reinterpret_cast<float3 *>(this->d_modelSpaceEdgeOrigins);
    rayGenRecord.data.directions = reinterpret_cast<float3 *>(this->d_modelSpaceEdgeDirections);
    CUdeviceptr d_raygenRecord;
    const size_t raygen_record_size = sizeof(RayGenSbtRecord);
    optixSbtRecordPackHeader(programGroups[0], &rayGenRecord);
    cudaMallocAsync(reinterpret_cast<void**>(&d_raygenRecord), raygen_record_size, this->cuStream[0]);
    cudaMemcpyAsync(reinterpret_cast<void *>(d_raygenRecord), &rayGenRecord, sizeof(RayGenSbtRecord), cudaMemcpyHostToDevice, this->cuStream[0]);
    sbt.raygenRecord = d_raygenRecord;

    EdgeIntersectionSbtRecord hitGroupRecord = {};
    hitGroupRecord.data.edgeIntersection = false;
    CUdeviceptr d_hitGroupRecord;
    optixSbtRecordPackHeader(programGroups[1], &hitGroupRecord);
    cudaMallocAsync(reinterpret_cast<void **>(&d_hitGroupRecord), sizeof(EdgeIntersectionSbtRecord), this->cuStream[0]);
    cudaMemcpyAsync(reinterpret_cast<void *>(d_hitGroupRecord), &hitGroupRecord, sizeof(EdgeIntersectionSbtRecord), cudaMemcpyHostToDevice, this->cuStream[0]);
    sbt.hitgroupRecordBase = d_hitGroupRecord;
    sbt.hitgroupRecordStrideInBytes = sizeof(EdgeIntersectionSbtRecord);
    sbt.hitgroupRecordCount = 1;

    CUdeviceptr d_missRecord;
    char missHeader[OPTIX_SBT_RECORD_HEADER_SIZE];
    optixSbtRecordPackHeader(programGroups[2], &missHeader);
    cudaMallocAsync(reinterpret_cast<void **>(&d_missRecord), OPTIX_SBT_RECORD_HEADER_SIZE, this->cuStream[0]);
    cudaMemcpyAsync(reinterpret_cast<void *>(d_missRecord), &missHeader, OPTIX_SBT_RECORD_HEADER_SIZE, cudaMemcpyHostToDevice, this->cuStream[0]);
    sbt.missRecordBase = d_missRecord;
    sbt.missRecordCount = 1;
    sbt.missRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;

    //    4.    Launch a device-side kernel that will invoke a ray generation program with a multitude of threads calling
    //          optixTrace to begin traversal and the execution of the other programs.

    // Set the launch parameters and copy them to device memory
    Transformation otherToCurrentWorldSpaceTransformation = glm::inverse(this->modelTransformation) * other.modelTransformation;

    Transformation otherToCurrentModelSpaceTransformationTransposed = glm::transpose(otherToCurrentWorldSpaceTransformation);
    Transformation inverseOtherToCurrentModelSpaceTransformationTransposed = glm::transpose(glm::inverse(otherToCurrentWorldSpaceTransformation));
    const float* transform = glm::value_ptr(otherToCurrentModelSpaceTransformationTransposed);
    const float* inverseTransform = glm::value_ptr(inverseOtherToCurrentModelSpaceTransformationTransposed);

    OptixStaticTransform optixOtherToCurrentModelTransformation;
    memcpy(optixOtherToCurrentModelTransformation.transform, transform, sizeof(float)*12);
    memcpy(optixOtherToCurrentModelTransformation.invTransform, inverseTransform, sizeof(float)*12);
    optixOtherToCurrentModelTransformation.child = other.modelSpaceHandle;
    cudaMemcpyAsync(reinterpret_cast<void *>(d_modelTransformation), &optixOtherToCurrentModelTransformation, sizeof(OptixStaticTransform), cudaMemcpyHostToDevice, this->cuStream[0]);

    OptixTraversableHandle currentModelSpaceHandle = {};
    optixConvertPointerToTraversableHandle(this->optixDeviceContext[0], d_modelTransformation, OPTIX_TRAVERSABLE_TYPE_STATIC_TRANSFORM, &currentModelSpaceHandle);

    OptixLaunchParameters optixLaunchParameters = {};
    optixLaunchParameters.handle = currentModelSpaceHandle;

    CUdeviceptr d_optixLaunchParameters;
    cudaMallocAsync(reinterpret_cast<void**>(&d_optixLaunchParameters), sizeof(optixLaunchParameters), this->cuStream[0]);
    cudaMemcpyAsync(reinterpret_cast<void*>(d_optixLaunchParameters), &optixLaunchParameters, sizeof(optixLaunchParameters),cudaMemcpyHostToDevice, this->cuStream[0]);

    bool* d_hitPointer = &(reinterpret_cast<EdgeIntersectionSbtRecord*>(d_hitGroupRecord))->data.edgeIntersection;

    optixLaunch(edgeIntersectionPipeline, this->cuStream[0], d_optixLaunchParameters, sizeof(OptixLaunchParameters), &sbt, numberOfEdges, 1, 1);

    cudaMemcpyAsync(&hitGroupRecord.data.edgeIntersection, d_hitPointer, sizeof(bool), cudaMemcpyDeviceToHost, this->cuStream[0]);
//    cudaMemcpyAsync(&hitGroupRecord, reinterpret_cast<void *>(d_hitGroupRecord),  sizeof(EdgeIntersectionSbtRecord), cudaMemcpyDeviceToHost, this->cuStream[0]);
    cudaStreamSynchronize(this->cuStream[0]);

    return !hitGroupRecord.data.edgeIntersection;
}

bool OptixWorldSpaceMesh::isFullyOutside(const OptixWorldSpaceMesh &other) const{
    return false;
}

void OptixWorldSpaceMesh::setModelTransformation(const Transformation &transformation) {
    this->modelTransformation = transformation;
}
