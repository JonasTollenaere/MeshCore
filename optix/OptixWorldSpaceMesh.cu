//
// Created by Jonas on 17/12/2020.
//

#include "OptixWorldSpaceMesh.h"
#include <optix_stubs.h>
#include <glm/gtc/type_ptr.hpp>

static float3 vertexToFloat3(const Vertex& vertex){
    return make_float3(vertex.x, vertex.y, vertex.z);
}

OptixWorldSpaceMesh::OptixWorldSpaceMesh(const WorldSpaceMesh& worldSpaceMesh, const OptixDeviceContext& optixDeviceContext, const CUstream& cuStream):
optixDeviceContext(&optixDeviceContext), cuStream(&cuStream){

    // Load the necessary data on the device
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
    cudaMallocAsync(reinterpret_cast<void **>(&d_modelSpaceVertices), sizeof(float3) * modelSpaceVertices.size(), this->cuStream[0]);
    cudaMallocAsync(reinterpret_cast<void **>(&d_triangleIndices), sizeof(uint3) * triangleIndices.size(), this->cuStream[0]);
    cudaMallocAsync(reinterpret_cast<void **>(&d_edgeOrigins), sizeof(float3) * edgeOrigins.size(), this->cuStream[0]);
    cudaMallocAsync(reinterpret_cast<void **>(&d_edgeDirections), sizeof(float3) * edgeDirections.size(), this->cuStream[0]);
    cudaMallocAsync(reinterpret_cast<void **>(&d_modelTransformation), sizeof(OptixStaticTransform), this->cuStream[0]);

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
    unsigned int triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT }; // Setting this flag usuallyimproves performance even if no any-hit program is present in the SBT
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

    const float* transform = glm::value_ptr(glm::transpose(worldSpaceMesh.getModelTransformation()));
    OptixStaticTransform modelTransformation;
    memcpy(modelTransformation.transform, transform, sizeof(float)*12);
    memcpy(modelTransformation.invTransform, transform, sizeof(float)*12);
    modelTransformation.child = modelSpaceHandle;
    cudaMemcpyAsync(reinterpret_cast<void *>(d_modelTransformation), &modelTransformation, sizeof(OptixStaticTransform), cudaMemcpyHostToDevice, this->cuStream[0]);
    optixConvertPointerToTraversableHandle(this->optixDeviceContext[0], d_modelTransformation, OPTIX_TRAVERSABLE_TYPE_STATIC_TRANSFORM, &worldSpaceHandle);
}

OptixWorldSpaceMesh::~OptixWorldSpaceMesh() {
    cudaFreeAsync(reinterpret_cast<void *>(d_modelSpaceVertices), this->cuStream[0]);
    cudaFreeAsync(reinterpret_cast<void *>(d_triangleIndices), this->cuStream[0]);
    cudaFreeAsync(reinterpret_cast<void *>(d_edgeOrigins), this->cuStream[0]);
    cudaFreeAsync(reinterpret_cast<void *>(d_edgeDirections), this->cuStream[0]);
    cudaFreeAsync(reinterpret_cast<void *>(d_modelTransformation), this->cuStream[0]);
}

bool OptixWorldSpaceMesh::isInside(const OptixWorldSpaceMesh &other) const {
    return false;
}

bool OptixWorldSpaceMesh::isOutside(const OptixWorldSpaceMesh &other) const{
    return false;
}

void OptixWorldSpaceMesh::setModelTransformation(const Transformation &transformation) {
    const float* transform = glm::value_ptr(glm::transpose(transformation));
    const float* invTransform = glm::value_ptr(glm::transpose(glm::inverse(transformation)));
    OptixStaticTransform modelTransformation;
    memcpy(modelTransformation.transform, transform, sizeof(float)*12);
    memcpy(modelTransformation.invTransform, invTransform, sizeof(float)*12);
    modelTransformation.child = modelSpaceHandle;
    cudaMemcpyAsync(reinterpret_cast<void *>(d_modelTransformation), &modelTransformation, sizeof(OptixStaticTransform), cudaMemcpyHostToDevice, this->cuStream[0]);
    optixConvertPointerToTraversableHandle(this->optixDeviceContext[0], d_modelTransformation, OPTIX_TRAVERSABLE_TYPE_STATIC_TRANSFORM, &worldSpaceHandle);
}
