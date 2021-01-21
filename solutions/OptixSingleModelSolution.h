//
// Created by Jonas on 16/01/2021.
//

#ifndef MESHCORE_OPTIXSINGLEMODELSOLUTION_H
#define MESHCORE_OPTIXSINGLEMODELSOLUTION_H


#include <optix.h>

#include "AbstractSolution.h"
#include "../core/WorldSpaceMesh.h"
#include "NewOptixData.h"

struct StreamContext{
    OptixDeviceContext optixDeviceContext;
    CUstream cuStream;
};

class OptixSingleModelSolution: public AbstractSolution {
private:

    OptixTraversableHandle outerModelSpaceHandle; // Handle for the outer GAS
    OptixTraversableHandle innerModelSpaceHandle; // Handle for the inner GAS
    OptixTraversableHandle instanceWorldSpaceHandle; // Handle for the inner instance

    Transformation innerTransformation;

    CUdeviceptr d_outerGAS;
    OptixInstance* d_instances;
    CUdeviceptr d_innerGAS;

    CUdeviceptr d_outerTriangleIndices;
    CUdeviceptr d_outerModelSpaceVertices;
    float3* d_outerModelSpaceEdgeOrigins;
    float3* d_outerModelSpaceEdgeDirections;

    CUdeviceptr d_innerTriangleIndices;
    CUdeviceptr d_innerModelSpaceVertices;
    float3* d_innerModelSpaceEdgeOrigins;
    float3* d_innerModelSpaceEdgeDirections;

    unsigned int numberOfRays;

    float3* d_rayOrigins;
    float3* d_rayDirections;
    float* d_transformations; // contains 12 floats for each transform
    CUdeviceptr d_rayTransformIndices; // contains the index of transform/instance per ray


    OptixInstance outerInstance = {};
    OptixInstance innerInstance = {};

    OptixBuildInput instanceBuildInput = {};
    OptixAccelBufferSizes instanceBufferSizes;
    CUdeviceptr d_outputInstance;
    CUdeviceptr d_temp; // Used for building and updating the instance

    OptixPipeline edgeIntersectionPipeline;
    OptixShaderBindingTable sbt = {};

    CUdeviceptr d_raygenRecord;
    EdgeIntersectionSbtRecord hitGroupRecord = {};
    EdgeIntersectionSbtRecord* h_hitGroupRecord;
    CUdeviceptr d_missRecord;
    CUdeviceptr d_structEdgeIntersectionSbtRecord;
    CUdeviceptr d_optixLaunchParameters;

    OptixLaunchParameters* h_optixLaunchParameters;
    bool* d_EdgeIntersectionPointer;

public:
    OptixSingleModelSolution(const WorldSpaceMesh &outerWorldSpaceMesh, const WorldSpaceMesh &innerWorldSpaceMesh,
                             const StreamContext &streamContext);
    ~OptixSingleModelSolution();
    bool isFeasible(const StreamContext& streamContext);
    void setInnerTransformation(const Transformation& innerTransformation);
};


#endif //MESHCORE_OPTIXSINGLEMODELSOLUTION_H
