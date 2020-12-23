//
// Created by Jonas on 17/12/2020.
//

#ifndef MESHCORE_OPTIXWORLDSPACEMESH_H
#define MESHCORE_OPTIXWORLDSPACEMESH_H

#include <optix.h>
#include "../core/WorldSpaceMesh.h"
#include "OptixPipeLineContext.h"
#include "OptixData.h"

class OptixWorldSpaceMesh {

private:

    CUdeviceptr d_modelSpaceVertices{};
    CUdeviceptr d_triangleIndices{};
    CUdeviceptr d_modelSpaceEdgeOrigins{};
    CUdeviceptr d_modelSpaceEdgeDirections{};
    CUdeviceptr d_modelTransformation{};
    CUdeviceptr d_optixLaunchParameters{};

    // SBT
    CUdeviceptr d_missRecord;
    CUdeviceptr d_hitGroupRecord;
    CUdeviceptr d_raygenRecord;

    CUdeviceptr d_outputGAS;

    // Host pinned memory
    EdgeIntersectionSbtRecord* h_hitGroupRecord;
    OptixStaticTransform* h_modelTransformation;
    OptixLaunchParameters* h_optixLaunchParameters;

    IterationData* h_iterationData;
    IterationData* d_iterationData;
    CUdeviceptr d_structModelTransformation;
    CUdeviceptr d_structOptixLaunchParameters;
    CUdeviceptr d_structEdgeIntersectionSbtRecord;
    EdgeIntersectionSbtRecord hitGroupRecord = {};

    bool* d_EdgeIntersectionPointer;

    unsigned int numberOfVertices;
    unsigned int numberOfEdges;
    unsigned int numberOfTriangles;

    const OptixDeviceContext optixDeviceContext;
    const CUstream* cuStream;

    OptixPipeline edgeIntersectionPipeline;
    OptixShaderBindingTable sbt{};
    OptixTraversableHandle modelSpaceHandle{};
    Transformation modelTransformation;

public:
    OptixWorldSpaceMesh() = delete;
    OptixWorldSpaceMesh(const OptixWorldSpaceMesh& other) = delete;
    explicit OptixWorldSpaceMesh(const WorldSpaceMesh& worldSpaceMesh, const CUstream& cUStream, const OptixDeviceContext& optixDeviceContext);
    ~OptixWorldSpaceMesh();

    void setModelTransformation(const Transformation& transformation);

//    bool isFullyInside(const OptixWorldSpaceMesh &other) const;
    bool isFullyOutside(const OptixWorldSpaceMesh& other) const;

    bool isFullyInsideWithStruct(const OptixWorldSpaceMesh &other) const;
};


#endif //MESHCORE_OPTIXWORLDSPACEMESH_H
