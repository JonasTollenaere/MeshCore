//
// Created by Jonas on 17/12/2020.
//

#ifndef MESHCORE_OPTIXWORLDSPACEMESH_H
#define MESHCORE_OPTIXWORLDSPACEMESH_H

#include <optix.h>
#include "../core/WorldSpaceMesh.h"

class OptixWorldSpaceMesh {

private:

    CUdeviceptr d_modelSpaceVertices{};
    CUdeviceptr d_triangleIndices{};
    CUdeviceptr d_modelSpaceEdgeOrigins{};
    CUdeviceptr d_modelSpaceEdgeDirections{};
    CUdeviceptr d_modelTransformation{};

    unsigned int numberOfVertices;
    unsigned int numberOfEdges;
    unsigned int numberOfTriangles;

    const OptixDeviceContext* optixDeviceContext;
    const CUstream* cuStream;

    OptixTraversableHandle modelSpaceHandle{};

    Transformation modelTransformation{};

public:
    OptixWorldSpaceMesh() = delete;
    OptixWorldSpaceMesh(const OptixWorldSpaceMesh& other) = delete;
    explicit OptixWorldSpaceMesh(const WorldSpaceMesh& worldSpaceMesh, const OptixDeviceContext& optixDeviceContext, const CUstream& cuStream);
    ~OptixWorldSpaceMesh();

    void setModelTransformation(const Transformation& transformation);

    bool isFullyInside(const OptixWorldSpaceMesh &other) const;
    bool isFullyOutside(const OptixWorldSpaceMesh& other) const;
};


#endif //MESHCORE_OPTIXWORLDSPACEMESH_H
