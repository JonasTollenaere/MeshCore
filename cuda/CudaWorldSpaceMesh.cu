#include <glm/gtc/type_ptr.hpp>
#include "CudaWorldSpaceMesh.h"
#include "CudaRay.h"
#include "../cuda/CudaTriangleTriangleIntersect.h"

#define BLOCK_SIZE 32

CudaWorldSpaceMesh::CudaWorldSpaceMesh(const WorldSpaceMesh &worldSpaceMesh, const CudaStream& cudaStream):
    modelSpaceVertices(nullptr),
    worldSpaceVertices(nullptr),
    triangles(nullptr),
    transformation(nullptr),
    flags(nullptr),
    cudaStream(&cudaStream.getStream())
{
    // Allocate and the transformation
    Transformation transform(worldSpaceMesh.getModelTransformation());
    cudaMalloc(&this->transformation, sizeof(Transformation));

    // Allocate and copy the modelSpaceVertices
    this->numberOfVertices = worldSpaceMesh.getModelSpaceMesh().getVertices().size();
    unsigned int verticesBytes = this->numberOfVertices * sizeof(Vertex);
    cudaMalloc(&this->modelSpaceVertices, verticesBytes);
    cudaMemcpyAsync(this->modelSpaceVertices, worldSpaceMesh.getModelSpaceMesh().getVertices().data(), verticesBytes, cudaMemcpyHostToDevice, *this->cudaStream);

    // Allocate and update the transformation, worldSpaceVertices
    cudaMalloc(&this->worldSpaceVertices, verticesBytes);
    this->setModelTransformation(worldSpaceMesh.getModelTransformation());

    // Allocate and copy the triangles
    this->numberOfTriangles = worldSpaceMesh.getModelSpaceMesh().getTriangles().size();
    unsigned int triangleBytes = this->numberOfTriangles * sizeof(Triangle);
    cudaMalloc(&this->triangles, triangleBytes);
    cudaMemcpyAsync(this->triangles, worldSpaceMesh.getModelSpaceMesh().getTriangles().data(), triangleBytes, cudaMemcpyHostToDevice, *this->cudaStream);

    // Allocate flags
    cudaMalloc(&this->flags, 2 * sizeof(bool));

}

CudaWorldSpaceMesh::~CudaWorldSpaceMesh() {
    cudaFree(this->transformation);
    cudaFree(this->modelSpaceVertices);
    cudaFree(this->triangles);
    cudaFree(this->worldSpaceVertices);
    cudaFree(this->flags);
}

__global__
void calculateWorldSpaceVertices(const Transformation* transformation, const Vertex* modelSpaceVertices, Vertex* worldSpaceVertices, const unsigned numberOfVertices)
{
    unsigned int vertexId = threadIdx.x + blockIdx.x * blockDim.x;
    if(vertexId < numberOfVertices){
        worldSpaceVertices[vertexId] = Vertex(*transformation * glm::vec4(modelSpaceVertices[vertexId], 1));
    }
}

void CudaWorldSpaceMesh::setModelTransformation(const Transformation &transformation) {
    cudaMemcpyAsync(this->transformation, &transformation, sizeof(Transformation), cudaMemcpyHostToDevice, *this->cudaStream);
    unsigned blockSize = BLOCK_SIZE;
    unsigned int nBlocks= this->numberOfVertices / blockSize + (this->numberOfVertices % blockSize == 0 ? 0 : 1);
    calculateWorldSpaceVertices <<< nBlocks, blockSize, 0, *this->cudaStream >>> (this->transformation, this->modelSpaceVertices, this->worldSpaceVertices, this->numberOfVertices);
}


__global__
void rayTriangleInsideKernel(const Triangle* innerTriangles, const Vertex* innerVertices, const unsigned numberOfInnerVertices, const unsigned int numberOfInnerTriangles,
                             const Triangle* outerTriangles, const Vertex* outerVertices, const unsigned int numberOfOuterVertices, const unsigned int numberOfOuterTriangles,
                             bool* inside)
{
    unsigned int innerVertexId = threadIdx.x + blockIdx.x * blockDim.x;

    if(innerVertexId < numberOfInnerVertices){

        unsigned int numberOfIntersections = 0;

        CudaRay ray(innerVertices[innerVertexId], glm::vec3(1, 0, 0));

        for(int outerTriangleIndex=0; outerTriangleIndex<numberOfOuterTriangles; outerTriangleIndex++){
            Triangle outerTriangle = outerTriangles[outerTriangleIndex];
            Vertex v0 = outerVertices[outerTriangle.vertexIndex0];
            Vertex v1 = outerVertices[outerTriangle.vertexIndex1];
            Vertex v2 = outerVertices[outerTriangle.vertexIndex2];
            if(ray.intersectsTriangle(v0,v1,v2)){
                numberOfIntersections++;
            }
        }
        if(numberOfIntersections % 2 == 0){
            *inside = false;
            return;
        }
    }

}

bool CudaWorldSpaceMesh::rayTriangleInside(const CudaWorldSpaceMesh &other) const {

    unsigned blockSize = BLOCK_SIZE;
    unsigned int nBlocks= this->numberOfVertices / blockSize + (this->numberOfVertices % blockSize == 0 ? 0 : 1);

    bool inside = true;
    cudaMemcpyAsync(&flags[0], &inside, sizeof(bool), cudaMemcpyHostToDevice, *this->cudaStream);

    rayTriangleInsideKernel <<< nBlocks, blockSize, 0, *this->cudaStream >>>
    (this->triangles, this->worldSpaceVertices, this->numberOfVertices, this->numberOfTriangles,
     other.triangles, other.worldSpaceVertices, other.numberOfVertices, other.numberOfTriangles, &flags[0]);

    cudaMemcpyAsync(&inside, &flags[0], sizeof(bool), cudaMemcpyDeviceToHost, *this->cudaStream);

    cudaStreamSynchronize(*this->cudaStream);

    return inside;
}

__global__
void triangleTriangleIntersectKernel(const Triangle* innerTriangles, const Vertex* innerVertices, const unsigned numberOfInnerVertices, const unsigned int numberOfInnerTriangles,
                                     const Triangle* outerTriangles, const Vertex* outerVertices, const unsigned int numberOfOuterVertices, const unsigned int numberOfOuterTriangles,
                                     bool* intersects)
{
    unsigned int threadId = threadIdx.x + blockIdx.x * blockDim.x;

    unsigned int innerTriangleId = threadId;

//    unsigned int innerTriangleId = threadId/numberOfOuterTriangles;
//    unsigned int outerTriangleId = threadId%numberOfOuterTriangles;

    if(innerTriangleId < numberOfInnerTriangles){
        Triangle innerTriangle = innerTriangles[innerTriangleId];
        Vertex innerVertex0 = innerVertices[innerTriangle.vertexIndex0];  // It's worth it transforming all vertices in advance
        Vertex innerVertex1 = innerVertices[innerTriangle.vertexIndex1];
        Vertex innerVertex2 = innerVertices[innerTriangle.vertexIndex2];

        for(int outerTriangleIndex=0; outerTriangleIndex<numberOfOuterTriangles; outerTriangleIndex++){
            Triangle outerTriangle = outerTriangles[outerTriangleIndex];
            Vertex outerVertex0 = outerVertices[outerTriangle.vertexIndex0];
            Vertex outerVertex1 = outerVertices[outerTriangle.vertexIndex1];
            Vertex outerVertex2 = outerVertices[outerTriangle.vertexIndex2];
            bool test = CudaIntersection::NoDivTriTriIsect(glm::value_ptr(innerVertex0),
                                                           glm::value_ptr(innerVertex1),
                                                           glm::value_ptr(innerVertex2),
                                                           glm::value_ptr(outerVertex0),
                                                           glm::value_ptr(outerVertex1),
                                                           glm::value_ptr(outerVertex2));
            if (test) {
                *intersects = true;
            }
        }
    }
}

bool CudaWorldSpaceMesh::triangleTriangleIntersects(const CudaWorldSpaceMesh &other) const {

    unsigned blockSize = BLOCK_SIZE;
//    unsigned int nThreads = this->numberOfTriangles * other.numberOfTriangles;
    unsigned int nThreads = this->numberOfTriangles;
    unsigned int nBlocks= nThreads / blockSize + (nThreads % blockSize == 0 ? 0 : 1);

    bool intersects = false;

    cudaMemcpyAsync(&flags[1], &intersects, sizeof(bool), cudaMemcpyHostToDevice, *this->cudaStream);

    triangleTriangleIntersectKernel <<< nBlocks, blockSize, 0, *this->cudaStream >>>
            (this->triangles, this->worldSpaceVertices, this->numberOfVertices, this->numberOfTriangles,
             other.triangles, other.worldSpaceVertices, other.numberOfVertices, other.numberOfTriangles, &flags[1]);

    cudaMemcpyAsync(&intersects, &flags[1], sizeof(bool), cudaMemcpyDeviceToHost, *this->cudaStream);

    cudaStreamSynchronize(*this->cudaStream);

    return intersects;
}