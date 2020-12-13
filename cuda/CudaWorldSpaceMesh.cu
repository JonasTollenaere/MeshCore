#include <glm/gtc/type_ptr.hpp>
#include "CudaWorldSpaceMesh.h"
#include "CudaRay.h"
#include "../cuda/CudaTriangleTriangleIntersect.h"

CudaWorldSpaceMesh::~CudaWorldSpaceMesh() {
    cudaFree(this->transformation);
    cudaFree(this->modelSpaceVertices);
    cudaFree(this->triangles);
    cudaFree(this->worldSpaceVertices);
}

CudaWorldSpaceMesh::CudaWorldSpaceMesh(const WorldSpaceMesh &worldSpaceMesh):
    modelSpaceVertices(nullptr), worldSpaceVertices(nullptr), triangles(nullptr), transformation(nullptr)
{
    // Allocate and the transformation
    Transformation transform(worldSpaceMesh.getModelTransformation());
    cudaMalloc(&this->transformation, sizeof(Transformation));

    // Allocate and copy the modelSpaceVertices
    this->numberOfVertices = worldSpaceMesh.getModelSpaceMesh().vertices.size();
    unsigned int verticesBytes = this->numberOfVertices * sizeof(Vertex);
    cudaMalloc(&this->modelSpaceVertices, verticesBytes);
    cudaMemcpy(this->modelSpaceVertices, worldSpaceMesh.getModelSpaceMesh().vertices.data(), verticesBytes, cudaMemcpyHostToDevice);

    // Allocate and update the transformation, worldSpaceVertices
    cudaMalloc(&this->worldSpaceVertices, verticesBytes);
    this->setModelTransformationMatrix(worldSpaceMesh.getModelTransformation());

    // Allocate and copy the triangles
    this->numberOfTriangles = worldSpaceMesh.getModelSpaceMesh().triangles.size();
    unsigned int triangleBytes = this->numberOfTriangles * sizeof(Triangle);
    cudaMalloc(&this->triangles, triangleBytes);
    cudaMemcpy(this->triangles, worldSpaceMesh.getModelSpaceMesh().triangles.data(), triangleBytes, cudaMemcpyHostToDevice);
}

__global__
void calculateWorldSpaceVertices(const Transformation* transformation, const Vertex* modelSpaceVertices, Vertex* worldSpaceVertices, const unsigned numberOfVertices)
{
    unsigned int vertexId = threadIdx.x + blockIdx.x * blockDim.x;
    if(vertexId < numberOfVertices){
        worldSpaceVertices[vertexId] = Vertex(*transformation * glm::vec4(modelSpaceVertices[vertexId], 1));
    }
}

void CudaWorldSpaceMesh::setModelTransformationMatrix(const Transformation &newTransformation) {
    cudaMemcpy(this->transformation, &newTransformation, sizeof(Transformation), cudaMemcpyHostToDevice);
    unsigned blocksize = 512;
    unsigned int nBlocks= this->numberOfVertices/blocksize+(this->numberOfVertices%blocksize == 0?0:1);
    calculateWorldSpaceVertices <<< nBlocks, blocksize >>> (this->transformation, this->modelSpaceVertices, this->worldSpaceVertices, this->numberOfVertices);
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
        }
    }

}

bool CudaWorldSpaceMesh::rayTriangleInside(const CudaWorldSpaceMesh &other) const {

    unsigned blockSize = 512; // TODO
    unsigned int nBlocks= this->numberOfVertices / blockSize + (this->numberOfVertices % blockSize == 0 ? 0 : 1);

    bool inside = true;
    bool* d_inside;

    cudaMalloc(&d_inside, sizeof(bool));
    cudaMemcpy(d_inside, &inside, sizeof(bool), cudaMemcpyHostToDevice);

    rayTriangleInsideKernel <<< nBlocks, blockSize >>>
    (this->triangles, this->worldSpaceVertices, this->numberOfVertices, this->numberOfTriangles,
     other.triangles, other.worldSpaceVertices, other.numberOfVertices, other.numberOfTriangles, d_inside);

    cudaMemcpy(&inside, d_inside, sizeof(bool), cudaMemcpyDeviceToHost);

    return inside;
}

__global__
void triangleTriangleIntersectKernel(const Triangle* innerTriangles, const Vertex* innerVertices, const unsigned numberOfInnerVertices, const unsigned int numberOfInnerTriangles,
                                     const Triangle* outerTriangles, const Vertex* outerVertices, const unsigned int numberOfOuterVertices, const unsigned int numberOfOuterTriangles,
                                     bool* intersects)
{
    unsigned int threadId = threadIdx.x + blockIdx.x * blockDim.x;

    unsigned int innerTriangleId = threadId;

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

    unsigned blockSize = 512; // TODO
    unsigned int nBlocks= this->numberOfTriangles / blockSize + (this->numberOfTriangles % blockSize == 0 ? 0 : 1);

    bool intersects = false;
    bool* d_intersects;

    cudaMalloc(&d_intersects, sizeof(bool));
    cudaMemcpy(d_intersects, &intersects, sizeof(bool), cudaMemcpyHostToDevice);

    triangleTriangleIntersectKernel <<< nBlocks, blockSize >>>
            (this->triangles, this->worldSpaceVertices, this->numberOfVertices, this->numberOfTriangles,
             other.triangles, other.worldSpaceVertices, other.numberOfVertices, other.numberOfTriangles, d_intersects);

    cudaMemcpy(&intersects, d_intersects, sizeof(bool), cudaMemcpyDeviceToHost);

    return intersects;
}
