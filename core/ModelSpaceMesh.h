//
// Created by Jonas on 9/11/2020.
//

#ifndef OPTIX_SAMPLES_MODELSPACEMESH_H
#define OPTIX_SAMPLES_MODELSPACEMESH_H


#include <vector>
#include "Vertex.h"
#include "Triangle.h"
#include "Ray.h"
#include "Edge.h"

class ModelSpaceMesh {
public:
    const std::vector<Vertex> vertices;
    const std::vector<Triangle> triangles;
public:
    ModelSpaceMesh();
    ModelSpaceMesh(const std::vector<Vertex> & vertices, const std::vector<Triangle> & triangles);
    ModelSpaceMesh(const ModelSpaceMesh& other);
    bool includes(const Vertex& modelSpaceVertex) const;
    ~ModelSpaceMesh();
    std::vector<Edge> getEdges() const;
private:
    unsigned int calculateNumberOfIntersections(Ray modelSpaceRay) const;

};

#endif //OPTIX_SAMPLES_MODELSPACEMESH_H
