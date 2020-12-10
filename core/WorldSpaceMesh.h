//
// Created by Jonas on 9/11/2020.
//

#ifndef OPTIX_SAMPLES_WORLDSPACEMESH_H
#define OPTIX_SAMPLES_WORLDSPACEMESH_H


#include "ModelSpaceMesh.h"
#include "Transformation.h"

class WorldSpaceMesh {
private:
    static int nextId;
    std::string id;
    Transformation modelTransformation;
    const ModelSpaceMesh modelSpaceMesh;
//    std::vector<Vertex> transformedVertices; // TODO should these be cached?

public:
    WorldSpaceMesh();
    WorldSpaceMesh(const WorldSpaceMesh& other);
    ~WorldSpaceMesh();

    explicit WorldSpaceMesh(const ModelSpaceMesh & modelSpaceMesh);
    explicit WorldSpaceMesh(const ModelSpaceMesh & modelSpaceMesh, const Transformation& modelTransformationMatrix);

    void setModelTransformationMatrix(const Transformation& transformation);
    Transformation getModelTransformationMatrix() const;
    void transform(const Transformation& transformation);
    bool includes(const Vertex& worldSpaceVertex) const;

    const ModelSpaceMesh &getModelSpaceMesh() const;

    const std::string &getId() const;

    bool triangleTriangleIntersects(const WorldSpaceMesh &other) const;
    bool rayTriangleInside(const WorldSpaceMesh &other) const;
};


#endif //OPTIX_SAMPLES_WORLDSPACEMESH_H