//
// Created by Jonas on 9/11/2020.
//

#ifndef OPTIX_SAMPLES_WORLDSPACEMESH_H
#define OPTIX_SAMPLES_WORLDSPACEMESH_H


#include "ModelSpaceMesh.h"

class WorldSpaceMesh {
private:
    glm::mat4 modelTransformationMatrix;
//    std::vector<Vertex> transformedVertices; // TODO should these be cached?
public:
    const ModelSpaceMesh& modelSpaceMesh;
public:
//    WorldSpaceMesh();
//    WorldSpaceMesh(const WorldSpaceMesh& other);
//    WorldSpaceMesh(WorldSpaceMesh&& other) noexcept ;
//    WorldSpaceMesh& operator=(WorldSpaceMesh&& other) noexcept ;
//    ~WorldSpaceMesh();

    explicit WorldSpaceMesh(const ModelSpaceMesh & modelSpaceMesh);
    explicit WorldSpaceMesh(const ModelSpaceMesh & modelSpaceMesh, const glm::mat4& modelTransformationMatrix);

    void setModelTransformationMatrix(const glm::mat4& modelTransformationMatrix);
    glm::mat4 getModelTransformationMatrix() const;
    void transform(const glm::mat4& transformationMatrix);
};


#endif //OPTIX_SAMPLES_WORLDSPACEMESH_H
