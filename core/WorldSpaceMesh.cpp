//
// Created by Jonas on 9/11/2020.
//

#include "WorldSpaceMesh.h"

WorldSpaceMesh::WorldSpaceMesh(const ModelSpaceMesh &modelSpaceMesh):
        modelSpaceMesh(modelSpaceMesh),
        modelTransformationMatrix(1.0f)
{

}

WorldSpaceMesh::WorldSpaceMesh(const ModelSpaceMesh &modelSpaceMesh, const glm::mat4 &modelTransformationMatrix):
        modelSpaceMesh(modelSpaceMesh),
        modelTransformationMatrix(modelTransformationMatrix)
{

}

void WorldSpaceMesh::setModelTransformationMatrix(const glm::mat4 &newModelTransformationMatrix) {
    this->modelTransformationMatrix = newModelTransformationMatrix;
}

glm::mat4 WorldSpaceMesh::getModelTransformationMatrix() const {
    return modelTransformationMatrix;
}

void WorldSpaceMesh::transform(const glm::mat4 &transformationMatrix) {
    this->modelTransformationMatrix *= transformationMatrix;
}