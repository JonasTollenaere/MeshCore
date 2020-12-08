//
// Created by Jonas on 9/11/2020.
//

#include <string>
#include "WorldSpaceMesh.h"

int WorldSpaceMesh::nextId = 0;

WorldSpaceMesh::WorldSpaceMesh(const ModelSpaceMesh &modelSpaceMesh):
        modelSpaceMesh(modelSpaceMesh),
        modelTransformationMatrix(1.0f),
        id(std::to_string(nextId++))
{

}

WorldSpaceMesh::WorldSpaceMesh(const ModelSpaceMesh &modelSpaceMesh, const glm::mat4 &modelTransformationMatrix):
        modelSpaceMesh(modelSpaceMesh),
        modelTransformationMatrix(modelTransformationMatrix),
        id(std::to_string(nextId++))
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

WorldSpaceMesh::~WorldSpaceMesh() {

}

const ModelSpaceMesh &WorldSpaceMesh::getModelSpaceMesh() const {
    return modelSpaceMesh;
}

const std::string &WorldSpaceMesh::getId() const {
    return id;
}
