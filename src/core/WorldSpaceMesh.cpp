//
// Created by Jonas on 27/01/2021.
//

#include "meshcore/core/WorldSpaceMesh.h"
#include "meshcore/core/Ray.h"
#include <string>

int WorldSpaceMesh::nextId = 0;

WorldSpaceMesh::WorldSpaceMesh():
id(std::to_string(nextId++)),
modelSpaceMesh(std::make_shared<ModelSpaceMesh>(ModelSpaceMesh()))
{}

WorldSpaceMesh::WorldSpaceMesh(const std::shared_ptr<ModelSpaceMesh>& modelSpaceMesh):
        modelSpaceMesh(modelSpaceMesh),
        modelTransformation(),
        id(std::to_string(nextId++))
{}

const std::string& WorldSpaceMesh::getId() const {
    return id;
}

WorldSpaceMesh::WorldSpaceMesh(ModelSpaceMesh &&modelSpaceMesh):
        WorldSpaceMesh(std::make_shared<ModelSpaceMesh>(modelSpaceMesh))
{}

std::shared_ptr<ModelSpaceMesh> WorldSpaceMesh::getModelSpaceMesh() const {
    return modelSpaceMesh;
}

std::shared_ptr<WorldSpaceMesh> WorldSpaceMesh::clone() const{
    return std::make_shared<WorldSpaceMesh>(*this);
}

std::shared_ptr<WorldSpaceMesh> WorldSpaceMesh::duplicate() const {
    std::shared_ptr<WorldSpaceMesh> returnWorldSpaceMesh = std::make_shared<WorldSpaceMesh>(*this);
    returnWorldSpaceMesh->id = std::to_string(nextId++);
    return returnWorldSpaceMesh;
}

void WorldSpaceMesh::setModelTransformation(const Transformation &transformation) {
    this->modelTransformation = transformation;
}

float WorldSpaceMesh::getVolume() const {
    const auto scale = this->modelTransformation.getScale();
    return modelSpaceMesh->getVolume() * (scale * scale * scale);
}

const Transformation& WorldSpaceMesh::getModelTransformation() const {
    return modelTransformation;
}

Transformation& WorldSpaceMesh::getModelTransformation() {
    return modelTransformation;
}

std::shared_ptr<ModelSpaceMesh> WorldSpaceMesh::getTransformedModelSpaceMesh() const {
    const auto& modelSpaceVertices = this->modelSpaceMesh->getVertices();
    std::vector<Vertex> transformedVertices;
    transformedVertices.reserve(modelSpaceVertices.size());
    for (auto modelSpaceVertex : modelSpaceVertices){
        transformedVertices.emplace_back(this->getModelTransformation().transformVertex(modelSpaceVertex));
    }
    auto result = std::make_shared<ModelSpaceMesh>(transformedVertices, this->modelSpaceMesh->getTriangles());
    result->setName(this->modelSpaceMesh->getName() + "-Transformed");
    return result;
}

AABB WorldSpaceMesh::computeWorldSpaceAABB() const {
    Vertex minimum(std::numeric_limits<float>::max());
    Vertex maximum(-std::numeric_limits<float>::max());
    for (const auto &vertex: this->modelSpaceMesh->getVertices()){
        auto transformedVertex = this->modelTransformation.transformVertex(vertex);
        minimum = glm::min(minimum, transformedVertex);
        maximum = glm::max(maximum, transformedVertex);
    }
    return {minimum, maximum};
}

glm::vec3 WorldSpaceMesh::computeSupport(const glm::vec3 &direction) const {

    // Transform the direction to modelSpace
    auto& transformation = this->getModelTransformation();
    auto modelSpaceDirection = transformation.getRotation().inverseRotateVertex(direction);

    // Determine support in model space
    auto modelSpaceSupport = this->modelSpaceMesh->computeSupport(modelSpaceDirection);

    // Transform the support back to world space
    return this->getModelTransformation().transformVertex(modelSpaceSupport);
}

glm::vec3 WorldSpaceMesh::getCenter() const {
    auto modelSpaceCenter = this->modelSpaceMesh->getCenter();
    return this->getModelTransformation().transformVertex(modelSpaceCenter);
}