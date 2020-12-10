//
// Created by Jonas on 9/11/2020.
//

#include <string>
#include "WorldSpaceMesh.h"
#include "TriangleTriangleIntersect.h"
#include <glm/gtc/type_ptr.hpp>

int WorldSpaceMesh::nextId = 0;

WorldSpaceMesh::WorldSpaceMesh(const ModelSpaceMesh &modelSpaceMesh):
        modelSpaceMesh(modelSpaceMesh),
        modelTransformation(1.0f),
        id(std::to_string(nextId++))
{

}

WorldSpaceMesh::WorldSpaceMesh(const ModelSpaceMesh &modelSpaceMesh, const Transformation& transformation):
        modelSpaceMesh(modelSpaceMesh),
        modelTransformation(transformation),
        id(std::to_string(nextId++))
{

}

void WorldSpaceMesh::setModelTransformationMatrix(const Transformation &transformation) {
    this->modelTransformation = transformation;
}

Transformation WorldSpaceMesh::getModelTransformationMatrix() const {
    return modelTransformation;
}

void WorldSpaceMesh::transform(const Transformation& transformation) {
    this->modelTransformation *= transformation;
    // TODO check multiplication order
}

WorldSpaceMesh::~WorldSpaceMesh() = default;

const ModelSpaceMesh &WorldSpaceMesh::getModelSpaceMesh() const {
    return modelSpaceMesh;
}

const std::string &WorldSpaceMesh::getId() const {
    return id;
}

WorldSpaceMesh::WorldSpaceMesh():
        modelSpaceMesh(ModelSpaceMesh()),
        modelTransformation(1.0f),
        id(std::to_string(nextId++))
{}

WorldSpaceMesh::WorldSpaceMesh(const WorldSpaceMesh &other) = default;

bool WorldSpaceMesh::includes(const Vertex &worldSpaceVertex) const {
    Vertex modelSpaceVertex = Vertex(glm::inverse(this->modelTransformation) * glm::vec4(worldSpaceVertex, 1));
    return this->modelSpaceMesh.includes(modelSpaceVertex);
}

bool WorldSpaceMesh::triangleTriangleIntersects(const WorldSpaceMesh& other) const {
    for (Triangle thisTriangle: this->modelSpaceMesh.triangles) {

        Vertex innerVertex0 = Vertex(this->modelTransformation *
                                     glm::vec4(this->modelSpaceMesh.vertices[thisTriangle.vertexIndex0], 1));
        Vertex innerVertex1 = Vertex(this->modelTransformation *
                                     glm::vec4(this->modelSpaceMesh.vertices[thisTriangle.vertexIndex1], 1));
        Vertex innerVertex2 = Vertex(this->modelTransformation *
                                     glm::vec4(this->modelSpaceMesh.vertices[thisTriangle.vertexIndex2], 1));

        for (Triangle otherTriangle: other.modelSpaceMesh.triangles) {
            Vertex roughVertex0 = Vertex(other.modelTransformation *
                                         glm::vec4(other.modelSpaceMesh.vertices[otherTriangle.vertexIndex0], 1));
            Vertex roughVertex1 = Vertex(other.modelTransformation *
                                         glm::vec4(other.modelSpaceMesh.vertices[otherTriangle.vertexIndex1], 1));
            Vertex roughVertex2 = Vertex(other.modelTransformation *
                                         glm::vec4(other.modelSpaceMesh.vertices[otherTriangle.vertexIndex2], 1));

            int test = Intersection::NoDivTriTriIsect(glm::value_ptr(innerVertex0), glm::value_ptr(innerVertex1),
                                                      glm::value_ptr(innerVertex2),
                                                      glm::value_ptr(roughVertex0), glm::value_ptr(roughVertex1),
                                                      glm::value_ptr(roughVertex2));

            if (test == 1) {
                return true;
            }
        }
    }
    return false;
}

bool WorldSpaceMesh::rayTriangleInside(const WorldSpaceMesh &other) const {
    for(Vertex vertex: this->modelSpaceMesh.vertices){
        Vertex worldSpaceVertex = glm::vec3(this->modelTransformation * glm::vec4(vertex, 1.0f));
        if(!other.includes(worldSpaceVertex)){ //186490 ms. without inverse caching // 186666 ms. with inverse transformation cached
            return false;
        }
    }
    return true;
}

