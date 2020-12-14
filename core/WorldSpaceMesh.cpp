//
// Created by Jonas on 9/11/2020.
//

#include <string>
#include "WorldSpaceMesh.h"
#include "TriangleTriangleIntersect.h"
#include "TriangleTriangleIntersectModified.h"
#include <glm/gtc/type_ptr.hpp>

int WorldSpaceMesh::nextId = 0;

WorldSpaceMesh::WorldSpaceMesh(const ModelSpaceMesh &modelSpaceMesh):
        modelSpaceMesh(modelSpaceMesh),
        modelTransformation(1.0f),
        id(std::to_string(nextId++)),
        worldSpaceVertices(modelSpaceMesh.vertices)
{

}

WorldSpaceMesh::WorldSpaceMesh(const ModelSpaceMesh &modelSpaceMesh, const Transformation& transformation):
        modelSpaceMesh(modelSpaceMesh),
        modelTransformation(transformation),
        id(std::to_string(nextId++)),
        worldSpaceVertices()
{
    for(Vertex vertex: modelSpaceMesh.vertices){
        worldSpaceVertices.emplace_back(Vertex(transformation * glm::vec4(vertex, 1.0f)));
    }
}

void WorldSpaceMesh::setModelTransformationMatrix(const Transformation &transformation) {
    this->modelTransformation = transformation;
    for(int i=0; i < worldSpaceVertices.size(); i++){
        worldSpaceVertices[i] = Vertex(this->modelTransformation * glm::vec4(this->modelSpaceMesh.vertices[i], 1.0f));
    }
}

Transformation WorldSpaceMesh::getModelTransformation() const {
    return modelTransformation;
}

void WorldSpaceMesh::transform(const Transformation& transformation) {
    this->modelTransformation *= transformation;
    for(int i=0; i < worldSpaceVertices.size(); i++){
        worldSpaceVertices[i] = Vertex(this->modelTransformation * glm::vec4(this->modelSpaceMesh.vertices[i], 1.0f));
    }
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
    Ray ray(worldSpaceVertex, glm::vec3(1, 0, 0));
    if(this->calculateNumberOfIntersections(ray)%2 == 0){
        return false;
    }
    return true;

//     Alternative to modelspace if worldspaceVertices are not cached
//    Vertex modelSpaceVertex = Vertex(glm::inverse(this->modelTransformation) * glm::vec4(worldSpaceVertex, 1));
//    return this->modelSpaceMesh.includes(modelSpaceVertex);

}

unsigned int WorldSpaceMesh::calculateNumberOfIntersections(Ray modelSpaceRay) const {
    unsigned int numberOfIntersections = 0;
    for (Triangle triangle: this->modelSpaceMesh.triangles) {
        bool intersects = modelSpaceRay.intersectsTriangle(this->worldSpaceVertices[triangle.vertexIndex0],
                                                           this->worldSpaceVertices[triangle.vertexIndex1],
                                                           this->worldSpaceVertices[triangle.vertexIndex2]);
        if (intersects) {
            numberOfIntersections++;
        }
    }
    return numberOfIntersections;
}

bool WorldSpaceMesh::triangleTriangleIntersects(const WorldSpaceMesh& other) const {
    for (Triangle thisTriangle: this->modelSpaceMesh.triangles) {

        Vertex innerVertex0 = this->worldSpaceVertices[thisTriangle.vertexIndex0];
        Vertex innerVertex1 = this->worldSpaceVertices[thisTriangle.vertexIndex1];
        Vertex innerVertex2 = this->worldSpaceVertices[thisTriangle.vertexIndex2];

        for (Triangle otherTriangle: other.modelSpaceMesh.triangles) {
            Vertex roughVertex0 = other.worldSpaceVertices[otherTriangle.vertexIndex0];
            Vertex roughVertex1 = other.worldSpaceVertices[otherTriangle.vertexIndex1];
            Vertex roughVertex2 = other.worldSpaceVertices[otherTriangle.vertexIndex2];
//            bool test = Intersection::NoDivTriTriIsect(glm::value_ptr(innerVertex0), glm::value_ptr(innerVertex1),
//                                                      glm::value_ptr(innerVertex2),
//                                                      glm::value_ptr(roughVertex0), glm::value_ptr(roughVertex1),
//                                                      glm::value_ptr(roughVertex2));
//            bool test2 = MIntersection::NoDivTriTriIsect(innerVertex0, innerVertex1, innerVertex2,
//                                                        roughVertex0, roughVertex1, roughVertex2);
            bool test = MIntersection::ExperimentalTriangleTriangleIntersection(
                    innerVertex0, innerVertex1, innerVertex2,roughVertex0, roughVertex1, roughVertex2);

            if(test != Intersection::NoDivTriTriIsect(glm::value_ptr(innerVertex0), glm::value_ptr(innerVertex1),
                                                      glm::value_ptr(innerVertex2),
                                                      glm::value_ptr(roughVertex0), glm::value_ptr(roughVertex1),
                                                      glm::value_ptr(roughVertex2))){
                std::cout << "Different result!!!" << std::endl;
            }

            if (test) {
                return true;
            }
        }
    }
    return false;
}

bool WorldSpaceMesh::rayTriangleInside(const WorldSpaceMesh &other) const {
    for(Vertex worldSpaceVertex: this->worldSpaceVertices){
        if(!other.includes(worldSpaceVertex)){
            return false;
        }
    }
    return true;
}

