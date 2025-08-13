//
// Created by Jonas on 10/12/2020.
//

#ifndef MESHCORE_TRANSFORMATION_H
#define MESHCORE_TRANSFORMATION_H

#include <glm/glm.hpp>
#include "Vertex.h"
#include "Core.h"
#include "Quaternion.h"

class Transformation {

    float scale;
    Quaternion rotation;
    glm::vec3 position;

    mutable glm::mat4 matrix;
    mutable glm::mat4 inverseMatrix;
    mutable bool validMatrix = false;
    mutable bool validInverseMatrix = false;

public:
    MC_FUNC_QUALIFIER Transformation(): scale(1.0f), rotation(), position(), matrix(), inverseMatrix() {}

    MC_FUNC_QUALIFIER explicit Transformation(const Quaternion quaternion): position(), scale(1.0f), rotation(quaternion), matrix(), inverseMatrix() {}

    MC_FUNC_QUALIFIER bool operator==(const Transformation &other) const {
        return scale == other.scale &&
               rotation == other.rotation &&
               position == other.position;
    }
    MC_FUNC_QUALIFIER bool operator!=(const Transformation &other) const {
        return !(other == *this);
    }

    MC_FUNC_QUALIFIER Transformation operator*(const Transformation &other) const {
        Transformation result = *this;
        result *= other;

#if !NDEBUG
        auto expectedResultMatrix = this->getMatrix() * other.getMatrix();
        assert(glm::all(glm::epsilonEqual(expectedResultMatrix[0], result.getMatrix()[0], glm::vec4(1e-5f))));
        assert(glm::all(glm::epsilonEqual(expectedResultMatrix[1], result.getMatrix()[1], glm::vec4(1e-5f))));
        assert(glm::all(glm::epsilonEqual(expectedResultMatrix[2], result.getMatrix()[2], glm::vec4(1e-5f))));
        assert(glm::all(glm::epsilonEqual(expectedResultMatrix[3], result.getMatrix()[3], glm::vec4(1e-5f))));
#endif //NDEBUG

        return result;
    }

    MC_FUNC_QUALIFIER void operator*=(const Transformation &other) {

#if !NDEBUG
        auto expectedResultMatrix = this->getMatrix() * other.getMatrix();
        auto expectedRotationMatrix = this->rotation.computeMatrix() * other.getRotation().computeMatrix();
#endif

        this->position = this->transformVertex(other.position);
        this->scale *= other.scale;
        this->rotation *= other.rotation;
        this->invalidateMatrices();

#if !NDEBUG
        auto resultRotationMatrix = this->rotation.computeMatrix();
        assert(glm::all(glm::epsilonEqual(expectedRotationMatrix[0], resultRotationMatrix[0], glm::vec4(1e-4f))));
        assert(glm::all(glm::epsilonEqual(expectedRotationMatrix[1], resultRotationMatrix[1], glm::vec4(1e-4f))));
        assert(glm::all(glm::epsilonEqual(expectedRotationMatrix[2], resultRotationMatrix[2], glm::vec4(1e-4f))));
#endif //NDEBUG

#if !NDEBUG
        assert(glm::epsilonEqual(glm::length(glm::fquat(this->rotation)),1.0f,1e-5f)); // Prefer to keep quaternions normalized
        auto resultMatrix = this->getMatrix();
        assert(glm::all(glm::epsilonEqual(expectedResultMatrix[0], resultMatrix[0], glm::vec4(1e-5f))));
        assert(glm::all(glm::epsilonEqual(expectedResultMatrix[1], resultMatrix[1], glm::vec4(1e-5f))));
        assert(glm::all(glm::epsilonEqual(expectedResultMatrix[2], resultMatrix[2], glm::vec4(1e-5f))));
        assert(glm::all(glm::epsilonEqual(expectedResultMatrix[3], resultMatrix[3], glm::vec4(1e-5f))));
#endif //NDEBUG
    }

private:
    MC_FUNC_QUALIFIER void invalidateMatrices() const {
        validMatrix = false;
        validInverseMatrix = false;
    }

public:

    MC_FUNC_QUALIFIER glm::mat4 getMatrix() const {
        if(!this->validMatrix){
            glm::mat4 newModelTransformationMatrix(1.0f);
            newModelTransformationMatrix = glm::translate(newModelTransformationMatrix, position);
            newModelTransformationMatrix = glm::scale(newModelTransformationMatrix, glm::vec3(scale));
            newModelTransformationMatrix *= rotation.computeMatrix();
            this->matrix = newModelTransformationMatrix;
            this->validMatrix = true;
        }
        return matrix;
    }


    MC_FUNC_QUALIFIER Transformation getInverse() const {
        Transformation result(this->rotation.getInverse());
        result.scale = 1.0f / this->scale;
        result.position = result.scale * result.rotation.rotateVertex(-this->position);
        return result;
    }

    MC_FUNC_QUALIFIER glm::mat4 getInverseMatrix() const {
        if(!this->validInverseMatrix){
            this->inverseMatrix = glm::inverse(this->getMatrix());
            this->validInverseMatrix = true;
        }
        return inverseMatrix;
    }

    MC_FUNC_QUALIFIER float getScale() const {
        return scale;
    }

    MC_FUNC_QUALIFIER void setScale(float newScale) {
        this->invalidateMatrices();
        this->scale = newScale;
    }

    MC_FUNC_QUALIFIER void deltaScale(float deltaScale) {
        this->invalidateMatrices();
        this->scale += deltaScale;
    }

    MC_FUNC_QUALIFIER void factorScale(float scaleFactor){
        this->invalidateMatrices();
        this->scale *= scaleFactor;
    }

    MC_FUNC_QUALIFIER void factorRotation(const Quaternion& quaternion) {
        this->invalidateMatrices();
        this->rotation *= quaternion;
    }

    MC_FUNC_QUALIFIER void setRotation(Quaternion newRotation) {
        this->invalidateMatrices();
        this->rotation = newRotation;
    }

    MC_FUNC_QUALIFIER const Quaternion &getRotation() const {
        return rotation;
    }

    MC_FUNC_QUALIFIER const glm::vec3& getPosition() const {
        return position;
    }

    MC_FUNC_QUALIFIER void setPosition(const glm::vec3 &newPosition) {
        this->invalidateMatrices();
        this->position = newPosition;
    }

    MC_FUNC_QUALIFIER void setPosition(float newX, float newY, float newZ) {
        this->setPosition(glm::vec3(newX, newY, newZ));
    }

    MC_FUNC_QUALIFIER void setPositionX(const float newX) {
        this->invalidateMatrices();
        this->position.x = newX;
    }

    MC_FUNC_QUALIFIER void setPositionY(const float newY) {
        this->invalidateMatrices();
        this->position.y = newY;
    }

    MC_FUNC_QUALIFIER void setPositionZ(const float newZ) {
        this->invalidateMatrices();
        this->position.z = newZ;
    }

    MC_FUNC_QUALIFIER void deltaPosition(const glm::vec3 &deltaPosition) {
        this->invalidateMatrices();
        this->position += deltaPosition;
    }

    MC_FUNC_QUALIFIER void deltaPositionX(const float deltaX) {
        this->invalidateMatrices();
        this->position.x += deltaX;
    }

    MC_FUNC_QUALIFIER void deltaPositionY(const float deltaY) {
        this->invalidateMatrices();
        this->position.y += deltaY;
    }

    MC_FUNC_QUALIFIER void deltaPositionZ(const float deltaZ) {
        this->invalidateMatrices();
        this->position.z += deltaZ;
    }

    MC_FUNC_QUALIFIER Vertex transformVertex(const Vertex &vertex) const {
        return this->getMatrix() * glm::vec4(vertex, 1);
    }
    MC_FUNC_QUALIFIER Vertex inverseTransformVertex(const Vertex& vertex) const {
        return this->getInverseMatrix() * glm::vec4(vertex, 1);
    }

    MC_FUNC_QUALIFIER Vertex transformVector(const Vertex &vertex) const {
        return this->getMatrix() * glm::vec4(vertex, 0);
    }
    MC_FUNC_QUALIFIER Vertex inverseTransformVector(const Vertex& vertex) const {
        return this->getInverseMatrix() * glm::vec4(vertex, 0);
    }

    MC_FUNC_QUALIFIER static Transformation fromRotationMatrix(const glm::mat3& rotationMatrix){

        Quaternion rotation(rotationMatrix);
        Transformation returnObject;
        returnObject.setRotation(rotation);
        
#if !NDEBUG

        // Assert the matrices are equal, with at least a limited level of accuracy
        auto testRotationMatrix = glm::mat3(returnObject.rotation.computeMatrix());

        assert(glm::all(glm::epsilonEqual(testRotationMatrix[0],rotationMatrix[0], 0.0001f)));
        assert(glm::all(glm::epsilonEqual(testRotationMatrix[1],rotationMatrix[1], 0.0001f)));
        assert(glm::all(glm::epsilonEqual(testRotationMatrix[2],rotationMatrix[2], 0.0001f)));
#endif

        return returnObject;
    }

    MC_FUNC_QUALIFIER static Transformation interpolate(const Transformation& a, const Transformation& b, float t){
        Transformation returnObject;

        // When interpolating a transformation, we interpolate it's meaningful components separately
        returnObject.position = glm::mix(a.position, b.position, t);
        returnObject.scale = glm::mix(a.scale, b.scale, t);
        returnObject.rotation = Quaternion(glm::slerp(a.rotation, b.rotation, t)); // Spherical linear interpolation
        return returnObject;
    }

};

#endif //MESHCORE_TRANSFORMATION_H