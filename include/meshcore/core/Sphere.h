//
// Created by Jonas on 1/07/2022.
//

#ifndef MESHCORE_SPHERE_H
#define MESHCORE_SPHERE_H

#include "Vertex.h"
#include "Core.h"
#include "Transformation.h"

class Sphere{
private:
    Vertex center;
    float radius, radiusSquared;
public:
    MC_FUNC_QUALIFIER [[nodiscard]] const Vertex &getCenter() const {
        return center;
    }

    MC_FUNC_QUALIFIER [[nodiscard]] float getRadius() const {
        return radius;
    }

    MC_FUNC_QUALIFIER [[nodiscard]] float getRadiusSquared() const {
        return radiusSquared;
    }

    MC_FUNC_QUALIFIER Sphere(): center(0.0f), radius(0.0f), radiusSquared(0.0f){}

    MC_FUNC_QUALIFIER Sphere(Vertex center, float radius): center(center), radius(radius), radiusSquared(radius * radius){
        assert(radius>=0.0f);
    }

    MC_FUNC_QUALIFIER [[nodiscard]] Sphere getTransformed(const Transformation &transformation) const {
        auto transformedCenter = transformation.transformVertex(this->center);
        auto transformedRadius = this->radius * transformation.getScale();
        return {transformedCenter, transformedRadius};
    }

    MC_FUNC_QUALIFIER [[nodiscard]] Sphere getRotated(const Quaternion &quaternion) const {
        Vertex rotatedCenter = quaternion.rotateVertex(this->center);
        return {rotatedCenter, this->radius};
    }

    MC_FUNC_QUALIFIER [[nodiscard]] Sphere getTransformed(const glm::mat4 &transformationMatrix) const {
        auto transformedCenter = transformationMatrix * glm::vec4(this->center, 1);
        auto transformedRadius = glm::length(transformationMatrix[0]); // Extract x-component scaling from the matrix
        return {transformedCenter, transformedRadius};
    }

    MC_FUNC_QUALIFIER [[nodiscard]] float getVolume() const {
        return (4.0f / 3.0f) * glm::pi<float>() * getRadiusSquared() * radius;
    }

    MC_FUNC_QUALIFIER [[nodiscard]] float getSurfaceArea() const {
        return 4.0f * glm::pi<float>() * getRadiusSquared();
    }

    MC_FUNC_QUALIFIER [[nodiscard]] bool containsPoint(const Vertex& point) const {
        auto deltaCenter = point - this->center;
        return glm::dot(deltaCenter, deltaCenter) <= getRadiusSquared();
    }

    MC_FUNC_QUALIFIER [[nodiscard]] Vertex getClosestPoint(const Vertex& point) const {
        const auto deltaCenter = point - this->center;
        const auto distanceCenterSquared = glm::dot(deltaCenter, deltaCenter);

        // Return the point itself if it is inside the sphere
        if(distanceCenterSquared <= getRadiusSquared()){
            return point;
        }

        // Normalize the deltaCenter vector
        const auto deltaCenterNormalized = deltaCenter * glm::inversesqrt(distanceCenterSquared);

        return  this->center + deltaCenterNormalized * this->radius;
    }

    MC_FUNC_QUALIFIER [[nodiscard]] float getDistanceSquaredTo(const Vertex& point) const {
        auto deltaCenter = point - this->center;
        auto distanceCenterSquared = glm::dot(deltaCenter, deltaCenter);

        // Return 0 if the point is inside the sphere
        if(distanceCenterSquared <= getRadiusSquared()){
            return 0.0f;
        }

        auto distance = glm::sqrt(distanceCenterSquared) - this->radius;
        return distance*distance;
    }
};

#endif //MESHCORE_SPHERE_H
