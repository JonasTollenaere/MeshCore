//
// Created by Jonas on 9/11/2020.
//

#ifndef MESHCORE_RAY_H
#define MESHCORE_RAY_H

#include <glm/glm.hpp>
#include "Vertex.h"
#include "Transformation.h"
#include "Core.h"
#include "Quaternion.h"

class Ray {
public:
    const Vertex origin{};
    const glm::vec3 direction{};
    const glm::vec3 inverseDirection{};

    MC_FUNC_QUALIFIER Ray() = default;

    MC_FUNC_QUALIFIER Ray(const Vertex& origin, const glm::vec3& direction): origin(origin), direction(direction), inverseDirection(1.0f/direction) {}

    MC_FUNC_QUALIFIER [[nodiscard]] Ray getTransformed(const Transformation &transformation) const {
        return this->getTransformed(transformation.getMatrix());
    }

    MC_FUNC_QUALIFIER [[nodiscard]] Ray getRotated(const Quaternion &quaternion) const {
        Vertex rotatedOrigin = quaternion.rotateVertex(this->origin);
        glm::vec3 rotatedDirection = quaternion.rotateVertex(this->direction);
        return {rotatedOrigin, rotatedDirection};
    }

    MC_FUNC_QUALIFIER [[nodiscard]] Ray getTransformed(const glm::mat4 &transformationMatrix) const {
        Vertex transformedOrigin = transformationMatrix * glm::vec4(this->origin, 1);
        glm::vec3 transformedDirection = transformationMatrix * glm::vec4(this->direction, 0);
        return {transformedOrigin, transformedDirection};
    }

    MC_FUNC_QUALIFIER [[nodiscard]] const Vertex &getOrigin() const {
        return origin;
    }

    MC_FUNC_QUALIFIER [[nodiscard]] const glm::vec3 &getDirection() const {
        return direction;
    }
};

#endif //MESHCORE_RAY_H
