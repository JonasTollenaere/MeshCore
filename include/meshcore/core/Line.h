//
// Created by Jonas Tollenaere on 20/01/2025.
//

#ifndef EXTENDEDMESHCORE_LINE_H
#define EXTENDEDMESHCORE_LINE_H

#include <glm/glm.hpp>
#include "Vertex.h"
#include "Transformation.h"
#include "Core.h"
#include "Quaternion.h"

class Line {
public:
    Vertex point;
    glm::vec3 direction;
    glm::vec3 inverseDirection;

    MC_FUNC_QUALIFIER Line() = default;

    MC_FUNC_QUALIFIER Line(const Vertex& point, const glm::vec3& direction): point(point), direction(direction), inverseDirection(1.0f/direction) {}

    MC_FUNC_QUALIFIER [[nodiscard]] Line getTransformed(const Transformation &transformation) const {
        return this->getTransformed(transformation.getMatrix());
    }

    MC_FUNC_QUALIFIER [[nodiscard]] Line getRotated(const Quaternion &quaternion) const {
        Vertex rotatedPoint = quaternion.rotateVertex(this->point);
        glm::vec3 rotatedDirection = quaternion.rotateVertex(this->direction);
        return {rotatedPoint, rotatedDirection};
    }

    MC_FUNC_QUALIFIER [[nodiscard]] Line getTransformed(const glm::mat4 &transformationMatrix) const {
        Vertex transformedPoint = transformationMatrix * glm::vec4(this->point, 1);
        glm::vec3 transformedDirection = transformationMatrix * glm::vec4(this->direction, 0);
        return {transformedPoint, transformedDirection};
    }

    MC_FUNC_QUALIFIER [[nodiscard]] const Vertex &getPoint() const {
        return point;
    }

    MC_FUNC_QUALIFIER [[nodiscard]] const glm::vec3 &getDirection() const {
        return direction;
    }
};

#endif //EXTENDEDMESHCORE_LINE_H
