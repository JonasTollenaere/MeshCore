//
// Created by Jonas on 7/10/2021.
//

#ifndef MESHCORE_AABB_H
#define MESHCORE_AABB_H

#include "Core.h"
#include "Vertex.h"

/** Axis Aligned Bounding Box **/
class AABB {
    Vertex minimum;
    Vertex maximum;

public:
    MC_FUNC_QUALIFIER AABB(): minimum(), maximum(){};

    MC_FUNC_QUALIFIER AABB(const Vertex& minimum, const Vertex& maximum): minimum(minimum), maximum(maximum) {
        assert(minimum.x <= maximum.x);
        assert(minimum.y <= maximum.y);
        assert(minimum.z <= maximum.z);
    }

    MC_FUNC_QUALIFIER [[nodiscard]] Vertex getMinimum() const {
        return this->minimum;
    }

    MC_FUNC_QUALIFIER [[nodiscard]] Vertex getMaximum() const {
        return this->maximum;
    }

    MC_FUNC_QUALIFIER [[nodiscard]] Vertex getCenter() const {
        return (this->maximum + this->minimum) / 2.0f;
    }

    MC_FUNC_QUALIFIER [[nodiscard]] Vertex getHalf() const {
        return (this->maximum - this->minimum) / 2.0f;
    }

    MC_FUNC_QUALIFIER [[nodiscard]] float getSurfaceArea() const {
        auto delta = this->maximum - this->minimum;
        return 2 * (delta.x * delta.y + delta.x * delta.z + delta.y * delta.z);
    }

    MC_FUNC_QUALIFIER [[nodiscard]] float getVolume() const {
        const auto delta = this->maximum - this->minimum;
        return delta.x * delta.y * delta.z;
    }

    MC_FUNC_QUALIFIER [[nodiscard]] bool containsPoint(Vertex point) const {
        return glm::all(glm::greaterThanEqual(point, this->minimum)) &&
               glm::all(glm::lessThanEqual(point, this->maximum));
    }

    MC_FUNC_QUALIFIER [[nodiscard]] bool containsAABB(const AABB& aabb) const {
        return minimum.x <= aabb.minimum.x &&
               minimum.y <= aabb.minimum.y &&
               minimum.z <= aabb.minimum.z &&
               maximum.x >= aabb.maximum.x &&
               maximum.y >= aabb.maximum.y &&
               maximum.z >= aabb.maximum.z;
    }

    MC_FUNC_QUALIFIER [[nodiscard]] Vertex getClosestPoint(const Vertex& point) const {
        return glm::clamp(point, this->getMinimum(), this->getMaximum());
    }

    MC_FUNC_QUALIFIER [[nodiscard]] float getDistanceSquaredTo(const Vertex& point) const {
        const auto delta = this->getClosestPoint(point) - point;
        return glm::dot(delta, delta);
    }

    MC_FUNC_QUALIFIER bool operator==(const AABB &other) const {
        return minimum == other.minimum && maximum == other.maximum;
    }

    MC_FUNC_QUALIFIER bool operator!=(const AABB &other) const {
        return !(other == *this);
    }

    MC_FUNC_QUALIFIER [[nodiscard]] AABB getTranslated(const glm::vec3 &translation) const {
        return {minimum + translation, maximum + translation};
    }
};

#endif //MESHCORE_AABB_H
