//
// Created by Jonas on 15/03/2022.
//

#ifndef MESHCORE_OBB_H
#define MESHCORE_OBB_H

#include "AABB.h"
#include "Quaternion.h"
#include "Transformation.h"

/** OBB as an AABB in its own model space, defined by a rotation only **/
class OBB {

    AABB aabb;
    Quaternion rotation;

public:
    MC_FUNC_QUALIFIER OBB() = default;

    MC_FUNC_QUALIFIER OBB(const AABB& aabb, const Quaternion& rotation): aabb(aabb), rotation(rotation){}

    MC_FUNC_QUALIFIER OBB(const AABB& aabb, const Transformation& transformation) {
        this->rotation = transformation.getRotation();
        auto rotatedTranslation = this->rotation.inverseRotateVertex(transformation.getPosition());
        auto scale = transformation.getScale();
        this->aabb = AABB(aabb.getMinimum()*scale+rotatedTranslation, aabb.getMaximum()*scale+rotatedTranslation);
    }

    MC_FUNC_QUALIFIER [[nodiscard]] float getSurfaceArea() const {
        return aabb.getSurfaceArea();
    }

    MC_FUNC_QUALIFIER [[nodiscard]] float getVolume() const {
        return aabb.getVolume();
    }

    MC_FUNC_QUALIFIER [[nodiscard]] Vertex getCenter() const {
        return this->rotation.rotateVertex(aabb.getCenter());
    }

    MC_FUNC_QUALIFIER [[nodiscard]] const AABB &getAabb() const {
        return aabb;
    }

    MC_FUNC_QUALIFIER [[nodiscard]] bool containsPoint(const Vertex& point) const {
        return this->aabb.containsPoint(this->rotation.inverseRotateVertex(point));
    }

    MC_FUNC_QUALIFIER [[nodiscard]] Vertex getClosestPoint(const Vertex& point) const {
        return this->rotation.rotateVertex(this->aabb.getClosestPoint(this->rotation.inverseRotateVertex(point)));
    }

    MC_FUNC_QUALIFIER [[nodiscard]] float getDistanceSquaredTo(const Vertex& point) const {
        return this->aabb.getDistanceSquaredTo(this->rotation.inverseRotateVertex(point));
    }

    MC_FUNC_QUALIFIER [[nodiscard]] const Quaternion &getRotation() const {
        return rotation;
    }
};

#endif //MESHCORE_OBB_H
