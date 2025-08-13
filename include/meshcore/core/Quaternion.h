//
// Created by Jonas on 20/05/2022.
//

#ifndef MESHCORE_QUATERNION_H
#define MESHCORE_QUATERNION_H

#include "Core.h"
#include <glm/gtc/quaternion.hpp>

class Quaternion: public glm::fquat{
public:

    MC_FUNC_QUALIFIER Quaternion(): glm::fquat(1,0,0,0){};

    MC_FUNC_QUALIFIER explicit Quaternion(const glm::fquat &quaternion): glm::fquat(quaternion){}

    MC_FUNC_QUALIFIER Quaternion(float w, float x, float y, float z): glm::fquat(w, x, y, z){}

    MC_FUNC_QUALIFIER explicit Quaternion(float yaw, float pitch, float roll): glm::fquat(){
        const auto sx = glm::sin(roll/2);
        const auto sy = glm::sin(pitch/2);
        const auto sz = glm::sin(yaw/2);
        const auto cx = glm::cos(roll/2);
        const auto cy = glm::cos(pitch/2);
        const auto cz = glm::cos(yaw/2);
        this->w = cx*cy*cz - sx*sy*sz;
        this->x = sx*cy*cz + cx*sy*sz;
        this->y = cx*sy*cz - sx*cy*sz;
        this->z = cx*cy*sz + sx*sy*cz;
    }

    MC_FUNC_QUALIFIER explicit Quaternion(const glm::mat3 &rotationMatrix): glm::fquat(glm::quat_cast(rotationMatrix)){}

    MC_FUNC_QUALIFIER Quaternion(const glm::vec3& axis, float angle): glm::fquat(glm::angleAxis(angle, glm::normalize(axis))){}

    MC_FUNC_QUALIFIER explicit Quaternion(const glm::mat4 &rotationMatrix): glm::fquat(glm::quat_cast(rotationMatrix)){}

    MC_FUNC_QUALIFIER [[nodiscard]] Vertex rotateVertex(const glm::vec3& vertex) const{
        return (*this)*vertex; // glm quaternion multiplication does what we need
    };

    MC_FUNC_QUALIFIER [[nodiscard]] Vertex inverseRotateVertex(const glm::vec3& vertex) const{
        return glm::inverse(*this)*vertex;
    };

    MC_FUNC_QUALIFIER [[nodiscard]] glm::mat4 computeMatrix() const{
        return glm::mat4_cast(*this);
    }

    MC_FUNC_QUALIFIER [[nodiscard]] glm::mat4 computeInverseMatrix() const{
        return glm::mat4_cast(glm::inverse(*this));
    }

    MC_FUNC_QUALIFIER [[nodiscard]] Quaternion getInverse() const{
        return Quaternion(glm::inverse(*this));
    }

    MC_FUNC_QUALIFIER Quaternion operator*(const Quaternion& other) const{
        return Quaternion(glm::fquat(*this) * glm::fquat(other));
    }

    MC_FUNC_QUALIFIER [[nodiscard]] float getYaw() const{
        const auto qyy = this->y * this->y;
        const auto qzz = this->z * this->z;
        const auto qxy = this->x * this->y;
        const auto qwz = this->w * this->z;

        const auto m00 = 1.0f - 2.0f * (qyy +  qzz);
        const auto m10 = 2.0f * (qxy - qwz);

        return std::atan2(-m10, m00);
    }

    MC_FUNC_QUALIFIER [[nodiscard]] float getPitch() const {
        const auto qxz(this->x * this->z);
        const auto qwy(this->w * this->y);

        const auto m20 = 2.0f * (qxz + qwy);

        return glm::asin(glm::clamp(m20, -1.0f, 1.0f));
    }

    MC_FUNC_QUALIFIER [[nodiscard]] float getRoll() const {
        const auto qxx = this->x * this->x;
        const auto qxz = this->x * this->z;
        const auto qyz = this->y * this->z;
        const auto qwx = this->w * this->x;
        const auto qwy = this->w * this->y;

        const auto m20 = 2.0f * (qxz + qwy);

        if(glm::abs(m20) < (1-1e-8)){
            const auto qyy = this->y * this->y;
            const auto m21 = 2.0f * (qyz - qwx);
            const auto m22 = 1.0f - 2.0f * (qxx +  qyy);
            return std::atan2(-m21, m22);
        }
        const auto qzz = this->z * this->z;
        const auto m11 = 1.0f - 2.0f * (qxx +  qzz);
        const auto m12 = 2.0f * (qyz + qwx);
        return std::atan2(m12, m11);
    }

    using glm::fquat::operator*=;
    MC_FUNC_QUALIFIER void operator*=(const Quaternion& other){
        *this *= glm::fquat(other);
    }
};
#endif //MESHCORE_QUATERNION_H
