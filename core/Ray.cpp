//
// Created by Jonas on 9/11/2020.
//

#include "Ray.h"

Ray::Ray(Vertex origin, Vertex direction) {
    this->origin = origin;
    this->direction = direction;
}

inline std::ostream &operator<<(std::ostream &os, const Ray &ray) {
    return os << "Ray(Origin" << ray.origin << ", Direction" << ray.direction << ")";
}

bool Ray::intersectsTriangle(const Vertex& v0, const Vertex& v1, const Vertex& v2) {

    //Möller–Trumbore
    const float EPSILON = 0.0000001f;

    glm::vec3 edge1 = v1 - v0;
    glm::vec3 edge2 = v2 - v0;

    glm::vec3 h = glm::cross(this->direction, edge2);

    float a = glm::dot(edge1, h);
    if (a > -EPSILON && a < EPSILON) {
        return false;    // This ray is parallel to this triangle.
    }
    float f = 1.0f / a;
    glm::vec3 s = this->origin - v0;
    float u = f * (glm::dot(s, h));
    if (u < 0.0 || u > 1.0) {
        return false;
    }
    glm::vec3 q = glm::cross(s, edge1);
    float v = f * glm::dot(this->direction, q);
    if (v < 0.0 || u + v > 1.0) {
        return false;
    }
    // At this stage we can compute t to find out where the intersection point is on the line.
    double t = f * glm::dot(edge2, q);
    if (t > EPSILON) // ray intersection
    {
        return true;
    }
    else { // This means that there is a line intersection but not a ray intersection.
        return false;
    }
}

void Ray::transform(const Transformation& transformation) {
    this->origin = glm::vec3(transformation * glm::vec4(this->origin, 1.0f));
    this->direction = glm::vec3(transformation * glm::vec4(this->direction, 1.0f));
}

