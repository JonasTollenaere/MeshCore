//
// Created by Jonas on 9/11/2020.
//

#include "Ray.h"

#define EPSILON 0.000001f

Ray::Ray(Vertex origin, Vertex direction) {
    this->origin = origin;
    this->direction = direction;
}

inline std::ostream &operator<<(std::ostream &os, const Ray &ray) {
    return os << "Ray(Origin" << ray.origin << ", Direction" << ray.direction << ")";
}

bool Ray::intersectsTriangle(const Vertex& v0, const Vertex& v1, const Vertex& v2) const {

    //Möller–Trumbore
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
    float t = f * glm::dot(edge2, q);
    if (t > EPSILON) // ray intersection
    {
        return true;
    }
    else { // This means that there is a line intersection but not a ray intersection.
        return false;
    }
}

float Ray::intersectionDistance(const Vertex& v0, const Vertex& v1, const Vertex& v2) const {

    //Möller–Trumbore
    glm::vec3 edge1 = v1 - v0;
    glm::vec3 edge2 = v2 - v0;

    glm::vec3 h = glm::cross(this->direction, edge2);

    float a = glm::dot(edge1, h);
    if (a > -EPSILON && a < EPSILON) {
        return - std::numeric_limits<float>::max();    // This ray is parallel to this triangle, no line intersection
    }
    float f = 1.0f / a;
    glm::vec3 s = this->origin - v0;
    float u = f * (glm::dot(s, h));
    if (u < 0.0 || u > 1.0) {
        return - std::numeric_limits<float>::max(); // No line intersection
    }
    glm::vec3 q = glm::cross(s, edge1);
    float v = f * glm::dot(this->direction, q);
    if (v < 0.0 || u + v > 1.0) {
        return - std::numeric_limits<float>::max(); // No line intersection
    }
    // At this stage we can compute t to find out where the intersection point is on the line.
    float t = f * glm::dot(edge2, q);
    return t;
}

float Ray::intersectionDistanceJGT(const Vertex &v0, const Vertex &v1, const Vertex &v2) const {

    //Möller–Trumbore  // https://github.com/erich666/jgt-code/blob/master/Volume_02/Number_1/Moller1997a/raytri.c
    glm::vec3 edge1 = v1 - v0;
    glm::vec3 edge2 = v2 - v0;

    glm::vec3 h = glm::cross(this->direction, edge2);


    /* if determinant is near zero, ray lies in plane of triangle */
    float a = glm::dot(edge1, h);

    glm::vec3 q;
    if (a > EPSILON)
    {
        /* calculate distance from vert0 to ray origin */
        glm::vec3 s = this->origin - v0;

        /* calculate U parameter and test bounds */
        float u = glm::dot(s, h);
        if (u < 0.0 || u > a)
            return 0;

        /* prepare to test V parameter */
        q = glm::cross(s, edge1);

        /* calculate V parameter and test bounds */
        float v = glm::dot(this->direction, q);
        if (v < 0.0 || u + v > a)
            return 0;

    }
    else if(a < -EPSILON)
    {
        /* calculate distance from vert0 to ray origin */
        glm::vec3 s = this->origin - v0;

        /* calculate U parameter and test bounds */
        float u = glm::dot(s, h);

        if (u > 0.0 || u < a)
            return 0;

        /* prepare to test V parameter */
        q = glm::cross(s, edge1);

        /* calculate V parameter and test bounds */
        float v = glm::dot(this->direction, q);
        if (v > 0.0 || u + v < a)
            return 0;
    }
    else return 0;  /* ray is parallell to the plane of the triangle */


    float f = 1.0f / a;

    /* calculate t, ray intersects triangle */
    float t = glm::dot(edge2, q) * f;
    return t;
}

void Ray::transform(const Transformation& transformation) {
    this->origin = glm::vec3(transformation * glm::vec4(this->origin, 1.0f));
    this->direction = glm::vec3(transformation * glm::vec4(this->direction, 1.0f));
}

