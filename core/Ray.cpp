//
// Created by Jonas on 9/11/2020.
//

#include "Ray.h"
#include "MeshTriangle.h"
#include <glm/glm.hpp>
#include <glm/gtc/epsilon.hpp>
#include <glm/gtx/intersect.hpp>

Ray::Ray(Vertex origin, Vertex direction) {
    this->origin = origin;
    this->direction = direction;
}

std::ostream &operator<<(std::ostream &os, const Ray &ray) {
    return os << "Ray(Origin" << ray.origin << ", Direction" << ray.direction << ")";
}

double Ray::intersects(const MeshTriangle &triangle) {

    return 0;
//    // Values returned by the glm::intersectRayTriangle
//    double distance;
//    glm::dvec2 baryPosition = glm::dvec2();
//
//    bool intersects = glm::intersectRayTriangle(this->origin, this->direction,
//                                                triangle.vertices[0], triangle.vertices[1], triangle.vertices[2],
//                                     baryPosition, distance);
//
//    assert((distance >= 0) == intersects);
//
//    return distance;
}

//bool
//)


