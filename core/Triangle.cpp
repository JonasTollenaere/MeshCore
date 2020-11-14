//
// Created by Jonas on 9/11/2020.
//

#include <iostream>
#include "Triangle.h"


Triangle::Triangle(const Vector& v0, const Vector& v1, const Vector& v2) : vertices() {
    vertices[0] = v0;
    vertices[1] = v1;
    vertices[2] = v2;
}

std::ostream &operator<<(std::ostream &os, const Triangle &t) {
    return os << "Triangle(" << t.vertices[0] << ", " << t.vertices[1] << ", " << t.vertices[2] << ")";
}