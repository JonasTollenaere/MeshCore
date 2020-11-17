//
// Created by Jonas on 9/11/2020.
//

#include <iostream>
#include "Vertex.h"

std::ostream &operator<<(std::ostream &os, const Vertex &v) {
    return os << "(" << v.x << "," << v.y << "," << v.z << ")";
}