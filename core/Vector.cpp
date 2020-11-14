////
//// Created by Jonas on 9/11/2020.
////
//
#include <iostream>
#include "Vector.h"

std::ostream &operator<<(std::ostream &os, const Vector &v) {
    return os << "(" << v.x << "," << v.y << "," << v.z << ")";
}