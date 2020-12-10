//
// Created by Jonas on 10/12/2020.
//

#include "Transformation.h"

std::ostream& operator<<(std::ostream& o, const Transformation& transform) {
    o << "\n\t[" << transform.operator[](0) << "\n\t " << transform.operator[](1) << "\n\t " << transform.operator[](2) << "\n\t " << transform.operator[](3) << ']';
    return o;
}
