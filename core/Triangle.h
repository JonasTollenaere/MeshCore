//
// Created by Jonas on 9/11/2020.
//

#ifndef OPTIX_SAMPLES_TRIANGLE_H
#define OPTIX_SAMPLES_TRIANGLE_H


#include <vector>
#include "Vector.h"

class Triangle {
private:
public:
    Vector vertices[3];
    Triangle(const Vector& v0, const Vector& v1, const Vector& v2);
    friend std::ostream & operator<<(std::ostream & os, const Triangle& triangle);
};


#endif //OPTIX_SAMPLES_TRIANGLE_H
