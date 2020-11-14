//
// Created by Jonas on 9/11/2020.
//

#ifndef OPTIX_SAMPLES_RAY_H
#define OPTIX_SAMPLES_RAY_H

#include <glm/glm.hpp>
#include <iostream>
#include "Vector.h"
#include "Triangle.h"

class Ray {
private:
    Vector origin{};
    Vector direction{};
public:
    Ray(Vector origin, Vector direction);
    friend std::ostream & operator<<(std::ostream & os, const Ray& ray);
    double intersects(const Triangle &triangle);
};

#endif //OPTIX_SAMPLES_RAY_H
