//
// Created by Jonas on 9/11/2020.
//

#ifndef OPTIX_SAMPLES_RAY_H
#define OPTIX_SAMPLES_RAY_H

#include <glm/glm.hpp>
#include <iostream>
#include "Vertex.h"
#include "MeshTriangle.h"

class Ray {
private:
    Vertex origin{};
    Vertex direction{};
public:
    Ray(Vertex origin, Vertex direction);
    friend std::ostream & operator<<(std::ostream & os, const Ray& ray);
    double intersects(const MeshTriangle &triangle);
};

#endif //OPTIX_SAMPLES_RAY_H
