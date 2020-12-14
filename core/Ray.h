//
// Created by Jonas on 9/11/2020.
//

#ifndef OPTIX_SAMPLES_RAY_H
#define OPTIX_SAMPLES_RAY_H

#include <glm/glm.hpp>
#include <iostream>
#include "Vertex.h"
#include "Transformation.h"

class Ray {
private:
    Vertex origin{};
    Vertex direction{};
public:
    Ray(Vertex origin, Vertex direction);
    friend std::ostream & operator<<(std::ostream & os, const Ray& ray);
    void transform(const Transformation &transformation);
    bool intersectsTriangle(const Vertex &v0, const Vertex &v1, const Vertex &v2) const;

    float intersectionDistance(const Vertex &v0, const Vertex &v1, const Vertex &v2) const;

    float intersectionDistanceJGT(const Vertex &v0, const Vertex &v1, const Vertex &v2) const;
};

#endif //OPTIX_SAMPLES_RAY_H
