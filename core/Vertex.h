//
// Created by Jonas on 9/11/2020.
//

#ifndef OPTIX_SAMPLES_VERTEX_H
#define OPTIX_SAMPLES_VERTEX_H

#include <glm/glm.hpp>
#include <ostream>

typedef glm::vec3 Vertex;
std::ostream & operator<<(std::ostream & os, const Vertex& v);

#endif //OPTIX_SAMPLES_VERTEX_H
