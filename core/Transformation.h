//
// Created by Jonas on 10/12/2020.
//

#ifndef MESHCORE_TRANSFORMATION_H
#define MESHCORE_TRANSFORMATION_H

#include <glm/glm.hpp>
#include <glm/gtx/io.hpp>
typedef glm::mat4 Transformation;
std::ostream& operator<<(std::ostream& o, const Transformation& transform);
#endif //MESHCORE_TRANSFORMATION_H
