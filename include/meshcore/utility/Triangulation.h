//
// Created by Jonas on 13/02/2025.
//

#ifndef MESHCORE_TRIANGULATE_H
#define MESHCORE_TRIANGULATE_H

#include <vector>
#include "meshcore/core/Vertex.h"
#include "meshcore/core/IndexFace.h"
#include "meshcore/core/IndexTriangle.h"

class Triangulation{
public:
    static std::vector<IndexTriangle> triangulateFace(const std::vector<Vertex>& vertices, const IndexFace& face);
    static glm::vec3 computeFaceNormal(const std::vector<Vertex>& vertices, const IndexFace& face);
};

#endif //MESHCORE_TRIANGULATE_H
