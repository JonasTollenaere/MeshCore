//
// Created by Jonas on 7/10/2021.
//

#ifndef MESHCORE_VERTEXTRIANGLE_H
#define MESHCORE_VERTEXTRIANGLE_H

#include "Vertex.h"
#include "AABB.h"
#include "Transformation.h"
#include "Quaternion.h"

class VertexTriangle{
public:
    const Vertex vertices[3];
    const glm::vec3 edges[3];
    const glm::vec3 normal;
    const AABB bounds;

    MC_FUNC_QUALIFIER VertexTriangle(const Vertex vertex0, const Vertex vertex1, const Vertex vertex2):
            vertices{vertex0, vertex1, vertex2},
            edges{vertices[1]-vertices[0], vertices[2]-vertices[1], vertices[0]-vertices[2]},
            normal(glm::cross(edges[0], edges[1])),
            bounds((glm::min)((glm::min)(vertices[0], vertices[1]), vertices[2]), (glm::max)((glm::max)(vertices[0], vertices[1]), vertices[2]))
            {}

    MC_FUNC_QUALIFIER VertexTriangle(Vertex vertex0, Vertex vertex1, Vertex vertex2, const glm::vec3 edges[3],
                                     glm::vec3 normal, AABB bounds):
        vertices{vertex0, vertex1, vertex2},
        edges{edges[0], edges[1], edges[2]},
        normal(normal),
        bounds(bounds)
    {}

    MC_FUNC_QUALIFIER bool operator==(const VertexTriangle &other) const {
        return this->vertices[0] == other.vertices[0] && this->vertices[1] == other.vertices[1] && this->vertices[2] == other.vertices[2];
    }

    MC_FUNC_QUALIFIER [[nodiscard]] VertexTriangle getTransformed(const Transformation& transformation) const {
        return this->getTransformed(transformation.getMatrix());
    }

    MC_FUNC_QUALIFIER [[nodiscard]] VertexTriangle getRotated(const Quaternion& quaternion) const {
        Vertex rVertex0 = quaternion.rotateVertex(this->vertices[0]);
        Vertex rVertex1 = quaternion.rotateVertex(this->vertices[1]);
        Vertex rVertex2 = quaternion.rotateVertex(this->vertices[2]);
        return {rVertex0, rVertex1, rVertex2};
    }

    MC_FUNC_QUALIFIER [[nodiscard]] VertexTriangle getTransformed(const glm::mat4& transformationMatrix) const{
        Vertex rVertex0 = transformationMatrix * glm::vec4(this->vertices[0], 1);
        Vertex rVertex1 = transformationMatrix * glm::vec4(this->vertices[1], 1);
        Vertex rVertex2 = transformationMatrix * glm::vec4(this->vertices[2], 1);
        return {rVertex0, rVertex1, rVertex2};
    }

    MC_FUNC_QUALIFIER [[nodiscard]] VertexTriangle getTranslated(const glm::vec3& translation) const {
        const Vertex rVertex0 = this->vertices[0] + translation;
        const Vertex rVertex1 = this->vertices[1] + translation;
        const Vertex rVertex2 = this->vertices[2] + translation;
        const AABB newBounds = this->bounds.getTranslated(translation);
        return VertexTriangle{rVertex0, rVertex1, rVertex2, &this->edges[0], this->normal, newBounds};

    }

    MC_FUNC_QUALIFIER [[nodiscard]] Vertex getClosestPoint(const Vertex &point) const {

        // As described in "Real-Time Collision Detection" by Christer Ericson

        // Check if point lies in the vertex region outside A
        auto ab = edges[0];
        auto ac = - edges[2];
        auto ap = point - vertices[0];
        auto d1 = glm::dot(ab, ap);
        auto d2 = glm::dot(ac, ap);

        if(d1 <= 0.0f && d2 <= 0.0f) return vertices[0];

        // Check if point lies in the vertex region outside B
        auto bp = point - vertices[1];
        auto d3 = glm::dot(ab, bp);
        auto d4 = glm::dot(ac, bp);

        if(d3 >= 0.0f && d4 <= d3) return vertices[1];

        // Check if point lies in de edge region of AB, if so return the projection of point onto AB
        auto vc = d1*d4 - d3 *d2;
        if(vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f){
            auto v = d1 / (d1 - d3);
            return vertices[0] + v * ab;
        }

        // Check if point lies in the vertex region outside C
        auto cp = point - vertices[2];
        auto d5 = glm::dot(ab, cp);
        auto d6 = glm::dot(ac, cp);

        if(d6 >= 0.0f && d5 <= d6) return vertices[2];

        // Check if point lies in the edge region of AC, if so return the projection of point onto AC
        auto vb = d5*d2 - d1*d6;
        if(vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f){
            auto w = d2 / (d2 - d6);
            return vertices[0] + w * ac;
        }

        // Check if point lies in the edge region of BC, if so return the projection of point onto BC
        auto va = d3*d6 - d5*d4;
        if(va <= 0.0f && (d4-d3) >= 0.0f && (d5 - d6) >= 0.0f){
            auto w = (d4-d3) / ((d4-d3) + (d5-d6));
            return vertices[1] + w * edges[1];
        }

        // The point lies in the face region. Compute the closest point through barycentric coordinates
        auto denom = 1.0f / (va + vb + vc);
        auto v = vb * denom;
        auto w = vc * denom;
        return vertices[0] + ab*v + ac*w;
    }

    MC_FUNC_QUALIFIER [[nodiscard]] float getSurfaceArea() const {
        return glm::length(glm::cross(edges[0], edges[1])) / 2.0f;
    }

    MC_FUNC_QUALIFIER [[nodiscard]] Vertex getCentroid() const {
        return (vertices[0] + vertices[1] + vertices[2]) / 3.0f;
    }
};

#endif //MESHCORE_VERTEXTRIANGLE_H
