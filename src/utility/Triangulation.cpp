//
// Created by Jonas on 13/02/2025.
//

#include "meshcore/utility/Triangulation.h"

#include <mapbox/earcut.hpp>
#include <glm/gtx/vector_angle.hpp>
#include <array>

std::vector<IndexTriangle> Triangulation::triangulateFace(const std::vector<Vertex>& vertices, const IndexFace& face) {

    const auto& indices = face.vertexIndices;
    assert(indices.size()>=3);
    if(indices.size()==3){
        return std::vector<IndexTriangle>({IndexTriangle{indices[0], indices[1], indices[2]}});
    }

    // TransformUtil the vertices to a plane with constant z coordinates
    auto facetNormal = glm::normalize(computeFaceNormal(vertices, face));

    // Find the rotation for which the z-coordinates of all vertices are equal
    // (the rotation that maps the normal to the z-axis)
    glm::vec3 zAxis(0, 0, 1);
    float angle = glm::angle(zAxis, facetNormal);
    glm::vec3 cross = glm::cross(facetNormal, zAxis);
    if(glm::all(glm::epsilonEqual(cross, glm::vec3(), 1e-8f))){
        // Choose an arbitrary axis to rotate around
        cross = glm::vec3(1,0,0);
    }
    glm::mat4 transformation = glm::rotate(angle, cross);

    // Pass the projected vertices as 2D to the mapbox earcut heuristics
    std::vector<std::vector<std::array<float, 2>>> polygon;
    std::vector<std::array<float, 2>> polyline;
    polyline.reserve(indices.size());
    for (const auto &index : indices) {
        Vertex transformedVertex = transformation * glm::vec4(vertices[index], 1);
        polyline.emplace_back(std::array<float, 2>({transformedVertex.x, transformedVertex.y}));
    }
    polygon.emplace_back(polyline);
    std::vector<int> triangleIndices = mapbox::earcut<int>(polygon);
    assert(triangleIndices.size()%3==0);

    // Vector of triangles to be returned
    std::vector<IndexTriangle> triangles;
    triangles.reserve(triangleIndices.size()/3);
    for (auto iterator = triangleIndices.begin(); iterator!=triangleIndices.end(); iterator++) {
        triangles.emplace_back(IndexTriangle{indices[*iterator++], indices[*iterator++], indices[*iterator]});
    }
    return triangles;
}

glm::vec3 Triangulation::computeFaceNormal(const std::vector<Vertex>& vertices, const IndexFace& face){
    glm::vec3 facetNormal(0.0f);

    // Newell's Method to calculate the facet normal
    const auto& indices = face.vertexIndices;
    for (auto current = indices.begin(); current != indices.end(); ++current) {
        auto next = std::next(current);
        if(next==indices.end()) next = indices.begin(); // If wrapped

        unsigned int indexA = *current;
        unsigned int indexB = *next;

        Vertex vertexA = vertices[indexA];
        Vertex vertexB = vertices[indexB];

        facetNormal.x += (vertexA.y - vertexB.y) * (vertexA.z + vertexB.z);
        facetNormal.y += (vertexA.z - vertexB.z) * (vertexA.x + vertexB.x);
        facetNormal.z += (vertexA.x - vertexB.x) * (vertexA.y + vertexB.y);
    }
    return glm::normalize(facetNormal);
}
