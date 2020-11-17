//
// Created by Jonas on 12/11/2020.
//

#include "VertexBuffer.h"
#include "Renderer.h"
#include <glm/gtx/normal.hpp>
#include <glm/gtx/component_wise.hpp>

VertexBuffer::VertexBuffer(const void *data, unsigned int size): m_RendererId(0) {
    GLCall(glGenBuffers(1, &m_RendererId));
    bind();
    GLCall(glBufferData(GL_ARRAY_BUFFER,size, data, GL_STATIC_DRAW));
}

VertexBuffer::VertexBuffer(const ModelSpaceMesh& mesh, const glm::vec4& color): m_RendererId(0) {
    GLCall(glGenBuffers(1, &m_RendererId));
    bind();
    const std::vector<Vertex> vertices = mesh.vertices;
    const std::vector<Triangle> triangles = mesh.triangles;

    // Calculate normals
    std::vector<glm::vec3> normals(vertices.size(), glm::vec3(0.0f));
    for(Triangle triangle: triangles){
        Vertex v0 = vertices[triangle.vertexIndex0];
        Vertex v1 = vertices[triangle.vertexIndex1];
        Vertex v2 = vertices[triangle.vertexIndex2];

        normals[triangle.vertexIndex0] += glm::vec3(glm::triangleNormal(v0, v1, v2));
    }


    std::vector<float> data;
//    for(Vertex v: vertices){
    for(int i=0; i<vertices.size(); i++) {
        Vertex v = vertices[i];
        glm::vec3 n= normals[i];
        data.emplace_back(v.x);
        data.emplace_back(v.y);
        data.emplace_back(v.z);
        data.emplace_back(n.x);
        data.emplace_back(n.y);
        data.emplace_back(n.z);
        data.emplace_back(color.r);
        data.emplace_back(color.g);
        data.emplace_back(color.b);
        data.emplace_back(color.a);
    }

    GLCall(glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), &data.front(), GL_STATIC_DRAW));
}

VertexBuffer::~VertexBuffer() {
    GLCall(glDeleteBuffers(1, &m_RendererId));
}

void VertexBuffer::bind() const {
    GLCall(glBindBuffer(GL_ARRAY_BUFFER, m_RendererId));
}

void VertexBuffer::unbind() const {
    GLCall(glBindBuffer(GL_ARRAY_BUFFER, 0));
}
