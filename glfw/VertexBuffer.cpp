//
// Created by Jonas on 12/11/2020.
//

#include "VertexBuffer.h"
#include "Renderer.h"
#include <glm/gtx/normal.hpp>

VertexBuffer::VertexBuffer(): m_RendererId(0) {
    // This is a hollow object, 0-value as a buffer name is always implicitly ignored in OpenGL
}

VertexBuffer::VertexBuffer(const VertexBuffer& other): m_RendererId(0) {
    int size;
    GLCall(glGenBuffers(1, &this->m_RendererId))
    GLCall(glBindBuffer(GL_COPY_READ_BUFFER, other.m_RendererId))
    GLCall(glBindBuffer(GL_COPY_WRITE_BUFFER, this->m_RendererId))
    GLCall(glGetBufferParameteriv(GL_COPY_READ_BUFFER, GL_BUFFER_SIZE, &size))
    GLCall(glBufferData(GL_COPY_WRITE_BUFFER, size, nullptr, GL_STATIC_DRAW))
    GLCall(glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, size))
}

VertexBuffer::VertexBuffer(VertexBuffer&& other) noexcept {
    this->m_RendererId = other.m_RendererId;
    other.m_RendererId = 0;
}

VertexBuffer &VertexBuffer::operator=(VertexBuffer&& other) noexcept {
    if(this != &other){
        GLCall(glDeleteBuffers(1, &this->m_RendererId))
        this->m_RendererId = other.m_RendererId;
        other.m_RendererId = 0;
    }
    return *this;
}

VertexBuffer::VertexBuffer(const void *data, unsigned int size): m_RendererId(0) {
    GLCall(glGenBuffers(1, &m_RendererId))
    bind();
    GLCall(glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW))
}

VertexBuffer::VertexBuffer(const WorldSpaceMesh& worldSpaceMesh): m_RendererId(0) {

    GLCall(glGenBuffers(1, &m_RendererId))
    bind();
    const std::vector<Vertex> vertices = worldSpaceMesh.getModelSpaceMesh().vertices;
    const std::vector<Triangle> triangles = worldSpaceMesh.getModelSpaceMesh().triangles;

    // Calculate normals
    std::vector<glm::vec3> normals(vertices.size(), glm::vec3(0.0f));
    for(Triangle triangle: triangles){
        Vertex v0 = vertices[triangle.vertexIndex0];
        Vertex v1 = vertices[triangle.vertexIndex1];
        Vertex v2 = vertices[triangle.vertexIndex2];

        normals[triangle.vertexIndex0] += glm::triangleNormal(v0, v1, v2);
    }

    std::vector<float> data;
    for(int i=0; i<vertices.size(); i++) {
        Vertex v = vertices[i];
        glm::vec3 n= normals[i];
        data.emplace_back(v.x);
        data.emplace_back(v.y);
        data.emplace_back(v.z);
        data.emplace_back(n.x);
        data.emplace_back(n.y);
        data.emplace_back(n.z);
    }
    GLCall(glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), &data.front(), GL_STATIC_DRAW))
}

VertexBuffer::~VertexBuffer() {
    GLCall(glDeleteBuffers(1, &m_RendererId))
}

void VertexBuffer::bind() const {
    GLCall(glBindBuffer(GL_ARRAY_BUFFER, m_RendererId))
}