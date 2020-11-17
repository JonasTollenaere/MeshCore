//
// Created by Jonas on 12/11/2020.
//

#include "VertexBuffer.h"
#include "Renderer.h"

VertexBuffer::VertexBuffer(const void *data, unsigned int size): m_RendererId(0) {
    GLCall(glGenBuffers(1, &m_RendererId));
    bind();
    GLCall(glBufferData(GL_ARRAY_BUFFER,size, data, GL_STATIC_DRAW));
}

VertexBuffer::VertexBuffer(const std::vector<Vertex>& vertices, const glm::vec4& color): m_RendererId(0) {
    GLCall(glGenBuffers(1, &m_RendererId));
    bind();

    std::vector<float> data;
    for(Vertex v: vertices){
        data.emplace_back(v.x);
        data.emplace_back(v.y);
        data.emplace_back(v.z);
        data.emplace_back(color.r);
        data.emplace_back(color.g);
        data.emplace_back(color.b);
        data.emplace_back(color.a);
    }

    std::cout << "#Vertices data size: " << data.size() << std::endl;

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
