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

VertexBuffer::~VertexBuffer() {
    GLCall(glDeleteBuffers(1, &m_RendererId));
}

void VertexBuffer::bind() const {
    GLCall(glBindBuffer(GL_ARRAY_BUFFER, m_RendererId));
}

void VertexBuffer::unbind() const {
    GLCall(glBindBuffer(GL_ARRAY_BUFFER, 0));
}
