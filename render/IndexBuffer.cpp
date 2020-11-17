//
// Created by Jonas on 12/11/2020.
//

#include "IndexBuffer.h"
#include "Renderer.h"

IndexBuffer::IndexBuffer(const unsigned int *indices, unsigned int count): m_Count(count), m_RendererId(0) {

    ASSERT(sizeof(unsigned int) == sizeof(GLuint));

    GLCall(glGenBuffers(1, &m_RendererId));
    bind();
    GLCall(glBufferData(GL_ELEMENT_ARRAY_BUFFER, count*sizeof(unsigned int), indices, GL_STATIC_DRAW));
}

IndexBuffer::IndexBuffer(const std::vector<Triangle> &triangles): m_RendererId(0) {
    ASSERT(sizeof(unsigned int) == sizeof(GLuint));
    GLCall(glGenBuffers(1, &m_RendererId));
    bind();

    std::vector<unsigned int> data;
    for(Triangle t: triangles){
        data.emplace_back(t.vertexIndex0);
        data.emplace_back(t.vertexIndex1);
        data.emplace_back(t.vertexIndex2);
    }

    m_Count = data.size();

    GLCall(glBufferData(GL_ELEMENT_ARRAY_BUFFER, data.size()*sizeof(unsigned int), &data.front(), GL_STATIC_DRAW));
}


IndexBuffer::~IndexBuffer() {
        GLCall(glDeleteBuffers(1, &m_RendererId));
}

void IndexBuffer::bind() const {
    GLCall(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_RendererId));
}

void IndexBuffer::unbind() const {
    GLCall(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
}
