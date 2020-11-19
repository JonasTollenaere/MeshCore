//
// Created by Jonas on 12/11/2020.
//

#include "IndexBuffer.h"
#include "Renderer.h"

IndexBuffer::IndexBuffer(): m_RendererId(0), m_Count(0) {
    // This is a hollow object, 0-value as a buffer name is always implicitly ignored in OpenGL
}

IndexBuffer::IndexBuffer(const IndexBuffer &other): m_RendererId(0), m_Count(other.m_Count) {
    int size;
    GLCall(glGenBuffers(1, &this->m_RendererId))
    GLCall(glBindBuffer(GL_COPY_READ_BUFFER, other.m_RendererId))
    GLCall(glBindBuffer(GL_COPY_WRITE_BUFFER, this->m_RendererId))
    GLCall(glGetBufferParameteriv(GL_COPY_READ_BUFFER, GL_BUFFER_SIZE, &size))
    GLCall(glBufferData(GL_COPY_WRITE_BUFFER, size, nullptr, GL_STATIC_DRAW))
    GLCall(glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, size))
}

IndexBuffer::IndexBuffer(IndexBuffer &&other) noexcept {
    this->m_RendererId = other.m_RendererId;
    this->m_RendererId = other.m_RendererId;
    other.m_RendererId = 0;
    other.m_Count = 0;
}

IndexBuffer &IndexBuffer::operator=(IndexBuffer &&other) {
    if(this != &other){
        GLCall(glDeleteBuffers(1, &this->m_RendererId))
        this->m_RendererId = other.m_RendererId;
        this->m_Count = other.m_Count;
        other.m_RendererId = 0;
        other.m_Count = 0;
    }
    return *this;
}

IndexBuffer::~IndexBuffer() {
    GLCall(glDeleteBuffers(1, &m_RendererId))
}

IndexBuffer::IndexBuffer(const unsigned int *indices, unsigned int count): m_Count(count), m_RendererId(0) {
    ASSERT(sizeof(unsigned int) == sizeof(GLuint));

    GLCall(glGenBuffers(1, &m_RendererId));
    bind();
    GLCall(glBufferData(GL_ELEMENT_ARRAY_BUFFER, count*sizeof(unsigned int), indices, GL_STATIC_DRAW))
}

IndexBuffer::IndexBuffer(const std::vector<Triangle> &triangles): m_RendererId(0) {
    ASSERT(sizeof(unsigned int) == sizeof(GLuint))
    GLCall(glGenBuffers(1, &m_RendererId))
    bind();

    std::vector<unsigned int> data;
    for(Triangle t: triangles){
        data.emplace_back(t.vertexIndex0);
        data.emplace_back(t.vertexIndex1);
        data.emplace_back(t.vertexIndex2);
    }

    m_Count = (unsigned int) data.size();

    GLCall(glBufferData(GL_ELEMENT_ARRAY_BUFFER, data.size()*sizeof(unsigned int), &data.front(), GL_STATIC_DRAW))
}


void IndexBuffer::bind() const {
    GLCall(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_RendererId))
}



//void IndexBuffer::unbind() const {
//    GLCall(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
//}
