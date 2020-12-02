//
// Created by Jonas on 12/11/2020.
//

#include "VertexArray.h"

#include <utility>
#include "Renderer.h"
#include "VertexBufferLayout.h"

VertexArray::VertexArray(): m_RendererId(0) {
    GLCall(glGenVertexArrays(1, &m_RendererId))
}

VertexArray::VertexArray(VertexBuffer vertexBuffer): m_RendererId(0), vertexBuffer(std::move(vertexBuffer)) {
    GLCall(glGenVertexArrays(1, &m_RendererId))
    VertexBufferLayout vertexBufferLayout;
    vertexBufferLayout.push<float>(3);
    vertexBufferLayout.push<float>(3);
    bind();
    this->vertexBuffer.bind();
    const std::vector<VertexBufferElement> elements = vertexBufferLayout.getElements();
    unsigned int offset = 0;
    for(unsigned int i=0; i< elements.size(); i++){
        const VertexBufferElement element = elements[i];
        GLCall(glEnableVertexAttribArray(i))
        GLCall(glVertexAttribPointer(i, element.count, element.type, element.normalized, vertexBufferLayout.getStride(), (void*) offset))
        offset += element.count * VertexBufferElement::getSizeOfType(element.type);
    }
}

VertexArray &VertexArray::operator=(VertexArray &&other) noexcept {
    if(this != &other){
        GLCall(glDeleteVertexArrays(1, &m_RendererId))
        this->m_RendererId = other.m_RendererId;
        this->vertexBuffer = std::move(other.vertexBuffer);
        other.m_RendererId = 0;
    }
    return *this;
}


VertexArray::~VertexArray() {
    GLCall(glDeleteVertexArrays(1, &m_RendererId))
}

void VertexArray::addBuffer(const VertexBuffer &vertexBuffer, const VertexBufferLayout &layout) {
    bind();
    vertexBuffer.bind();
    const std::vector<VertexBufferElement> elements = layout.getElements();
    unsigned int offset = 0;
    for(unsigned int i=0; i< elements.size(); i++){
        const VertexBufferElement element = elements[i];
        GLCall(glEnableVertexAttribArray(i))
        GLCall(glVertexAttribPointer(i, element.count, element.type, element.normalized, layout.getStride(), (void*) offset))
        offset += element.count * VertexBufferElement::getSizeOfType(element.type);
    }
}

void VertexArray::bind() const {
    GLCall(glBindVertexArray(m_RendererId))
}