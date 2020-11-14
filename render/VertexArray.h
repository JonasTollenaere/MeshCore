//
// Created by Jonas on 12/11/2020.
//

#ifndef MESHCORE2_VERTEXARRAY_H
#define MESHCORE2_VERTEXARRAY_H


#include "VertexBuffer.h"

class VertexBufferLayout;

class VertexArray {
private:
    unsigned int m_RendererId;
public:
    VertexArray();
    ~VertexArray();

    void addBuffer(const VertexBuffer& vertexBuffer, const VertexBufferLayout& layout);

    void bind() const;
    void unbind() const;
};


#endif //MESHCORE2_VERTEXARRAY_H
