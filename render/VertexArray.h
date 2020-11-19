//
// Created by Jonas on 12/11/2020.
//

#ifndef MESHCORE2_VERTEXARRAY_H
#define MESHCORE2_VERTEXARRAY_H


#include "VertexBuffer.h"
#include "../core/WorldSpaceMesh.h"
#include <vector>

class VertexBufferLayout;

class VertexArray {
private:
    unsigned int m_RendererId;
//    VertexBuffer vertexBuffer;
public:
    VertexArray();
    explicit VertexArray(const VertexBuffer &vertexBuffer);
    ~VertexArray();

    void addBuffer(const VertexBuffer& vertexBuffer, const VertexBufferLayout& layout);

    void bind() const;
    void unbind() const;
};


#endif //MESHCORE2_VERTEXARRAY_H
