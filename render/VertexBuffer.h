//
// Created by Jonas on 12/11/2020.
//

#ifndef MESHCORE_VERTEXBUFFER_H
#define MESHCORE_VERTEXBUFFER_H


#include <vector>
#include "../core/Vertex.h"
#include "../core/WorldSpaceMesh.h"

class VertexBuffer {
private:
    unsigned int m_RendererId;
public:
    VertexBuffer();
    VertexBuffer(const VertexBuffer& other);
    VertexBuffer(VertexBuffer&& other) noexcept ;
    VertexBuffer& operator=(VertexBuffer&& other) noexcept ;
    ~VertexBuffer();

    VertexBuffer(const void* data, unsigned int size);
    explicit VertexBuffer(const WorldSpaceMesh& worldSpaceMesh);

    void bind() const;
};

#endif //MESHCORE_VERTEXBUFFER_H
