//
// Created by Jonas on 12/11/2020.
//

#ifndef MESHCORE_VERTEXBUFFER_H
#define MESHCORE_VERTEXBUFFER_H


#include <vector>
#include "../core/Vertex.h"
#include "../core/ModelSpaceMesh.h"

class VertexBuffer {
private:
    unsigned int m_RendererId;
public:
    VertexBuffer(const void* data, unsigned int size);
    VertexBuffer(const ModelSpaceMesh& mesh, const glm::vec4& color);
    ~VertexBuffer();

    void bind() const;
    void unbind() const;


};


#endif //MESHCORE_VERTEXBUFFER_H
