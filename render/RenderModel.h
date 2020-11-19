//
// Created by Jonas on 18/11/2020.
//

#ifndef MESHCORE2_RENDERMODEL_H
#define MESHCORE2_RENDERMODEL_H


#include "../core/WorldSpaceMesh.h"
#include "VertexArray.h"
#include "IndexBuffer.h"
#include "Shader.h"

typedef glm::vec4 Color;

class RenderModel {
private:
    const WorldSpaceMesh& worldSpaceMesh;
    VertexArray vertexArray;
    IndexBuffer indexBuffer;
    VertexBuffer vertexBuffer;
    Color color;

public:
    explicit RenderModel(const WorldSpaceMesh& worldSpaceMesh);
    void draw(Shader& shader) const;

    void setColor(const Color& color);
//    void setShader(const Shader& shader);
};


#endif //MESHCORE2_RENDERMODEL_H
