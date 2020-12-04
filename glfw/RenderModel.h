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
    Color color;

public:
//    RenderModel();
//    RenderModel(const RenderModel& other);
    RenderModel(RenderModel&& other) noexcept;

    explicit RenderModel(const WorldSpaceMesh& worldSpaceMesh);
    void draw(Shader &shader, const glm::mat4 &projectionViewMatrix, const glm::vec3& viewSpaceLightDirection) const;

    void setColor(const Color& color);
//    void setShader(const Shader& shader);
};


#endif //MESHCORE2_RENDERMODEL_H
