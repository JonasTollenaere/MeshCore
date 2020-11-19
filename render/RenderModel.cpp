//
// Created by Jonas on 18/11/2020.
//

#include "RenderModel.h"
#include "VertexBufferLayout.h"

RenderModel::RenderModel(const WorldSpaceMesh &worldSpaceMesh):
worldSpaceMesh(worldSpaceMesh),
indexBuffer(IndexBuffer(worldSpaceMesh.modelSpaceMesh.triangles)),
color(Color(1, 0, 0, 1))
{
    this->vertexBuffer = VertexBuffer(worldSpaceMesh);
    this->vertexArray = VertexArray(vertexBuffer);
    this->indexBuffer = IndexBuffer(worldSpaceMesh.modelSpaceMesh.triangles);
}

void RenderModel::draw(Shader& shader) const {

    ASSERT(sizeof(GLuint) == sizeof(unsigned int))
    shader.bind();
    shader.setUniform4fv("u_Color", this->color);
    vertexArray.bind();
    indexBuffer.bind();
    GLCall(glDrawElements(GL_TRIANGLES, indexBuffer.getCount(), GL_UNSIGNED_INT, nullptr))
}

void RenderModel::setColor(const Color &newColor) {
    this->color = newColor;
}
