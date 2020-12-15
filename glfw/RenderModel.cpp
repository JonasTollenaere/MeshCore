//
// Created by Jonas on 18/11/2020.
//

#include "RenderModel.h"
#include "VertexBufferLayout.h"
#include <glm/gtx/normal.hpp>

RenderModel::RenderModel(const WorldSpaceMesh &worldSpaceMesh):
worldSpaceMesh(worldSpaceMesh),
color(Color(1,1,1,1))
{
    const std::vector<Vertex> vertices = worldSpaceMesh.getModelSpaceMesh().vertices;
    const std::vector<Triangle> triangles = worldSpaceMesh.getModelSpaceMesh().triangles;

    std::vector<unsigned int> indices;
    std::vector<float> data;
    for(unsigned int i=0; i<triangles.size(); i++) {

        Triangle t = triangles[i];
        std::vector<Vertex> triangleVertices;
        triangleVertices.emplace_back(vertices[t.vertexIndex0]);
        triangleVertices.emplace_back(vertices[t.vertexIndex1]);
        triangleVertices.emplace_back(vertices[t.vertexIndex2]);
        glm::vec3 normal = glm::triangleNormal(triangleVertices[0], triangleVertices[1], triangleVertices[2]);

        // We push the vertices once for each triangle because the normal is different for each triangle
        // (Adding up the normals doesn't provide visually satisfying results
        for(unsigned int j=0; j<3; j++){
            data.emplace_back(triangleVertices[j].x);
            data.emplace_back(triangleVertices[j].y);
            data.emplace_back(triangleVertices[j].z);
            data.emplace_back(normal.x);
            data.emplace_back(normal.y);
            data.emplace_back(normal.z);
            indices.emplace_back(i * 3 + j);
        }
    }

    VertexBufferLayout layout;
    layout.push<float>(3);
    layout.push<float>(3);
    VertexBuffer vertexBuffer = VertexBuffer(&data.front(), (unsigned int) data.size()*sizeof(float));
    this->vertexArray = VertexArray(std::move(vertexBuffer));
    this->indexBuffer = IndexBuffer(&indices.front(), (unsigned int) indices.size());
}

void RenderModel::draw(Shader& shader, const glm::mat4& projectionViewMatrix, const glm::vec3& viewSpaceLightDirection) const {

    ASSERT(sizeof(GLuint) == sizeof(unsigned int))
    shader.bind();
    shader.setUniform4fv("u_Color", this->color);
    const glm::mat4 modelViewProjectionMatrix = projectionViewMatrix * this->worldSpaceMesh.getModelTransformation();
    shader.setUniformMat4f("u_ModelViewProjectionMatrix", modelViewProjectionMatrix);
    const glm::vec3 modelLightDirection = glm::vec3(glm::vec4(viewSpaceLightDirection, 1.0f) * this->worldSpaceMesh.getModelTransformation());
    shader.setUniform3fv("u_LightDirection", modelLightDirection);
    const float ambientLighting = 0.05f;
    shader.setUniform1f("u_Ambient", ambientLighting);

    vertexArray.bind();
    indexBuffer.bind();
    GLCall(glDrawElements(GL_TRIANGLES, indexBuffer.getCount(), GL_UNSIGNED_INT, nullptr))
}

RenderModel::RenderModel(RenderModel &&other) noexcept: worldSpaceMesh(other.worldSpaceMesh) {
    this->vertexArray = std::move(other.vertexArray);
    this->indexBuffer = std::move(other.indexBuffer);
    this->color = other.color;
}

void RenderModel::setColor(const Color &c) {
    this->color = c;
}