//
// Created by Jonas on 1/12/2020.
//

#include "OpenGLModel.h"
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/normal.hpp>

#include "OpenGLWidget.h"
#include "ShaderProgramSource.h"
#include <QOpenGLShaderProgram>

OpenGLModel::OpenGLModel(const WorldSpaceMesh &worldSpaceMesh):
        worldSpaceMesh(worldSpaceMesh),
        color(Color(1,1,1,1)),
        vertexBuffer(new QOpenGLBuffer(QOpenGLBuffer::Type::VertexBuffer)),
        indexBuffer(new QOpenGLBuffer(QOpenGLBuffer::Type::IndexBuffer))
{
    const std::vector<Vertex> vertices = worldSpaceMesh.modelSpaceMesh.vertices;
    const std::vector<Triangle> triangles = worldSpaceMesh.modelSpaceMesh.triangles;

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

    vertexBuffer->create();
    vertexBuffer->bind();
    vertexBuffer->allocate(&data.front(), data.size() * sizeof(float));

    vertexArray->create();
    vertexArray->bind();
    QOpenGLVertexArrayObject::Binder vaoBinder(vertexArray);

    initializeOpenGLFunctions();
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    indexBuffer->create();
    indexBuffer->bind();
    indexBuffer->allocate(&indices.front(), indices.size() * sizeof(unsigned int));

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat),
                             nullptr);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat),
                          (void*) (3 * sizeof(GLfloat)));

    vertexBuffer->release();
}

void OpenGLModel::draw(QOpenGLShaderProgram& shader, const glm::mat4& viewMatrix, const glm::mat4& projectionMatrix) {
    vertexArray->bind();
    indexBuffer->bind();
    std::cout << "Size: " << indexBuffer->size() << std::endl;

    glm::mat4 projectionViewMatrix = projectionMatrix * viewMatrix;
    glm::vec3 viewSpaceLightDirection = glm::vec4(glm::vec3(0, 0, 1), 1.0f) * viewMatrix;
    const glm::vec3 modelLightDirection = glm::vec3(glm::vec4(viewSpaceLightDirection, 1.0f) * this->worldSpaceMesh.getModelTransformationMatrix());
    const glm::mat4 modelViewProjectionMatrix = projectionViewMatrix * this->worldSpaceMesh.getModelTransformationMatrix();

    shader.setUniformValue("u_ModelViewProjectionMatrix", QMatrix4x4(glm::value_ptr(modelViewProjectionMatrix)).transposed());
    shader.setUniformValue("u_Color", QVector4D(this->color.r, this->color.g, this->color.b, this->color.a));
    shader.setUniformValue("u_LightDirection", QVector3D(modelLightDirection.x, modelLightDirection.y, modelLightDirection.z));

    glDrawElements(GL_TRIANGLES, indexBuffer->size()/sizeof(unsigned int),  GL_UNSIGNED_INT, nullptr); // TODO pass amount of indices
}
