//
// Created by Jonas on 1/12/2020.
//

#ifndef MESHCORE_OPENGLMODEL_H
#define MESHCORE_OPENGLMODEL_H

#include "../core/WorldSpaceMesh.h"
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>

typedef glm::vec4 Color;

class OpenGLModel: protected QOpenGLFunctions {
private:
    const WorldSpaceMesh* worldSpaceMesh;
    QOpenGLBuffer* indexBuffer;
    QOpenGLBuffer* vertexBuffer;
    QOpenGLVertexArrayObject* vertexArray;
    Color color;
public:

    OpenGLModel();
//    OpenGLModel(const OpenGLModel& other);
    OpenGLModel(OpenGLModel&& other) noexcept;
    OpenGLModel& operator=(OpenGLModel&& other) noexcept;
    ~OpenGLModel();

    explicit OpenGLModel(const WorldSpaceMesh* worldSpaceMesh);
    void draw(QOpenGLShaderProgram& shader, const glm::mat4& viewMatrix, const glm::mat4& projectionMatrix);
};


#endif //MESHCORE_OPENGLMODEL_H
