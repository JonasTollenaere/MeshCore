//
// Created by Jonas on 1/12/2020.
//

#ifndef MESHCORE_RENDERMODEL_H
#define MESHCORE_RENDERMODEL_H

#include <glm/glm.hpp>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include "../core/WorldSpaceMesh.h"

typedef glm::vec4 Color;

class RenderModel: protected QOpenGLFunctions {
private:
    QOpenGLBuffer* indexBuffer;
    QOpenGLBuffer* vertexBuffer;
    QOpenGLVertexArrayObject* vertexArray;
    Transformation transformation;

    bool visible = true;
    bool wireframeEnabled = false;
    bool cullingEnabled = true;
    Color color;

public:
    void setTransformation(const Transformation &transformation);
    void setColor(const Color& c);
    void setVisible(bool visible);
    void setWireframeEnabled(bool showWireframe);
    void setCullingEnabled(bool cullingEnabled);

    [[nodiscard]] bool isVisible() const ;
    [[nodiscard]] bool isWireframeEnabled() const ;
    [[nodiscard]] bool isCullingEnabled() const ;

    RenderModel();
//    RenderModel(const RenderModel& other);
    RenderModel(RenderModel&& other) noexcept;
    RenderModel& operator=(RenderModel&& other) noexcept;
    ~RenderModel();

    explicit RenderModel(const WorldSpaceMesh& worldSpaceMesh);
    void draw(QOpenGLShaderProgram& shader, const glm::mat4& viewMatrix, const glm::mat4& projectionMatrix);
};


#endif //MESHCORE_RENDERMODEL_H
