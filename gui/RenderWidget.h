//
// Created by Jonas on 30/11/2020.
//

#ifndef MESHCORE_RENDERWIDGET_H
#define MESHCORE_RENDERWIDGET_H

#include "RenderModel.h"
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QMouseEvent>
#include <QOpenGLShaderProgram>
#include <glm/glm.hpp>
#include <unordered_map>
#include <mutex>
#include <shared_mutex>


class RenderWidget: public QOpenGLWidget, protected QOpenGLFunctions {
private:
    float fieldOfView = 75.0f;
    int width;
    int height;
    QPoint lastMousePosition;

    glm::dmat4 viewMatrix;
    glm::dmat4 projectionMatrix;

    std::unordered_map<std::string, RenderModel> renderModelsMap;
    std::shared_mutex sharedMutex;


    QOpenGLShaderProgram shader;
//    QOpenGLShaderProgram axisShader;

public:
    explicit RenderWidget(QWidget *parent = nullptr);
    void resetView();
    void addWorldSpaceMesh(const WorldSpaceMesh& worldSpaceMesh);
    void addWorldSpaceMesh(const WorldSpaceMesh &worldSpaceMesh, const Color &color);
    void updateWorldSpaceMesh(const WorldSpaceMesh& worldSpaceMesh);

protected:

    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

    void mouseMoveEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent* event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseDoubleClickEvent(QMouseEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;

};

#endif //MESHCORE_RENDERWIDGET_H
