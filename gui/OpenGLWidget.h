//
// Created by Jonas on 30/11/2020.
//

#ifndef MESHCORE_OPENGLWIDGET_H
#define MESHCORE_OPENGLWIDGET_H

#include "OpenGLModel.h"
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QMouseEvent>
#include <QOpenGLShaderProgram>
#include <glm/glm.hpp>
#include <iostream>

class OpenGLWidget: public QOpenGLWidget, protected QOpenGLFunctions {
private:
    float fieldOfView = 75.0f;
    int width;
    int height;

    glm::dmat4 viewMatrix;
    glm::dmat4 projectionMatrix;

    std::vector<OpenGLModel> renderModels;
    QOpenGLShaderProgram shader;

public:
    explicit OpenGLWidget(QWidget *parent = nullptr);

protected:

    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

    void mouseMoveEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent* event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseDoubleClickEvent(QMouseEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;

private:
    void clear();
    QPoint lastMousePosition;


};


#endif //MESHCORE_OPENGLWIDGET_H
