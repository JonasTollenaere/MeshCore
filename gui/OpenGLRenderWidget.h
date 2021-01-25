//
// Created by Jonas on 30/11/2020.
//

#ifndef MESHCORE_OPENGLRENDERWIDGET_H
#define MESHCORE_OPENGLRENDERWIDGET_H

#include "RenderModel.h"
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QMouseEvent>
#include <QOpenGLShaderProgram>
#include <glm/glm.hpp>
#include <unordered_map>
#include <mutex>
#include <shared_mutex>



#define INITIAL_VIEW_DISTANCE 50.0f
#define INITIAL_FOV 86.0f

class OpenGLRenderWidget: public QOpenGLWidget, protected QOpenGLFunctions {
Q_OBJECT
private:
    float fieldOfView = INITIAL_FOV;
    int width;
    int height;
    QPoint lastMousePosition;

    glm::dmat4 viewMatrix;
    glm::dmat4 projectionMatrix;

    std::unordered_map<std::string, RenderModel> renderModelsMap;
    std::shared_mutex sharedMutex;

    QOpenGLShaderProgram betterShader;
    QOpenGLShaderProgram basicShader;

public:
    explicit OpenGLRenderWidget(QWidget *parent = nullptr);

private:
    void resetView();

public:
    void addWorldSpaceMesh(const WorldSpaceMesh& worldSpaceMesh);
    void addWorldSpaceMesh(const WorldSpaceMesh &worldSpaceMesh, const Color &color);
    void updateWorldSpaceMesh(const WorldSpaceMesh& worldSpaceMesh);
    friend class ApplicationWindow;

protected:

    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

    void mouseMoveEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent* event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseDoubleClickEvent(QMouseEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;

public slots:
    Q_INVOKABLE [[maybe_unused]] void addWorldSpaceMeshSlot(const WorldSpaceMesh &worldSpaceMesh, const Color &color);
};

#endif //MESHCORE_OPENGLRENDERWIDGET_H
