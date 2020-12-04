//
// Created by Jonas on 30/11/2020.
//

#include "OpenGLWidget.h"
#include "ShaderProgramSource.h"
#include <glm/gtc/matrix_transform.hpp>
#include <QOpenGLShaderProgram>

#define INITIAL_VIEW_DISTANCE 50.0f

OpenGLWidget::OpenGLWidget(QWidget *parent): QOpenGLWidget(parent) {}

void OpenGLWidget::initializeGL() {

    viewMatrix = glm::translate(glm::dmat4(1.0f), glm::dvec3(0.0f,0.0f, - INITIAL_VIEW_DISTANCE));

    initializeOpenGLFunctions();

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_MULTISAMPLE);

    // Options (should be available in GUI, per model)
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    ShaderProgramSource shaderProgramSource = ShaderProgramSource::parseShader("../../glfw/res/shaders/Intermediate.shader");
    shader.addShaderFromSourceCode(QOpenGLShader::Vertex, shaderProgramSource.VertexSource);
    shader.addShaderFromSourceCode(QOpenGLShader::Fragment, shaderProgramSource.FragmentSource);
    shader.bindAttributeLocation("vertex", 0);
    shader.bindAttributeLocation("normal", 1);
    shader.link();
}

void OpenGLWidget::resetView() {
    viewMatrix = glm::translate(glm::dmat4(1.0f), glm::dvec3(0.0f,0.0f, - INITIAL_VIEW_DISTANCE));
}

void OpenGLWidget::resizeGL(int w, int h) {
    this->width = w;
    this->height = h;
    projectionMatrix = glm::perspective(glm::radians(fieldOfView), float(width)/float(height), 0.1f, 10000.0f);
}

void OpenGLWidget::paintGL() {

    for(OpenGLModel& model: this->renderModels){
        model.draw(shader, viewMatrix, projectionMatrix);
    }
}

void OpenGLWidget::mouseMoveEvent(QMouseEvent *event) {
    int dx = event->x() - lastMousePosition.x();
    int dy = event->y() - lastMousePosition.y();
    lastMousePosition = event->pos();
    const double rotationSpeed = 0.01f;

    glm::dvec3 cameraUp    = glm::vec3(0.0f, 1.0f,  0.0f);
    glm::dvec3 cameraLeft    = glm::vec3(- 1.0f, 0.0f,  0.0f);

    this->viewMatrix = glm::rotate(viewMatrix, rotationSpeed * dx, glm::dvec3(glm::inverse(viewMatrix) * glm::dvec4(cameraUp,0.0f)));
    this->viewMatrix = glm::rotate(viewMatrix, - rotationSpeed * dy, glm::dvec3(glm::inverse(viewMatrix) * glm::dvec4(cameraLeft,0.0f)));
    this->update();
}

void OpenGLWidget::wheelEvent(QWheelEvent *event) {
    auto factor = event->angleDelta().y() / 1200.0;
    auto distance = glm::length(glm::vec3(viewMatrix[3]));
    viewMatrix = glm::translate(viewMatrix, glm::dvec3(glm::inverse(viewMatrix) * glm::dvec4(glm::dvec3(0.0f, 0.0f, factor * distance), 0.0f)));
    this->update();
}

void OpenGLWidget::mousePressEvent(QMouseEvent *event) {
     lastMousePosition = event->pos();
     this->update();
     this->setFocus();
}

void OpenGLWidget::mouseDoubleClickEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton)
    {
        if(!this->parentWidget()->isMaximized()){
            this->parentWidget()->showMaximized();
        }
        else{
            this->parentWidget()->showNormal();
        }
    }
}

void OpenGLWidget::keyPressEvent(QKeyEvent* event){
    int key = event->key();
    if(key == Qt::Key_Plus){
        auto factor = 0.1;
        auto distance = glm::length(glm::vec3(viewMatrix[3]));
        viewMatrix = glm::translate(viewMatrix, glm::dvec3(glm::inverse(viewMatrix) * glm::dvec4(glm::dvec3(0.0f, 0.0f, factor * distance), 0.0f)));
        this->update();
    }
    else if (key == Qt::Key_Minus) {
        auto factor = 0.1;
        auto distance = glm::length(glm::vec3(viewMatrix[3]));
        viewMatrix = glm::translate(viewMatrix, glm::dvec3(glm::inverse(viewMatrix) * glm::dvec4(glm::dvec3(0.0f, 0.0f, - factor * distance), 0.0f)));
        this->update();
    }
}

void OpenGLWidget::addWorldSpaceMesh(const WorldSpaceMesh* worldspaceMesh) {
    this->makeCurrent();
    this->renderModels.emplace_back(OpenGLModel(worldspaceMesh));
}