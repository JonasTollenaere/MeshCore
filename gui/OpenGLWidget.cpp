//
// Created by Jonas on 30/11/2020.
//

#include "OpenGLWidget.h"
#include "../core/FileParser.h"
#include "ShaderProgramSource.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <QOpenGLShaderProgram>

OpenGLWidget::OpenGLWidget(QWidget *parent): QOpenGLWidget(parent) {}

void OpenGLWidget::initializeGL() {

    double distance = 20.0f;
    viewMatrix = glm::translate(glm::dmat4(1.0f), glm::dvec3(0.0f,0.0f, - distance));

    initializeOpenGLFunctions();

    std::cout << glGetString(GL_VERSION) << std::endl;

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_MULTISAMPLE);

    // Options (should be available in GUI, per model)
//    glClearColor(100.0f/255.0f, 149.0f/255.0f, 237.0f/255.0f, 1.0f);
    glEnable(GL_CULL_FACE);
//    GLCall(glDisable(GL_CULL_FACE))
    glCullFace(GL_BACK);

    const std::string path4 = "../../data/models/dragon.obj";
    const ModelSpaceMesh dragonMesh = FileParser::parseFile(path4);
    const WorldSpaceMesh dragonWorldSpaceMesh(dragonMesh);
    auto model = OpenGLModel(dragonWorldSpaceMesh);

    ShaderProgramSource shaderProgramSource = ShaderProgramSource::parseShader("../../render/res/shaders/Intermediate.shader");
    shader.addShaderFromSourceCode(QOpenGLShader::Vertex, shaderProgramSource.VertexSource);
    shader.addShaderFromSourceCode(QOpenGLShader::Fragment, shaderProgramSource.FragmentSource);
    shader.bindAttributeLocation("vertex", 0);
    shader.bindAttributeLocation("normal", 1);
    shader.link();
}

void OpenGLWidget::resizeGL(int w, int h) {
    std::cout << "Resize" << std::endl;
    this->width = w;
    this->height = h;
    projectionMatrix = glm::perspective(glm::radians(fieldOfView), float(width)/float(height), 0.1f, 10000.0f);
}

void OpenGLWidget::paintGL() {
    std::cout << "Paint" << std::endl;

    const std::string path4 = "../../data/models/dragon.obj";
    const ModelSpaceMesh dragonMesh = FileParser::parseFile(path4);
    const WorldSpaceMesh dragonWorldSpaceMesh(dragonMesh);
    auto model = OpenGLModel(dragonWorldSpaceMesh);

    this->clear();

    shader.bind();
    glm::vec3 viewSpaceLightDirection = glm::vec4(glm::vec3(0, 0, 1), 1.0f) * viewMatrix;

    const float ambientLighting = 0.05f;
    shader.setUniformValue("u_Ambient", ambientLighting);

    model.draw(shader, viewMatrix, projectionMatrix);
//    for(OpenGLModel& openGLModel: renderModels){
//        openGLModel.draw(shader, viewMatrix, projectionMatrix);
//    }
}

void OpenGLWidget::clear() {
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
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
    printf("\nkey event in OpenGLWidget: %i", event->key());
    std::cout << event->key() << std::endl;

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
