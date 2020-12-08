//
// Created by Jonas on 30/11/2020.
//

#include "RenderWidget.h"
#include "ShaderProgramSource.h"
#include <glm/gtc/matrix_transform.hpp>
#include <QOpenGLShaderProgram>

#define INITIAL_VIEW_DISTANCE 50.0f

RenderWidget::RenderWidget(QWidget *parent): QOpenGLWidget(parent) {}

void RenderWidget::initializeGL() {

    const std::thread::id MAIN_THREAD_ID = std::this_thread::get_id();
    std::cout << MAIN_THREAD_ID << std::endl;

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

void RenderWidget::resetView() {
    viewMatrix = glm::translate(glm::dmat4(1.0f), glm::dvec3(0.0f,0.0f, - INITIAL_VIEW_DISTANCE));
}

void RenderWidget::resizeGL(int w, int h) {
    this->width = w;
    this->height = h;
    projectionMatrix = glm::perspective(glm::radians(fieldOfView), float(width)/float(height), 0.1f, 10000.0f);
}

void RenderWidget::paintGL() {

    std::shared_lock<std::shared_mutex> lock(sharedMutex); // Lock with destructor that releases the mutex
    for(auto& [id, model]:  this->renderModelsMap){
        model.draw(shader, viewMatrix, projectionMatrix);
    }

}

void RenderWidget::mouseMoveEvent(QMouseEvent *event) {
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

void RenderWidget::wheelEvent(QWheelEvent *event) {
    auto factor = event->angleDelta().y() / 1200.0;
    auto distance = glm::length(glm::vec3(viewMatrix[3]));
    viewMatrix = glm::translate(viewMatrix, glm::dvec3(glm::inverse(viewMatrix) * glm::dvec4(glm::dvec3(0.0f, 0.0f, factor * distance), 0.0f)));
    this->update();
}

void RenderWidget::mousePressEvent(QMouseEvent *event) {
     lastMousePosition = event->pos();
     this->update();
     this->setFocus();
}

void RenderWidget::mouseDoubleClickEvent(QMouseEvent *event) {
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

void RenderWidget::keyPressEvent(QKeyEvent* event){
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

void RenderWidget::addWorldSpaceMesh(const WorldSpaceMesh& worldSpaceMesh) {
    this->addWorldSpaceMesh(worldSpaceMesh, Color(1,1,1,1));
}

void RenderWidget::addWorldSpaceMesh(const WorldSpaceMesh& worldSpaceMesh, const Color& color) {
    this->makeCurrent();
    auto model = RenderModel(worldSpaceMesh);
    model.setColor(color);
    std::unique_lock<std::shared_mutex> lock(sharedMutex); // Unique lock for writing with destructor that releases the mutex
    this->renderModelsMap[worldSpaceMesh.getId()] = std::move(model);
    this->update();
}

void RenderWidget::updateWorldSpaceMesh(const WorldSpaceMesh &worldSpaceMesh) {
    const std::string& id = worldSpaceMesh.getId();
    std::unique_lock<std::shared_mutex> lock(sharedMutex); // Unique lock for writing with destructor that releases the mutex
    if(renderModelsMap.find(id) != renderModelsMap.end()){
        renderModelsMap[id].setTransformation(worldSpaceMesh.getModelTransformationMatrix());
    }
    else{
        this->makeCurrent(); // TODO makeCurrent can only be executed on the GUI thread
        auto model = RenderModel(worldSpaceMesh);
        this->renderModelsMap[id] = std::move(model);
    }
    this->update();
}
