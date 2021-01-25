//
// Created by Jonas on 30/11/2020.
//

#include "OpenGLRenderWidget.h"
#include "ShaderProgramSource.h"
#include <QOpenGLShaderProgram>

OpenGLRenderWidget::OpenGLRenderWidget(QWidget *parent): QOpenGLWidget(parent) {}

void OpenGLRenderWidget::initializeGL() {

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
    betterShader.addShaderFromSourceCode(QOpenGLShader::Vertex, shaderProgramSource.VertexSource);
    betterShader.addShaderFromSourceCode(QOpenGLShader::Fragment, shaderProgramSource.FragmentSource);
    betterShader.bindAttributeLocation("vertex", 0);
    betterShader.bindAttributeLocation("normal", 1);
    betterShader.link();

    ShaderProgramSource basicShaderProgramSource = ShaderProgramSource::parseShader("../../glfw/res/shaders/Basic.shader");
    basicShader.addShaderFromSourceCode(QOpenGLShader::Vertex, basicShaderProgramSource.VertexSource);
    basicShader.addShaderFromSourceCode(QOpenGLShader::Fragment, basicShaderProgramSource.FragmentSource);
    basicShader.bindAttributeLocation("vertex", 0);
    basicShader.bindAttributeLocation("normal", 1);
    basicShader.link();

    currentShader = &betterShader;
}

void OpenGLRenderWidget::resetView() {
    viewMatrix = glm::translate(glm::dmat4(1.0f), glm::dvec3(0.0f,0.0f, - INITIAL_VIEW_DISTANCE));
}

void OpenGLRenderWidget::resizeGL(int w, int h) {
    this->width = w;
    this->height = h;
    projectionMatrix = glm::perspective(glm::radians(fieldOfView), float(width)/float(height), 0.1f, 10000.0f);
}

void OpenGLRenderWidget::paintGL() {
    std::shared_lock<std::shared_mutex> lock(sharedMutex); // A lock with destructor that releases the mutex
    for(auto& [id, model]:  this->renderModelsMap){
        model.draw(*currentShader, viewMatrix, projectionMatrix);
    }
}

void OpenGLRenderWidget::mouseMoveEvent(QMouseEvent *event) {
    int dx = event->x() - lastMousePosition.x();
    int dy = event->y() - lastMousePosition.y();
    lastMousePosition = event->pos();
    const double rotationSpeed = 0.01f;

    glm::dvec3 cameraUp = glm::vec3(0.0f, 1.0f,  0.0f);
    glm::dvec3 cameraRight = glm::vec3(1.0f, 0.0f, 0.0f);

    this->viewMatrix = glm::rotate(viewMatrix, rotationSpeed * dx, glm::dvec3(glm::inverse(viewMatrix) * glm::dvec4(cameraUp,0.0f)));
    this->viewMatrix = glm::rotate(viewMatrix, rotationSpeed * dy, glm::dvec3(glm::inverse(viewMatrix) * glm::dvec4(cameraRight, 0.0f)));
    this->update();
}

void OpenGLRenderWidget::wheelEvent(QWheelEvent *event) {
    auto factor = event->angleDelta().y() / 1200.0;
    auto distance = glm::length(glm::vec3(viewMatrix[3]));
    viewMatrix = glm::translate(viewMatrix, glm::dvec3(glm::inverse(viewMatrix) * glm::dvec4(glm::dvec3(0.0f, 0.0f, factor * distance), 0.0f)));
    this->update();
}

void OpenGLRenderWidget::mousePressEvent(QMouseEvent *event) {
     lastMousePosition = event->pos();
     this->update();
     this->setFocus();
}

void OpenGLRenderWidget::mouseDoubleClickEvent(QMouseEvent *event) {
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

void OpenGLRenderWidget::keyPressEvent(QKeyEvent* event){
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
    else if (key == Qt::Key_Left) {
        const double rotationSpeed = 0.05f;
        glm::dvec3 cameraUp = glm::vec3(0.0f, 1.0f,  0.0f);
        glm::dvec3 cameraRight = glm::vec3(1.0f, 0.0f, 0.0f);
        this->viewMatrix = glm::rotate(viewMatrix, - rotationSpeed, glm::dvec3(glm::inverse(viewMatrix) * glm::dvec4(cameraUp,0.0f)));
        this->update();
    }
    else if (key == Qt::Key_Right) {
        const double rotationSpeed = 0.05f;
        glm::dvec3 cameraUp = glm::vec3(0.0f, 1.0f,  0.0f);
        this->viewMatrix = glm::rotate(viewMatrix, rotationSpeed, glm::dvec3(glm::inverse(viewMatrix) * glm::dvec4(cameraUp,0.0f)));
        this->update();
    }
    else if (key == Qt::Key_Up) {
        const double rotationSpeed = 0.05f;
        glm::dvec3 cameraRight = glm::vec3(1.0f, 0.0f, 0.0f);
        this->viewMatrix = glm::rotate(viewMatrix, - rotationSpeed, glm::dvec3(glm::inverse(viewMatrix) * glm::dvec4(cameraRight,0.0f)));
        this->update();
    }
    else if (key == Qt::Key_Down) {
        const double rotationSpeed = 0.05f;
        glm::dvec3 cameraRight = glm::vec3(1.0f, 0.0f, 0.0f);
        this->viewMatrix = glm::rotate(viewMatrix, rotationSpeed, glm::dvec3(glm::inverse(viewMatrix) * glm::dvec4(cameraRight,0.0f)));
        this->update();
    }
}

/*** This is threadsafe, can be called from anywhere ***/
void OpenGLRenderWidget::addWorldSpaceMesh(const WorldSpaceMesh& worldSpaceMesh) {
    this->addWorldSpaceMesh(worldSpaceMesh, Color(1, 1, 1, 1));
}

/*** This is threadsafe, can be called from anywhere ***/
Q_DECLARE_METATYPE(WorldSpaceMesh)
Q_DECLARE_METATYPE(Color)
void OpenGLRenderWidget::addWorldSpaceMesh(const WorldSpaceMesh& worldSpaceMesh, const Color& color) {

    // This way the actions are executed on the main thread
    qRegisterMetaType<const WorldSpaceMesh&>();
    qRegisterMetaType<const Color&>();
    QMetaObject::invokeMethod(this, "addWorldSpaceMeshSlot", Qt::AutoConnection, Q_ARG(WorldSpaceMesh, worldSpaceMesh), Q_ARG(const Color&, color));
}

[[maybe_unused]] void OpenGLRenderWidget::addWorldSpaceMeshSlot(const WorldSpaceMesh& worldSpaceMesh, const Color& color){
    this->makeCurrent();
    auto model = RenderModel(worldSpaceMesh);
    model.setColor(color);
    std::unique_lock<std::shared_mutex> lock(sharedMutex); // Unique lock for writing with destructor that releases the mutex
    this->renderModelsMap[worldSpaceMesh.getId()] = std::move(model);
    this->update();
}

/*** This is threadsafe, can be called from anywhere ***/
void OpenGLRenderWidget::updateWorldSpaceMesh(const WorldSpaceMesh &worldSpaceMesh) {
    const std::string& id = worldSpaceMesh.getId();
    std::unique_lock<std::shared_mutex> lock(sharedMutex); // Unique lock for writing with destructor that releases the mutex
    if(renderModelsMap.find(id) != renderModelsMap.end()){
        renderModelsMap[id].setTransformation(worldSpaceMesh.getModelTransformation());
        this->update();
    }
}

// TODO setting per rendermodel
void OpenGLRenderWidget::toggleWireFrame() {
    this->makeCurrent();
    GLint currentPolygonMode[2];
    glGetIntegerv(GL_POLYGON_MODE, currentPolygonMode);
    if(currentPolygonMode[0] == GL_FILL){
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        currentShader = &basicShader;
    }
    else{
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glDisable(GL_CULL_FACE);
        currentShader = &betterShader;
    }
}

// TODO culling per rendermodel
void OpenGLRenderWidget::toggleCullFace() {
    this->makeCurrent();
    GLboolean enabled = glIsEnabled(GL_CULL_FACE);
    if(enabled){
        glDisable(GL_CULL_FACE);
    }
    else{
        glEnable(GL_CULL_FACE);
    }
}
