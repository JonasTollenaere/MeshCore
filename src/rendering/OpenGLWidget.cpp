//
// Created by Jonas on 18/05/2022.
//

#include <qfiledialog.h>
#include "meshcore/rendering/OpenGLWidget.h"
#include "ShaderProgramSource.h"
#include "RenderMesh.h"
#include "RenderPlane.h"
#include "RenderSphere.h"
#include "include/meshcore/rendering/RenderWidget.h"
#include "Exception.h"
#include "src/external/gifencoder/GifEncoder.h"
#include "GifWrapper.h"
#include "meshcore/utility/io.h"
#include "RenderRay.h"
#include <glm/gtx/hash.hpp>
#include <QProgressDialog>

#ifndef MESHCORE_SHADER_DIR
#define MESHCORE_SHADER_DIR "undefined_shader_directory"
#endif

[[maybe_unused]] OpenGLWidget::OpenGLWidget(QWidget *parent): QOpenGLWidget(parent) {}

Q_DECLARE_METATYPE(Color)
Q_DECLARE_METATYPE(PhongMaterial)
Q_DECLARE_METATYPE(std::string)
Q_DECLARE_METATYPE(std::shared_ptr<WorldSpaceMesh>)
Q_DECLARE_METATYPE(RenderWidget*)
Q_DECLARE_METATYPE(AABB)
Q_DECLARE_METATYPE(Sphere)
Q_DECLARE_METATYPE(Transformation)
Q_DECLARE_METATYPE(glm::vec3)
Q_DECLARE_METATYPE(Plane)
Q_DECLARE_METATYPE(Ray)

void OpenGLWidget::initializeGL() {

    this->resetView();

    initializeOpenGLFunctions();

    qRegisterMetaType<Color>();
    qRegisterMetaType<PhongMaterial>();
    qRegisterMetaType<std::string>();
    qRegisterMetaType<std::shared_ptr<WorldSpaceMesh>>();
    qRegisterMetaType<RenderWidget*>();
    qRegisterMetaType<AABB>();
    qRegisterMetaType<Transformation>();
    qRegisterMetaType<glm::vec3>();
    qRegisterMetaType<Sphere>();
    qRegisterMetaType<Plane>();
    qRegisterMetaType<Ray>();
    qRegisterMetaType<size_t>("size_t");

    GL_CALL(glClearColor(0,0,0,1));

    GL_CALL(glEnable(GL_DEPTH_TEST));
    GL_CALL(glDepthFunc(GL_LESS));
    GL_CALL(glEnable(GL_BLEND));
    GL_CALL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
    GL_CALL(glEnable(GL_MULTISAMPLE));

    GL_CALL(glPixelStorei(GL_PACK_ALIGNMENT, 1));

    // Options (should be available in GUI, per model)
    GL_CALL(glEnable(GL_CULL_FACE));
    GL_CALL(glCullFace(GL_BACK));

    ShaderProgramSource basicShaderProgramSource = ShaderProgramSource::parseShader(MESHCORE_SHADER_DIR + std::string("Ambient.shader"));
    ambientShader = std::make_shared<QOpenGLShaderProgram>();
    ambientShader->addShaderFromSourceCode(QOpenGLShader::Vertex, basicShaderProgramSource.VertexSource);
    ambientShader->addShaderFromSourceCode(QOpenGLShader::Fragment, basicShaderProgramSource.FragmentSource);
    ambientShader->bindAttributeLocation("vertex", 0);
    ambientShader->bindAttributeLocation("normal", 1);
    ambientShader->link();

    ShaderProgramSource diffuseShaderProgramSource = ShaderProgramSource::parseShader(MESHCORE_SHADER_DIR + std::string("Diffuse.shader"));
    diffuseShader = std::make_shared<QOpenGLShaderProgram>();
    diffuseShader->addShaderFromSourceCode(QOpenGLShader::Vertex, diffuseShaderProgramSource.VertexSource);
    diffuseShader->addShaderFromSourceCode(QOpenGLShader::Fragment, diffuseShaderProgramSource.FragmentSource);
    diffuseShader->bindAttributeLocation("vertex", 0);
    diffuseShader->bindAttributeLocation("normal", 1);
    diffuseShader->link();

    ShaderProgramSource pongShaderProgramSource = ShaderProgramSource::parseShader(MESHCORE_SHADER_DIR + std::string("Phong.shader"));
    phongShader = std::make_shared<QOpenGLShaderProgram>();
    phongShader->addShaderFromSourceCode(QOpenGLShader::Vertex, pongShaderProgramSource.VertexSource);
    phongShader->addShaderFromSourceCode(QOpenGLShader::Fragment, pongShaderProgramSource.FragmentSource);
    phongShader->bindAttributeLocation("vertex", 0);
    phongShader->bindAttributeLocation("normal", 1);
    phongShader->link();

    ShaderProgramSource polyChromeShaderProgramSource = ShaderProgramSource::parseShader(MESHCORE_SHADER_DIR + std::string("PolyChrome.shader"));
    polyChromeShader = std::make_shared<QOpenGLShaderProgram>();
    polyChromeShader->addShaderFromSourceCode(QOpenGLShader::Vertex, polyChromeShaderProgramSource.VertexSource);
    polyChromeShader->addShaderFromSourceCode(QOpenGLShader::Fragment, polyChromeShaderProgramSource.FragmentSource);
    polyChromeShader->bindAttributeLocation("vertex", 0);
    polyChromeShader->bindAttributeLocation("normal", 1);
    polyChromeShader->bindAttributeLocation("color", 2);
    polyChromeShader->link();

    // Store the axis render models
    axisRenderModels.emplace_back(std::make_shared<RenderLine>(glm::vec3(0,0,0), glm::vec3(1e8,0,0), Transformation()));
    axisRenderModels.emplace_back(std::make_shared<RenderLine>(glm::vec3(0,0,0), glm::vec3(0,1e8,0), Transformation()));
    axisRenderModels.emplace_back(std::make_shared<RenderLine>(glm::vec3(0,0,0), glm::vec3(0,0,1e8), Transformation()));
    axisRenderModels[0]->setMaterial(PhongMaterial(Color::Red()));
    axisRenderModels[1]->setMaterial(PhongMaterial(Color::Green()));
    axisRenderModels[2]->setMaterial(PhongMaterial(Color::Blue()));
}

void OpenGLWidget::resetView() {

    this->viewTransformation = Transformation();
    viewTransformation.setPosition(glm::vec3(0.0f, 0.0f, -INITIAL_VIEW_DISTANCE));
    viewTransformation *= Transformation::fromRotationMatrix(glm::mat3({0, 0, 1, 1, 0, 0, 0, 1, 0})); // Default viewspace rotation: y-axis to the right and z-axis upwards

    this->update();
}

void OpenGLWidget::setView(size_t preset) {

    auto newViewTransformation = Transformation();
    newViewTransformation.setPosition(glm::vec3(0.0f, 0.0f, -INITIAL_VIEW_DISTANCE));
    switch (preset) {
        case 5:
        case 0: // Default front view: y-axis to the right and z-axis upwards
            newViewTransformation *= Transformation::fromRotationMatrix(glm::mat3({0, 0, 1, 1, 0, 0, 0, 1, 0}));
            break;
        case 4: // Default left view: x-axis to the right and z-axis upwards
            newViewTransformation *= Transformation::fromRotationMatrix(glm::mat3({1, 0, 0, 0, 0, -1, 0, 1, 0}));
            break;
        case 6: // Default right view: x-axis to the left and z-axis upwards
            newViewTransformation *= Transformation::fromRotationMatrix(glm::mat3({-1, 0, 0, 0, 0, 1, 0, 1, 0}));
            break;
        case 8: // Default top view: x-axis to the bottom and y-axis to the right
            newViewTransformation *= Transformation::fromRotationMatrix(glm::mat3({0, -1, 0, 1, 0, 0, 0, 0, 1}));
            break;
        case 2: // Default bottom view: x-axis to the top and y-axis to the right
            newViewTransformation *= Transformation::fromRotationMatrix(glm::mat3({0, 1, 0, 1, 0, 0, 0, 0, -1}));
            break;
        case 9: // Default top right front
            newViewTransformation *= Transformation::fromRotationMatrix(glm::mat3({-glm::sqrt(2.0f)/2.0f, -glm::sqrt(2.0f)/6.0f,      2.0f/3.0f,
                                                                                    glm::sqrt(2.0f)/2.0f, -glm::sqrt(2.0f)/6.0f,      2.0f/3.0f,
                                                                                    0.0f,                  2.0f*glm::sqrt(2.0f)/3.0f, 1.0f/3.0f}));
            break;
        case 7: // Default top left front
            newViewTransformation *= Transformation::fromRotationMatrix(glm::mat3({glm::sqrt(2.0f)/2.0f, -glm::sqrt(2.0f)/6.0f,      2.0f/3.0f,
                                                                                   glm::sqrt(2.0f)/2.0f,  glm::sqrt(2.0f)/6.0f,     -2.0f/3.0f,
                                                                                   0.0f,                  2.0f*glm::sqrt(2.0f)/3.0f, 1.0f/3.0f}));
            break;
        case 1: // Default bottom left front
            newViewTransformation *= Transformation::fromRotationMatrix(glm::mat3({glm::sqrt(2.0f)/2.0f,  glm::sqrt(2.0f)/6.0f,       2.0f/3.0f,
                                                                                   glm::sqrt(2.0f)/2.0f, -glm::sqrt(2.0f)/6.0f,      -2.0f/3.0f,
                                                                                   0.0f,                  2.0f*glm::sqrt(2.0f)/3.0f, -1.0f/3.0f}));
            break;
        case 3: // Default bottom right front
            newViewTransformation *= Transformation::fromRotationMatrix(glm::mat3({-glm::sqrt(2.0f)/2.0f, glm::sqrt(2.0f)/6.0f,       2.0f/3.0f,
                                                                                    glm::sqrt(2.0f)/2.0f, glm::sqrt(2.0f)/6.0f,       2.0f/3.0f,
                                                                                    0.0f,                 2.0f*glm::sqrt(2.0f)/3.0f, -1.0f/3.0f}));
            break;
        default:
            // Do nothing
            return;
            break;
    }

    // Animate the view change
    this->makeCurrent();
    const auto initialViewTransformation = this->viewTransformation;
    auto steps = 30;
    for(auto i=0;i<steps;i++){

        QCoreApplication::processEvents();

        // Sleep for 16 ms;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

        // Interpolate the view matrix
        float t = float(i)/(float(steps) - 1); // Linear interpolation
        this->viewTransformation = Transformation::interpolate(initialViewTransformation, newViewTransformation, t);

        GL_CALL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));
        this->update();
    }

    this->viewTransformation = newViewTransformation;
    GL_CALL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));
    this->update();
}

void OpenGLWidget::resizeGL(int w, int h) {
    this->width = w * this->devicePixelRatio();
    this->height = h * this->devicePixelRatio();

    this->calculateProjectionMatrix();
}

void OpenGLWidget::calculateProjectionMatrix(){
    if(this->usePerspective){
        this->projectionMatrix = glm::perspective(glm::radians(INITIAL_FOV), float(width)/float(height), 0.001f, 10000.0f);
    }
    else{
        float halfWidth = float(width)/20.0f;
        float halfHeight = float(height)/20.0f;
        this->projectionMatrix = glm::ortho(-halfWidth, halfWidth, -halfHeight, halfHeight, -1000.0f, 1000.0f);
    }
}

void OpenGLWidget::paintGL() {

    //
    GL_CALL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));

    if(this->axisEnabled){
        auto normalizedViewTransformation = viewTransformation;
        normalizedViewTransformation.setScale(1.0f);
        for (const auto &axisRenderModel: axisRenderModels){
            axisRenderModel->draw(this, normalizedViewTransformation.getMatrix(), projectionMatrix, lightMode);
        }
    }

    for(auto& renderModel: this->sortedRenderModels){
        renderModel->draw(this, viewTransformation.getMatrix(), projectionMatrix, lightMode);
    }
}

void OpenGLWidget::mouseMoveEvent(QMouseEvent *event) {
    int dx = event->x() - lastMousePosition.x();
    int dy = event->y() - lastMousePosition.y();
    lastMousePosition = event->pos();
    const float rotationSpeed = 0.01f;

    glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f,  0.0f);
    glm::vec3 cameraRight = glm::vec3(1.0f, 0.0f, 0.0f);

    this->viewTransformation.factorRotation(Quaternion(viewTransformation.inverseTransformVector(cameraUp), rotationSpeed * float(dx)));
    this->viewTransformation.factorRotation(Quaternion(viewTransformation.inverseTransformVector(cameraRight), rotationSpeed * float(dy)));

    this->update();
}

void OpenGLWidget::wheelEvent(QWheelEvent *event) {
    auto factor = float(event->angleDelta().y()) / 1200.0f;

    viewTransformation.factorScale(1 + factor);

    this->update();
}

void OpenGLWidget::mousePressEvent(QMouseEvent *event) {
    lastMousePosition = event->pos();
    this->update();
    this->setFocus();
}

void OpenGLWidget::mouseDoubleClickEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton) {
        QWidget* window = this->parentWidget();
        while(!window->isWindow()){
            window = window->parentWidget();
        }

        if(!window->isMaximized()){
            window->showMaximized();
        }
        else{
            window->showNormal();
        }
    }
}

void OpenGLWidget::keyPressEvent(QKeyEvent* event){
    const auto key = event->key();
    const auto distance = glm::length(viewTransformation.getPosition());
    if(key == Qt::Key_Plus){
        float zoomFactor = 0.1f;
        viewTransformation.factorScale(1 + zoomFactor);
        this->update();
    }
    if (key == Qt::Key_Minus) {
        float zoomFactor = 0.1f;
        viewTransformation.factorScale(1 - zoomFactor);
        this->update();
    }
    if (key == Qt::Key_Left) {
        const float rotationSpeed = glm::pi<float>()/60.0f;
        const glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f,  0.0f);
        Quaternion rotation = Quaternion(viewTransformation.inverseTransformVector(cameraUp), -rotationSpeed);
        this->viewTransformation.factorRotation(rotation);
        this->update();
    }
    if (key == Qt::Key_Right) {
        const float rotationSpeed = glm::pi<float>()/60.0f;
        const glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f,  0.0f);
        Quaternion rotation = Quaternion(viewTransformation.inverseTransformVector(cameraUp), rotationSpeed);
        this->viewTransformation.factorRotation(rotation);
        this->update();
    }
    if (key == Qt::Key_Up) {
        const float rotationSpeed = glm::pi<float>()/60.0f;
        const glm::vec3 cameraRight = glm::vec3(1.0f, 0.0f, 0.0f);
        Quaternion rotation = Quaternion(viewTransformation.inverseTransformVector(cameraRight), -rotationSpeed);
        this->viewTransformation.factorRotation(rotation);
        this->update();
    }
    if (key == Qt::Key_Down) {
        const float rotationSpeed = glm::pi<float>()/60.0f;
        const glm::vec3 cameraRight = glm::vec3(1.0f, 0.0f, 0.0f);
        Quaternion rotation = Quaternion(viewTransformation.inverseTransformVector(cameraRight), rotationSpeed);
        this->viewTransformation.factorRotation(rotation);
        this->update();
    }
    if (key == Qt::Key_A) {
        const float rotationSpeed = glm::pi<float>()/60.0f;
        const glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, 1.0f);
        Quaternion rotation = Quaternion(viewTransformation.inverseTransformVector(cameraFront), rotationSpeed);
        this->viewTransformation.factorRotation(rotation);
        this->update();
    }
    if (key == Qt::Key_E) {
        const float rotationSpeed = glm::pi<float>()/60.0f;
        const glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, 1.0f);
        Quaternion rotation = Quaternion(viewTransformation.inverseTransformVector(cameraFront), -rotationSpeed);
        this->viewTransformation.factorRotation(rotation);
        this->update();
    }
    if (key == Qt::Key_Z) {

        const float translationSpeed = 0.025f * distance;
        const glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f,  0.0f);
        Transformation translation;
        translation.setPosition(-translationSpeed * viewTransformation.inverseTransformVector(cameraUp));
        this->viewTransformation *= translation;
        this->update();
    }
    if (key == Qt::Key_S) {
        const float translationSpeed = 0.025f * distance;
        const glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f,  0.0f);
        Transformation translation;
        translation.setPosition(translationSpeed * viewTransformation.inverseTransformVector(cameraUp));
        this->viewTransformation *= translation;

        this->update();
    }
    if (key == Qt::Key_Q) {
        const float translationSpeed = 0.025f * distance;
        const glm::vec3 cameraRight = glm::vec3(1.0f, 0.0f, 0.0f);
        Transformation translation;
        translation.setPosition(translationSpeed * viewTransformation.inverseTransformVector(cameraRight));
        this->viewTransformation *= translation;
        this->update();
    }
    if (key == Qt::Key_D) {
        const float translationSpeed = 0.025f * distance;
        const glm::vec3 cameraRight = glm::vec3(1.0f, 0.0f, 0.0f);
        Transformation translation;
        translation.setPosition(-translationSpeed * viewTransformation.inverseTransformVector(cameraRight));
        this->viewTransformation *= translation;
        this->update();
    }
}

void OpenGLWidget::setLightMode(bool newLightMode){
    this->lightMode = newLightMode;
    this->makeCurrent();
    if(this->lightMode){
        GL_CALL(glClearColor(1,1,1,1));
    }
    else{
        GL_CALL(glClearColor(0,0,0,1));
    }

    for (const auto &groupEntry : this->groupedRenderModelsMap){
        for (const auto &modelEntry: groupEntry.second){
            if(auto renderMesh = std::dynamic_pointer_cast<RenderMesh>(modelEntry.second)){
                if(renderMesh->getMaterial().getDiffuseColor().a<1.0){
                    renderMesh->setCullingEnabled(this->lightMode);
                }
            }
        }
    }
    this->parentWidget()->update();
    this->update();
}

void OpenGLWidget::captureAnimationSlot() {

    this->makeCurrent();
    auto initialViewTransformation = this->viewTransformation;
    QString fileName = QFileDialog::getSaveFileName(this, QString("Save animation"), "example", QString("GIF (*.gif)"));

    // Check if the user has selected a file
    if (fileName.isEmpty()) {
        return;
    }

    // Calculate how many steps we need to take to get a full rotation, in radians
    const float fullRotation = 2 * glm::pi<float>();
    int delay = 5;
    int steps = 200;
    const float rotationSpeed = fullRotation / float(steps);
    const glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f,  0.0f);

    // Show processing bar
    QProgressDialog progressDialog("Exporting GIF animation...", "Abort", 0, steps, this);
    progressDialog.setWindowModality(Qt::ApplicationModal);
    progressDialog.show();

    // Initialise the gif encoder
    GifWrapper gifWrapper(width, height);
    gifWrapper.open(fileName.toStdString());

    for(int i=0;i<steps;i++){

        // Check if aborted
        if(progressDialog.wasCanceled()){
            gifWrapper.close();
            viewTransformation = initialViewTransformation;
            GL_CALL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));
            this->update();
            QFile::remove(fileName);
            progressDialog.close();
            return;
        }

        progressDialog.setValue(i);

        this->viewTransformation.factorRotation(Quaternion(viewTransformation.inverseTransformVector(cameraUp), -rotationSpeed));
        GL_CALL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));
        this->update();
        auto capture = this->grabFramebuffer();

        gifWrapper.appendFrame(capture, delay);
    }
    progressDialog.setValue(steps);

    // Close the file handle and free memory.
    gifWrapper.close();
    viewTransformation = initialViewTransformation;
    GL_CALL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));
    this->update();
    progressDialog.close();
}

void OpenGLWidget::captureSceneSlot() {

    this->makeCurrent();

    // We grab the frame before the user inputs the file name
    GL_CALL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));
    this->update();
    auto capture = this->grabFramebuffer();

    QString fileName = QFileDialog::getSaveFileName(this, QString("Save screenshot"), "example", QString("Image Files (*.png)"));

    // Check if the user has selected a file
    if (fileName.isEmpty()) {
        return;
    }

    // Eliminate the alpha channel
    QImage rgbImage(capture.size(), QImage::Format_ARGB32);
    for (int x = 0; x < width; ++x){
        for (int y = 0; y < height; ++y){
            auto rgb = capture.pixel(x, y);
            rgbImage.setPixelColor(x,y, QColor(rgb));
        }
    }
    rgbImage.save(fileName);
}

void OpenGLWidget::captureSceneToFileSlot(const QString& fileName) {
    this->makeCurrent();
    GL_CALL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));
    this->update();
    auto capture = this->grabFramebuffer();
    std::cout<< fileName.toStdString() << std::endl;

    // Eliminate the alpha channel
    QImage rgbImage(capture.size(), QImage::Format_ARGB32);
    for (int x = 0; x < width; ++x){
        for (int y = 0; y < height; ++y){
            auto rgb = capture.pixel(x, y);
            rgbImage.setPixelColor(x,y, QColor(rgb));
        }
    }
    rgbImage.save(fileName);
}

bool OpenGLWidget::isUsePerspective() const {
    return usePerspective;
}

void OpenGLWidget::setUsePerspective(bool newUsePerspective) {
    OpenGLWidget::usePerspective = newUsePerspective;
    this->calculateProjectionMatrix();
}

bool OpenGLWidget::isLightMode() const {
    return lightMode;
}

bool OpenGLWidget::isAxisEnabled() const {
    return axisEnabled;
}

void OpenGLWidget::setAxisEnabled(bool enabled) {
    OpenGLWidget::axisEnabled = enabled;
    this->update();
}

void OpenGLWidget::openAnimationSlot(const std::string& path) {
    if (gifWrapper) {
        std::cout << "Open animation is closed before creating new one." << std::endl;
        gifWrapper->close();
    }

    gifWrapper = std::make_shared<GifWrapper>(width, height);
    gifWrapper->open(path);
}

void OpenGLWidget::appendAnimationSlot() {
    if (!gifWrapper) {
        std::cout << "No animation to append to." << std::endl;
        return;
    }
    auto capture = this->grabFramebuffer();
    int delay = 5;
    gifWrapper->appendFrame(capture, delay);
}

void OpenGLWidget::closeAnimationSlot() {
    if (!gifWrapper) {
        std::cout << "No animation to save." << std::endl;
        return;
    }

    gifWrapper->close();
    gifWrapper = nullptr;
}

std::unordered_map<std::string, std::shared_ptr<AbstractRenderModel>>& OpenGLWidget::getOrInsertRenderModelsMap(const std::string& group) const {

    // Find the group
    auto iterator = groupedRenderModelsMap.find(group);

    // Add new group if not found
    if(iterator == groupedRenderModelsMap.end()){
        iterator = groupedRenderModelsMap.insert({group, {}}).first;
    }
    return iterator->second;
}

void OpenGLWidget::renderWorldSpaceMeshSlot(const std::string &group, const std::shared_ptr<WorldSpaceMesh> &worldSpaceMesh, const PhongMaterial &material, RenderWidget* renderWidget){

    // Find the group
    auto& renderModelsMap = this->getOrInsertRenderModelsMap(group);

    // Find the model in this group
    auto modelIterator = renderModelsMap.find(worldSpaceMesh->getId());
    if(modelIterator == renderModelsMap.end()){

        this->makeCurrent();

        // No entry present yet, create new render Model
        auto renderMesh = std::make_shared<RenderMesh>(*worldSpaceMesh);

        // Insert it in the renderModelsMap
        modelIterator = renderModelsMap.insert({worldSpaceMesh->getId(), renderMesh}).first;

        // Add required listeners
        this->addRenderModelListeners(group, renderMesh);

        // Add control widget to the renderWidget
        renderWidget->addControlWidget(group, renderMesh);

        this->updateSortedRenderModels();
    }

    // Set the color
    modelIterator->second->setMaterial(material);

    // Update the transformation
    modelIterator->second->setTransformation(worldSpaceMesh->getModelTransformation());

    this->update();
}

void OpenGLWidget::clear() {
    this->groupedRenderModelsMap.clear();
    this->updateSortedRenderModels();
    this->update();
}

void OpenGLWidget::clearGroup(const std::string &group) {
    this->groupedRenderModelsMap.erase(group);
    this->updateSortedRenderModels();
    this->update();
}

void OpenGLWidget::setGroupVisible(const std::string &group, bool visible) {
    for (const auto &modelEntry: this->groupedRenderModelsMap[group]){
        modelEntry.second->setVisible(visible);
    }
    this->update();
}

void OpenGLWidget::updateSortedRenderModels(){

    // Add all models to the vector
    this->sortedRenderModels.clear();
    for(auto& groupEntry: this->groupedRenderModelsMap){
        for (const auto &renderModelEntry: groupEntry.second){
            this->sortedRenderModels.emplace_back(renderModelEntry.second);
        }
    }

    // Sort them by transparency
    std::sort(this->sortedRenderModels.begin(), this->sortedRenderModels.end(), [](const std::shared_ptr<AbstractRenderModel>& a, const std::shared_ptr<AbstractRenderModel>& b){
        return a->getMaterial().getDiffuseColor().a > b->getMaterial().getDiffuseColor().a;
    });
}

void OpenGLWidget::renderBoxSlot(const std::string &group, const std::string &name, const AABB &aabb, const Transformation& transformation, const PhongMaterial& material, RenderWidget *renderWidget) {

    // Find the group
    auto& renderModelsMap = this->getOrInsertRenderModelsMap(group);

    // Find the model in this group
//    auto hash = std::hash<AABB>{}(aabb);
//    glm::detail::hash_combine(hash, std::hash<glm::mat4>{}(transformation.getMatrix()));
    auto hash = std::hash<AABB>{}(aabb);
    glm::detail::hash_combine(hash, std::hash<std::string>{}(name));
    auto renderId = std::to_string(hash);
    auto modelIterator = renderModelsMap.find(renderId);

    // Create a new model if not found
    if(modelIterator == renderModelsMap.end()){

        this->makeCurrent();

        // No entry present yet, create new render Model
        auto renderAABB = std::make_shared<RenderAABB>(aabb, transformation);

        // Insert it in the renderModelsMap
        modelIterator = renderModelsMap.insert({renderId, renderAABB}).first;

        // Add required listeners
        this->addRenderModelListeners(group, renderAABB);

        // Add control widget to the renderWidget
        renderWidget->addControlWidget(group, renderAABB);

        // Set the color
        modelIterator->second->setMaterial(PhongMaterial(Color::White()));
        modelIterator->second->setName(name);

        this->updateSortedRenderModels();
    }

    // Update the color
    modelIterator->second->setMaterial(material);

    // Update the transformation
    modelIterator->second->setTransformation(transformation);

    this->update();

}

void OpenGLWidget::renderSphereSlot(const std::string &group, const std::string& name, const Sphere &sphere, const PhongMaterial& material, RenderWidget *renderWidget) {

    // Find the group
    auto& renderModelsMap = this->getOrInsertRenderModelsMap(group);

    // Find the model in this group
    auto renderId = std::to_string(std::hash<Sphere>{}(sphere));
    auto modelIterator = renderModelsMap.find(renderId);

    // Create a new model if not found
    if(modelIterator == renderModelsMap.end()){

        this->makeCurrent();

        // No entry present yet, create new render Model
        auto renderSphere = std::make_shared<RenderSphere>(sphere, Transformation());

        // Insert it in the renderModelsMap
        modelIterator = renderModelsMap.insert({renderId, renderSphere}).first;

        // Add required listeners
        this->addRenderModelListeners(group, renderSphere);

        // Add control widget to the renderWidget
        renderWidget->addControlWidget(group, renderSphere);

        // Set the color
        modelIterator->second->setMaterial(material);
        modelIterator->second->setName(name);

        this->updateSortedRenderModels();
    }

    // Update the transformation
    modelIterator->second->setTransformation(Transformation());

    this->update();

}

void OpenGLWidget::renderTriangleSlot(const std::string &group, const std::string &name, const VertexTriangle &triangle,
                                      const PhongMaterial &material, RenderWidget *renderWidget) {

    // Find the group
    auto& renderModelsMap = this->getOrInsertRenderModelsMap(group);

    // Find the model in this group
    auto renderId = std::to_string(std::hash<VertexTriangle>{}(triangle));
    auto modelIterator = renderModelsMap.find(renderId);

    // Create a new model if not found
    if(modelIterator == renderModelsMap.end()){

        this->makeCurrent();

        // No entry present yet, create new render Model
//        auto renderTriangle = std::make_shared<RenderTriangle>(triangle, Transformation(), this->ambientShader, this->diffuseShader);
        // TODO create a renderTriangle or handle it as a mesh
    }

    // Update the transformation
    modelIterator->second->setTransformation(Transformation());

    this->update();
}

void OpenGLWidget::renderLineSlot(const std::string &group, const std::string &name, const Vertex &start,
                                  const Vertex &end, const PhongMaterial &material, RenderWidget *renderWidget) {

    // Find the group
    auto& renderModelsMap = this->getOrInsertRenderModelsMap(group);

    // Find the model in this group
    auto renderId = std::to_string(std::hash<glm::vec3>{}(start)) + std::to_string(std::hash<glm::vec3>{}(end));
    auto modelIterator = renderModelsMap.find(renderId);

    // Create a new model if not found
    if(modelIterator == renderModelsMap.end()){

        this->makeCurrent();

        // No entry present yet, create new render Model
        auto renderLine = std::make_shared<RenderLine>(start, end, Transformation());

        // Insert it in the renderModelsMap
        modelIterator = renderModelsMap.insert({renderId, renderLine}).first;

        // Add required listeners
        this->addRenderModelListeners(group, renderLine);

        // Add control widget to the renderWidget
        renderWidget->addControlWidget(group, renderLine);

        // Set the color
        modelIterator->second->setMaterial(material);
        modelIterator->second->setName(name);

        this->updateSortedRenderModels();
    }

    // Update the transformation
    modelIterator->second->setTransformation(Transformation());

    this->update();

}

const std::shared_ptr<QOpenGLShaderProgram> &OpenGLWidget::getAmbientShader() const {
    return ambientShader;
}

const std::shared_ptr<QOpenGLShaderProgram> &OpenGLWidget::getDiffuseShader() const {
    return diffuseShader;
}

const std::shared_ptr<QOpenGLShaderProgram> &OpenGLWidget::getPhongShader() const {
    return phongShader;
}

const std::shared_ptr<QOpenGLShaderProgram> &OpenGLWidget::getPolyChromeShader() const {
    return polyChromeShader;
}

void OpenGLWidget::addOrUpdateRenderModelSlot(const std::string& group, const std::string& id, std::shared_ptr<AbstractRenderModel> renderModel, RenderWidget* renderWidget) {
    // Find the group
    auto& renderModelsMap = this->getOrInsertRenderModelsMap(group);

    // Find the model in this group
    auto modelIterator = renderModelsMap.find(id);
    if(modelIterator == renderModelsMap.end()){

        this->makeCurrent();

        // No entry present yet, insert new renderModel it in the renderModelsMap
        modelIterator = renderModelsMap.insert({id, renderModel}).first;

        // Add required listeners
        this->addRenderModelListeners(group, renderModel);

        // Add control widget to the renderWidget
        renderWidget->addControlWidget(group, renderModel);

        this->updateSortedRenderModels();
    }

    modelIterator->second->setTransformation(renderModel->getTransformation());

    this->update();
}

void OpenGLWidget::addGroupListener(const std::string &group, const std::shared_ptr<AbstractRenderGroupListener> &listener) {
    this->groupListeners[group].emplace_back(listener);
}

void OpenGLWidget::removeGroupListener(const std::string &group,const std::shared_ptr<AbstractRenderGroupListener> &listener) {
    this->groupListeners[group].erase(std::remove(this->groupListeners[group].begin(), this->groupListeners[group].end(), listener), this->groupListeners[group].end());
}

void OpenGLWidget::addRenderModelListeners(const std::string &group, const std::shared_ptr<AbstractRenderModel> &renderModel) {

    // Add listener to redraw when model is changed
    const auto listener = std::make_shared<SimpleRenderModelListener>();
    listener->setOnMaterialChanged([this](const PhongMaterial &oldMaterial, const PhongMaterial &newMaterial) {
        if (oldMaterial.getDiffuseColor().a != newMaterial.getDiffuseColor().a) {
            this->updateSortedRenderModels();
        }
    });

    // Check visibility of all models in this group when changed
    auto onVisibleChanged = [this, group](bool oldVisible, bool newVisible){
        // Test if all models in this group have equal visibility
        bool uniform = true;
        for (const auto &renderModelInGroup: this->groupedRenderModelsMap[group]){
            if(renderModelInGroup.second->isVisible() != newVisible){
                uniform = false;
            }
        }

        // Notify group listeners accordingly
        for (const auto &groupListener: this->groupListeners[group]){
            groupListener->notifyVisible(newVisible, uniform);
        }
    };
    onVisibleChanged(renderModel->isVisible(), renderModel->isVisible()); // Trigger this listener once to keep group listeners in sync
    listener->setOnVisibleChanged(onVisibleChanged);

    listener->setOnChanged([this](){
        this->update();
    });
    renderModel->addListener(listener);
}

int OpenGLWidget::getWidth() const {
    return width;
}

int OpenGLWidget::getHeight() const {
    return height;
}

std::vector<uint8_t> OpenGLWidget::capturePixelBufferSlot() {

    this->makeCurrent();
    std::vector<uint8_t> pixelBuffer(width * height * 4, 0);
    GL_CALL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));
    this->update();
    auto capture = this->grabFramebuffer();
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            auto rgb = capture.pixel(x, y);
            pixelBuffer[4 * (y * width + x) + 0] = (rgb >> 16) & 0xFF;
            pixelBuffer[4 * (y * width + x) + 1] = (rgb >> 8) & 0xFF;
            pixelBuffer[4 * (y * width + x) + 2] = (rgb >> 0) & 0xFF;
            pixelBuffer[4 * (y * width + x) + 3] = 255;
        }

    }
    return pixelBuffer;
}

const Transformation &OpenGLWidget::getViewTransformation() const {
    return viewTransformation;
}

void OpenGLWidget::setViewTransformation(const Transformation &newViewTransformation) {
    OpenGLWidget::viewTransformation = newViewTransformation;
}

void OpenGLWidget::captureLinearAnimationSlot(const Transformation &initialViewTransformation,
                                              const Transformation &finalViewTransformation,
                                              const KeyFrame &initialKeyFrame, const KeyFrame &finalKeyFrame,
                                              const QString &fileName, int steps, int delay,
                                              RenderWidget *renderWidget) {
    // Animation time (s) = delay * steps / 100
    this->makeCurrent();

    // Show processing bar
    QProgressDialog progressDialog("Exporting " + fileName + " animation...", "Abort", 0, steps, this);
    progressDialog.setWindowModality(Qt::ApplicationModal);
    progressDialog.show();

    // Initialise the gif encoder
    GifWrapper gifWrapper(width, height);
    gifWrapper.open(fileName.toStdString());

    for(int i=0;i<steps;i++){
        // Check if aborted
        if(progressDialog.wasCanceled()){
            gifWrapper.close();
            GL_CALL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));
            this->update();
            QFile::remove(fileName);
            progressDialog.close();
            return;
        }

        progressDialog.setValue(i);

        // Interpolate the view matrix
        float t = float(i)/(steps - 1); // Linear interpolation
        this->viewTransformation = Transformation::interpolate(initialViewTransformation, finalViewTransformation, t);

        // Do the linear interpolation between transformations, color and viewMatrix for each object
        assert(initialKeyFrame.getObjects().size() == finalKeyFrame.getObjects().size());

        for (int objectId = 0; objectId < initialKeyFrame.getObjects().size(); ++objectId){
            const auto& initialKeyFrameObject = initialKeyFrame.getObjects()[objectId];
            const auto& finalKeyFrameObject = finalKeyFrame.getObjects()[objectId];

            assert(initialKeyFrameObject.mesh->getId() == finalKeyFrameObject.mesh->getId());
            assert(initialKeyFrameObject.group == finalKeyFrameObject.group);
            const auto& group = initialKeyFrameObject.group;
            const auto& mesh = initialKeyFrameObject.mesh;

            auto interpolatedMaterial = PhongMaterial::interpolate(initialKeyFrameObject.material, finalKeyFrameObject.material, t);
            auto interpolatedTransformation = Transformation::interpolate(initialKeyFrameObject.transformation, finalKeyFrameObject.transformation, t);
            mesh->setModelTransformation(interpolatedTransformation);

            renderWorldSpaceMeshSlot(group, initialKeyFrameObject.mesh, interpolatedMaterial, renderWidget);
        }

        GL_CALL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));

        this->update();
        auto capture = this->grabFramebuffer();
        gifWrapper.appendFrame(capture, delay);
    }
    progressDialog.setValue(steps);

    // Close the file handle and free memory.
    gifWrapper.close();
    GL_CALL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));
    this->update();
    progressDialog.close();
}

void OpenGLWidget::renderPlaneSlot(const std::string &group, const std::string &name, const Plane &plane, const PhongMaterial &material, RenderWidget *renderWidget) {
    // Find the group
    auto& renderModelsMap = this->getOrInsertRenderModelsMap(group);

    // Find the model in this group
    auto hash = std::hash<Plane>{}(plane);
    glm::detail::hash_combine(hash, std::hash<std::string>{}(name));
    auto renderId = std::to_string(hash);
    auto modelIterator = renderModelsMap.find(renderId);

    // Create a new model if not found
    if(modelIterator == renderModelsMap.end()){

        this->makeCurrent();

        // No entry present yet, create new render Model
        auto renderPlane = std::make_shared<RenderPlane>(plane);

        // Insert it in the renderModelsMap
        modelIterator = renderModelsMap.insert({renderId, renderPlane}).first;

        // Add required listeners
        this->addRenderModelListeners(group, renderPlane);

        // Add control widget to the renderWidget
        renderWidget->addControlWidget(group, renderPlane);

        // Set the color
        modelIterator->second->setMaterial(PhongMaterial(Color::White()));
        modelIterator->second->setName(name);

        this->updateSortedRenderModels();
    }

    // Update the color
    modelIterator->second->setMaterial(material);

    // Update the transformation
//    modelIterator->second->setTransformation(transformation);

    this->update();
}

void OpenGLWidget::renderRaySlot(const std::string &group, const std::string &name, const Ray &ray, const PhongMaterial &material, float widthLengthRatio, RenderWidget *renderWidget) {

    // Find the group
    auto& renderModelsMap = this->getOrInsertRenderModelsMap(group);

    // Find the model in this group
    auto hash = std::hash<Ray>{}(ray);
    glm::detail::hash_combine(hash, std::hash<std::string>{}(name));
    auto renderId = std::to_string(hash);
    auto modelIterator = renderModelsMap.find(renderId);

    // Create a new model if not found
    if(modelIterator == renderModelsMap.end()){

        this->makeCurrent();

        // No entry present yet, create new render Model
        auto renderRay = std::make_shared<RenderRay>(ray, Transformation(), widthLengthRatio);

        // Insert it in the renderModelsMap
        modelIterator = renderModelsMap.insert({renderId, renderRay}).first;

        // Add required listeners
        this->addRenderModelListeners(group, renderRay);

        // Add control widget to the renderWidget
        renderWidget->addControlWidget(group, renderRay);

        // Set the color
        modelIterator->second->setMaterial(PhongMaterial(Color::White()));
        modelIterator->second->setName(name);

        this->updateSortedRenderModels();
    }

    // Update the color
    modelIterator->second->setMaterial(material);

    // Update the transformation
    // modelIterator->second->setTransformation(transformation);

    this->update();
}

OpenGLWidget::~OpenGLWidget() {
    makeCurrent();
}

