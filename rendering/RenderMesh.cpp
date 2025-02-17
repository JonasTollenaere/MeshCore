//
// Created by Jonas on 18/05/2022.
//

#include "RenderMesh.h"

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/normal.hpp>

#include "ShaderProgramSource.h"
#include <QOpenGLShaderProgram>
#include <utility>
#include <QLabel>
#include <QFrame>
#include <QCheckBox>
#include <QGridLayout>
#include <QComboBox>
#include "Exception.h"
#include "OpenGLWidget.h"
#include "utility/random.h"
#include "utility/Triangulation.h"

RenderMesh::RenderMesh(const WorldSpaceMesh& worldSpaceMesh):
        AbstractRenderModel(worldSpaceMesh.getModelTransformation(), worldSpaceMesh.getModelSpaceMesh()->getName()),
        boundingBox(worldSpaceMesh),
        faceVertexBuffer(new QOpenGLBuffer(QOpenGLBuffer::Type::VertexBuffer)),
        faceIndexBuffer(new QOpenGLBuffer(QOpenGLBuffer::Type::IndexBuffer)),
        faceVertexArray(new QOpenGLVertexArrayObject())
{
    const std::vector<Vertex>& vertices = worldSpaceMesh.getModelSpaceMesh()->getVertices();
    const std::vector<IndexTriangle>& triangles = worldSpaceMesh.getModelSpaceMesh()->getTriangles();
    const std::vector<IndexFace>& faces = triangles.size() < 1000 ? worldSpaceMesh.getModelSpaceMesh()->getFaces() : std::vector<IndexFace>();

    this->numberOfVertices = vertices.size();
    this->numberOfFaces = faces.size();
    this->numberOfTriangles = triangles.size();
    this->unscaledSurfaceArea = worldSpaceMesh.getModelSpaceMesh()->getSurfaceArea();
    this->unscaledVolume = worldSpaceMesh.getModelSpaceMesh()->getVolume();

    Random random;

    // Data for triangles
    std::vector<unsigned int> indices;
    std::vector<float> data;
    for(const auto& t : triangles) {
        std::array<Vertex, 3> triangleVertices{vertices[t.vertexIndex0], vertices[t.vertexIndex1], vertices[t.vertexIndex2]};
        glm::vec3 normal = glm::triangleNormal(triangleVertices[0], triangleVertices[1], triangleVertices[2]);

        // Sample random pastel color
        auto pastelFactor = 0.2f;
        Color triangleColor = Color((random.nextFloat()+pastelFactor)/(1+pastelFactor),
                                    (random.nextFloat()+pastelFactor)/(1+pastelFactor),
                                    (random.nextFloat()+pastelFactor)/(1+pastelFactor),
                                    1.0);

        // We push each model space vertex once for each triangle because the normal is different for each triangle
        // (Adding up the normals in the shader doesn't provide visually satisfying results
        for (const auto &triangleVertex: triangleVertices){
            data.emplace_back(triangleVertex.x);
            data.emplace_back(triangleVertex.y);
            data.emplace_back(triangleVertex.z);
            data.emplace_back(normal.x);
            data.emplace_back(normal.y);
            data.emplace_back(normal.z);
            data.emplace_back(triangleColor.r);
            data.emplace_back(triangleColor.g);
            data.emplace_back(triangleColor.b);
            data.emplace_back(triangleColor.w);

            indices.emplace_back(indices.size());
        }
    }

    this->vertexBuffer->create();
    this->vertexBuffer->bind();
    this->vertexBuffer->allocate(&data.front(), data.size() * sizeof(float));

    this->vertexArray->create();
    this->vertexArray->bind();

    this->initializeOpenGLFunctions();

    GL_CALL(glEnableVertexAttribArray(0));
    GL_CALL(glEnableVertexAttribArray(1));
    GL_CALL(glEnableVertexAttribArray(2));

    this->indexBuffer->create();
    this->indexBuffer->bind();
    this->indexBuffer->allocate(&indices.front(), indices.size() * sizeof(unsigned int));

    GL_CALL(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 10 * sizeof(GLfloat), nullptr));
    GL_CALL(glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 10 * sizeof(GLfloat), (void*) (3 * sizeof(GLfloat))));
    GL_CALL(glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 10 * sizeof(GLfloat), (void*) (6 * sizeof(GLfloat))));

    // Prepare data for faces
    std::vector<float> faceData;
    std::vector<unsigned int> faceIndices;
    for (const auto & i : faces){

        const auto& face = i;

        // Compute the normal using Newell's method
        glm::vec3 normal(0.0f);
        for (int j = 0; j < i.vertexIndices.size(); ++j){
            auto indexA = i.vertexIndices[j];
            auto indexB = i.vertexIndices[(j + 1) % i.vertexIndices.size()];
            Vertex vertexA = vertices[indexA];
            Vertex vertexB = vertices[indexB];
            normal.x += (vertexA.y - vertexB.y) * (vertexA.z + vertexB.z);
            normal.y += (vertexA.z - vertexB.z) * (vertexA.x + vertexB.x);
            normal.z += (vertexA.x - vertexB.x) * (vertexA.y + vertexB.y);
        }
        normal = glm::normalize(normal);

        // Sample random pastel color
        auto pastelFactor = 0.2f;
        Color faceColor = Color((random.nextFloat()+pastelFactor)/(1+pastelFactor),
                                (random.nextFloat()+pastelFactor)/(1+pastelFactor),
                                (random.nextFloat()+pastelFactor)/(1+pastelFactor),
                                1.0);

        for (const auto &faceTriangle: Triangulation::triangulateFace(vertices, face)){
            std::array<Vertex, 3> triangleVertices{vertices[faceTriangle.vertexIndex0], vertices[faceTriangle.vertexIndex1], vertices[faceTriangle.vertexIndex2]};
            for (const auto &triangleVertex: triangleVertices){
                faceData.emplace_back(triangleVertex.x);
                faceData.emplace_back(triangleVertex.y);
                faceData.emplace_back(triangleVertex.z);
                faceData.emplace_back(normal.x);
                faceData.emplace_back(normal.y);
                faceData.emplace_back(normal.z);
                faceData.emplace_back(faceColor.r);
                faceData.emplace_back(faceColor.g);
                faceData.emplace_back(faceColor.b);
                faceData.emplace_back(faceColor.w);

                faceIndices.emplace_back(faceIndices.size());
            }
        }
    }

    this->faceVertexBuffer->create();
    this->faceVertexBuffer->bind();
    this->faceVertexBuffer->allocate(&faceData.front(), faceData.size() * sizeof(float));

    this->faceVertexArray->create();
    this->faceVertexArray->bind();

    GL_CALL(glEnableVertexAttribArray(0));
    GL_CALL(glEnableVertexAttribArray(1));
    GL_CALL(glEnableVertexAttribArray(2));

    this->faceIndexBuffer->create();
    this->faceIndexBuffer->bind();
    this->faceIndexBuffer->allocate(&faceIndices.front(), faceIndices.size() * sizeof(unsigned int));

    GL_CALL(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 10 * sizeof(GLfloat), nullptr));
    GL_CALL(glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 10 * sizeof(GLfloat), (void*) (3 * sizeof(GLfloat))));
    GL_CALL(glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 10 * sizeof(GLfloat), (void*) (6 * sizeof(GLfloat))));

    // Store the axis render line
    auto& aabb = worldSpaceMesh.getModelSpaceMesh()->getBounds();
    axisRenderLines.emplace_back(std::make_shared<RenderLine>(glm::vec3(0,0,0), glm::vec3(glm::max(0.0f, aabb.getMaximum().x),0,0), Transformation()));
    axisRenderLines.emplace_back(std::make_shared<RenderLine>(glm::vec3(0,0,0), glm::vec3(0,glm::max(0.0f, aabb.getMaximum().y),0), Transformation()));
    axisRenderLines.emplace_back(std::make_shared<RenderLine>(glm::vec3(0,0,0), glm::vec3(0,0,glm::max(0.0f, aabb.getMaximum().z)), Transformation()));
    axisRenderLines[0]->setMaterial(PhongMaterial(Color::Red()));
    axisRenderLines[1]->setMaterial(PhongMaterial(Color::Green()));
    axisRenderLines[2]->setMaterial(PhongMaterial(Color::Blue()));

    for (const auto &renderLine: axisRenderLines){
        renderLine->setTransformation(worldSpaceMesh.getModelTransformation());
    }

    // Store the normals
    for (const auto &triangle: triangles){
        auto vertexTriangle = VertexTriangle(vertices[triangle.vertexIndex0], vertices[triangle.vertexIndex1], vertices[triangle.vertexIndex2]);
        auto center = vertexTriangle.getCentroid();
        auto normal = vertexTriangle.normal;
        normal = glm::normalize(normal) * glm::sqrt(glm::length(normal)); // Normals scale with surface, but we want them to scale with the scaling factor of the mod
        this->normalRays.emplace_back(center, normal);
    }
}

void RenderMesh::initializeNormals() {
    normalRenderRays.clear();
    normalRenderRays.reserve(normalRays.size());
    for (const auto &normalRay: this->normalRays){
        auto renderRay = std::make_shared<RenderRay>(normalRay, this->getTransformation());
        renderRay->setMaterial(PhongMaterial(Color::Red()));
        renderRay->setTransformation(this->getTransformation());
        this->normalRenderRays.emplace_back(renderRay);
    }
}

bool RenderMesh::isCullingEnabled() const {
    return cullingEnabled;
}

void RenderMesh::setCullingEnabled(bool newCullingEnabled) {
    RenderMesh::cullingEnabled = newCullingEnabled;
    for (const auto &listener: this->listeners){
        listener->notify();
    }
}

bool RenderMesh::isWireframeEnabled() const {
    return wireframeEnabled;
}

void RenderMesh::setWireframeEnabled(bool newWireframeEnabled) {
    RenderMesh::wireframeEnabled = newWireframeEnabled;
    for (const auto &listener: this->listeners){
        listener->notify();
    }
}

void RenderMesh::draw(const OpenGLWidget* openGLWidget, const glm::mat4& viewMatrix, const glm::mat4& projectionMatrix, bool lightMode) {

    if(this->isVisible()){

        if(this->axisEnabled){
            for (const auto &axisRenderLine: this->axisRenderLines){
                axisRenderLine->draw(openGLWidget, viewMatrix, projectionMatrix, lightMode);
            }
        }

        if(this->boundingBoxEnabled) boundingBox.draw(openGLWidget, viewMatrix, projectionMatrix, lightMode);

        if(this->normalsEnabled) {
            if(this->normalRenderRays.empty()) this->initializeNormals();
            for (const auto &normalRenderRay: this->normalRenderRays) {
                normalRenderRay->draw(openGLWidget, viewMatrix, projectionMatrix, lightMode);
            }
        }

        this->initializeOpenGLFunctions();
        this->vertexArray->bind();
        this->indexBuffer->bind();

        if(this->cullingEnabled){
            GL_CALL(glEnable(GL_CULL_FACE));
        }
        else{
            GL_CALL(glDisable(GL_CULL_FACE));
        }

        if(this->wireframeEnabled){

            GL_CALL(glPolygonMode(GL_FRONT_AND_BACK, GL_LINE));

            auto& ambientShader = openGLWidget->getAmbientShader();

            ambientShader->bind();
            const glm::mat4 modelViewProjectionMatrix = projectionMatrix * viewMatrix * this->getTransformationMatrix();
            ambientShader->setUniformValue("u_ModelViewProjectionMatrix", QMatrix4x4(glm::value_ptr(modelViewProjectionMatrix)).transposed());
            QVector4D drawColor;
            const auto color = this->getMaterial().getDiffuseColor();
            const auto wireframeColorFactor = 0.5f; // TODO correct for light mode, but what if dark? It should become ligther no?
            drawColor = QVector4D(wireframeColorFactor * color.r, wireframeColorFactor * color.g, wireframeColorFactor * color.b, color.a);
            if(lightMode){
                if(glm::vec3(color) == glm::vec3(1,1,1)){
                    drawColor = QVector4D(0, 0, 0, color.a);
                }
                else if(glm::vec3(color) == glm::vec3(0,0,0)){
                    drawColor = QVector4D(1, 1, 1, color.a);
                }
            }
            ambientShader->setUniformValue("u_Color", drawColor);

            GL_CALL(glDrawElements(GL_TRIANGLES, this->indexBuffer->size()/sizeof(unsigned int), GL_UNSIGNED_INT, nullptr));
        }
        if(this->surfaceEnabled){

            GL_CALL(glPolygonMode(GL_FRONT_AND_BACK, GL_FILL));

            const glm::mat4 modelViewProjectionMatrix = projectionMatrix * viewMatrix * this->getTransformationMatrix();
            glm::vec3 viewSpaceLightDirection = glm::vec4(0, 0, 1, 1) * viewMatrix;
            const glm::vec3 modelLightDirection = glm::vec3(glm::vec4(viewSpaceLightDirection, 1.0f) *
                                                                    this->getTransformationMatrix());
            glm::vec3 cameraPosition = glm::inverse(viewMatrix) * glm::vec4(0,0,0,1000);
            glm::vec3 modelSpaceCameraPosition = glm::vec3(glm::inverse(this->getTransformationMatrix()) * glm::vec4(cameraPosition, 1.0f));
            const float ambientLighting = 0.25f; // TODO make this a configurable member of OpenGLWidget
            const auto& material = this->getMaterial();

            auto diffuseColor = material.getDiffuseColor();
            auto specularColor = material.getSpecularColor();

            if(this->wireframeEnabled){
                // Make the colors more pastel
                auto pastelFactor = 0.2f;
                diffuseColor.r = (diffuseColor.r + pastelFactor) / (1+pastelFactor);
                diffuseColor.g = (diffuseColor.g + pastelFactor) / (1+pastelFactor);
                diffuseColor.b = (diffuseColor.b + pastelFactor) / (1+pastelFactor);
            }

            auto& shader = this->getRenderedTexture() == DEFAULT ? openGLWidget->getPhongShader() : openGLWidget->getPolyChromeShader();

            if(this->getRenderedTexture() == MeshTexture::FACES){
                this->faceVertexArray->bind();
                this->faceIndexBuffer->bind();
            }

            shader->bind();
            shader->setUniformValue("u_ModelViewProjectionMatrix", QMatrix4x4(glm::value_ptr(modelViewProjectionMatrix)).transposed());
            shader->setUniformValue("u_LightDirection", QVector3D(modelLightDirection.x, modelLightDirection.y, modelLightDirection.z));
            shader->setUniformValue("u_ViewPosition", QVector3D(modelSpaceCameraPosition.x, modelSpaceCameraPosition.y, modelSpaceCameraPosition.z));
            shader->setUniformValue("u_AmbientIntensity", ambientLighting);
            shader->setUniformValue("u_DiffuseIntensity", 1.0f - ambientLighting);
            shader->setUniformValue("u_SpecularIntensity", 0.5f);
            shader->setUniformValue("u_DiffuseColor", QVector4D(diffuseColor.r, diffuseColor.g, diffuseColor.b, diffuseColor.a));
            shader->setUniformValue("u_SpecularColor", QVector4D(specularColor.r, specularColor.g, specularColor.b, specularColor.a));
            shader->setUniformValue("u_SpecularPower", 8.0f);

            GL_CALL(glDrawElements(GL_TRIANGLES, this->indexBuffer->size()/sizeof(unsigned int), GL_UNSIGNED_INT, nullptr));
        }
    }
}

RenderMesh &RenderMesh::operator=(RenderMesh &&other) noexcept {
    if(this != &other){
        this->indexBuffer = other.indexBuffer;
        this->vertexArray = other.vertexArray;
        this->vertexBuffer = other.vertexBuffer;
        this->cullingEnabled = other.cullingEnabled;
        this->wireframeEnabled = other.wireframeEnabled;
        this->boundingBoxEnabled = other.boundingBoxEnabled;
        this->axisEnabled = other.axisEnabled;
        this->normalsEnabled = other.normalsEnabled;
        this->surfaceEnabled = other.surfaceEnabled;
        this->boundingBox = std::move(other.boundingBox);
        this->numberOfTriangles = other.numberOfTriangles;
        this->numberOfFaces = other.numberOfFaces;
        this->numberOfVertices = other.numberOfVertices;
        this->unscaledSurfaceArea = other.unscaledSurfaceArea;
        this->unscaledVolume = other.unscaledVolume;
        this->axisRenderLines = std::move(other.axisRenderLines);
        this->normalRays = std::move(other.normalRays);
        this->normalRenderRays = std::move(other.normalRenderRays);

        other.indexBuffer = nullptr;
        other.vertexArray = nullptr;
        other.vertexBuffer = nullptr;
    }
    return *this;
}

bool RenderMesh::isAxisEnabled() const {
    return axisEnabled;
}

void RenderMesh::setAxisEnabled(bool newAxisEnabled) {
    RenderMesh::axisEnabled = newAxisEnabled;
    for (const auto &listener: this->listeners){
        listener->notify();
    }
}

bool RenderMesh::isBoundingBoxEnabled() const {
    return boundingBoxEnabled;
}

void RenderMesh::setBoundingBoxEnabled(bool newBoundingBoxEnabled) {
    RenderMesh::boundingBoxEnabled = newBoundingBoxEnabled;
    for (const auto &listener: this->listeners){
        listener->notify();
    }
}

QMenu* RenderMesh::getContextMenu() {

    auto* contextMenu = AbstractRenderModel::getContextMenu();

    contextMenu->addSeparator();

    QAction* surfaceAction = contextMenu->addAction(QString("Surface"));
    QObject::connect(surfaceAction, &QAction::triggered, [=](bool enabled){
        this->setSurfaceEnabled(enabled);
    });
    surfaceAction->setCheckable(true);
    surfaceAction->setChecked(this->isSurfaceEnabled());
    contextMenu->addAction(surfaceAction);

    auto surfaceMenu = contextMenu->addMenu(QString("Texture"));

    auto defaultAction = surfaceMenu->addAction(QString("Default"));
    defaultAction->setCheckable(true);
    defaultAction->setChecked(this->getRenderedTexture() == MeshTexture::DEFAULT);
    QObject::connect(defaultAction, &QAction::triggered, [=](){
        this->setRenderedTexture(MeshTexture::DEFAULT);
    });

    auto triangleAction = surfaceMenu->addAction(QString("Triangles"));
    triangleAction->setCheckable(true);
    triangleAction->setChecked(this->getRenderedTexture() == MeshTexture::TRIANGLES);
    QObject::connect(triangleAction, &QAction::triggered, [=](){
        this->setRenderedTexture(MeshTexture::TRIANGLES);
    });

    if(this->numberOfFaces){
        auto faceAction = surfaceMenu->addAction(QString("Faces"));
        faceAction->setCheckable(true);
        faceAction->setChecked(this->getRenderedTexture() == MeshTexture::FACES);
        QObject::connect(faceAction, &QAction::triggered, [=](){
            this->setRenderedTexture(MeshTexture::FACES);
        });
    }

    QAction* wireframeAction = contextMenu->addAction(QString("Wireframe"));
    QObject::connect(wireframeAction, &QAction::triggered, [=](bool enabled){
        this->setWireframeEnabled(enabled);
    });
    wireframeAction->setCheckable(true);
    wireframeAction->setChecked(this->isWireframeEnabled());
    contextMenu->addAction(wireframeAction);

    QAction* cullingAction = contextMenu->addAction(QString("Culling"));
    QObject::connect(cullingAction, &QAction::triggered, [=](bool enabled){
        this->setCullingEnabled(enabled);
    });
    cullingAction->setCheckable(true);
    cullingAction->setChecked(this->isCullingEnabled());
    contextMenu->addAction(cullingAction);

    QAction* boundingBoxAction = contextMenu->addAction(QString("Bounding box"));
    QObject::connect(boundingBoxAction, &QAction::triggered, [=](bool enabled){
        this->setBoundingBoxEnabled(enabled);
    });
    boundingBoxAction->setCheckable(true);
    boundingBoxAction->setChecked(this->isBoundingBoxEnabled());
    contextMenu->addAction(boundingBoxAction);

    QAction* axisAction = contextMenu->addAction(QString("Axis"));
    QObject::connect(axisAction, &QAction::triggered, [=](bool enabled){
        this->setAxisEnabled(enabled);
    });
    axisAction->setCheckable(true);
    axisAction->setChecked(this->isAxisEnabled());
    contextMenu->addAction(axisAction);

    QAction* normalsAction = contextMenu->addAction(QString("Normals"));
    QObject::connect(normalsAction, &QAction::triggered, [=](bool enabled){
        this->setNormalsEnabled(enabled);
    });
    normalsAction->setCheckable(true);
    normalsAction->setChecked(this->isNormalsEnabled());
    contextMenu->addAction(normalsAction);

    return contextMenu;

    // TODO add option to save the transformed mesh (int its current position) to a file
}

void RenderMesh::setMaterial(const PhongMaterial& newMaterial) {
    AbstractRenderModel::setMaterial(newMaterial);
    this->boundingBox.setMaterial(newMaterial);
}

RenderModelDetailDialog* RenderMesh::createRenderModelDetailDialog(QWidget* parent) {

    auto dialog = AbstractRenderModel::createRenderModelDetailDialog(parent);

    auto* detailsLayout = new QGridLayout();
    detailsLayout->addWidget(new QLabel(QString::fromStdString("Number of vertices: " + std::to_string(numberOfVertices))), 0, 0);
    detailsLayout->addWidget(new QLabel(QString::fromStdString("Number of faces: " + std::to_string(numberOfFaces))), 1, 0);
    detailsLayout->addWidget(new QLabel(QString::fromStdString("Number of triangles: " + std::to_string(numberOfTriangles))), 2, 0);
    detailsLayout->addWidget(new QLabel(QString::fromStdString("Unscaled surface area: " + std::to_string(unscaledSurfaceArea))), 3, 0);
    detailsLayout->addWidget(new QLabel(QString::fromStdString("Unscaled volume: " + std::to_string(unscaledVolume))), 4, 0);

    auto* detailsWidget = new QWidget();
    detailsWidget->setLayout(detailsLayout);
    dialog->addTab(detailsWidget, QString("Details"));

    auto* optionsLayout = new QGridLayout();

    auto visibleCheckBox = new QCheckBox(QString("Visible"));
    visibleCheckBox->setChecked(this->isVisible());
    QObject::connect(visibleCheckBox, &QCheckBox::clicked, [&](bool enabled) {
        this->setVisible(enabled);
    });
    optionsLayout->addWidget(visibleCheckBox, 0, 0);

    auto textureComboBox = new QComboBox();
    textureComboBox->addItem(QString("Default"));
    textureComboBox->addItem(QString("Triangles"));
    if(numberOfFaces) textureComboBox->addItem(QString("Faces"));
    textureComboBox->setCurrentIndex(static_cast<int>(this->getRenderedTexture()));
    QObject::connect(textureComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), [=](int index){
        this->setRenderedTexture(static_cast<MeshTexture>(index));
    });
    optionsLayout->addWidget(textureComboBox, 0, 1);

    auto surfaceCheckBox = new QCheckBox(QString("Show Surface"));
    surfaceCheckBox->setChecked(this->isSurfaceEnabled());
    QObject::connect(surfaceCheckBox, &QCheckBox::clicked, [&](bool enabled) {
        this->setSurfaceEnabled(enabled);
    });
    optionsLayout->addWidget(surfaceCheckBox, 1, 0);

    auto wireframeCheckBox = new QCheckBox(QString("Show Wireframe"));
    wireframeCheckBox->setChecked(this->isWireframeEnabled());
    QObject::connect(wireframeCheckBox, &QCheckBox::clicked, [&](bool enabled) {
        this->setWireframeEnabled(enabled);
    });
    optionsLayout->addWidget(wireframeCheckBox, 2, 0);

    auto cullingCheckBox = new QCheckBox(QString("Enable Culling"));
    cullingCheckBox->setChecked(this->isCullingEnabled());

    QObject::connect(cullingCheckBox, &QCheckBox::clicked, [&](bool enabled) {
        this->setCullingEnabled(enabled);
    });
    optionsLayout->addWidget(cullingCheckBox, 3, 0);

    auto boundingBoxCheckBox = new QCheckBox(QString("Show Bounding Box"));
    boundingBoxCheckBox->setChecked(this->isBoundingBoxEnabled());
    QObject::connect(boundingBoxCheckBox, &QCheckBox::clicked, [&](bool enabled) {
        this->setBoundingBoxEnabled(enabled);
    });
    optionsLayout->addWidget(boundingBoxCheckBox, 1, 1);

    auto axisCheckBox = new QCheckBox(QString("Show Axis"));
    axisCheckBox->setChecked(this->isAxisEnabled());
    QObject::connect(axisCheckBox, &QCheckBox::clicked, [&](bool enabled) {
        this->setAxisEnabled(enabled);
    });
    optionsLayout->addWidget(axisCheckBox, 2, 1);

    auto normalsCheckBox = new QCheckBox(QString("Show Normals"));
    normalsCheckBox->setChecked(this->isNormalsEnabled());
    QObject::connect(normalsCheckBox, &QCheckBox::clicked, [&](bool enabled) {
        this->setNormalsEnabled(enabled);
    });
    optionsLayout->addWidget(normalsCheckBox, 3, 1);

    auto listener = std::make_shared<SimpleRenderModelListener>();
    this->addListener(listener);
    listener->setOnVisibleChanged([=](bool oldVisible, bool newVisible) {
        visibleCheckBox->setChecked(newVisible);
    });
    listener->setOnChanged([=]() {
        wireframeCheckBox->setChecked(this->isWireframeEnabled());
        cullingCheckBox->setChecked(this->isCullingEnabled());
        boundingBoxCheckBox->setChecked(this->isBoundingBoxEnabled());
    });

    auto volumeWidget = new QLabel(QString::fromStdString("Volume: " + std::to_string(unscaledVolume * this->getTransformation().getScale() * this->getTransformation().getScale() * this->getTransformation().getScale())));
    detailsLayout->addWidget(volumeWidget, 5, 0);
    listener->setOnTransformationChanged([=](const Transformation& oldTransformation, const Transformation& newTransformation) {
        volumeWidget->setText(QString::fromStdString("Volume: " + std::to_string(unscaledVolume * newTransformation.getScale() * newTransformation.getScale() * newTransformation.getScale())));
    });

    auto* optionsWidget = new QWidget();
    optionsWidget->setLayout(optionsLayout);
    dialog->addTab(optionsWidget, QString("Options"));

    return dialog;
}

void RenderMesh::setTransformation(const Transformation &newTransformation) {
    AbstractRenderModel::setTransformation(newTransformation);
    this->boundingBox.setTransformation(newTransformation);
    for (const auto &axisRenderLine: this->axisRenderLines){
        axisRenderLine->setTransformation(newTransformation);
    }
    for (const auto &normalRenderRay: this->normalRenderRays){
        normalRenderRay->setTransformation(newTransformation);
    }
}

void RenderMesh::setSurfaceEnabled(bool newSurfaceEnabled) {
    RenderMesh::surfaceEnabled = newSurfaceEnabled;
    for (const auto &listener: this->listeners){
        listener->notify();
    }
}

bool RenderMesh::isSurfaceEnabled() const {
    return surfaceEnabled;
}

void RenderMesh::setNormalsEnabled(bool newNormalsEnabled) {
    RenderMesh::normalsEnabled = newNormalsEnabled;
    for (const auto &listener: this->listeners){
        listener->notify();
    }
}

bool RenderMesh::isNormalsEnabled() const {
    return normalsEnabled;
}

RenderMesh::MeshTexture RenderMesh::getRenderedTexture() const {
    return renderedTexture;
}

void RenderMesh::setRenderedTexture(MeshTexture newRenderedTexture) {
    RenderMesh::renderedTexture = newRenderedTexture;
    for (const auto &listener: this->listeners){
        listener->notify();
    }
}
