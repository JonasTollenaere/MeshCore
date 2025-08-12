//
// Created by Jonas on 8/11/2021.
//

#include "AbstractRenderModel.h"
#include <iostream>
#include <QColorDialog>

bool AbstractRenderModel::isVisible() const {
    return visible;
}

void AbstractRenderModel::setVisible(bool newVisible) {
    const auto oldValue = this->visible;
    this->visible = newVisible;
    for (const auto &listener: this->listeners){
        listener->notifyVisibleChanged(oldValue, this->visible);
    }
}

const PhongMaterial &AbstractRenderModel::getMaterial() const {
    return material;
}

void AbstractRenderModel::setMaterial(const PhongMaterial &newMaterial) {
    const auto oldValue = this->material;
    this->material = newMaterial;
    for (const auto &listener: this->listeners){
        listener->notifyMaterialChanged(oldValue, this->material);
    }
}

glm::mat4 AbstractRenderModel::getTransformationMatrix() const {
    return transformation.getMatrix();
}

AbstractRenderModel::AbstractRenderModel(const Transformation& transformation, const std::string& name):
        material(Color::White()),
        vertexBuffer(new QOpenGLBuffer(QOpenGLBuffer::Type::VertexBuffer)),
        indexBuffer(new QOpenGLBuffer(QOpenGLBuffer::Type::IndexBuffer)),
        vertexArray(new QOpenGLVertexArrayObject()),
        transformation(transformation)
{
    this->name = name;
    if(!name.empty()){
        this->name[0] = std::toupper(name[0]);
    }
}

AbstractRenderModel::~AbstractRenderModel() {

    delete vertexArray;
    delete indexBuffer;
    delete vertexBuffer;

    // We don't have to do this anymore, as the dialog will be removed by its parent (the RenderModelControlWidget)
//    if(this->detailDialog != nullptr){
//        this->detailDialog->close();
//        delete this->detailDialog;
//    }
}

AbstractRenderModel::AbstractRenderModel(AbstractRenderModel &&other) noexcept:
        material(other.material),
        transformation(other.transformation)
{
    this->indexBuffer = other.indexBuffer;
    this->vertexArray = other.vertexArray;
    this->vertexBuffer = other.vertexBuffer;
    this->visible = other.visible;
    this->name = other.name;

    other.indexBuffer = nullptr;
    other.vertexArray = nullptr;
    other.vertexBuffer = nullptr;
}

AbstractRenderModel &AbstractRenderModel::operator=(AbstractRenderModel &&other) noexcept {
    if(this != &other){
        this->indexBuffer = other.indexBuffer;
        this->vertexArray = other.vertexArray;
        this->vertexBuffer = other.vertexBuffer;
        this->material = other.material;
        this->transformation = other.transformation;
        this->visible = other.visible;
        this->name = other.name;

        other.indexBuffer = nullptr;
        other.vertexArray = nullptr;
        other.vertexBuffer = nullptr;
    }
    return *this;
}

const std::string &AbstractRenderModel::getName() const {
    return name;
}

void AbstractRenderModel::setName(const std::string &newName) {
    const auto oldValue = this->name;
    this->name = newName;
    for (const auto &listener: this->listeners){
        listener->notifyNameChanged(oldValue, this->name);
    }
}

QMenu* AbstractRenderModel::getContextMenu(){

    auto contextMenu = new QMenu(QString("Context menu"));

    QAction* visibleAction = contextMenu->addAction(QString("Visible"));
    QObject::connect(visibleAction, &QAction::triggered, [=](bool enabled){
        this->setVisible(enabled);
    });
    visibleAction->setCheckable(true);
    visibleAction->setChecked(this->isVisible());
    contextMenu->addAction(visibleAction);

    QAction* diffuseColorAction = contextMenu->addAction(QString("Change diffuse color..."));
    QObject::connect(diffuseColorAction, &QAction::triggered, [=](){
        auto& initialMaterial = this->getMaterial();
        auto& initialDiffuseColor = initialMaterial.getDiffuseColor();
        auto resultColor = QColorDialog::getColor(QColor(255.f * initialDiffuseColor.r, 255.f * initialDiffuseColor.g, 255.f * initialDiffuseColor.b, 255.f * initialDiffuseColor.a), nullptr, QString(), QColorDialog::ShowAlphaChannel);
        if(resultColor.isValid()){
            Color newDiffuseColor(resultColor.red() / 255.f, resultColor.green() / 255.f, resultColor.blue() / 255.f, resultColor.alpha() / 255.f);
            this->setMaterial(PhongMaterial(newDiffuseColor, initialMaterial.getSpecularColor()));
        }
    });

    QAction* specularColorAction = contextMenu->addAction(QString("Change specular color..."));
    QObject::connect(specularColorAction, &QAction::triggered, [=](){
        auto& initialMaterial = this->getMaterial();
        auto& initialSpecularColor = initialMaterial.getSpecularColor();
        auto resultColor = QColorDialog::getColor(QColor(255.f * initialSpecularColor.r, 255.f * initialSpecularColor.g, 255.f * initialSpecularColor.b, 255.f * initialSpecularColor.a), nullptr, QString(), QColorDialog::ShowAlphaChannel);
        if(resultColor.isValid()){
            Color newSpecularColor(resultColor.red() / 255.f, resultColor.green() / 255.f, resultColor.blue() / 255.f, resultColor.alpha() / 255.f);
            this->setMaterial(PhongMaterial(initialMaterial.getDiffuseColor(), newSpecularColor));
        }
    });

    contextMenu->addAction(diffuseColorAction);

    return contextMenu;
}

RenderModelDetailDialog *AbstractRenderModel::getDetailsDialog(QWidget* parent){

    // Show existing dialog if already exists
    if(this->detailDialog==nullptr){
        this->detailDialog = this->createRenderModelDetailDialog(parent);
    }
    return this->detailDialog;
}

RenderModelDetailDialog *AbstractRenderModel::createRenderModelDetailDialog(QWidget* parent) {
    return new RenderModelDetailDialog(this, parent);
}

void AbstractRenderModel::setTransformation(const Transformation &newTransformation) {
    const auto oldValue = this->transformation;
    AbstractRenderModel::transformation = newTransformation;
    for (const auto &listener: this->listeners){
        listener->notifyTransformationChanged(oldValue, this->transformation);
    }
}

const Transformation &AbstractRenderModel::getTransformation() const {
    return this->transformation;
}

Transformation &AbstractRenderModel::getTransformation() {
    return this->transformation;
};

