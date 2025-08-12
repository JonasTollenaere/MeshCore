//
// Created by Jonas on 2/10/2022.
//

#include "RenderBoundsTree.h"

#include <QGridLayout>
#include <QLabel>
#include <QSlider>

void RenderBoundsTree::draw(const OpenGLWidget* openGLWidget, const glm::mat4 &viewMatrix, const glm::mat4 &projectionMatrix, bool lightMode) {
    if(this->isVisible()) this->drawRecursive(openGLWidget, viewMatrix, projectionMatrix, lightMode, renderDepth);
}

void RenderBoundsTree::drawRecursive(const OpenGLWidget* openGLWidget, const glm::mat4 &viewMatrix, const glm::mat4 &projectionMatrix, bool lightMode, unsigned int depth) {
    if(depth > 0){
        for (const auto &child : this->children){
            child->drawRecursive(openGLWidget, viewMatrix, projectionMatrix, lightMode, depth-1);
        }
    }
    else if(depth == 0){
        renderNodeModel->draw(openGLWidget, viewMatrix, projectionMatrix, lightMode);
    }
}

void RenderBoundsTree::setTransformation(const Transformation &transformation) {
    AbstractRenderModel::setTransformation(transformation);
    renderNodeModel->setTransformation(transformation);
    for (const auto &child : this->children){
        child->setTransformation(transformation);
    }
}

void RenderBoundsTree::setRenderDepth(unsigned int newRenderDepth) {
    RenderBoundsTree::renderDepth = newRenderDepth;
    for (const auto &listener: this->listeners){
        listener->notify();
    }
}

unsigned int RenderBoundsTree::getRenderDepth() const {
    return renderDepth;
}

unsigned int RenderBoundsTree::getMaximumDepth() {
    unsigned int depth = 0;
    for (const auto &child : this->children){
        if(child) depth = std::max(depth, child->getMaximumDepth() + 1);
    }
    return depth;
}

void RenderBoundsTree::setMaterial(const PhongMaterial &newMaterial) {
    AbstractRenderModel::setMaterial(newMaterial);
    renderNodeModel->setMaterial(newMaterial);
    for (const auto &child : this->children){
        child->setMaterial(newMaterial);
    }
}

RenderModelDetailDialog *RenderBoundsTree::createRenderModelDetailDialog(QWidget* parent) {

    auto dialog = AbstractRenderModel::createRenderModelDetailDialog(parent);

    auto listener = std::make_shared<SimpleRenderModelListener>();
    this->addListener(listener);

    auto* detailsLayout = new QGridLayout();
    detailsLayout->addWidget(new QLabel(QString::fromStdString("Maximum depth: " + std::to_string(
            this->getMaximumDepth()))), 0, 0);

    auto* detailsWidget = new QWidget();
    detailsWidget->setLayout(detailsLayout);
    dialog->addTab(detailsWidget, QString("Details"));

    auto* optionsLayout = new QGridLayout();

    auto visibleCheckBox = new QCheckBox(QString("Visible"));
    visibleCheckBox->setChecked(this->isVisible());
    listener->setOnVisibleChanged([=](bool oldVisible, bool newVisible) {
        visibleCheckBox->setChecked(newVisible);
    });
    QObject::connect(visibleCheckBox, &QCheckBox::clicked, [&](bool enabled) {
        this->setVisible(enabled);
    });
    optionsLayout->addWidget(visibleCheckBox, 0, 0);

    auto animateRenderDepth = new QCheckBox(QString("Animate render depth"));
    animateRenderDepth->setChecked(false);
    QObject::connect(animateRenderDepth, &QCheckBox::clicked, [&](bool enabled) {
        animating = enabled;
        // Start a thread that increases the render depth every second until animate is false
        if(enabled){
            std::thread([&]() {
                while(animating){
                    this->setRenderDepth((this->getRenderDepth()+1)% this->getMaximumDepth());
                    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                }
            }).detach();
        }
    });
    optionsLayout->addWidget(animateRenderDepth, 1, 0);

    auto* slider = new QSlider(Qt::Horizontal);
    slider->setMinimum(0);
    slider->setValue(this->renderDepth);
    slider->setMaximum(this->getMaximumDepth());
    optionsLayout->addWidget(new QLabel("Rendering Depth"), 2, 0);
    optionsLayout->addWidget(slider, 3, 0);
    QObject::connect(slider, &QSlider::valueChanged, [=](){
        this->setRenderDepth(slider->value());
    });
    listener->setOnChanged([=](){
        slider->setValue(renderDepth);
    });


    auto* optionsWidget = new QWidget();
    optionsWidget->setLayout(optionsLayout);
    dialog->addTab(optionsWidget, QString("Options"));

    return dialog;
}

QMenu *RenderBoundsTree::getContextMenu() {
    return AbstractRenderModel::getContextMenu();

    // TODO, if necessary
}