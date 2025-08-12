//
// Created by Jonas on 19/05/2022.
//

#include "RenderModelControlWidget.h"
#include "forms/ui_rendermodelcontrolwidget.h"

#include <iostream>
#include <Qinputdialog.h>
#include <QLabel>
#include <QColorDialog>


RenderModelControlWidget::RenderModelControlWidget(const std::shared_ptr<AbstractRenderModel>& renderModel, QWidget* parent): QWidget(parent),
        ui(new Ui::RenderModelControlWidget),
        renderModel(renderModel),
        listener(std::make_shared<SimpleRenderModelListener>()) {

    ui->setupUi(this);


    ui->visibleCheckBox->setChecked(this->renderModel->isVisible());
    const auto &color = this->renderModel->getMaterial().getDiffuseColor();
    ui->colorPushButton->setStyleSheet("background-color: " + QColor(color.r*255, color.g*255, color.b*255, color.a*255).name());

    listener->setOnMaterialChanged([&](const PhongMaterial& oldMaterial, const PhongMaterial& newMaterial) {
        const auto& newColor = newMaterial.getDiffuseColor();
        this->ui->colorPushButton->setStyleSheet("background-color: " +
                                                 QColor(newColor.r * 255, newColor.g * 255, newColor.b * 255,
                                                        newColor.a * 255).name());
    });

    listener->setOnVisibleChanged([&](bool oldVisible, bool newVisible) {
        this->ui->visibleCheckBox->setChecked(newVisible);
    });

    QFontMetrics metrics(this->ui->nameLabel->font());
    QString elidedText = metrics.elidedText(QString::fromStdString(this->renderModel->getName()), Qt::ElideRight, 244); // Sad that this value is hard coded, but this is executed before the name label gets it's final size, we won't override the resize operator in a separate class just for this
    ui->nameLabel->setText(elidedText);
    listener->setOnNameChanged([&](const std::string& oldName, const std::string& newName) {
        QFontMetrics metrics(this->ui->nameLabel->font());
        QString elidedText = metrics.elidedText(QString::fromStdString(newName), Qt::ElideRight, this->ui->nameLabel->width());
        this->ui->nameLabel->setText(elidedText);
    });

    this->renderModel->addListener(listener);

    connect(ui->visibleCheckBox, &QCheckBox::clicked, [&](bool enabled) {
        this->renderModel->setVisible(enabled);
    });

    connect(ui->colorPushButton, &QPushButton::clicked, [=](){
        auto originalMaterial = this->renderModel->getMaterial();
        auto initialColor = originalMaterial.getDiffuseColor();
        auto resultColor = QColorDialog::getColor(QColor(255.f*initialColor.r, 255.f*initialColor.g, 255.f*initialColor.b, 255.f*initialColor.a), nullptr, QString(), QColorDialog::ShowAlphaChannel);
        if(resultColor.isValid()){
            Color newDiffuseColor(resultColor.red() / 255.f, resultColor.green() / 255.f, resultColor.blue() / 255.f, resultColor.alpha() / 255.f);
            this->renderModel->setMaterial(PhongMaterial(newDiffuseColor, originalMaterial.getSpecularColor())); // Keep original specular color
        }
    });

    connect(ui->settingsPushButton, &QPushButton::clicked, [=](){
        auto detailsDialog =  this->renderModel->getDetailsDialog(this);
        detailsDialog->setParent(this);
        detailsDialog->show();
    });

    // Context menu on right click
    this->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(this, &QLabel::customContextMenuRequested, [&](const QPoint &pos) {
        auto menu = this->renderModel->getContextMenu();
        menu->exec(this->mapToGlobal(pos));
    });
}

RenderModelControlWidget::~RenderModelControlWidget() {
    this->renderModel->removeListener(listener);
    delete ui;
}
