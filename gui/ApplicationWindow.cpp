//
// Created by Jonas on 30/11/2020.
//

#include "ApplicationWindow.h"
#include "../core/FileParser.h"
#include <QtWidgets>
#include <iostream>
#include <filesystem>

ApplicationWindow::ApplicationWindow() {

    QSurfaceFormat format;
    format.setDepthBufferSize(24);
    format.setStencilBufferSize(8);
    format.setVersion(3, 2);
    format.setProfile(QSurfaceFormat::CoreProfile);
    QSurfaceFormat::setDefaultFormat(format);

   this->openGlWidget = new OpenGLWidget(this);

    openGlWidget->setMinimumSize(200, 100);
    openGlWidget->setFocus();
    setCentralWidget(openGlWidget);

    menuBar = new QMenuBar;
    QMenu* fileMenu = new QMenu(QString("File"), this);
    QMenu* viewMenu = new QMenu(QString("View"), this);
    QMenu* helpMenu = new QMenu(QString("Help"), this);
    menuBar->addMenu(fileMenu);
    menuBar->addMenu(viewMenu);
    menuBar->addMenu(helpMenu);
    setMenuBar(menuBar);

    QAction* openAction = fileMenu->addAction(QString("Open..."));
    connect(openAction, &QAction::triggered, this, &ApplicationWindow::loadMesh);
    openAction->setShortcut(QKeySequence(QString("Ctrl+O")));

    QAction* exitAction = fileMenu->addAction(QString("Exit"));
    connect(exitAction, &QAction::triggered, this, &QWidget::close);

//    QAction* zoomInAction = viewMenu->addAction(QString("Zoom in"));
//    connect(zoomInAction, &QAction::triggered, this, &ApplicationWindow::loadMesh);
//    zoomInAction->setShortcut(QKeySequence::ZoomIn);

    QAction* resetViewAction = viewMenu->addAction(QString("Reset view"));
    connect(resetViewAction, &QAction::triggered, openGlWidget, &OpenGLWidget::resetView);
    resetViewAction->setShortcut(QKeySequence(QString("Ctrl+0")));


    QAction* aboutAction = helpMenu->addAction(QString("About Meshcore"));
    connect(aboutAction, &QAction::triggered, this, &ApplicationWindow::displayApplicationInfo);

    setMinimumSize(640,360);
    setWindowTitle(tr("MeshCore"));
}

void ApplicationWindow::keyPressEvent(QKeyEvent* event){
    printf("\nkey event in ApplicationWindow: %i", event->key());
    std::cout << event->key() << std::endl;
}

void ApplicationWindow::loadMesh(){
    QString fileName = QFileDialog::getOpenFileName(this, QString("Select mesh file"), R"(C:\Users\tolle\CLionProjects\MeshCore\data\models)", QString("Mesh Files (*.stl *.obj)"));
    if(std::filesystem::exists(fileName.toStdString())){
        const ModelSpaceMesh modelSpaceMesh = FileParser::parseFile(fileName.toStdString());
        const WorldSpaceMesh* worldSpaceMesh = new WorldSpaceMesh(modelSpaceMesh);
        openGlWidget->addWorldSpaceMesh(worldSpaceMesh);
    }
//    else{
//        auto messageBox = QMessageBox(this);
//        messageBox.setInformativeText("No valid file selected!");
//        messageBox.setWindowTitle("Warning");
//        messageBox.exec();
//    }
}

void ApplicationWindow::displayApplicationInfo(){
    auto messageBox = QMessageBox(this);
    messageBox.setInformativeText("Meshcore v0.0.1");
    messageBox.setWindowTitle("About MeshCore");
    messageBox.setText("Created by Jonas Tollenaere");
    messageBox.exec();
}
