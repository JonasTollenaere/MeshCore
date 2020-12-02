//
// Created by Jonas on 30/11/2020.
//

#include "ApplicationWindow.h"
#include <QtWidgets>
#include <iostream>

void chat(){
    std::cout << "Chat" << std::endl;
}

ApplicationWindow::ApplicationWindow() {

    menuBar = new QMenuBar;
    QMenu *fileMenu;
    QAction *exitAction;
    fileMenu = new QMenu(tr("&File"), this);
    exitAction = fileMenu->addAction(tr("E&xit"));
    menuBar->addMenu(fileMenu);
    connect(exitAction, &QAction::triggered, this, &QWidget::close);

    setMenuBar(menuBar);


    QSurfaceFormat format;
    format.setDepthBufferSize(24);
    format.setStencilBufferSize(8);
    format.setVersion(3, 2);
    format.setProfile(QSurfaceFormat::CoreProfile);
    QSurfaceFormat::setDefaultFormat(format);

    auto* openGLWidget = new OpenGLWidget(this);

    openGLWidget->setMinimumSize(200, 100);
    openGLWidget->setFocus();

    setCentralWidget(openGLWidget);
    setMinimumSize(640,360);
    setWindowTitle(tr("MeshCore"));
}

void ApplicationWindow::keyPressEvent(QKeyEvent* event){
    printf("\nkey event in ApplicationWindow: %i", event->key());
    std::cout << event->key() << std::endl;
}