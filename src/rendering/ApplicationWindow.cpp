//
// Created by Jonas on 19/05/2022.
//

#include "meshcore/rendering/ApplicationWindow.h"
#include "meshcore/utility/FileParser.h"
#include <QMenu>
#include <QMessageBox>
#include <QFileDialog>
#include <QMenuBar>
#include <iostream>
#include <filesystem>

#ifndef MESHCORE_ICON_DIR
#define MESHCORE_ICON_DIR ""
#endif

ApplicationWindow::ApplicationWindow() {

    auto icon = QIcon(QString::fromStdString(MESHCORE_ICON_DIR + std::string("logo3.svg")));
    QGuiApplication::setWindowIcon(icon);

    QSurfaceFormat format;
    format.setDepthBufferSize(24);
    format.setStencilBufferSize(8);
    format.setSamples(16);
    format.setVersion(3, 2);
    format.setProfile(QSurfaceFormat::CoreProfile);
    QSurfaceFormat::setDefaultFormat(format);

    this->renderWidget = new RenderWidget(this);

//    taskRenderWidget->setMinimumSize(200, 100);
//    taskRenderWidget->setFocus();
    setCentralWidget(renderWidget);

    menuBar = new QMenuBar;
    auto* fileMenu = new QMenu(QString("File"), this);
    auto* viewMenu = new QMenu(QString("View"), this);
    auto* helpMenu = new QMenu(QString("Help"), this);
    menuBar->addMenu(fileMenu);
    menuBar->addMenu(viewMenu);
    menuBar->addMenu(helpMenu);
    setMenuBar(menuBar);

    QAction* openAction = fileMenu->addAction(QString("Open..."));
    connect(openAction, &QAction::triggered, this, &ApplicationWindow::loadMesh);
    openAction->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_O));

    QAction* screenshotAction = fileMenu->addAction(QString("Export current frame..."));
    connect(screenshotAction, &QAction::triggered, renderWidget, &RenderWidget::captureScene);
    screenshotAction->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_P));

    QAction* animationAction = fileMenu->addAction(QString("Export GIF animation..."));
    connect(animationAction, &QAction::triggered, renderWidget, &RenderWidget::captureAnimation);
    animationAction->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_G));

    QAction* exitAction = fileMenu->addAction(QString("Exit"));
    connect(exitAction, &QAction::triggered, this, &QWidget::close);
    exitAction->setShortcut(QKeySequence(Qt::ALT + Qt::Key_F4));

//    QAction* zoomInAction = viewMenu->addAction(QString("Zoom in"));
//    connect(zoomInAction, &QAction::triggered, this, &ApplicationWindow::loadMesh);
//    zoomInAction->setShortcut(QKeySequence::ZoomIn);

    QAction* resetViewAction = viewMenu->addAction(QString("Reset view"));
    connect(resetViewAction, &QAction::triggered, [=]() { renderWidget->setView(0); });
    resetViewAction->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_0));

    QMenu* otherViewsMenu = viewMenu->addMenu(QString("Other views"));

    QAction* blfViewAction = otherViewsMenu->addAction(QString("Bottom left front view"));
    connect(blfViewAction, &QAction::triggered, [=]() { renderWidget->setView(1); });
    blfViewAction->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_1));

    QAction* bottomViewAction = otherViewsMenu->addAction(QString("Bottom view"));
    connect(bottomViewAction, &QAction::triggered, [=]() { renderWidget->setView(2); });
    bottomViewAction->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_2));

    QAction* brfViewAction = otherViewsMenu->addAction(QString("Bottom right front view"));
    connect(brfViewAction, &QAction::triggered, [=]() { renderWidget->setView(3); });
    brfViewAction->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_3));

    QAction* leftViewAction = otherViewsMenu->addAction(QString("Left view"));
    connect(leftViewAction, &QAction::triggered, [=]() { renderWidget->setView(4); });
    leftViewAction->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_4));

    QAction* frontViewAction = otherViewsMenu->addAction(QString("Front view"));
    connect(frontViewAction, &QAction::triggered, [=]() { renderWidget->setView(5); });
    frontViewAction->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_5));

    QAction* sideViewAction = otherViewsMenu->addAction(QString("Right view"));
    connect(sideViewAction, &QAction::triggered, [=]() { renderWidget->setView(6); });
    sideViewAction->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_6));

    QAction* tlfViewAction = otherViewsMenu->addAction(QString("Top left front view"));
    connect(tlfViewAction, &QAction::triggered, [=]() { renderWidget->setView(7); });
    tlfViewAction->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_7));

    QAction* topViewAction = otherViewsMenu->addAction(QString("Top view"));
    connect(topViewAction, &QAction::triggered, [=]() { renderWidget->setView(8); });
    topViewAction->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_8));

    QAction* trfViewAction = otherViewsMenu->addAction(QString("Top right front view"));
    connect(trfViewAction, &QAction::triggered, [=]() { renderWidget->setView(9); });
    trfViewAction->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_9));

//    QAction* toggleWireFrame = viewMenu->addAction(QString("Enable Wireframe"));
//    connect(toggleWireFrame, &QAction::triggered, renderWidget->getOpenGLWidget(),
//            &OpenGLWidget::toggleWireframe);
//    toggleWireFrame->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_W));
//    toggleWireFrame->setCheckable(true);
//
//    QAction* toggleCulling = viewMenu->addAction(QString("Enable Backface Culling"));
//    connect(toggleCulling, &QAction::triggered, renderWidget->getOpenGLWidget(), &OpenGLWidget::toggleCullFace);
//    toggleCulling->setCheckable(true);
//    toggleCulling->setChecked(true);

    {
        QAction* toggleAxis = viewMenu->addAction(QString("Enable Axis"));
        connect(toggleAxis, &QAction::triggered, [=](bool enabled){
            renderWidget->getOpenGLWidget()->setAxisEnabled(enabled);
            renderWidget->getOpenGLWidget()->update();
        });
        toggleAxis->setCheckable(true);
        toggleAxis->setChecked(renderWidget->getOpenGLWidget()->isAxisEnabled());
        toggleAxis->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_X));
    }
    {
        QAction* toggleLightMode = viewMenu->addAction(QString("Enable Light Mode"));
        connect(toggleLightMode, &QAction::triggered, [=](bool enabled){
            renderWidget->getOpenGLWidget()->setLightMode(enabled);
            renderWidget->getOpenGLWidget()->update();
        });
        toggleLightMode->setCheckable(true);
        toggleLightMode->setChecked(renderWidget->getOpenGLWidget()->isLightMode());
        toggleLightMode->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_L));
    }
    {
        QAction* togglePerspective = viewMenu->addAction(QString("Enable Perspective"));
        connect(togglePerspective, &QAction::triggered, [=](bool enabled){
            renderWidget->getOpenGLWidget()->setUsePerspective(enabled);
            renderWidget->getOpenGLWidget()->update();
        });
        togglePerspective->setCheckable(true);
        togglePerspective->setChecked(renderWidget->getOpenGLWidget()->isUsePerspective());
    }

    QAction* aboutAction = helpMenu->addAction(QString("About Meshcore"));
    connect(aboutAction, &QAction::triggered, this, &ApplicationWindow::displayApplicationInfo);

    setMinimumSize(1280,720);
    setWindowTitle(tr("MeshCore"));
}

RenderWidget *ApplicationWindow::getRenderWidget() const {
    return renderWidget;
}

void ApplicationWindow::loadMesh(){
    QString fileName = QFileDialog::getOpenFileName(this, QString("Select mesh file"), R"(C:\Users\tolle\CLionProjects\MeshCore\data\models)", QString("Mesh Files (*.stl *.obj)"));
    if(std::filesystem::exists(fileName.toStdString())){
        std::shared_ptr<ModelSpaceMesh> modelSpaceMesh = FileParser::loadMeshFile(fileName.toStdString());
        const auto worldSpaceMesh = std::make_shared<WorldSpaceMesh>(modelSpaceMesh);
        this->renderWidget->renderWorldSpaceMesh("File", worldSpaceMesh);
    }
}

void ApplicationWindow::displayApplicationInfo(){
    auto messageBox = QMessageBox(this);
    messageBox.setInformativeText("Meshcore v0.0.2");
    messageBox.setWindowTitle("About MeshCore");
    messageBox.setText("Created by Jonas Tollenaere");
    messageBox.exec();
}
