//
// Created by tolle on 1/02/2024.
//

#include <thread>

#include "meshcore/rendering/ApplicationWindow.h"
#include "meshcore/utility/FileParser.h"

void run(RenderWidget* renderWidget);

int main(int argc, char *argv[]){

    QApplication app(argc, argv);
    ApplicationWindow window;
    window.show();

    auto renderWidget = window.getRenderWidget();
    auto openGLWidget = renderWidget->getOpenGLWidget();
    openGLWidget->setLightMode(true);
    openGLWidget->setFixedWidth(1920/app.devicePixelRatio());
    openGLWidget->setFixedHeight(1080/app.devicePixelRatio());
    openGLWidget->setUsePerspective(false);
    openGLWidget->update();


    std::thread workerThread(run, renderWidget);

    int returnCode = QApplication::exec();
    workerThread.join();
    return returnCode;
}

void run(RenderWidget* renderWidget){

    /// Here is where you would start code running on a separate thread

    // Load some meshes
    std::shared_ptr<WorldSpaceMesh> bananaWorldSpaceMesh = std::make_shared<WorldSpaceMesh>(FileParser::loadMeshFile(MESHCORE_DATA_DIR + std::string("E. F. Silva et al. 2021/banana.stl")));
    std::shared_ptr<WorldSpaceMesh> rockWorldSpaceMesh = std::make_shared<WorldSpaceMesh>(FileParser::loadMeshFile(MESHCORE_DATA_DIR + std::string("E. F. Silva et al. 2021/stone_1.obj")));

    glm::vec3 position(10.0f,0,0);

    bananaWorldSpaceMesh->getModelTransformation().setPosition(position);
//    bananaWorldSpaceMesh->getModelTransformation().setYaw(1.0f);
    rockWorldSpaceMesh->getModelTransformation().setScale(0.5f);
    rockWorldSpaceMesh->getModelTransformation().setPosition(glm::vec3(0,0,-15));

    // Pass them to the renderer
    renderWidget->renderWorldSpaceMesh("Meshes", rockWorldSpaceMesh, Color(0.8, 0.8, 0.8, 0.6));
    renderWidget->renderWorldSpaceMesh("Meshes", bananaWorldSpaceMesh, Color(0.75, 0.75, 0, 1));

    // Render a primitive shape as well
    renderWidget->renderSphere("Sphere", "Sphere2", Sphere(glm::vec3(-15.0f, 0.0f, 0.0f), 5.0f), Color(1.0f, 0.2f, 0.2f, 0.5f));

    // Test quaternion yaw, pitch, roll conversions

    bananaWorldSpaceMesh->getModelTransformation().setRotation(Quaternion());
    bananaWorldSpaceMesh->getModelTransformation().factorScale(10.0f); // YPR
    bananaWorldSpaceMesh->getModelTransformation().factorRotation(Quaternion(0.23,0.96,0.14)); // YPR
    renderWidget->renderWorldSpaceMesh("Meshes", bananaWorldSpaceMesh, Color(0.75, 0.75, 0, 1));

    auto transformationA = bananaWorldSpaceMesh->getModelTransformation();
    auto transformationB = transformationA;
    transformationB.factorScale(2.0f);
    transformationB.factorRotation(Quaternion(glm::pi<float>()/2,0,0)); // YPR
    transformationB.deltaPosition(glm::vec3(0,0,-10));

    KeyFrame keyFrame1;
    keyFrame1.addObject("Meshes", bananaWorldSpaceMesh, transformationA, PhongMaterial(Color::Yellow()));
    KeyFrame keyFrame2;
    keyFrame2.addObject("Meshes", bananaWorldSpaceMesh, transformationB, PhongMaterial(Color::Yellow()));
    KeyFrame keyFrame3;
    keyFrame3.addObject("Meshes", bananaWorldSpaceMesh, transformationB, PhongMaterial(Color::Red()));
    KeyFrame keyFrame4;
    keyFrame4.addObject("Meshes", bananaWorldSpaceMesh, transformationA, PhongMaterial(Color::Yellow()));

    renderWidget->captureLinearAnimation(renderWidget->getOpenGLWidget()->getViewTransformation(), renderWidget->getOpenGLWidget()->getViewTransformation(), keyFrame1, keyFrame2, "test.gif", 200, 2);
    renderWidget->captureLinearAnimation(renderWidget->getOpenGLWidget()->getViewTransformation(), renderWidget->getOpenGLWidget()->getViewTransformation(), keyFrame2, keyFrame3, "test2.gif", 200, 2);
    renderWidget->captureLinearAnimation(renderWidget->getOpenGLWidget()->getViewTransformation(), renderWidget->getOpenGLWidget()->getViewTransformation(), keyFrame3, keyFrame4, "test3.gif", 200, 2);
}