//
// Created by Jonas Tollenaere on 4/11/2021.
//

#include <iostream>
#include <thread>

#include "meshcore/rendering/ApplicationWindow.h"
#include "meshcore/utility/FileParser.h"

void run(RenderWidget* renderWidget);

int main(int argc, char *argv[]){

    QApplication app(argc, argv);
    ApplicationWindow window;
    window.show();

    std::thread workerThread(run, window.getRenderWidget());

    int returnCode = QApplication::exec();
    workerThread.join();
    return returnCode;
}

void run(RenderWidget* renderWidget){

    /// Here is where you would start code running on a separate thread

    // Load some meshes
    std::shared_ptr<WorldSpaceMesh> bananaWorldSpaceMesh = std::make_shared<WorldSpaceMesh>(FileParser::loadMeshFile(MESHCORE_DATA_DIR + std::string("E. F. Silva et al. 2021/banana.stl")));
    std::shared_ptr<WorldSpaceMesh> rockWorldSpaceMesh = std::make_shared<WorldSpaceMesh>(FileParser::loadMeshFile(MESHCORE_DATA_DIR + std::string("E. F. Silva et al. 2021/stone_1.obj")));

    std::cout << "Number of vertices: " << bananaWorldSpaceMesh->getModelSpaceMesh()->getVertices().size() << std::endl;
    std::cout << "Number of vertices: " << rockWorldSpaceMesh->getModelSpaceMesh()->getVertices().size() << std::endl;

    glm::vec3 position(10.0f,0,0);

    bananaWorldSpaceMesh->getModelTransformation().setPosition(position);
    rockWorldSpaceMesh->getModelTransformation().setScale(0.5f);
    rockWorldSpaceMesh->getModelTransformation().setPosition(glm::vec3(0,0,-15));

    // Pass them to the renderer
    renderWidget->renderWorldSpaceMesh("Meshes", rockWorldSpaceMesh, Color(0.8, 0.8, 0.8, 0.6));
    renderWidget->renderWorldSpaceMesh("Meshes", bananaWorldSpaceMesh, Color(0.75, 0.75, 0, 1));

    // Render a primitive shape as well
    renderWidget->renderSphere("Sphere", "Sphere2", Sphere(glm::vec3(-15.0f, 0.0f, 0.0f), 5.0f), Color(1.0f, 0.2f, 0.2f, 0.5f));

}