//
// Created by Jonas on 4/12/2020.
//

#include <QtWidgets>
#include <glm/ext/matrix_transform.hpp>
#include "SingleModelOptimization.h"
#include "../core/FileParser.h"
#include <chrono>

void runRotation(RenderWidget *renderWidget, WorldSpaceMesh& worldSpaceMesh){
    std::cout << "Test 1" << std::endl;
    while(true){
        worldSpaceMesh.transform(glm::rotate(glm::dmat4(1.0f), 0.005, glm::dvec3(0, 0, 1)));
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        renderWidget->updateWorldSpaceMesh(worldSpaceMesh);
    }
}

void runRotation2(RenderWidget *renderWidget, WorldSpaceMesh& worldSpaceMesh){
    std::cout << "Test 2" << std::endl;
    while(true){
        worldSpaceMesh.transform(glm::rotate(glm::dmat4(1.0f), 0.5, glm::dvec3(1, 1, 0)));
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        renderWidget->updateWorldSpaceMesh(worldSpaceMesh);
    }
}

void runRotation3(RenderWidget *renderWidget, WorldSpaceMesh& worldSpaceMesh){
    std::cout << "Test 3" << std::endl;
    while(true){
        worldSpaceMesh.transform(glm::rotate(glm::dmat4(1.0f), 0.02, glm::dvec3(0, 1, 0)));
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
        renderWidget->updateWorldSpaceMesh(worldSpaceMesh);
    }
}

void test(RenderWidget *renderWidget){

    // This should happen on de main thread
    const ModelSpaceMesh modelSpaceMesh3 = FileParser::parseFile(R"(C:\Users\tolle\OneDrive\Doctoraat\Data\Models\banana.stl)");
    WorldSpaceMesh worldSpaceMesh3 = WorldSpaceMesh(modelSpaceMesh3);
    renderWidget->addWorldSpaceMesh(worldSpaceMesh3, glm::vec4(1,0,0,0.3));
}

void SingleModelOptimization::run() {
    const ModelSpaceMesh modelSpaceMesh = FileParser::parseFile(R"(C:\Users\tolle\OneDrive\Doctoraat\Data\Models\pig.obj)");
    WorldSpaceMesh worldSpaceMesh = WorldSpaceMesh(modelSpaceMesh);
    renderWidget->addWorldSpaceMesh(worldSpaceMesh);
    auto* first = new std::thread(runRotation, renderWidget, worldSpaceMesh);

    const ModelSpaceMesh modelSpaceMesh2 = FileParser::parseFile(R"(C:\Users\tolle\OneDrive\Doctoraat\Data\Models\dragon.obj)");
    WorldSpaceMesh worldSpaceMesh2 = WorldSpaceMesh(modelSpaceMesh2);
    worldSpaceMesh2.transform(glm::translate(glm::mat4(1.0f), glm::vec3(5,0,0)));
    renderWidget->addWorldSpaceMesh(worldSpaceMesh2, glm::vec4(1,1,1,0.3));
    auto* second = new std::thread(runRotation2, renderWidget, worldSpaceMesh2);

    const ModelSpaceMesh modelSpaceMesh3 = FileParser::parseFile(R"(C:\Users\tolle\OneDrive\Doctoraat\Data\Models\apple.stl)");
    WorldSpaceMesh worldSpaceMesh3 = WorldSpaceMesh(modelSpaceMesh3);
    renderWidget->addWorldSpaceMesh(worldSpaceMesh3, glm::vec4(1,0,0,0.3));
    auto* third = new std::thread(runRotation3, renderWidget, worldSpaceMesh3);

//    auto* fourth = new std::thread(test, renderWidget);

    // TODO clean up thread afterwards

}

void SingleModelOptimization::setRenderWidget(RenderWidget *renderWidget) {
    SingleModelOptimization::renderWidget = renderWidget;
}
