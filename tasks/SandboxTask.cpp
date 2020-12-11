//
// Created by Jonas on 4/12/2020.
//

#include <QtWidgets>
#include <glm/ext/matrix_transform.hpp>
#include "SandboxTask.h"
#include "../core/FileParser.h"
#include <chrono>

void randomWalk(RenderWidget *renderWidget, WorldSpaceMesh& innerMesh, const WorldSpaceMesh& roughMesh){

    auto startms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    std::mt19937 randomEngine(0);
    auto nextFloat = std::uniform_real_distribution<float>(0, 1);

    Transformation currentTransformation = innerMesh.getModelTransformationMatrix();
    std::cout << std::boolalpha;
    int moves = 500;
    for(int i=0; i<moves; i++){

        Transformation newTransformation = glm::scale(currentTransformation, glm::vec3(0.85 + 0.4 * nextFloat(randomEngine)));
        newTransformation = glm::rotate(newTransformation, nextFloat(randomEngine), glm::vec3(nextFloat(randomEngine) - 0.5, nextFloat(randomEngine)-0.5, nextFloat(randomEngine)-0.5));
        newTransformation = glm::translate(newTransformation, glm::vec3(nextFloat(randomEngine) - 0.5f, nextFloat(randomEngine) - 0.5f,nextFloat(randomEngine)  - 0.5f));
        innerMesh.setModelTransformationMatrix(newTransformation);

        // Ray triangle test
        bool feasible;
        if(!innerMesh.rayTriangleInside(roughMesh)){
            feasible = false;
        }
        else{
            feasible = !innerMesh.triangleTriangleIntersects(roughMesh);
        }


//        std::cout << "Modeltransformation: " << newTransformation << std::endl;
//        std::cout << "Feasible: " << feasible << std::endl;
//        std::cout << std::endl;

        if(feasible){
            currentTransformation = newTransformation;
            renderWidget->updateWorldSpaceMesh(innerMesh);
        }
        else{
//            innerMesh.setModelTransformationMatrix(currentTransformation);
        }
    }

    auto stopms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    auto totalms = stopms - startms;


    std::cout << totalms << std::endl;

    std::cout << "MPS: " << float(moves)/float(totalms)*1000.0f << std::endl;
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
    const ModelSpaceMesh modelSpaceMesh3 = FileParser::parseFile("../../data/models/DIAMCADrough.stl");
    WorldSpaceMesh worldSpaceMesh3 = WorldSpaceMesh(modelSpaceMesh3);
    renderWidget->addWorldSpaceMesh(worldSpaceMesh3, glm::vec4(1,1,0,1.0));

}

void SandboxTask::run() {

    const ModelSpaceMesh modelSpaceMesh = FileParser::parseFile("../../data/models/bobijn-ascii.stl");
    WorldSpaceMesh worldSpaceMesh = WorldSpaceMesh(modelSpaceMesh, glm::scale(Transformation(1.0f), glm::vec3(1.0f)));
    renderWidget->addWorldSpaceMesh(worldSpaceMesh, glm::vec4(1,0,0,1));

    const ModelSpaceMesh modelSpaceMesh5 = FileParser::parseFile("../../data/models/DIAMCADrough.obj");
    WorldSpaceMesh worldSpaceMesh5 = WorldSpaceMesh(modelSpaceMesh5, glm::translate(Transformation(1.0f), glm::vec3(0,-1,0)));
    worldSpaceMesh5.transform(glm::scale(Transformation(1.0f), glm::vec3(1.2f)));
//    worldSpaceMesh5.transform(Transformation(0.02f));
//    worldSpaceMesh5.transform(glm::rotate(Transformation(1.0f), 1.57f, glm::vec3(1,0,0)));
    renderWidget->addWorldSpaceMesh(worldSpaceMesh5, glm::vec4(1,1,1,0.4));
    auto* first = new std::thread(randomWalk, renderWidget, worldSpaceMesh, worldSpaceMesh5);

//    const ModelSpaceMesh modelSpaceMesh2 = FileParser::parseFile(R"(C:\Users\tolle\CLionProjects\MeshCore\data\models\dragon.obj)");
//    WorldSpaceMesh worldSpaceMesh2 = WorldSpaceMesh(modelSpaceMesh2);
//    worldSpaceMesh2.transform(glm::translate(Transformation(1.0f), glm::vec3(5,0,0)));
//    renderWidget->addWorldSpaceMesh(worldSpaceMesh2, glm::vec4(1,1,1,0.6));
//    auto* second = new std::thread(runRotation2, renderWidget, worldSpaceMesh2);
//
//    const ModelSpaceMesh modelSpaceMesh3 = FileParser::parseFile(R"(C:\Users\tolle\CLionProjects\MeshCore\data\models\apple.stl)");
//    WorldSpaceMesh worldSpaceMesh3 = WorldSpaceMesh(modelSpaceMesh3);
//    renderWidget->addWorldSpaceMesh(worldSpaceMesh3, glm::vec4(1,0,0,1));
//    auto* third = new std::thread(runRotation3, renderWidget, worldSpaceMesh3);
//
//    auto* fourth = new std::thread(test, renderWidget);

}

void SandboxTask::setRenderWidget(RenderWidget *renderWidget) {
    SandboxTask::renderWidget = renderWidget;
}
