//
// Created by Jonas on 4/12/2020.
//

#include <QtWidgets>
#include <glm/ext/matrix_transform.hpp>
#include "SandboxTask.h"
#include "../core/FileParser.h"
#include "../core/TriangleTriangleIntersectModified.h"
#include <chrono>

void SandboxTask::run(){

    Vertex v0(-1.17306,1.68758,0.467281);
    Vertex v1(-0.825573,1.34635,0.711619);
    Vertex v2(-0.603772,1.46099,0.592236);

    Vertex u0(-2.40213,0.923508,0.409477);
    Vertex u1(-7.58275,-1.51443,-1.14518);
    Vertex u2(-7.87082,-1.57415,-0.119689);

    MIntersection::ExperimentalTriangleTriangleIntersection(v0, v1, v2, u0, u1, u2);


    const ModelSpaceMesh innerModelMesh = FileParser::parseFile("../../data/models/bobijn-ascii.stl");
//    const ModelSpaceMesh innerMesh = FileParser::parseFile("../../data/models/DIAMCADbr1.obj");
    WorldSpaceMesh innerMesh = WorldSpaceMesh(innerModelMesh, glm::scale(Transformation(1.0f), glm::vec3(1.0f)));
    this->renderMesh(innerMesh, glm::vec4(1, 0, 0, 1));
    const ModelSpaceMesh modelSpaceMesh5 = FileParser::parseFile("../../data/models/DIAMCADrough.obj");
    WorldSpaceMesh roughMesh = WorldSpaceMesh(modelSpaceMesh5, glm::translate(Transformation(1.0f), glm::vec3(0, -1, 0)));
    roughMesh.transform(glm::scale(Transformation(1.0f), glm::vec3(1.2f)));
    this->renderMesh(roughMesh, glm::vec4(1, 1, 1, 0.4));

    std::cout << "Starting Random Walk CPU" << std::endl;

    std::mt19937 randomEngine(0); // TODO set back to 0
    auto nextFloat = std::uniform_real_distribution<float>(0, 1);

    Transformation currentTransformation = innerMesh.getModelTransformation();
    std::cout << std::boolalpha;
    int moves = 1000;

    auto startms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    for(int i=0; i<moves; i++){

        std::cout << i << std::endl;

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

        if(feasible){
            currentTransformation = newTransformation;
            this->updateRenderMesh(innerMesh);
        }
        else{
//            innerMesh.setModelTransformationMatrix(currentTransformation);
        }
    }

    auto stopms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    auto totalms = stopms - startms;

    std::cout << totalms << std::endl;

    std::cout << currentTransformation << std::endl;

    std::cout << "MPS: " << float(moves)/float(totalms)*1000.0f << std::endl;
}