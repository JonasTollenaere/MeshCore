//
// Created by Jonas on 12/12/2020.
//

#include <random>
#include "CudaTask.h"
#include "../core/FileParser.h"
#include "../cuda/CudaWorldSpaceMesh.h"
#include "../cuda/CudaStream.h"

void randomWalk2(RenderWidget *renderWidget, WorldSpaceMesh& innerMesh, const WorldSpaceMesh& roughMesh){

    std::cout << "Starting Random Walk GPU" << std::endl;

    auto startms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    std::mt19937 randomEngine(0);
    auto nextFloat = std::uniform_real_distribution<float>(0, 1);

//    const CudaStream cudaStream;

    CudaWorldSpaceMesh cudaInnerMesh(innerMesh);
    CudaWorldSpaceMesh cudaRoughMesh(roughMesh);

    Transformation currentTransformation = innerMesh.getModelTransformation();
    std::cout << std::boolalpha;
    int moves = 20;
    for(int i=0; i<moves; i++){

        Transformation newTransformation = glm::scale(currentTransformation, glm::vec3(0.85 + 0.4 * nextFloat(randomEngine)));
        newTransformation = glm::rotate(newTransformation, nextFloat(randomEngine), glm::vec3(nextFloat(randomEngine) - 0.5, nextFloat(randomEngine)-0.5, nextFloat(randomEngine)-0.5));
        newTransformation = glm::translate(newTransformation, glm::vec3(nextFloat(randomEngine) - 0.5f, nextFloat(randomEngine) - 0.5f,nextFloat(randomEngine)  - 0.5f));

        cudaInnerMesh.setModelTransformationMatrix(newTransformation);
        innerMesh.setModelTransformationMatrix(newTransformation);

        // Ray triangle test
        bool feasible;
        if(!cudaInnerMesh.rayTriangleInside(cudaRoughMesh)){
            feasible = false;

            assert(!innerMesh.rayTriangleInside(roughMesh));
//            std::cout << "Vertices not inside" << std::endl;
        }
        else{
//            std::cout << "Vertices inside" << std::endl;
            assert(innerMesh.rayTriangleInside(roughMesh));
            feasible = !cudaInnerMesh.triangleTriangleIntersects(cudaRoughMesh);
            if(feasible){
                assert(!innerMesh.triangleTriangleIntersects(roughMesh)); // TODO fails
//                std::cout << "Triangles don't intersect" << std::endl;
            }
            else{
                assert(innerMesh.triangleTriangleIntersects(roughMesh));
//                std::cout << "Triangles intersect" << std::endl;
            }
        }

        if(feasible){
            currentTransformation = newTransformation;
            innerMesh.setModelTransformationMatrix(currentTransformation);
            renderWidget->updateWorldSpaceMesh(innerMesh);
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

void CudaTask::run() {

    const ModelSpaceMesh modelSpaceMesh = FileParser::parseFile("../../data/models/bobijn-ascii.stl");
    WorldSpaceMesh worldSpaceMesh = WorldSpaceMesh(modelSpaceMesh, glm::scale(Transformation(1.0f), glm::vec3(1.0f)));
    renderWidget->addWorldSpaceMesh(worldSpaceMesh, glm::vec4(1,0,0,1));

    const ModelSpaceMesh modelSpaceMesh5 = FileParser::parseFile("../../data/models/DIAMCADrough.obj");
    WorldSpaceMesh worldSpaceMesh5 = WorldSpaceMesh(modelSpaceMesh5, glm::translate(Transformation(1.0f), glm::vec3(0,-1,0)));
    worldSpaceMesh5.transform(glm::scale(Transformation(1.0f), glm::vec3(1.2f)));
//    worldSpaceMesh5.transform(Transformation(0.02f));
//    worldSpaceMesh5.transform(glm::rotate(Transformation(1.0f), 1.57f, glm::vec3(1,0,0)));
    renderWidget->addWorldSpaceMesh(worldSpaceMesh5, glm::vec4(1,1,1,0.4));
    auto* first = new std::thread(randomWalk2, renderWidget, worldSpaceMesh, worldSpaceMesh5);

}

void CudaTask::setRenderWidget(RenderWidget *renderWidget) {
    CudaTask::renderWidget = renderWidget;
}
