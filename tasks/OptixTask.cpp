//
// Created by Jonas on 17/12/2020.
//

#include <cuda.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include "OptixTask.h"

#include "../core/FileParser.h"
#include "../optix/OptixWorldSpaceMesh.h"

//#include "../optix/Exception.h"

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* cbdata)
{
    std::cerr << "[" << level << "][" << tag << "]:\t"
              << message << "\n";
}

void OptixTask::run() {

    const ModelSpaceMesh innerModelMesh = FileParser::parseFile("../../data/models/bobijn-ascii.stl");
//    const ModelSpaceMesh innerMesh = FileParser::parseFile("../../data/models/DIAMCADbr1.obj");
    WorldSpaceMesh innerMesh = WorldSpaceMesh(innerModelMesh, glm::scale(Transformation(1.0f), glm::vec3(1.0f)));
    this->renderMesh(innerMesh, glm::vec4(1, 0, 0, 1));
    const ModelSpaceMesh modelSpaceMesh5 = FileParser::parseFile("../../data/models/DIAMCADrough.obj");
    WorldSpaceMesh roughMesh = WorldSpaceMesh(modelSpaceMesh5,
                                              glm::translate(Transformation(1.0f), glm::vec3(0, -1, 0)));
    roughMesh.transform(glm::scale(Transformation(1.0f), glm::vec3(1.2f)));
    this->renderMesh(roughMesh, glm::vec4(1, 1, 1, 0.4));

    //    0.    Create and initialise OptixDeviceContext
    OptixDeviceContext optixContext = nullptr;
    CUstream cuStream;
    {
        cudaFree(nullptr);
        CUcontext cuCtx = nullptr;
        optixInit();
        OptixDeviceContextOptions options = {};
//        options.logCallbackFunction = &context_log_cb;
//        options.logCallbackLevel = 4;
        optixDeviceContextCreate(cuCtx, &options, &optixContext);
        optixDeviceContextSetCacheEnabled(optixContext, 1);

        cudaStreamCreate(&cuStream);
    }

    OptixWorldSpaceMesh innerOptixWorldSpaceMesh(innerMesh, optixContext, cuStream);
    OptixWorldSpaceMesh roughOptixWorldSpaceMesh(roughMesh, optixContext, cuStream);

    Transformation currentTransformation = innerMesh.getModelTransformation();
    std::cout << std::boolalpha;
    int moves = 1500;

    auto startms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    for(int i=0; i<moves; i++){

        Transformation newTransformation = glm::scale(currentTransformation, glm::vec3(0.85 + 0.4 * this->getRandomFloat(1)));
        newTransformation = glm::rotate(newTransformation, this->getRandomFloat(1), glm::vec3(this->getRandomFloat(1) - 0.5, this->getRandomFloat(1)-0.5, this->getRandomFloat(1)-0.5));
        newTransformation = glm::translate(newTransformation, glm::vec3(this->getRandomFloat(1) - 0.5f, this->getRandomFloat(1) - 0.5f,this->getRandomFloat(1)  - 0.5f));

        innerOptixWorldSpaceMesh.setModelTransformation(newTransformation);
//        innerMesh.setModelTransformationMatrix(newTransformation);

        bool feasible;
        feasible = innerOptixWorldSpaceMesh.isFullyInside(roughOptixWorldSpaceMesh);

//        std::cout << "Feasible: " << feasible << std::endl;

//        if(!cudaInnerMesh.rayTriangleInside(cudaRoughMesh)){
//            feasible = false;
//
//            assert(!innerMesh.rayTriangleInside(roughMesh));
////            std::cout << "Vertices not inside" << std::endl;
//        }
//        else{
////            std::cout << "Vertices inside" << std::endl;
//            assert(innerMesh.rayTriangleInside(roughMesh));
//            feasible = !cudaInnerMesh.triangleTriangleIntersects(cudaRoughMesh);
//            if(feasible){
//                assert(!innerMesh.triangleTriangleIntersects(roughMesh)); // TODO fails
////                std::cout << "Triangles don't intersect" << std::endl;
//            }
//            else{
//                assert(innerMesh.triangleTriangleIntersects(roughMesh));
////                std::cout << "Triangles intersect" << std::endl;
//            }
//        }

        if(feasible){
            currentTransformation = newTransformation;

            innerMesh.setModelTransformationMatrix(currentTransformation);
            this->updateRenderMesh(innerMesh);

        }
        else{
//            innerMesh.setModelTransformation(currentTransformation);
        }
    }

    auto stopms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    auto totalms = stopms - startms;

    std::cout << totalms << std::endl;

    std::cout << currentTransformation << std::endl;
    innerMesh.setModelTransformationMatrix(currentTransformation);
    this->updateRenderMesh(innerMesh);
    std::cout << "MPS: " << float(moves)/float(totalms)*1000.0f << std::endl;
}