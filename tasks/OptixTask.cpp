//
// Created by Jonas on 17/12/2020.
//

#include <cuda.h>
#include <optix_stubs.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "OptixTask.h"

#include "../core/FileParser.h"
#include "../optix/Exception.h"
#include "../optix/OptixWorldSpaceMeshWithInstance.h"

void OptixTask::run() {

//    const ModelSpaceMesh innerModelMesh = FileParser::parseFile("../../data/models/bobijn-ascii.stl");
//    const ModelSpaceMesh innerModelMesh = FileParser::parseFile("../../data/models/DIAMCADrough.obj");
    const ModelSpaceMesh innerModelMesh = FileParser::parseFile("../../data/models/DIAMCADbr1.obj");
//    const ModelSpaceMesh innerMesh = FileParser::parseFile("../../data/models/DIAMCADbr1.obj");
    WorldSpaceMesh innerMesh = WorldSpaceMesh(innerModelMesh, glm::scale(Transformation(1.0f), glm::vec3(0.1f)));
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
        CUDA_CALL(cudaFree(nullptr)); // This initialises the cuda context
        OPTIX_CALL(optixInit());
        OptixDeviceContextOptions options = {};
#if !NDEBUG
        options.logCallbackFunction = &optix_context_log_cb;
        options.logCallbackLevel = 4;
#endif
        OPTIX_CALL(optixDeviceContextCreate(nullptr, &options, &optixContext));
        OPTIX_CALL(optixDeviceContextSetCacheEnabled(optixContext, 1));
        cuStream = nullptr;
//        CUDA_CALL(cudaStreamCreate(&cuStream));
        // Using streams seems to have a significant overhead when using the WDDM drivers, which is the only option for GeForce cards in windows.


    }

    OptixWorldSpaceMeshWithInstance innerOptixWorldSpaceMesh(innerMesh, cuStream, optixContext);
    OptixWorldSpaceMeshWithInstance roughOptixWorldSpaceMesh(roughMesh, cuStream, optixContext);

    Transformation currentTransformation = innerMesh.getModelTransformation();
    std::cout << std::boolalpha;
    int moves = 150000;

    auto startms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    for(int i=0; i<moves; i++){

        Transformation newTransformation = glm::scale(currentTransformation, glm::vec3(0.85 + 0.4 * this->getRandomFloat(1)));
        newTransformation = glm::rotate(newTransformation, this->getRandomFloat(1), glm::vec3(this->getRandomFloat(1) - 0.5, this->getRandomFloat(1)-0.5, this->getRandomFloat(1)-0.5));
        newTransformation = glm::translate(newTransformation, glm::vec3(this->getRandomFloat(1) - 0.5f, this->getRandomFloat(1) - 0.5f,this->getRandomFloat(1)  - 0.5f));

        innerOptixWorldSpaceMesh.setModelTransformation(newTransformation);

        bool feasible;
        feasible = innerOptixWorldSpaceMesh.intersectsWithInstance(roughOptixWorldSpaceMesh);
        if(feasible){
            currentTransformation = newTransformation;
            innerMesh.setModelTransformationMatrix(currentTransformation);
            updateRenderMesh(innerMesh);
        }
//        if(i%1000==0){
//            innerMesh.setModelTransformationMatrix(currentTransformation);
//            updateRenderMesh(innerMesh);
//        }
    }

    auto stopms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    auto totalms = stopms - startms;
    auto totals = totalms/1000.0f;

    std::cout << totalms << " ms." << std::endl;

    std::cout << currentTransformation << std::endl;
    innerMesh.setModelTransformationMatrix(currentTransformation);
    this->updateRenderMesh(innerMesh);
    std::cout << "MPS: " << float(moves)/totals << std::endl;
}