//
// Created by jonas on 11.01.21.
//

#include <cuda.h>
#include <optix_stubs.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>

#include "../core/FileParser.h"
#include "../optix/OptixWorldSpaceMesh.h"
#include "../optix/Exception.h"
#include <random>

float getRandomFloat(float maxValue, std::mt19937 randomEngine){
    auto nextFloat = std::uniform_real_distribution<float>(0, maxValue);
    return nextFloat(randomEngine);
}

int main(int argc, char *argv[]){

    std::mt19937 randomEngine(0);

//    const ModelSpaceMesh innerModelMesh = FileParser::parseFile("../../data/models/bobijn-ascii.stl");
//    const ModelSpaceMesh innerModelMesh = FileParser::parseFile("../../data/models/DIAMCADrough.obj");
    const ModelSpaceMesh innerModelMesh = FileParser::parseFile("../../data/models/DIAMCADbr1.obj");
//    const ModelSpaceMesh innerMesh = FileParser::parseFile("../../data/models/DIAMCADbr1.obj");
    WorldSpaceMesh innerMesh = WorldSpaceMesh(innerModelMesh, glm::scale(Transformation(1.0f), glm::vec3(0.1f)));
    const ModelSpaceMesh modelSpaceMesh5 = FileParser::parseFile("../../data/models/DIAMCADrough.obj");
    WorldSpaceMesh roughMesh = WorldSpaceMesh(modelSpaceMesh5,
                                              glm::translate(Transformation(1.0f), glm::vec3(0, -1, 0)));
    roughMesh.transform(glm::scale(Transformation(1.0f), glm::vec3(1.2f)));

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
    }

    OptixWorldSpaceMesh innerOptixWorldSpaceMesh(innerMesh, cuStream, optixContext);
    OptixWorldSpaceMesh roughOptixWorldSpaceMesh(roughMesh, cuStream, optixContext);

    Transformation currentTransformation = innerMesh.getModelTransformation();
    std::cout << std::boolalpha;
    int moves = 150000;

    auto startms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    for(int i=0; i<moves; i++){

        Transformation newTransformation = glm::scale(currentTransformation, glm::vec3(0.85 + 0.4 * getRandomFloat(1, randomEngine)));
        newTransformation = glm::rotate(newTransformation, getRandomFloat(1, randomEngine), glm::vec3(getRandomFloat(1, randomEngine) - 0.5, getRandomFloat(1, randomEngine)-0.5, getRandomFloat(1,randomEngine)-0.5));
        newTransformation = glm::translate(newTransformation, glm::vec3(getRandomFloat(1, randomEngine) - 0.5f, getRandomFloat(1,randomEngine) - 0.5f,getRandomFloat(1,randomEngine)  - 0.5f));

        innerOptixWorldSpaceMesh.setModelTransformation(newTransformation);

        bool feasible;
        feasible = innerOptixWorldSpaceMesh.isFullyInsideWithStruct(roughOptixWorldSpaceMesh);
        if(feasible){
            currentTransformation = newTransformation;
            innerMesh.setModelTransformationMatrix(currentTransformation);
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
    std::cout << "MPS: " << float(moves)/totals << std::endl;
}