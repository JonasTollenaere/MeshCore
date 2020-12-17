//
// Created by Jonas on 17/12/2020.
//

#include <cuda.h>
#include "OptixTask.h"
#include "../core/WorldSpaceMesh.h"
#include "../core/FileParser.h"
//#include "../optix/OptixWorldSpaceMesh.h"
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
//    OptixDeviceContext optixContext = nullptr;
//    CUstream cuStream;
//    {
//        CUDA_CHECK(cudaFree(nullptr));
//        CUcontext cuCtx = nullptr;
//        OPTIX_CHECK(optixInit());
//        OptixDeviceContextOptions options;
//        options.logCallbackFunction = &context_log_cb;
//        options.logCallbackLevel = 4;
//        OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &optixContext));
//        OPTIX_CHECK(optixDeviceContextSetCacheEnabled(optixContext, 1));
//
//        CUDA_CHECK(cudaStreamCreate(&cuStream));
//    }

//    OptixWorldSpaceMesh optixWorldSpaceMesh();
}