//
//
//#include <iostream>
//
//#include <chrono>
//
//#include <optix_function_table_definition.h>
//#include <optix_stubs.h>
//
//#include <glm/gtc/type_ptr.hpp>
//
//#include "Exception.h"
//#include "../core/WorldSpaceMesh.h"
//#include "../core/FileParser.h"
//
//#include <string>
//#include <fstream>
//
//#include "OptixLaunchParameters.h"
//#include "OptixData.h"
//
//template <typename T>
//struct SbtRecord
//{
//    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
//    T data;
//};
//typedef SbtRecord<RayGenData> RayGenSbtRecord;
//typedef SbtRecord<EdgeIntersectionTestData> EdgeIntersectionSbtRecord;
//
//
//static void optix_context_log_cb( unsigned int level, const char* tag, const char* message, void* cbdata)
//{
//    std::cerr << "[" << level << "][" << tag << "]:\t"
//              << message << "\n";
//}
//
//static float3 vec3ToFloat3(const Vertex& vertex){
//    return make_float3(vertex.x, vertex.y, vertex.z);
//}
//
//
//int main(){
//
//    const ModelSpaceMesh roughModelSpaceMesh = FileParser::parseFile("../../data/models/DIAMCADrough.obj");
//    WorldSpaceMesh roughWorldSpaceMesh = WorldSpaceMesh(roughModelSpaceMesh, glm::translate(Transformation(1.0f), glm::vec3(0, -1, 0)));
////    roughWorldSpaceMesh.transform(glm::scale(Transformation(1.0f), glm::vec3(1.2f)));
//
//    const ModelSpaceMesh innerModelSpaceMesh = FileParser::parseFile("../../data/models/DIAMCADbr1.obj");
//    WorldSpaceMesh innerWorldSpaceMesh = WorldSpaceMesh(innerModelSpaceMesh, glm::translate(Transformation(1.0f), glm::vec3(0, -1, 0)));
//    innerWorldSpaceMesh.transform(glm::scale(Transformation(1.0f), glm::vec3(0.2f)));
//
//
//    try {
//
////    0.    Create and initialise OptixDeviceContext
//        OptixDeviceContext optixContext = nullptr;
//        CUstream cuStream;
//        {
//            CUDA_CHECK(cudaFree(nullptr));
//            CUcontext cuCtx = nullptr;
//            OPTIX_CHECK(optixInit());
//            OptixDeviceContextOptions options;
//            options.logCallbackFunction = &optix_context_log_cb;
//            options.logCallbackLevel = 4;
//            OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &optixContext));
//            OPTIX_CHECK(optixDeviceContextSetCacheEnabled(optixContext, 1));
//
//            CUDA_CHECK(cudaStreamCreate(&cuStream));
//        }
////    1.    Create acceleration structures that represent a geometry mesh in the scene and select
////          on or more records in the shader binding table for each mesh.
//
//        // Specify options for the build.
//        OptixAccelBuildOptions accelOptions = {};
//        accelOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;
//        accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
//        accelOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
////        accelOptions.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_COMPACTION; // TODO see documentation 5.7 (don't do this if this degrades tracing performance)
//
//        // Prepare data to build
//        unsigned int numberOfVertices = roughWorldSpaceMesh.getModelSpaceMesh().vertices.size();
//        std::vector<float3> modelSpaceVertices;
//        CUdeviceptr d_outerModelSpaceVertices;
//        for(Vertex vertex: roughWorldSpaceMesh.getModelSpaceMesh().vertices){
//            modelSpaceVertices.emplace_back(vec3ToFloat3(vertex));
//        }
//        unsigned int verticesBytes = numberOfVertices * sizeof(float3);
//        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void **>(&d_outerModelSpaceVertices), verticesBytes, cuStream));
//        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(d_outerModelSpaceVertices), modelSpaceVertices.data(), verticesBytes, cudaMemcpyHostToDevice, cuStream));
//
//        unsigned int numberOfTriangles = roughWorldSpaceMesh.getModelSpaceMesh().triangles.size();
//        std::vector<unsigned int> triangleIndices;
//        CUdeviceptr d_outerTriangleIndices;
//        for(Triangle triangle: roughWorldSpaceMesh.getModelSpaceMesh().triangles){
//            triangleIndices.emplace_back(triangle.vertexIndex0);
//            triangleIndices.emplace_back(triangle.vertexIndex1);
//            triangleIndices.emplace_back(triangle.vertexIndex2);
//        }
//        unsigned int indicesBytes = numberOfTriangles * sizeof(float3);
//        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void **>(&d_outerTriangleIndices), indicesBytes, cuStream));
//        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(d_outerTriangleIndices), triangleIndices.data(), indicesBytes, cudaMemcpyHostToDevice, cuStream));
//
//        // Populate the build input struct with our triangle data as well as
//        // information about the sizes and types of our data
//        OptixBuildInput buildInput = {};
//        buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
//        buildInput.triangleArray.vertexBuffers = &d_outerModelSpaceVertices;
//        buildInput.triangleArray.numVertices = numberOfVertices;
//        buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
//        buildInput.triangleArray.vertexStrideInBytes = 0;
//        buildInput.triangleArray.indexBuffer = d_outerTriangleIndices;
//        buildInput.triangleArray.numIndexTriplets = numberOfTriangles;
//        buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
//        buildInput.triangleArray.indexStrideInBytes = 0;
//        buildInput.triangleArray.numSbtRecords = 1;
//        buildInput.triangleArray.preTransform = 0;
//        unsigned int triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT }; // Setting this flag usuallyimproves performance even if no any-edgeIntersection program is present in the SBT
//        buildInput.triangleArray.flags = triangle_input_flags;
//
//        accelOptions.motionOptions.numKeys = 0; // A numKeys value of zero specifies no motion blur
//
//        OptixAccelBufferSizes bufferSizes;
//
//        OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext, &accelOptions, &buildInput, 1, &bufferSizes));
//        CUdeviceptr d_outputGAS;
//        CUdeviceptr d_temp;
//        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_outputGAS), bufferSizes.outputSizeInBytes, cuStream));
//        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_temp), bufferSizes.tempSizeInBytes, cuStream));
//
//
//        OptixTraversableHandle modelSpaceGASHandle = 0;
//        OPTIX_CHECK(optixAccelBuild(optixContext, cuStream,       // This runs asyncrhonously on the GPU
//                                              &accelOptions, &buildInput, 1, d_temp,
//                                    bufferSizes.tempSizeInBytes, d_outputGAS,
//                                    bufferSizes.outputSizeInBytes, &modelSpaceGASHandle,
//                                    nullptr, 0)); // Last 2 elements used when compacting
//
//        float transform[12] = {2,0,0,0,
//                               0,2,0,0,
//                               0,0,2,0}; // TODO set current worldspacemesh transform
//        std::cout << "{";
//        for(int i=0; i<12; i++){
//            std::cout << transform[i] << ",";
//        }
//        std::cout << "}";
//        std::cout << std::endl;
//        std::cout << "{";
//        for(int i=0; i<16; i++){
//            std::cout << glm::value_ptr(glm::transpose(roughWorldSpaceMesh.getModelTransformation()))[i] << ",";
//        }
//        std::cout << "}";
//        std::cout << std::endl;
//
//        const float* transform2 = glm::value_ptr(glm::transpose(roughWorldSpaceMesh.getModelTransformation()));
//        std::cout << "{";
//        for(int i=0; i<12; i++){
//            std::cout << transform2[i] << ",";
//        }
//        std::cout << "}";
//        std::cout << std::endl;
//
//        OptixStaticTransform modelTransformation;
//        memcpy(modelTransformation.transform, transform, sizeof(float)*12);
//        memcpy(modelTransformation.invTransform, transform, sizeof(float)*12);
//        modelTransformation.child = modelSpaceGASHandle;
//
//        CUdeviceptr d_modelTransformation;
//        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void **>(&d_modelTransformation), sizeof(OptixStaticTransform), cuStream));
//        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(d_modelTransformation), &modelTransformation, sizeof(OptixStaticTransform), cudaMemcpyHostToDevice, cuStream));
//        OptixTraversableHandle worldSpaceHandle = 0;
//        OPTIX_CHECK(optixConvertPointerToTraversableHandle(optixContext, d_modelTransformation, OPTIX_TRAVERSABLE_TYPE_STATIC_TRANSFORM, &worldSpaceHandle));
//
////    2.    Create a pipeline of programs that contains all programs that will be invoked during a ray tracing launch.
//
//        // Set the options for module compilation
//        OptixModuleCompileOptions moduleCompileOptions = {};
//        moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
//        moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
//        moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
//
//        // Set the options for pipeline compilation
//        OptixPipelineCompileOptions pipelineCompileOptions = {};
//        pipelineCompileOptions.usesMotionBlur = false;
//        pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
//        pipelineCompileOptions.numPayloadValues = 0;
//        pipelineCompileOptions.numAttributeValues = 0;
//        pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParameters";
//        pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;  // Improves performance
//
//        // Depending on the scenario and combination of flags, enabling exceptions can lead to severe overhead, so some flags shouldbe mainly used in internal and debug builds.
////        pipelineCompileOptions.exceptionFlags = (OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_USER);
//        pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
//        char logString[4096];
//        size_t logStringSize = sizeof(logString);
//
//        // Compile the module based on the OptixProgram.ptx
//        OptixModule ptxModule;
//        std::ifstream t("../../optix/OptixPrograms/OptixPrograms.ptx");
//        std::string ptxString((std::istreambuf_iterator<char>(t)),
//                              std::istreambuf_iterator<char>());
//        OPTIX_CHECK(optixModuleCreateFromPTX(optixContext,
//                                                 &moduleCompileOptions,
//                                                 &pipelineCompileOptions,
//                                                 ptxString.c_str(),
//                                                 ptxString.size(),
//                                                 logString,
//                                                 &logStringSize,
//                                                 &ptxModule));
//
//        std::cout << logString << std::endl;
//
//        // Use the modules to create the necessary programgroups (RAYGEN + HITGROUP + MISS)
//        OptixProgramGroupDesc programGroupDescriptions[3] = {};
//
//        programGroupDescriptions[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
//        programGroupDescriptions[0].raygen.module = ptxModule;
//        programGroupDescriptions[0].raygen.entryFunctionName = "__raygen__edgeIntersectionTest__";
//        programGroupDescriptions[1].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
//        programGroupDescriptions[1].hitgroup.moduleCH = ptxModule;
//        programGroupDescriptions[1].hitgroup.entryFunctionNameCH = "__closesthit__edgeIntersectionTest__";
////      As a special case, the intersection program is not required – and is ignored – for triangle primitives.
//        programGroupDescriptions[2].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
//        programGroupDescriptions[2].miss.module = nullptr;
//        programGroupDescriptions[2].miss.entryFunctionName = nullptr;
//
//        OptixProgramGroupOptions pgOptions = {};
//        OptixProgramGroup programGroups[3];
//        OPTIX_CHECK(optixProgramGroupCreate(optixContext,
//                                                programGroupDescriptions,
//                                                3,
//                                                &pgOptions,
//                                                logString, &logStringSize,
//                                                programGroups));
//        std::cout << logString << std::endl;
//
//        // Create a pipeline with these program groups
//        OptixPipeline pipeline = nullptr;
//        OptixPipelineLinkOptions pipelineLinkOptions = {};
//        pipelineLinkOptions.maxTraceDepth = 16;              // TODO check what this does exactly
//        pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
//        OPTIX_CHECK(optixPipelineCreate(optixContext,
//                                        &pipelineCompileOptions,
//                                        &pipelineLinkOptions,
//                                        programGroups, 3,
//                                        logString, &logStringSize,
//                                        &pipeline));
//
//        std::cout << logString << std::endl;
//
////    3.    Create a shader binding table that includes references to these programs and their parameters.
//
//        OptixShaderBindingTable sbt = {};
//        RayGenSbtRecord rayGenRecord;
//        rayGenRecord.data = {};
//        std::vector<float3> origins;
//        std::vector<float3> directions;
//
//                for(Triangle triangle: innerWorldSpaceMesh.getModelSpaceMesh().triangles){
//            Vertex v0 = innerWorldSpaceMesh.getModelSpaceMesh().vertices[triangle.vertexIndex0];
//            Vertex v1 = innerWorldSpaceMesh.getModelSpaceMesh().vertices[triangle.vertexIndex1];
//            Vertex v2 = innerWorldSpaceMesh.getModelSpaceMesh().vertices[triangle.vertexIndex2];
//            origins.emplace_back(vec3ToFloat3(v0));
//            directions.emplace_back(vec3ToFloat3(v1-v0));
//            origins.emplace_back(vec3ToFloat3(v1));
//            directions.emplace_back(vec3ToFloat3(v2-v1));
//            origins.emplace_back(vec3ToFloat3(v2)); // TODO testing 2 edges should be enough?
//            directions.emplace_back(vec3ToFloat3(v0-v2));
//        }
//        float3* d_origins;
//        float3* d_directions;
//        CUDA_CHECK(cudaMallocAsync(&d_origins, origins.size() * sizeof(float3), cuStream));
//        CUDA_CHECK(cudaMallocAsync(&d_directions, directions.size() * sizeof(float3), cuStream));
//        CUDA_CHECK(cudaMemcpyAsync(d_origins, origins.data(), origins.size()*sizeof(float3), cudaMemcpyHostToDevice, cuStream));
//        CUDA_CHECK(cudaMemcpyAsync(d_directions, directions.data(), directions.size()*sizeof(float3), cudaMemcpyHostToDevice, cuStream));
//        rayGenRecord.data.origins = d_origins;
//        rayGenRecord.data.directions = d_directions;
//        CUdeviceptr d_raygenRecord;
//        const size_t raygen_record_size = sizeof( RayGenSbtRecord );
//        OPTIX_CHECK(optixSbtRecordPackHeader(programGroups[0], &rayGenRecord));
//        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_raygenRecord), raygen_record_size, cuStream));
//        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(d_raygenRecord), &rayGenRecord, sizeof(RayGenSbtRecord), cudaMemcpyHostToDevice, cuStream));
//        sbt.raygenRecord = d_raygenRecord;
//
//        EdgeIntersectionSbtRecord hitGroupRecord = {};
//        hitGroupRecord.data.edgeIntersection = false;
//        CUdeviceptr d_hitGroupRecord;
//        OPTIX_CHECK(optixSbtRecordPackHeader(programGroups[1], &hitGroupRecord));
//        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void **>(&d_hitGroupRecord), sizeof(EdgeIntersectionSbtRecord), cuStream));
//        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(d_hitGroupRecord), &hitGroupRecord, sizeof(EdgeIntersectionSbtRecord), cudaMemcpyHostToDevice, cuStream));
//        sbt.hitgroupRecordBase = d_hitGroupRecord;
//        sbt.hitgroupRecordStrideInBytes = sizeof(EdgeIntersectionSbtRecord);
//        sbt.hitgroupRecordCount = 1;
//
//
//        CUdeviceptr d_missRecord;
//        char missHeader[OPTIX_SBT_RECORD_HEADER_SIZE];
//        OPTIX_CHECK(optixSbtRecordPackHeader(programGroups[2], &missHeader));
//        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void **>(&d_missRecord), OPTIX_SBT_RECORD_HEADER_SIZE, cuStream));
//        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(d_missRecord), &missHeader, OPTIX_SBT_RECORD_HEADER_SIZE, cudaMemcpyHostToDevice, cuStream));
//        sbt.missRecordBase = d_missRecord;
//        sbt.missRecordCount = 1;
//        sbt.missRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
//
//        int width = origins.size();
//        std::cout << "Width: " << width << std::endl;
//
////    4.    Launch a device-side kernel that will invoke a ray generation program with a multitude of threads calling
////          optixTrace to begin traversal and the execution of the other programs.
//
//        // Set the launch parameters and copy them to device memory
//        OptixLaunchParameters optixLaunchParameters = {};
//        optixLaunchParameters.handle = worldSpaceHandle;
//
//        CUdeviceptr d_optixLaunchParameters;
//        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_optixLaunchParameters), sizeof(optixLaunchParameters), cuStream));
//        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_optixLaunchParameters), &optixLaunchParameters, sizeof(optixLaunchParameters),cudaMemcpyHostToDevice, cuStream));
//
//        CUDA_CHECK(cudaStreamSynchronize(cuStream));
//        bool* d_hitPointer = &(reinterpret_cast<EdgeIntersectionSbtRecord*>(d_hitGroupRecord))->data.edgeIntersection;
//        unsigned int tests=10000;
//
//        auto startms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
//
//
//        for(int i=0;i<tests;i++){
//            CUDA_CHECK(cudaMemcpyAsync(d_hitPointer, &hitGroupRecord.data.edgeIntersection, sizeof(bool), cudaMemcpyHostToDevice, cuStream));
//            OPTIX_CHECK(optixLaunch(pipeline, cuStream, d_optixLaunchParameters, sizeof(OptixLaunchParameters), &sbt, width, 1, 1));
//            CUDA_CHECK(cudaMemcpyAsync(&hitGroupRecord.data.edgeIntersection, d_hitPointer, sizeof(bool), cudaMemcpyDeviceToHost, cuStream));
//            CUDA_CHECK(cudaStreamSynchronize(cuStream));
//        }
//        auto stopms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
//        auto totalMs = stopms - startms;
//        auto totalS = totalMs/1000.0f;
//
//        std::cout << "Tracing time: " << totalMs << " ms." << std::endl;
//
//        std::cout << "MPS: " << float(tests)/totalS << " (potential maximum)" << std::endl;
//        std::cout << "RPS: " << float(tests*width)/totalS << std::endl;
//
//
//        CUDA_CHECK(cudaStreamSynchronize(cuStream));
//        std::cout << "Returnvalue: " << hitGroupRecord.data.edgeIntersection << std::endl;
//
//        // Finalize TODO check if these can be called earlier to save VRAM
//        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void *>(d_temp), cuStream));
//        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void *>(d_outputGAS), cuStream));
//        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void *>(d_outerModelSpaceVertices), cuStream));
//        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void *>(d_outerTriangleIndices), cuStream));
//        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void *>(d_optixLaunchParameters), cuStream));
//        CUDA_CHECK(cudaFreeAsync(d_origins, cuStream));
//        CUDA_CHECK(cudaFreeAsync(d_directions, cuStream));
//
//    }
//    catch( std::exception& e )
//    {
//        std::cerr << "Caught exception: " << e.what() << "\n";
//        return 1;
//    }
//
//    return 0;
//}