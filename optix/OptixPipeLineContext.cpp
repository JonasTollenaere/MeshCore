////
//// Created by Jonas on 19/12/2020.
////
//
//#include "OptixPipeLineContext.h"
//#include <optix.h>
//#include <fstream>
//#include <optix_stubs.h>
//#include "Exception.h"
//
//OptixPipeLineContext::OptixPipeLineContext(const OptixDeviceContext& optixDeviceContext) {
//
//    ///    2.    Create a edgeIntersectionPipeline of programs that contains all programs that will be invoked during a ray tracing launch.
//    // Set the options for module compilation
//    OptixModuleCompileOptions moduleCompileOptions = {};
//    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
//    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
//#if NDEBUG
//    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
//#else
//    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
//#endif
//
//    // Set the options for edgeIntersectionPipeline compilation
//    OptixPipelineCompileOptions pipelineCompileOptions = {};
//    pipelineCompileOptions.usesMotionBlur = false;
//    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
//    pipelineCompileOptions.numPayloadValues = 0;
//    pipelineCompileOptions.numAttributeValues = 0;
//    pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParameters";
//    pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;  // Improves performance
//
//    // Depending on the scenario and combination of flags, enabling exceptions can lead to severe overhead, so some flags shouldbe mainly used in internal and debug builds.
//#if NDEBUG
//    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
//#else
//    pipelineCompileOptions.exceptionFlags |= OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
//    pipelineCompileOptions.exceptionFlags |= OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_USER;
//    pipelineCompileOptions.exceptionFlags |= OPTIX_EXCEPTION_FLAG_TRACE_DEPTH;
//#endif
//
//    // Compile the module based on the OptixProgram.ptx
//    OptixModule ptxModule;
//    std::ifstream t("../../optix/OptixPrograms/OptixPrograms.ptx");
//    std::string ptxString((std::istreambuf_iterator<char>(t)),
//                          std::istreambuf_iterator<char>());
//
//    OPTIX_LOG_CALL(optixModuleCreateFromPTX(this->optixDeviceContext,
//                                            &moduleCompileOptions,
//                                            &pipelineCompileOptions,
//                                            ptxString.c_str(),
//                                            ptxString.size(),
//                                            LOG,
//                                            &LOG_SIZE,
//                                            &ptxModule));
//
//    // Use the modules to create the necessary programgroups (RAYGEN + HITGROUP + MISS)
//    OptixProgramGroupDesc programGroupDescriptions[3] = {};
//    programGroupDescriptions[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
//    programGroupDescriptions[0].raygen.module = ptxModule;
//    programGroupDescriptions[0].raygen.entryFunctionName = "__raygen__edgeIntersectionTest__";
//    programGroupDescriptions[1].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
//    programGroupDescriptions[1].hitgroup.moduleCH = ptxModule;
//    programGroupDescriptions[1].hitgroup.entryFunctionNameCH = "__closesthit__edgeIntersectionTest__";
////      As a special case, the intersection program is not required – and is ignored – for triangle primitives.
//    programGroupDescriptions[2].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
//    programGroupDescriptions[2].miss.module = nullptr;
//    programGroupDescriptions[2].miss.entryFunctionName = nullptr;
//
//    OptixProgramGroupOptions pgOptions = {};
//    OPTIX_LOG_CALL(optixProgramGroupCreate(this->optixDeviceContext,
//                                           programGroupDescriptions,
//                                           3,
//                                           &pgOptions,
//                                           LOG, &LOG_SIZE,
//                                           edgeIntersectionProgramGroups));
//
//    // Create a edgeIntersectionPipeline with these program groups
//    OptixPipelineLinkOptions pipelineLinkOptions = {};
//    pipelineLinkOptions.maxTraceDepth = 1;
//    pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
//    OPTIX_LOG_CALL(optixPipelineCreate(this->optixDeviceContext,
//                                       &pipelineCompileOptions,
//                                       &pipelineLinkOptions,
//                                       edgeIntersectionProgramGroups, 3,
//                                       LOG, &LOG_SIZE,
//                                       &edgeIntersectionPipeline));
//
//}
//
//OptixProgramGroup const *OptixPipeLineContext::getEdgeIntersectionProgramGroups() const {
//    return edgeIntersectionProgramGroups;
//}
////
////OptixPipeLineContext::~OptixPipeLineContext() {
////
////}
