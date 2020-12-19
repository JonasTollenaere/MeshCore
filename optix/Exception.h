#pragma once

#include <optix.h>

#include <stdexcept>
#include <sstream>
#include <iostream>


#if !NDEBUG
static void optix_context_log_cb(unsigned int level, const char* tag, const char* message, void* cbdata)
{
    std::string levelString;
    switch(level) {
        case 4: levelString = "STATUS"; break;
        case 3: levelString = "WARNING"; break;
        case 2: levelString = "ERROR"; break;
        case 1: levelString = "FATAL ERROR"; break;
        default: levelString = "UNKNOWN";
    }

    if(level <= 3) {
        std::cerr << "[OPTIX " << levelString << "][" << tag << "]:\t"
                  << message << "\n";
    }
    else {
        std::cout << "[OPTIX " << levelString << "][" << tag << "]:\t"
                  << message << "\n";
    }
}
#else
static void optix_context_log_cb(unsigned int level, const char* tag, const char* message, void* cbdata) {
}
#endif

#ifdef NDEBUG
#define OPTIX_CALL(call)(call)
#else
#define OPTIX_CALL(call)                                              \
    do                                                                         \
    {                                                                          \
        OptixResult res = call;                                                \
        if( res != OPTIX_SUCCESS )                                             \
        {                                                                      \
            std::cerr << "Optix call '" << #call                               \
                      << "' failed: " __FILE__ ":" << __LINE__ << ")\n";       \
            std::terminate();                                                  \
        }                                                                      \
    } while( 0 )
#endif //NDEBUG

// This version of the log-check macro doesn't require the user do setup
// a log buffer and size variable in the surrounding context; rather the
// macro defines a log buffer and log size variable (LOG and LOG_SIZE)
// respectively that should be passed to the message being checked.
// E.g.:
//  OPTIX_LOG_CALL( optixProgramGroupCreate( ..., LOG, &LOG_SIZE, ... );
#ifdef NDEBUG
#define OPTIX_LOG_CALL(call)                                                   \
    {                                                                          \
        char LOG[1];                                                           \
        size_t LOG_SIZE=1;                                                     \
        (call);                                                                \
    }                                                                          \
    (void)0
#else
#define OPTIX_LOG_CALL(call)                                                   \
    do                                                                         \
    {                                                                          \
        char               LOG[2048];                                          \
        size_t             LOG_SIZE = sizeof( LOG );                           \
                                                                               \
        OptixResult res = call;                                                \
        if( res != OPTIX_SUCCESS )                                             \
        {                                                                      \
            std::cerr                                                          \
                << "Optix call '" << #call << "' failed: " __FILE__ ":"        \
                << __LINE__ << ")\nLog:\n" << LOG                              \
                << ( LOG_SIZE > sizeof( LOG ) ? "<TRUNCATED>" : "" )           \
                << "\n";                                                       \
            std::terminate();                                                  \
        }                                                                      \
    } while(0)
#endif //NDEBUG

#ifdef NDEBUG
#define CUDA_CALL(call)(call)
#else
#define CUDA_CALL(call)                                                        \
    do                                                                         \
    {                                                                          \
        cudaError_t error = (call);                                            \
        if( error != cudaSuccess )                                             \
        {                                                                      \
            std::cerr << "CUDA call (" << #call << " ) failed with error: '"   \
               << cudaGetErrorString( error )                                  \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n";                  \
            std::terminate();                                                  \
        }                                                                      \
    } while(0)
#endif //NDEBUG


