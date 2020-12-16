#pragma once

#include <optix.h>

#include <stdexcept>
#include <sstream>
#include <iostream>

//------------------------------------------------------------------------------
//
// GL error-checking
//
//------------------------------------------------------------------------------

#define DO_GL_CHECK
#ifdef DO_GL_CHECK
#    define GL_CHECK( call )                                                   \
        do                                                                     \
        {                                                                      \
            call;                                                              \
            GLenum err = glGetError();                                         \
            if( err != GL_NO_ERROR )                                           \
            {                                                                  \
                std::stringstream ss;                                          \
                ss << "GL error " <<  sutil::getGLErrorString( err ) << " at " \
                   << __FILE__  << "(" <<  __LINE__  << "): " << #call         \
                   << std::endl;                                               \
                std::cerr << ss.str() << std::endl;                            \
                throw sutil::Exception( ss.str().c_str() );                    \
            }                                                                  \
        }                                                                      \
        while (0)


#    define GL_CHECK_ERRORS( )                                                 \
        do                                                                     \
        {                                                                      \
            GLenum err = glGetError();                                         \
            if( err != GL_NO_ERROR )                                           \
            {                                                                  \
                std::stringstream ss;                                          \
                ss << "GL error " <<  sutil::getGLErrorString( err ) << " at " \
                   << __FILE__  << "(" <<  __LINE__  << ")";                   \
                std::cerr << ss.str() << std::endl;                            \
                throw sutil::Exception( ss.str().c_str() );                    \
            }                                                                  \
        }                                                                      \
        while (0)

#else
#    define GL_CHECK( call )   do { call; } while(0)
#    define GL_CHECK_ERRORS( ) do { ;     } while(0)
#endif


//------------------------------------------------------------------------------
//
// OptiX error-checking
//
//------------------------------------------------------------------------------

#define OPTIX_CHECK( call )                                                    \
    do                                                                         \
    {                                                                          \
        OptixResult res = call;                                                \
        if( res != OPTIX_SUCCESS )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "Optix call '" << #call << "' failed: " __FILE__ ":"         \
               << __LINE__ << ")\n";                                           \
            throw sutil::Exception( res, ss.str().c_str() );                   \
        }                                                                      \
    } while( 0 )


#define OPTIX_CHECK_LOG( call )                                                \
    do                                                                         \
    {                                                                          \
        OptixResult res = call;                                                \
        const size_t sizeof_log_returned = sizeof_log;                         \
        sizeof_log = sizeof( log ); /* reset sizeof_log for future calls */    \
        if( res != OPTIX_SUCCESS )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "Optix call '" << #call << "' failed: " __FILE__ ":"         \
               << __LINE__ << ")\nLog:\n" << log                               \
               << ( sizeof_log_returned > sizeof( log ) ? "<TRUNCATED>" : "" ) \
               << "\n";                                                        \
            throw sutil::Exception( res, ss.str().c_str() );                   \
        }                                                                      \
    } while( 0 )

// This version of the log-check macro doesn't require the user do setup
// a log buffer and size variable in the surrounding context; rather the
// macro defines a log buffer and log size variable (LOG and LOG_SIZE)
// respectively that should be passed to the message being checked.
// E.g.:
//  OPTIX_CHECK_LOG2( optixProgramGroupCreate( ..., LOG, &LOG_SIZE, ... );
//
#define OPTIX_CHECK_LOG2( call )                                               \
    do                                                                         \
    {                                                                          \
        char               LOG[400];                                           \
        size_t             LOG_SIZE = sizeof( LOG );                           \
                                                                               \
        OptixResult res = call;                                                \
        if( res != OPTIX_SUCCESS )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "Optix call '" << #call << "' failed: " __FILE__ ":"         \
               << __LINE__ << ")\nLog:\n" << LOG                               \
               << ( LOG_SIZE > sizeof( LOG ) ? "<TRUNCATED>" : "" )            \
               << "\n";                                                        \
            throw sutil::Exception( res, ss.str().c_str() );                   \
        }                                                                      \
    } while( 0 )

#define OPTIX_CHECK_NOTHROW( call )                                            \
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

//------------------------------------------------------------------------------
//
// CUDA error-checking
//
//------------------------------------------------------------------------------

#define CUDA_CHECK( call )                                                     \
    do                                                                         \
    {                                                                          \
        cudaError_t error = call;                                              \
        if( error != cudaSuccess )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "CUDA call (" << #call << " ) failed with error: '"          \
               << cudaGetErrorString( error )                                  \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n";                  \
            throw sutil::Exception( ss.str().c_str() );                        \
        }                                                                      \
    } while( 0 )


#define CUDA_SYNC_CHECK()                                                      \
    do                                                                         \
    {                                                                          \
        cudaDeviceSynchronize();                                               \
        cudaError_t error = cudaGetLastError();                                \
        if( error != cudaSuccess )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "CUDA error on synchronize with error '"                     \
               << cudaGetErrorString( error )                                  \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n";                  \
            throw sutil::Exception( ss.str().c_str() );                        \
        }                                                                      \
    } while( 0 )


// A non-throwing variant for use in destructors.
// An iostream must be provided for output (e.g. std::cerr).
#define CUDA_CHECK_NOTHROW( call )                                             \
    do                                                                         \
    {                                                                          \
        cudaError_t error = (call);                                            \
        if( error != cudaSuccess )                                             \
        {                                                                      \
            std::cerr << "CUDA call (" << #call << " ) failed with error: '"  \
               << cudaGetErrorString( error )                                  \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n";                  \
            std::terminate();                                                  \
        }                                                                      \
    } while( 0 )

//------------------------------------------------------------------------------
//
// Assertions
//
//------------------------------------------------------------------------------

#define SUTIL_ASSERT( cond )                                                   \
    do                                                                         \
    {                                                                          \
        if( !(cond) )                                                          \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << __FILE__ << " (" << __LINE__ << "): " << #cond;              \
            throw sutil::Exception( ss.str().c_str() );                        \
        }                                                                      \
    } while( 0 )


#define SUTIL_ASSERT_MSG( cond, msg )                                          \
    do                                                                         \
    {                                                                          \
        if( !(cond) )                                                          \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << (msg) << ": " << __FILE__ << " (" << __LINE__ << "): " << #cond ; \
            throw sutil::Exception( ss.str().c_str() ); \
        }                                                                      \
    } while( 0 )

namespace sutil
{

class Exception : public std::runtime_error
{
 public:
     Exception( const char* msg )
         : std::runtime_error( msg )
     { }

     Exception( OptixResult res, const char* msg )
         : std::runtime_error( createMessage( res, msg ).c_str() )
     { }

 private:
     std::string createMessage( OptixResult res, const char* msg )
     {
         std::ostringstream out;
         out << optixGetErrorName( res ) << ": " << msg;
         return out.str();
     }
};
} // end namespace sutil
