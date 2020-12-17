### CUDA ###
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
find_package(CUDA REQUIRED)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

### Optix ###
set(CMAKE_MODULE_PATH "${OptiX_INSTALL_DIR}/SDK/CMake")
find_package(OptiX REQUIRED)
include_directories("${OptiX_INCLUDE}")

### HelloOptix.exe ###
file(GLOB SOURCES *.cpp *.cu ../core/*.cpp *.h)
add_executable(HelloOptix ${SOURCES})
set_target_properties(HelloOptix PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

### PTX Compilation ### TODO
#add_library(OptixProgramPtx OBJECT OptixPrograms/OptixPrograms.cu)
#set_property(TARGET OptixProgramPtx PROPERTY CUDA_PTX_COMPILATION ON)
#set_target_properties(OptixProgramPtx PROPERTIES LIBRARY_OUTPUT_DIRECTORY testlibrary )
#add_dependencies(HelloOptix OptixProgramPtx)