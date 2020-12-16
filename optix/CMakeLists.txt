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
file(GLOB SOURCES *.cpp *.cu ../core/*.cpp)
add_executable(HelloOptix ${SOURCES})
set_target_properties(HelloOptix PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
