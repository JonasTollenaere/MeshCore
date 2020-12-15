### CUDA ###
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
find_package(CUDA REQUIRED)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

### Optix ###
set(CMAKE_MODULE_PATH "${OptiX_INSTALL_DIR}/SDK/CMake")
find_package(OptiX REQUIRED)
include_directories("${OptiX_INCLUDE}")

add_executable(HelloOptix ../optix/HelloOptix.cu)
set_target_properties(HelloOptix PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
