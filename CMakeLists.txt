project("MeshCore")
cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

### CPP ###
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

### GLM ###
find_package(glm REQUIRED)
message (STATUS "GLM_INCLUDE_DIRS=${GLM_INCLUDE_DIRS}")
include_directories(${GLM_INCLUDE_DIRS})

### CUDA ###
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1/include")  # TODO this shouldn't be necessary

#add_subdirectory(glfw)
add_subdirectory(core)
add_subdirectory(gui)