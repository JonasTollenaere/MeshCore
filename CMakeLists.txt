project("MeshCore2")
cmake_minimum_required(VERSION 3.3 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)

### GLM ###
find_package(glm 0.9.9 REQUIRED)
message (STATUS "GLM_INCLUDE_DIRS=${GLM_INCLUDE_DIRS}")
include_directories(${GLM_INCLUDE_DIRS})

add_subdirectory(render)
add_subdirectory(core)

