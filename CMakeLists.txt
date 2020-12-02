project("MeshCore")
cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

### GLM ###
find_package(glm 0.9.9 REQUIRED)
message (STATUS "GLM_INCLUDE_DIRS=${GLM_INCLUDE_DIRS}")
include_directories(${GLM_INCLUDE_DIRS})

add_subdirectory(render)
add_subdirectory(core)
add_subdirectory(gui)
add_subdirectory(guiTest)

