file(GLOB SOURCES ../core/*.cpp *.cpp)

add_executable(meshGUI ${SOURCES})

message((GUI) Sources: ${SOURCES})

### OpenGL ###
find_package(OpenGL REQUIRED)
message (STATUS "OPENGL_INCLUDE_DIR=${OPENGL_INCLUDE_DIR}")
include_directories(${OPENGL_INCLUDE_DIR})

### GLEW ###
add_definitions(-DGLEW_STATIC)
include_directories(glew-2.1.0/include)
link_libraries("C:/Users/tolle/CLionProjects/OptiXSandBox/meshJonas/glew-2.1.0/lib/Release/x64/glew32s.lib")

### GLFW ###
add_subdirectory(glfw)
target_link_libraries(meshGUI glfw)
target_link_libraries(meshGUI OpenGL::GL)