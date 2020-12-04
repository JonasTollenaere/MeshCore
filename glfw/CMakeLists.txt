file(GLOB SOURCES ../core/*.cpp *.cpp)

add_executable(GLFWRenderer ${SOURCES})

message((GUI) Sources: ${SOURCES})

### OpenGL ###
find_package(OpenGL REQUIRED)
message (STATUS "OPENGL_INCLUDE_DIR=${OPENGL_INCLUDE_DIR}")
include_directories(${OPENGL_INCLUDE_DIR})
link_libraries(OpenGL::GL)

### GLEW ###  TODO get find_package to work to make this a robust project
add_definitions(-DGLEW_STATIC)
include_directories(${GLEW_DIR}/include)
link_libraries("${GLEW_DIR}/lib/Release/x64/glew32s.lib")

### GLFW ### TODO get find_package to work to make this a robust project
add_subdirectory(glfw)
target_link_libraries(GLFWRenderer glfw)
link_libraries(glfw)