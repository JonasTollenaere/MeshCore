set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOMOC ON)
#set(CMAKE_AUTORCC ON)
#set(CMAKE_AUTOUIC ON)

file(GLOB SOURCES *.cpp ../core/*.cpp ../tasks/*.cpp ../cuda/*.cpp ../cuda/*.cu)

find_package(Qt5 COMPONENTS Core Widgets REQUIRED)

### OpenGL ###
find_package(OpenGL REQUIRED)
message (STATUS "OPENGL_INCLUDE_DIR=${OPENGL_INCLUDE_DIR}")
include_directories(${OPENGL_INCLUDE_DIR})
link_libraries(OpenGL::GL)

### GLEW ###  TODO get find_package to work to make this a robust project
add_definitions(-DGLEW_STATIC)
include_directories(${GLEW_DIR}/include)
link_libraries("${GLEW_DIR}/lib/Release/x64/glew32s.lib")

### CUDA ###
enable_language(CUDA)

add_executable(Application ${SOURCES})
target_link_libraries(Application Qt5::Widgets)
set_target_properties(Application PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
