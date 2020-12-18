set(CMAKE_INCLUDE_CURRENT_DIR ON)

### QT ###
set(CMAKE_AUTOMOC ON)
#set(CMAKE_AUTORCC ON)
#set(CMAKE_AUTOUIC ON)
find_package(Qt5 COMPONENTS Core Widgets REQUIRED)
link_libraries(Qt5::Widgets)

### OpenGL ###
find_package(OpenGL REQUIRED)
link_libraries(OpenGL::GL)

### GLEW ###
add_definitions(-DGLEW_STATIC)
include_directories(${GLEW_DIR}/include)
link_libraries("${GLEW_DIR}/lib/Release/x64/glew32s.lib")

### CUDA ###
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
find_package(CUDA 11.2 REQUIRED)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

### Optix ###
set(CMAKE_MODULE_PATH "${OptiX_INSTALL_DIR}/SDK/CMake")
find_package(OptiX REQUIRED)
include_directories("${OptiX_INCLUDE}")

### Application.exe ###
file(GLOB SOURCES *.cpp ../core/*.cpp ../tasks/*.cpp ../cuda/*.cpp ../cuda/*.cu ../optix/*.cu ../optix/*.h)
add_executable(Application ${SOURCES})
set_target_properties(Application PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
