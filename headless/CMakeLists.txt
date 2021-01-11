set(CMAKE_INCLUDE_CURRENT_DIR ON)

### OpenGL ###
find_package(OpenGL REQUIRED)
link_libraries(OpenGL::GL)

#### GLEW ###
#add_definitions(-DGLEW_STATIC)
#include_directories(${GLEW_DIR}/include)
#link_libraries("${GLEW_DIR}/lib/Release/x64/glew32s.lib")

### CUDA ###
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
find_package(CUDA 11.2 REQUIRED)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
#message(CUDA_LIBRARIES: ${CUDA_LIBRARIES})
#message(CUDA_CUDA_LIBRARY: ${CUDA_CUDA_LIBRARY})
#link_libraries("${CUDA_LIBRARIES}")
#link_libraries("${CUDA_CUDA_LIBRARY}")

### Optix ###
set(CMAKE_MODULE_PATH "${OptiX_INSTALL_DIR}/SDK/CMake")
find_package(OptiX REQUIRED)
include_directories("${OptiX_INCLUDE}")

### Headless.exe ###
file(GLOB SOURCES ../core/*.cpp ../cuda/*.cpp ../cuda/*.cu ../optix/*.cu ../optix/*.h)
add_executable(Headless ${SOURCES} Main.cpp)
set_target_properties(Headless PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(Latency ${SOURCES} Latency.cpp)
set_target_properties(Latency PROPERTIES CUDA_SEPARABLE_COMPILATION ON)