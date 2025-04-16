project("MeshCore")
cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

### CPP ###
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

### REQUIRED Dependencies for core functionality: GLM, Boost, GSL, TBB ###
add_compile_definitions(GLM_ENABLE_EXPERIMENTAL)
find_package(glm REQUIRED)
find_package(Boost REQUIRED)
find_package(GSL REQUIRED)
find_package(TBB REQUIRED COMPONENTS tbb tbbmalloc)

### MeshCoreBase library can be used for targets that run without UI ###
file(GLOB SOURCES tasks/*.cpp utility/*.cpp core/*.cpp geometric/*.cpp acceleration/*.cpp external/quickhull/*.cpp )
add_library(MeshCoreBase STATIC ${SOURCES})
target_link_libraries(MeshCoreBase PUBLIC TBB::tbb TBB::tbbmalloc Boost::boost GSL::gsl glm::glm)

### Tests ###
# Only uses MeshCoreBase
include_directories(${CMAKE_CURRENT_SOURCE_DIR}) # include this directory
add_subdirectory(tests)

### Dependencies for user interface: QT, OpenGL, GLEW and giflib ###
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
find_package(Qt6 COMPONENTS Core Widgets Svg OpenGL OpenGLWidgets REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GIF REQUIRED)

### MeshCore is the complete library with UI that expands upon MeshCoreBase, requires Qt, OpenGL, ...
file(GLOB UI_SOURCES rendering/*.cpp external/gifencoder/*.cpp)
add_library(MeshCore STATIC ${UI_SOURCES})
target_link_libraries(MeshCore PUBLIC MeshCoreBase Qt6::Widgets Qt6::Core Qt6::Svg Qt6::OpenGL Qt6::OpenGLWidgets OpenGL::GL GLEW::GLEW GIF::GIF)

### SimpleRenderDemo.exe ###
add_executable(A-SimpleRenderDemo demos/renderDemo.cpp ${SOURCES})
target_link_libraries(A-SimpleRenderDemo PRIVATE MeshCore)

### SimpleCollisionDemo.exe ###
add_executable(B-SimpleCollisionDemo demos/collisionDemo.cpp ${SOURCES})
target_link_libraries(B-SimpleCollisionDemo MeshCore)

### TaskDemo.exe ###
add_executable(C-TaskDemo demos/taskDemo.cpp ${SOURCES})
target_link_libraries(C-TaskDemo MeshCore)

### ConvexHull.exe ###
add_executable(D-ConvexHull demos/convexHullDemo.cpp ${SOURCES})
target_link_libraries(D-ConvexHull MeshCore)

### DistanceDemo.exe ###
add_executable(E-DistanceDemo demos/distanceDemo.cpp ${SOURCES})
target_link_libraries(E-DistanceDemo MeshCore)

### AnimationDemo.exe ###
add_executable(F-AnimationDemo demos/animationDemo.cpp ${SOURCES})
target_link_libraries(F-AnimationDemo MeshCore)
