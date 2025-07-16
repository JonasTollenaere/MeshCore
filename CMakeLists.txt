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
file(GLOB SOURCES src/acceleration/*.cpp src/core/*.cpp src/geometric/*.cpp src/tasks/*.cpp src/utility/*.cpp src/external/quickhull/*.cpp src/factories/*.cpp src/optimization/*.cpp)
add_library(MeshCoreBase STATIC ${SOURCES})
target_link_libraries(MeshCoreBase PUBLIC TBB::tbb TBB::tbbmalloc Boost::boost GSL::gsl glm::glm)
target_include_directories(MeshCoreBase PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include) # include this directory
target_compile_definitions(MeshCoreBase PUBLIC MESHCORE_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/datasets/")

### Tests ###
# Only uses MeshCoreBase
add_subdirectory(tests)

### Dependencies for user interface: QT, OpenGL, GLEW and giflib ###
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
find_package(Qt5 COMPONENTS Core Widgets Svg REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GIF REQUIRED)
find_package(nlohmann_json REQUIRED)

### MeshCore is the complete library with UI that expands upon MeshCoreBase, requires Qt, OpenGL, ...
file(GLOB UI_SOURCES src/rendering/*.cpp src/rendering/*.h src/external/gifencoder/*.cpp include/meshcore/rendering/*.h)
add_library(MeshCore STATIC ${UI_SOURCES})
target_link_libraries(MeshCore PUBLIC MeshCoreBase Qt5::Widgets Qt5::Core Qt5::Svg OpenGL::GL GLEW::GLEW GIF::GIF nlohmann_json::nlohmann_json)

# Set the shader directory for the MeshCore library
target_compile_definitions(MeshCore PRIVATE MESHCORE_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/rendering/shaders/")
target_compile_definitions(MeshCore PRIVATE MESHCORE_ICON_DIR="${CMAKE_CURRENT_SOURCE_DIR}/icons/")

# Option to build demos, default ON
option(MESHCORE_BUILD_DEMOS "Build demo applications" ON)

# Only include demos if MESHCORE_BUILD_DEMOS is ON
if(MESHCORE_BUILD_DEMOS)

    ### SimpleRenderDemo.exe ###
    add_executable(A-SimpleRenderDemo demos/renderDemo.cpp)
    target_link_libraries(A-SimpleRenderDemo PRIVATE MeshCore)

    ### SimpleCollisionDemo.exe ###
    add_executable(B-SimpleCollisionDemo demos/collisionDemo.cpp)
    target_link_libraries(B-SimpleCollisionDemo MeshCore)

    ### TaskDemo.exe ###
    add_executable(C-TaskDemo demos/taskDemo.cpp)
    target_link_libraries(C-TaskDemo MeshCore)

    ### ConvexHull.exe ###
    add_executable(D-ConvexHull demos/convexHullDemo.cpp)
    target_link_libraries(D-ConvexHull MeshCore)

    ### DistanceDemo.exe ###
    add_executable(E-DistanceDemo demos/distanceDemo.cpp)
    target_link_libraries(E-DistanceDemo MeshCore)

    ### AnimationDemo.exe ###
    add_executable(F-AnimationDemo demos/animationDemo.cpp)
    target_link_libraries(F-AnimationDemo MeshCore)

    ### ViewStripPackingSolution.exe ###
    add_executable(G-ViewStripPackingSolution demos/viewStripPackingSolution.cpp)
    target_link_libraries(G-ViewStripPackingSolution MeshCore)
endif()
