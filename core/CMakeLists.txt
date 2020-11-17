file(GLOB SOURCES *.cpp)
message(Sources: ${SOURCES})
#add_executable(meshMain ${SOURCES})

add_executable(meshMain Main.cpp Triangle.h ModelSpaceMesh.cpp Vertex.cpp FileParser.cpp)