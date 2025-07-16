### SimpleRenderDemo.exe ###
add_executable(A-SimpleRenderDemo renderDemo.cpp)
target_link_libraries(A-SimpleRenderDemo MeshCore)

### SimpleCollisionDemo.exe ###
add_executable(B-SimpleCollisionDemo collisionDemo.cpp)
target_link_libraries(B-SimpleCollisionDemo MeshCore)

### TaskDemo.exe ###
add_executable(C-TaskDemo taskDemo.cpp)
target_link_libraries(C-TaskDemo MeshCore)

### ConvexHull.exe ###
add_executable(D-ConvexHull convexHullDemo.cpp)
target_link_libraries(D-ConvexHull MeshCore)

### DistanceDemo.exe ###
add_executable(E-DistanceDemo distanceDemo.cpp)
target_link_libraries(E-DistanceDemo MeshCore)

### AnimationDemo.exe ###
add_executable(F-AnimationDemo animationDemo.cpp)
target_link_libraries(F-AnimationDemo MeshCore)

### ViewStripPackingSolution.exe ###
add_executable(G-ViewStripPackingSolution viewStripPackingSolution.cpp)
target_link_libraries(G-ViewStripPackingSolution MeshCore)