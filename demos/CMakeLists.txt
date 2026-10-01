### SimpleRenderDemo.exe ###
qt_add_executable(A-SimpleRenderDemo renderDemo.cpp)
target_link_libraries(A-SimpleRenderDemo PRIVATE MeshCore)

### SimpleCollisionDemo.exe ###
qt_add_executable(B-SimpleCollisionDemo collisionDemo.cpp)
target_link_libraries(B-SimpleCollisionDemo PRIVATE MeshCore)

### TaskDemo.exe ###
qt_add_executable(C-TaskDemo taskDemo.cpp)
target_link_libraries(C-TaskDemo PRIVATE MeshCore)

### ConvexHull.exe ###
qt_add_executable(D-ConvexHull convexHullDemo.cpp)
target_link_libraries(D-ConvexHull PRIVATE MeshCore)

### DistanceDemo.exe ###
qt_add_executable(E-DistanceDemo distanceDemo.cpp)
target_link_libraries(E-DistanceDemo PRIVATE MeshCore)

### AnimationDemo.exe ###
qt_add_executable(F-AnimationDemo animationDemo.cpp)
target_link_libraries(F-AnimationDemo PRIVATE MeshCore)

### ViewStripPackingSolution.exe ###
qt_add_executable(G-ViewStripPackingSolution viewStripPackingSolution.cpp)
target_link_libraries(G-ViewStripPackingSolution PRIVATE MeshCore)