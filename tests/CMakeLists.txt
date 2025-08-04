### GTest ###
find_package(GTest REQUIRED)

### Tests ###
file(GLOB TEST_SRC "*.cpp")
add_executable(Tests ${TEST_SRC} ${DCOPT_SRC})
target_link_libraries(Tests PUBLIC MeshCoreBase GTest::gtest)
gtest_discover_tests(Tests)