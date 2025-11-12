//
// Created by GitHub Copilot on 12/11/2025.
//

#include "gtest/gtest.h"
#include "meshcore/utility/FileParser.h"
#include "meshcore/core/ModelSpaceMesh.h"
#include <fstream>
#include <filesystem>

using namespace std;

static std::string writeTempOBJ(const std::string &content, const std::string &name){
    auto tmpPath = std::filesystem::temp_directory_path() / name;
    std::ofstream ofs(tmpPath.string());
    ofs << content;
    ofs.close();
    return tmpPath.string();
}

TEST(FileParserOBJ, HandlesVariousWhitespaceInVAndF) {
    // OBJ with extra spaces and tabs between vertex components and face indices
    const std::string obj1 =
        "# Test OBJ with varied whitespace\n"
        "v   1.0\t2.0    3.0\n"
        "v  0 0 0\n"
        "v  -1 0 0\n"
        "f   1   2   3\n";

    const std::string file1 = writeTempOBJ(obj1, "meshcore_test_ws1.obj");
    auto mesh1 = FileParser::loadMeshFile(file1);
    ASSERT_NE(mesh1, nullptr);
    EXPECT_EQ(mesh1->getVertices().size(), 3u);
    EXPECT_EQ(mesh1->getTriangles().size(), 1u);
    // Check first vertex coordinates
    const auto &v0 = mesh1->getVertices().at(0);
    EXPECT_FLOAT_EQ(v0.x, 1.0f);
    EXPECT_FLOAT_EQ(v0.y, 2.0f);
    EXPECT_FLOAT_EQ(v0.z, 3.0f);

    // OBJ with faces using slashes and mixed whitespace
    const std::string obj2 =
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "f\t1/1  2/2\t3/3\n";

    const std::string file2 = writeTempOBJ(obj2, "meshcore_test_ws2.obj");
    auto mesh2 = FileParser::loadMeshFile(file2);
    ASSERT_NE(mesh2, nullptr);
    EXPECT_EQ(mesh2->getVertices().size(), 3u);
    EXPECT_EQ(mesh2->getTriangles().size(), 1u);

    // OBJ with leading/trailing whitespace and empty lines
    const std::string obj3 =
        "  \n"
        "\tv  2.5   3.5  -1.25  \n"
        "\n"
        "   f   1   1   1   \n"; // degenerate face but should be parsed into indices

    const std::string file3 = writeTempOBJ(obj3, "meshcore_test_ws3.obj");
    auto mesh3 = FileParser::loadMeshFile(file3);
    ASSERT_NE(mesh3, nullptr);
    EXPECT_EQ(mesh3->getVertices().size(), 1u);
    // Triangulation of degenerate face will likely produce zero triangles; ensure parsing didn't crash
}

