//
// Created by Jonas on 10/11/2020.
//

#include "Renderer.h"
#include "VertexBufferLayout.h"
#include "../core/FileParser.h"


int main() {

    const std::string path = "../../data/models/DIAMCADbr1.obj";
    const ModelSpaceMesh brilliantMesh = FileParser::parseFile(path);
    WorldSpaceMesh brilliantWorldSpaceMesh(brilliantMesh);

    const std::string path2 = "../../data/models/DIAMCADrough.obj";
    const ModelSpaceMesh roughMesh = FileParser::parseFile(path2);
    WorldSpaceMesh roughWorldSpaceMesh(roughMesh);

    const std::string path3 = "../../data/models/DIAMCADbr2.obj";
    const ModelSpaceMesh brilliantMesh2 = FileParser::parseFile(path3);
    WorldSpaceMesh brilliantWorldSpaceMesh2(brilliantMesh2);

    const std::string path4 = "../../data/models/dragon.obj";
    const ModelSpaceMesh dragonMesh = FileParser::parseFile(path4);
    const WorldSpaceMesh dragonWorldSpaceMesh(dragonMesh);

    const std::string path5 = "../../data/models/dagger.stl";
    const ModelSpaceMesh appleMesh = FileParser::parseFile(path5);
    const WorldSpaceMesh appleWorldSpaceMesh(appleMesh);

    Renderer renderer = Renderer();
    renderer.addWorldSpaceMesh(appleWorldSpaceMesh);
    renderer.addWorldSpaceMesh(brilliantWorldSpaceMesh2, Color(0,0.5f,0,1));
    renderer.addWorldSpaceMesh(brilliantWorldSpaceMesh, Color(1,0,0,1));
    renderer.addWorldSpaceMesh(roughWorldSpaceMesh, Color(0.6627f, 0.6627f, 0.6627f, 0.4f));

    renderer.run();

    return 0;
}



