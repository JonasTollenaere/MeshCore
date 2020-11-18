//
// Created by Jonas on 16/11/2020.
//

#include "FileParser.h"
#include <string>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

ModelSpaceMesh FileParser::parseFile(const std::string &filePath) {
    std::string extension = filePath.substr(filePath.find_last_of('.') + 1);

    if (extension ==  "stl") return parseFileSTL(filePath);
    else if( extension == "obj")  return parseFileOBJ(filePath);
    // Return empty mesh if file extension not supported
    return ModelSpaceMesh(std::vector<Vertex>(), std::vector<Triangle>());
}

ModelSpaceMesh FileParser::parseFileOBJ(const std::string &filePath) {

    // TODO add support for polygonal faces, normals, etc..

    std::ifstream stream(filePath);
    std::vector<Vertex> vertices;
    std::vector<Triangle> triangles;

    std::string line;
    while(getline(stream, line)){
        auto typeLength = line.find_first_of(' ');
        if(typeLength != std::string::npos){
            std::string type = line.substr(0, typeLength);
            std::string content = line.substr(typeLength + 1);


            if(type == "v"){
                auto whitespace0 = content.find_first_of(' ');
                auto whitespace1 = content.find_last_of(' ');
                auto value0 = stod(content.substr(0, whitespace0));
                auto value1 = stod(content.substr(whitespace0 + 1, whitespace1 - whitespace0 - 1));
                auto value2 = stod(content.substr(whitespace1 + 1));
                vertices.emplace_back(value0, value1, value2);
            }
            else if (type == "f"){

                std::vector<unsigned int> indices;
                unsigned int whitespace = content.find_first_of(' ');
                while(content.find_first_of(' ')!=std::string::npos){
                    auto string = content.substr(0, whitespace);
                    indices.emplace_back(stoul(string) - 1);
                    content = content.substr(whitespace + 1);
                    whitespace = content.find_first_of(' ');
                }
                indices.emplace_back(stoul(content) - 1);

                for(int i=1; i+1 < indices.size(); i++){
                    triangles.emplace_back(Triangle{indices[0], indices[i], indices[i+1]});
                }




//                auto whitespace0 = content.find_first_of(' ');
//                auto whitespace1 = content.find_last_of(' ');
//                auto string0 = content.substr(0, whitespace0);
//                auto string1 = content.substr(whitespace0 + 1, whitespace1 - whitespace0 - 1);
//                auto string2 = content.substr(whitespace1 + 1);
//                auto slash0 = string0.find_first_of('/');
//                auto slash1 = string1.find_first_of('/');
//                auto slash2 = string2.find_first_of('/');
//                auto index0 = stoul(string0.substr(0, slash0)) - 1;
//                auto index1 = stoul(string1.substr(0, slash1)) - 1;
//                auto index2 = stoul(string2.substr(0, slash2)) - 1;
//                triangles.emplace_back(Triangle{index0, index1, index2});
            }
        }
    }
    return ModelSpaceMesh(vertices, triangles);
}

ModelSpaceMesh FileParser::parseFileSTL(const std::string &filePath) {
    return ModelSpaceMesh(std::vector<Vertex>(), std::vector<Triangle>());
}
