//
// Created by Jonas on 16/11/2020.
//

#include "FileParser.h"
#include <string>
#include <iostream>
#include <fstream>
#include <vector>

#define ASSERT(x) if (!(x)) __debugbreak();  //Compiler specific

ModelSpaceMesh FileParser::parseFile(const std::string &filePath) {
    std::string extension = filePath.substr(filePath.find_last_of('.') + 1);

    if (extension ==  "stl") return parseFileSTL(filePath);
    else if( extension == "obj")  return parseFileOBJ(filePath);
    // Return empty mesh if file extension not supported
    return ModelSpaceMesh(std::vector<Vertex>(), std::vector<Triangle>());
}

ModelSpaceMesh FileParser::parseFileOBJ(const std::string &filePath) {

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
                float value0 = stof(content.substr(0, whitespace0));
                float value1 = stof(content.substr(whitespace0 + 1, whitespace1 - whitespace0 - 1));
                float value2 = stof(content.substr(whitespace1 + 1));
                vertices.emplace_back(Vertex(value0, value1, value2));
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
            }
            else if (type == "vp") {
                std::cout << "vp strings in .obj files not supported" << std::endl;
            }
            else if (type == "l"){
                std::cout << "l strings in .obj files not supported" << std::endl;
            }
        }
    }
    return ModelSpaceMesh(vertices, triangles);
}

ModelSpaceMesh FileParser::parseFileSTL(const std::string &filePath) {

    std::ifstream stream(filePath);
    std::vector<Vertex> vertices;
    std::vector<Triangle> triangles;

    std::string line;
    getline(stream, line);
    ASSERT(line.find("solid") != std::string::npos) // Make sure this isn't a binary stl file


    while(getline(stream, line)){
        auto firstLength = line.find("facet normal");
        if(firstLength != std::string::npos){

            // Begin parsing polygon
            std::vector<Vertex> facetVertices;

            getline(stream, line);
            ASSERT(line.find("outer loop") != std::string::npos)

            getline(stream, line);
            auto vertexIndex = line.find("vertex");
            while(vertexIndex!=std::string::npos){

                line = line.substr(vertexIndex + 6); // Remove leading vertex word

                while(line.find_first_of(' ') == 0) line = line.substr(1); // Remove leading whitespace
                auto doubleIndex = line.find_first_of(' ');
                float x = stof(line.substr(0, doubleIndex));
                line = line.substr(doubleIndex);

                while(line.find_first_of(' ') == 0) line = line.substr(1); // Remove leading whitespace
                doubleIndex = line.find_first_of(' ');
                float y = stof(line.substr(0, doubleIndex));
                line = line.substr(doubleIndex);

                while(line.find_first_of(' ') == 0) line = line.substr(1); // Remove leading whitespace
                doubleIndex = line.find_first_of(' ');
                float z = stof(line.substr(0, doubleIndex));

                vertices.emplace_back(Vertex(x,y,z));

                getline(stream, line);
                vertexIndex = line.find("vertex");
            }

            unsigned int firstIndex = vertices.size();
            triangles.emplace_back(Triangle{firstIndex-3, firstIndex-2, firstIndex-1});

            // End parsing polygon
            ASSERT(line.find("endloop") != std::string::npos)
            getline(stream, line);
            ASSERT(line.find("endfacet") != std::string::npos)
        }
    }

    return ModelSpaceMesh(vertices, triangles);

//    return ModelSpaceMesh(std::vector<Vertex>(), std::vector<Triangle>());
}
