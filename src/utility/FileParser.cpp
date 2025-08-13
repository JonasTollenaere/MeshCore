//
// Created by Jonas on 16/11/2020.
//

#include "meshcore/utility/FileParser.h"
#include "meshcore/utility/io.h"
#include "meshcore/utility/Triangulation.h"
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include <unordered_set>
#include <glm/gtx/hash.hpp>
#include <array>
#include <cstring>

std::mutex FileParser::cacheMapMutex{};
std::unordered_map<std::string, std::weak_ptr<ModelSpaceMesh>> FileParser::meshCacheMap{};

[[maybe_unused]] std::vector<std::shared_ptr<ModelSpaceMesh>> FileParser::parseFolder(const std::string &folderPath) {

    std::vector<std::shared_ptr<ModelSpaceMesh>> modelSpaceMeshes;

    for(const auto& file: std::filesystem::directory_iterator(folderPath)){
        modelSpaceMeshes.emplace_back(FileParser::loadMeshFile(file.path().string()));
    }
    return modelSpaceMeshes;
}

std::shared_ptr<ModelSpaceMesh> FileParser::loadMeshFile(const std::string &filePath) {

    std::locale::global(std::locale("en_US.UTF-8"));

    if(!std::filesystem::exists(filePath)){
        auto absolutePath = std::filesystem::absolute(filePath);
        std::cout << "Warning: File " << absolutePath << " does not exist!" << std::endl;
        return nullptr;
    }

    auto fileName = std::filesystem::path(filePath).filename().string();

    // Check if the file is already cached
    cacheMapMutex.lock();
    const auto cacheIterator = meshCacheMap.find(filePath);
    cacheMapMutex.unlock();
    if(cacheIterator != meshCacheMap.end()){

        if(auto lockedPointer = cacheIterator->second.lock()){
            // We return this if this is no nullptr
            std::cout << "FileParser: cache hit for file " << fileName << std::endl;
            return lockedPointer;
        }

        // Expired weak pointer present in map, erase it
        cacheMapMutex.lock();
        meshCacheMap.erase(cacheIterator);

        // Likely that there are other expired weak pointers in the map, remove them as well
        for(auto iterator = meshCacheMap.begin(); iterator != meshCacheMap.end();){
            if(iterator->second.expired()){
                iterator = meshCacheMap.erase(iterator);
            }
            else{
                ++iterator;
            }
        }
        cacheMapMutex.unlock();
    }

    // Parse the file according to its extension
    std::shared_ptr<ModelSpaceMesh> returnModelSpaceMesh;
    std::string extension = filePath.substr(filePath.find_last_of('.') + 1);
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c){ return std::tolower(c); });
    if (extension ==  "stl") returnModelSpaceMesh = parseFileSTL(filePath);
    else if(extension == "obj") returnModelSpaceMesh = parseFileOBJ(filePath);
    else if(extension == "binvox") returnModelSpaceMesh = parseFileBinvox(filePath);
    else{
        // Return empty mesh if file extension not supported
        std::cout << "Warning: Extension ." << extension << " of file " << filePath << " not supported!" << std::endl;
        return nullptr;
    }
    cacheMapMutex.lock();
    meshCacheMap[filePath] = returnModelSpaceMesh;
    cacheMapMutex.unlock();

    returnModelSpaceMesh->setName(fileName);
    return returnModelSpaceMesh;

}

void FileParser::saveFile(const std::string &filePath, const std::shared_ptr<ModelSpaceMesh>& mesh) {
    std::locale::global(std::locale("en_US.UTF-8"));

    std::string extension = filePath.substr(filePath.find_last_of('.') + 1);

    if(std::filesystem::exists(filePath)){
        std::cout << "Warning: File " << filePath << " already exists!" << std::endl;
        return;
    }

    if(extension == "obj") saveFileOBJ(filePath, mesh);
    else{
        // Return empty mesh if file extension not supported
        std::cout << "Warning: Extension ." << extension << " for saving " << filePath << " not supported!" << std::endl;
        return;
    }
}

std::shared_ptr<ModelSpaceMesh> FileParser::parseFileOBJ(const std::string &filePath) {

    std::ifstream stream(filePath);
    std::vector<Vertex> vertices;
    std::vector<IndexTriangle> triangles;

    std::string line;
    while(getline(stream, line)){

        // Remove comments, if present
        auto commentStartIndex = line.find_first_of('#');
        if(commentStartIndex!=std::string::npos){
            line = line.substr(0, commentStartIndex);
        }

        // Remove trailing whitespace
        auto lastWhiteSpace = line.find_last_of(' ');
        while(lastWhiteSpace!=std::string::npos && lastWhiteSpace == line.size() - 1){
            line = line.substr(0, line.size() - 1);
            lastWhiteSpace = line.find_last_of(' ');
        }

        auto typeLength = line.find_first_of(' ');
        if(typeLength != std::string::npos){
            std::string type = line.substr(0, typeLength);
            std::string content = line.substr(typeLength + 1);

            if(type == "v"){
                auto whitespace0 = content.find_first_of(' ');
                auto whitespace1 = content.find_last_of(' ');
                float value0 = std::stof(content.substr(0, whitespace0));
                float value1 = std::stof(content.substr(whitespace0 + 1, whitespace1 - whitespace0 - 1));
                float value2 = std::stof(content.substr(whitespace1 + 1));
                vertices.emplace_back(value0, value1, value2);
            }
            else if (type == "f"){

                std::vector<size_t> indices;
                auto whitespace = content.find_first_of(' ');
                while(whitespace!=std::string::npos){
                    auto string = content.substr(0, whitespace);
                    indices.emplace_back(stoul(string) - 1);
                    content = content.substr(whitespace + 1);
                    whitespace = content.find_first_of(' ');
                }
                indices.emplace_back(stoul(content) - 1);

                for(const IndexTriangle& triangle: Triangulation::triangulateFace(vertices, IndexFace{indices})){
                    triangles.emplace_back(triangle);
                }

            }
#if !NDEBUG
            else if (type == "vp") {
                std::cout << "vp strings in .obj files not supported" << std::endl;
            }
            else if (type == "l"){
                std::cout << "l strings in .obj files not supported" << std::endl;
            }
#endif

        }
    }
//    return {vertices, triangles};

    // Remove unused vertices O(V) in time, O(V) in memory
    std::vector<Vertex> finalVertices;
    std::vector<IndexTriangle> finalTriangles;
    std::vector<unsigned int> vertexMapping(vertices.size(), -1); // Element at position i contains the index of vertex i in the hullVertices vector, -1 if not present in hullVertices
    auto addVertexIfNotPresent = [&](unsigned int index){
        if(vertexMapping.at(index)==-1){
            vertexMapping.at(index) = finalVertices.size();
            finalVertices.emplace_back(vertices.at(index));
        }
    };

    for (const auto &indexTriangle : triangles){
        // Put the vertices in the hullVertices vector if not present yet
        addVertexIfNotPresent(indexTriangle.vertexIndex0);
        addVertexIfNotPresent(indexTriangle.vertexIndex1);
        addVertexIfNotPresent(indexTriangle.vertexIndex2);

        unsigned int newIndex0 = vertexMapping.at(indexTriangle.vertexIndex0);
        unsigned int newIndex1 = vertexMapping.at(indexTriangle.vertexIndex1);
        unsigned int newIndex2 = vertexMapping.at(indexTriangle.vertexIndex2);
        finalTriangles.emplace_back(newIndex0, newIndex1, newIndex2);
    }

    return std::make_shared<ModelSpaceMesh>(finalVertices, finalTriangles);
}

glm::vec3 readASCIISTLNormalLine(std::string line){
    assert(line.find("facet normal") != std::string::npos);
    auto vertexIndex = line.find("facet normal");
    line = line.substr(vertexIndex + 12); // Remove leading vertex word

    while(line.find_first_of(' ') == 0) line = line.substr(1); // Remove leading whitespace
    auto floatIndex = line.find_first_of(' ');
    float x = std::stof(line.substr(0, floatIndex));
    line = line.substr(floatIndex);

    while(line.find_first_of(' ') == 0) line = line.substr(1); // Remove leading whitespace
    floatIndex = line.find_first_of(' ');
    float y = std::stof(line.substr(0, floatIndex));
    line = line.substr(floatIndex);

    while(line.find_first_of(' ') == 0) line = line.substr(1); // Remove leading whitespace
    floatIndex = line.find_first_of(' ');
    float z = std::stof(line.substr(0, floatIndex));

    return {x,y,z};
}

Vertex readASCIISTLVertexLine(std::ifstream& stream){
    std::string line;
    getline(stream, line);
    assert(line.find("vertex") != std::string::npos);
    auto vertexIndex = line.find("vertex");
    line = line.substr(vertexIndex + 6); // Remove leading vertex word

    while(line.find_first_of(' ') == 0) line = line.substr(1); // Remove leading whitespace
    auto floatIndex = line.find_first_of(' ');
    float x = std::stof(line.substr(0, floatIndex));
    line = line.substr(floatIndex);

    while(line.find_first_of(' ') == 0) line = line.substr(1); // Remove leading whitespace
    floatIndex = line.find_first_of(' ');
    float y = std::stof(line.substr(0, floatIndex));
    line = line.substr(floatIndex);

    while(line.find_first_of(' ') == 0) line = line.substr(1); // Remove leading whitespace
    floatIndex = line.find_first_of(' ');
    float z = std::stof(line.substr(0, floatIndex));

    return {x,y,z};
}

std::shared_ptr<ModelSpaceMesh> FileParser::parseFileSTL(const std::string &filePath) {

    std::ifstream stream(filePath);
    std::string line;
    getline(stream, line);
    if(line.find("solid")==std::string::npos){
        stream.close();
        return FileParser::parseFileBinarySTL(filePath);
    }

    std::unordered_set<Vertex> vertexSet;
    std::vector<Vertex> vertices;
    std::vector<IndexTriangle> triangles;
    while(getline(stream, line)){
        auto firstLength = line.find("facet normal");
        if(firstLength != std::string::npos){

            // Begin parsing polygon
            glm::vec3 facetNormal = readASCIISTLNormalLine(line);
            getline(stream, line);
            assert(line.find("outer loop") != std::string::npos);

            std::vector<unsigned int> indices;
            for(int i=0; i<3; i++){
                Vertex vertex = readASCIISTLVertexLine(stream);

                // We try to find if the vertex already exists, as we don't want to duplicate it
                if(vertexSet.find(vertex)!=vertexSet.end()){
                    auto position = std::find(vertices.rbegin(), vertices.rend(), vertex); // Searching the list backwards significantly speeds up the search
                    auto index = (unsigned int)(vertices.size() - std::distance(vertices.rbegin(), position) - 1);
                    assert(index < vertices.size());
                    assert(index >= 0);
                    indices.emplace_back(index);
                }
                else{
                    indices.emplace_back((unsigned int)vertices.size());
                    vertices.emplace_back(vertex);
                    vertexSet.insert(vertex);
                }
            }
            assert(indices.size()==3);
            triangles.emplace_back(indices[0], indices[1], indices[2]);

            VertexTriangle triangle{vertices[indices[0]], vertices[indices[1]], vertices[indices[2]]};

#if !NDEBUG
            auto calculatedUnitNormal = glm::normalize(triangle.normal);
            if(!glm::all(glm::epsilonEqual(calculatedUnitNormal,facetNormal, 1e-3f))){
                std::cout << "Warning: Parsed normal and calculated normals not equal:" << std::endl;
                std::cout << "Parsed normal: " << facetNormal << std::endl;
                std::cout << "Calculated normal: " << triangle.normal << std::endl;
                std::cout << "Calculated normalized normal: " << glm::normalize(triangle.normal) << std::endl;
            }
#endif


            // Finish parsing polygon
            getline(stream, line);
            assert(line.find("endloop") != std::string::npos);
            getline(stream, line);
            assert(line.find("endfacet") != std::string::npos);
        }
    }
    return std::make_shared<ModelSpaceMesh>(vertices, triangles);
}

float readBinaryFloatLittleEndian(std::ifstream& stream){
    char buffer[4];
    float f;
    stream.read(buffer, 4);
    assert(sizeof(float)==4);
    std::memcpy(&f, buffer, 4);

    return f;
}

Vertex readBinaryVertexLittleEndian(std::ifstream& stream){
    float x = readBinaryFloatLittleEndian(stream);
    float y = readBinaryFloatLittleEndian(stream);
    float z = readBinaryFloatLittleEndian(stream);
    return {x,y,z};
}

unsigned int readBinaryUnsignedIntegerLittleEndian(std::ifstream& stream){
    char buffer[4];
    unsigned int i;
    stream.read(buffer, 4);
    assert(sizeof(unsigned int)==4);
    std::memcpy(&i, buffer, 4);
    return i;
}

std::shared_ptr<ModelSpaceMesh> FileParser::parseFileBinarySTL(const std::string &filePath) {

    std::ifstream stream(filePath, std::ios::binary);

    char header[80];
    stream.read(header, 80);

    unsigned int numTriangles = readBinaryUnsignedIntegerLittleEndian(stream);

    std::unordered_set<Vertex> vertexSet;
    std::vector<Vertex> vertices;
    std::vector<IndexTriangle> triangles;
    triangles.reserve(numTriangles);
    char attributes[2];
    for(unsigned int i=0; i<numTriangles; i++){
        std::vector<unsigned int> indices;
        glm::vec3 normalVector = readBinaryVertexLittleEndian(stream); // TODO verify
        for(int j=0; j<3; j++){
            Vertex vertex = readBinaryVertexLittleEndian(stream);
            // We try to find if the vertex already exists, as we don't want to duplicate it
            if(vertexSet.find(vertex)!=vertexSet.end()){
                auto position = std::find(vertices.rbegin(), vertices.rend(), vertex); // Searching the list backwards significantly speeds up the search
                auto index = (unsigned int)(vertices.size() - std::distance(vertices.rbegin(), position) - 1);
                assert(index < vertices.size());
                indices.emplace_back(index);
            }
            else{
                indices.emplace_back((unsigned int) vertices.size());
                vertices.emplace_back(vertex);
                vertexSet.insert(vertex);
            }
        }
        assert(indices.size()==3);
        triangles.emplace_back(indices[0], indices[1], indices[2]);
        stream.read(attributes, 2);
    }
    return std::make_shared<ModelSpaceMesh>(vertices, triangles);
}

std::string formatDouble(float input){
    std::string str = std::to_string(input);
    str.erase(str.find_last_not_of('0') + 1, std::string::npos);
    str.erase(str.find_last_not_of('.') + 1, std::string::npos);
    return str;
}

std::string formatInteger(unsigned int input){
    std::string str = std::to_string(input);
    // Remove commas if present
    str.erase(std::remove(str.begin(), str.end(), ','), str.end());
    return str;
}

void FileParser::saveFileOBJ(const std::string &filePath, const std::shared_ptr<ModelSpaceMesh> &mesh) {
    std::ofstream basicOfstream(filePath);
    if(!basicOfstream.is_open()){
        throw std::runtime_error("Could not open file " + filePath);
    }

    // TODO think about required saving precision
    // Write vertices

    for (const auto &vertex : mesh->getVertices()) {
        basicOfstream << "v " << formatDouble(vertex.x) << " " << formatDouble(vertex.y) << " " << formatDouble(vertex.z) << std::endl;
    }

    // Write triangles
    for (const auto &triangle : mesh->getTriangles()) {
        basicOfstream << "f " << formatInteger(triangle.vertexIndex0 + 1) << " " << formatInteger(triangle.vertexIndex1 + 1) << " " << formatInteger(triangle.vertexIndex2 + 1) << std::endl;
    }
    basicOfstream.close();
}

std::shared_ptr<ModelSpaceMesh> FileParser::parseFileBinvox(const std::string &filePath) {

    // Open a file stream
    // It's important to add the ios::binary flag to avoid false EOF detection before the end of the file
    std::ifstream stream(filePath, std::ios::in | std::ios::binary);
    std::string line;

    // Parse the first line
    std::getline(stream, line);
    if(line!="#binvox 1"){
        std::cout << "Warning: File " << filePath << " is not a valid binvox file" << std::endl;
    }

    // Parse the dimensions of the grid
    std::getline(stream, line);
    assert(line.substr(0, line.find_first_of(' '))=="dim");
    unsigned int dimensions[3];
    for(unsigned int & dimension : dimensions){
        line = line.substr(line.find_first_of(' ')+1);
        auto currentString = line.substr(0, line.find_first_of(' '));
        dimension = std::stoul(currentString);
    }

    // Parse the translation
    std::getline(stream, line);
    assert(line.substr(0, line.find_first_of(' '))=="translate");
    float translation[3];
    for(float & translationComponent : translation){
        line = line.substr(line.find_first_of(' ')+1);
        auto currentString = line.substr(0, line.find_first_of(' '));
        translationComponent = std::stof(currentString);
    }

    // Parse the scale
    std::getline(stream, line);
    assert(line.substr(0, line.find_first_of(' '))=="scale");
    line = line.substr(line.find_first_of(' ')+1);
    float scale = std::stof(line);
    glm::vec3 scalingVector(scale/float(dimensions[0]), scale/float(dimensions[1]), scale/float(dimensions[2]));

    // Read the data line
    std::getline(stream, line);
    assert(line=="data");

    // Start reading voxel data
    unsigned int voxelCount = dimensions[0]*dimensions[1]*dimensions[2];
    std::vector<bool> voxels;
    voxels.reserve(voxelCount);
    stream.unsetf(std::ios::skipws); // No white space skipping!
    while(!stream.eof()){
        unsigned char isVoxelByte;
        stream >> isVoxelByte;
        if(stream.eof()) break;
        unsigned char newVoxelsByte;
        stream >> newVoxelsByte;
        if(stream.eof()) break;
        bool isVoxel = (isVoxelByte & 0b00000001);
        unsigned int newVoxels = newVoxelsByte;
        assert(newVoxels >= 1u);
        assert(newVoxels <= 255);
        for(unsigned int i=0; i < newVoxels; i++){
            voxels.emplace_back(isVoxel);
        }
    }
    assert(voxels.size() == voxelCount);

    // Convert the voxels to a mesh
    std::vector<Vertex> vertices;
    std::vector<IndexTriangle> triangles;
    for (unsigned int x = 0; x < dimensions[0]; ++x) {
        for (unsigned int z = 0; z < dimensions[1]; ++z) {
            for (unsigned int y = 0; y < dimensions[2]; ++y) {

                // depth, width, height
                // dimensions[0], dimensions[1], dimensions[2]
                // x, z, y
                unsigned int index = x * dimensions[1] * dimensions[2] + z * dimensions[1] + y;  // wxh = width * height = d * d
                if(voxels[index]){
//                if(voxels[z + y * dimensions[0] + x * dimensions[0] * dimensions[1]]){

                    // Add the vertices
                    auto min = glm::vec3(translation[0] + float(x), translation[1] + float(y),translation[2] + float(z));
                    auto max = glm::vec3(translation[0] + float(x + 1), translation[1] + float(y + 1),translation[2] + float(z + 1));
                    min *= scalingVector;
                    max *= scalingVector;

                    vertices.emplace_back(min.x, min.y, min.z);
                    vertices.emplace_back(max.x, min.y, min.z);
                    vertices.emplace_back(max.x, max.y, min.z);
                    vertices.emplace_back(min.x, max.y, min.z);
                    vertices.emplace_back(min.x, min.y, max.z);
                    vertices.emplace_back(max.x, min.y, max.z);
                    vertices.emplace_back(max.x, max.y, max.z);
                    vertices.emplace_back(min.x, max.y, max.z);

                    bool nextVoxelX = (x + 1 < dimensions[0]) && voxels[index + dimensions[1] * dimensions[2]];
                    bool previousVoxelX = (x > 0) && voxels[index - dimensions[1] * dimensions[2]];

                    bool nextVoxelY = (y + 1 < dimensions[2]) && voxels[index + 1];
                    bool previousVoxelY = (y > 0) && voxels[index - 1];

                    bool nextVoxelZ = (z + 1 < dimensions[1]) && voxels[index + dimensions[1]];
                    bool previousVoxelZ = (z > 0) && voxels[index - dimensions[1]];

                    // Add the triangles
                    unsigned int numberOfVertices = vertices.size();
                    if(!previousVoxelZ){
                        triangles.emplace_back(numberOfVertices - 7, numberOfVertices - 8, numberOfVertices - 6);
                        triangles.emplace_back(numberOfVertices - 6, numberOfVertices - 8, numberOfVertices - 5);
                    }
                    if(!nextVoxelZ){
                        triangles.emplace_back(numberOfVertices - 1, numberOfVertices - 4, numberOfVertices - 2);
                        triangles.emplace_back(numberOfVertices - 2, numberOfVertices - 4, numberOfVertices - 3);
                    }
                    if(!previousVoxelX){
                        triangles.emplace_back(numberOfVertices - 5, numberOfVertices - 8, numberOfVertices - 4);
                        triangles.emplace_back(numberOfVertices - 1, numberOfVertices - 5, numberOfVertices - 4);
                    }
                    if(!nextVoxelY){
                        triangles.emplace_back(numberOfVertices - 2, numberOfVertices - 6, numberOfVertices - 5);
                        triangles.emplace_back(numberOfVertices - 2, numberOfVertices - 5, numberOfVertices - 1);
                    }
                    if(!previousVoxelY){
                        triangles.emplace_back(numberOfVertices - 4, numberOfVertices - 7, numberOfVertices - 3);
                        triangles.emplace_back(numberOfVertices - 4, numberOfVertices - 8, numberOfVertices - 7);
                    }

                    if(!nextVoxelX){
                        triangles.emplace_back(numberOfVertices - 3, numberOfVertices - 7, numberOfVertices - 6);
                        triangles.emplace_back(numberOfVertices - 3, numberOfVertices - 6, numberOfVertices - 2);
                    }

                }
            }
        }
    }

    // Remove unused vertices O(V) in time, O(V) in memory
    std::vector<Vertex> finalVertices;
    std::vector<IndexTriangle> finalTriangles;
    std::vector<unsigned int> vertexMapping(vertices.size(), -1); // Element at position i contains the index of vertex i in the hullVertices vector, -1 if not present in hullVertices
    auto addVertexIfNotPresent = [&](unsigned int index){
        if(vertexMapping.at(index)==-1){
            vertexMapping.at(index) = finalVertices.size();
            finalVertices.emplace_back(vertices.at(index));
        }
    };

    for (const auto &indexTriangle : triangles){
        // Put the vertices in the hullVertices vector if not present yet
        addVertexIfNotPresent(indexTriangle.vertexIndex0);
        addVertexIfNotPresent(indexTriangle.vertexIndex1);
        addVertexIfNotPresent(indexTriangle.vertexIndex2);

        unsigned int newIndex0 = vertexMapping.at(indexTriangle.vertexIndex0);
        unsigned int newIndex1 = vertexMapping.at(indexTriangle.vertexIndex1);
        unsigned int newIndex2 = vertexMapping.at(indexTriangle.vertexIndex2);
        finalTriangles.emplace_back(newIndex0, newIndex1, newIndex2);
    }

    return std::make_shared<ModelSpaceMesh>(finalVertices, finalTriangles);
}

void FileParser::clearCache() {
    cacheMapMutex.lock();
    meshCacheMap.clear();
    cacheMapMutex.unlock();
}