//
// Created by Jonas on 16/11/2020.
//

#ifndef MESHCORE2_FILEPARSER_H
#define MESHCORE2_FILEPARSER_H
#include <string>
#include "ModelSpaceMesh.h"

class FileParser {
public:
    static ModelSpaceMesh parseFile(const std::string& filePath);
private:
    static ModelSpaceMesh parseFileSTL(const std::string& filePath);
    static ModelSpaceMesh parseFileOBJ(const std::string& filePath);
};

#endif //MESHCORE2_FILEPARSER_H
