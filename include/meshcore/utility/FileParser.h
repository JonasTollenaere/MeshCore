//
// Created by Jonas on 16/11/2020.
//

#ifndef MESHCORE2_FILEPARSER_H
#define MESHCORE2_FILEPARSER_H
#include <string>
#include <unordered_map>
#include <mutex>
#include <memory>

#include "meshcore/core/ModelSpaceMesh.h"

class FileParser {
public:
    static std::shared_ptr<ModelSpaceMesh> loadMeshFile(const std::string& filePath);
    static void saveFile(const std::string& filePath, const std::shared_ptr<ModelSpaceMesh>&);

    static void clearCache();

    [[maybe_unused]] static std::vector<std::shared_ptr<ModelSpaceMesh>> parseFolder(const std::string& folderPath);
private:

    static std::mutex cacheMapMutex;
    static std::unordered_map<std::string, std::weak_ptr<ModelSpaceMesh>> meshCacheMap;

    static std::shared_ptr<ModelSpaceMesh> parseFileSTL(const std::string& filePath);
    static std::shared_ptr<ModelSpaceMesh> parseFileOBJ(const std::string& filePath);
    static std::shared_ptr<ModelSpaceMesh> parseFileBinvox(const std::string& filePath);
    static void saveFileOBJ(const std::string& filePath, const std::shared_ptr<ModelSpaceMesh>& mesh);
    static std::shared_ptr<ModelSpaceMesh> parseFileBinarySTL(const std::string &filePath);
};

#endif //MESHCORE2_FILEPARSER_H
