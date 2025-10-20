//
// Created by tolle on 4/02/2023.
//

#ifndef MESHCORE_CACHINGBOUNDSTREEFACTORY_H
#define MESHCORE_CACHINGBOUNDSTREEFACTORY_H

#include <tbb/concurrent_hash_map.h>

#include <memory>
#include "AbstractBoundsTree.h"

template<class Tree>
class CachingBoundsTreeFactory {
public:
    static std::shared_ptr<Tree> getBoundsTree(const std::shared_ptr<ModelSpaceMesh> &modelSpaceMesh) {

        // This map keeps every ModelSpaceMesh allocated because it stores their shared pointers,
        // it might be better to make these trees part of the ModelSpaceMesh class itself.
        // Storing raw or weak pointers causes undefined behavior because another ModelSpaceMesh in the same location would be considered the same key.
        using CacheMap = tbb::concurrent_hash_map<std::shared_ptr<ModelSpaceMesh>, std::shared_ptr<Tree>>;
        static CacheMap cacheMap;

        // 1. Check if cached already
        {
            typename CacheMap::const_accessor accessor; // Read-only lock
            if (cacheMap.find(accessor, modelSpaceMesh)) {
                return accessor->second;
            }
        }

        // 2. Create a new tree and insert into the cache
        {
            auto tree = std::make_shared<Tree>(modelSpaceMesh);
            typename CacheMap::accessor accessor; // Lock for writing
            if (cacheMap.insert(accessor, modelSpaceMesh)) {
                accessor->second = tree;
            } else {
                tree = accessor->second;
            }
            return tree;
        }
    }
};

#endif //MESHCORE_CACHINGBOUNDSTREEFACTORY_H