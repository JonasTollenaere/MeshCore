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
        using CacheMap = tbb::concurrent_hash_map<ModelSpaceMesh*, std::shared_ptr<Tree>>;
        static CacheMap cacheMap;
        ModelSpaceMesh* rawPtr = modelSpaceMesh.get();

        // 1. Check if cached already
        {
            typename CacheMap::const_accessor accessor;
            if (cacheMap.find(accessor, rawPtr)) {
                return accessor->second;
            }
        }

        // 2. Create a new tree
        {
            auto tree = std::make_shared<Tree>(modelSpaceMesh);
            typename CacheMap::accessor accessor;
            if (cacheMap.insert(accessor, rawPtr)) {
                accessor->second = tree;
            } else {
                tree = accessor->second;
            }
            return tree;
        }
    }
};

#endif //MESHCORE_CACHINGBOUNDSTREEFACTORY_H
