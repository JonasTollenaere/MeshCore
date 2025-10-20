//
// Created by Jonas on 27/01/2021.
//

#ifndef MESHCORE_WORLDSPACEMESH_H
#define MESHCORE_WORLDSPACEMESH_H

#include "ModelSpaceMesh.h"
#include "Transformation.h"
#include <memory>

class WorldSpaceMesh: public GJKConvexShape {

    static int nextId;
    std::string id;
    Transformation modelTransformation;
    std::shared_ptr<ModelSpaceMesh> modelSpaceMesh;

public:
    WorldSpaceMesh();
    explicit WorldSpaceMesh(const std::shared_ptr<ModelSpaceMesh>& modelSpaceMesh);
    explicit WorldSpaceMesh(ModelSpaceMesh&& modelSpaceMesh);
    WorldSpaceMesh(const WorldSpaceMesh &other) = default;

    WorldSpaceMesh(WorldSpaceMesh &&other) noexcept = default;
    WorldSpaceMesh& operator=(WorldSpaceMesh &&other) = default;
    ~WorldSpaceMesh() override = default;

    [[nodiscard]] const std::string& getId() const;

    [[nodiscard]] std::shared_ptr<WorldSpaceMesh> duplicate () const;
    [[nodiscard]] std::shared_ptr<WorldSpaceMesh> clone () const;

    [[nodiscard]] std::shared_ptr<ModelSpaceMesh> getModelSpaceMesh() const;
    [[nodiscard]] std::shared_ptr<ModelSpaceMesh> getTransformedModelSpaceMesh() const;

    [[nodiscard]] AABB computeWorldSpaceAABB() const;

    [[nodiscard]] Transformation& getModelTransformation();
    [[nodiscard]] const Transformation& getModelTransformation() const;
    void setModelTransformation(const Transformation &transformation);

    float getVolume() const;

    // GJKConvexShape interface
    glm::vec3 computeSupport(const glm::vec3 &direction) const override;
    glm::vec3 getCenter() const override;
};


#endif //MESHCORE_WORLDSPACEMESH_H
