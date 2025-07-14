//
// Created by Jonas Tollenaere on 02/07/2025.
//

#include "meshcore/optimization/SingleVolumeMaximisationSolution.h"

#include "meshcore/acceleration/BoundingVolumeHierarchy.h"
#include "meshcore/acceleration/CachingBoundsTreeFactory.h"
#include "meshcore/core/Ray.h"

SingleVolumeMaximisationSolution::SingleVolumeMaximisationSolution(
    const std::shared_ptr<WorldSpaceMesh> &itemWorldSpaceMesh,
    const std::shared_ptr<WorldSpaceMesh> &containerWorldSpaceMesh):
    itemWorldSpaceMesh(itemWorldSpaceMesh), containerWorldSpaceMesh(containerWorldSpaceMesh) {}

const std::shared_ptr<WorldSpaceMesh> & SingleVolumeMaximisationSolution::getItemWorldSpaceMesh() const {
    return itemWorldSpaceMesh;
}

const std::shared_ptr<WorldSpaceMesh> & SingleVolumeMaximisationSolution::getContainerWorldSpaceMesh() const {
    return containerWorldSpaceMesh;
}

bool SingleVolumeMaximisationSolution::isFeasible() const {

    // Containment: At least one vertex of the item should be inside the outer mesh, point inclusion check using ray`
    const Transformation& itemTransformation =  itemWorldSpaceMesh->getModelTransformation();
    const Transformation& outerTransformation =  containerWorldSpaceMesh->getModelTransformation();
    glm::mat4 itemModelToOuterModelSpaceTransformation = outerTransformation.getInverseMatrix() * itemTransformation.getMatrix();
    Vertex transformedInnerVertex = itemModelToOuterModelSpaceTransformation * glm::vec4(itemWorldSpaceMesh->getModelSpaceMesh()->getVertices()[0],1);
    const auto& tree = CachingBoundsTreeFactory<BoundingVolumeHierarchy>::getBoundsTree(containerWorldSpaceMesh->getModelSpaceMesh());
    if(!tree->containsPoint(transformedInnerVertex)){
        return false;
    }

    // Intersection
    return !Intersection::intersect(*this->itemWorldSpaceMesh, *this->containerWorldSpaceMesh);

}

std::shared_ptr<AbstractSolution> SingleVolumeMaximisationSolution::clone() const {
    return std::make_shared<SingleVolumeMaximisationSolution>(itemWorldSpaceMesh->clone(), containerWorldSpaceMesh->clone());
}
