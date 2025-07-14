//
// Created by Jonas Tollenaere on 02/07/2025.
//

#ifndef SINGLEVOLUMEMAXIMISATIONPROBLEM_H
#define SINGLEVOLUMEMAXIMISATIONPROBLEM_H

#include "AbstractSolution.h"
#include "meshcore/core/WorldSpaceMesh.h"

class SingleVolumeMaximisationSolution: public AbstractSolution {
    std::shared_ptr<WorldSpaceMesh> itemWorldSpaceMesh;
    std::shared_ptr<WorldSpaceMesh> containerWorldSpaceMesh;

public:
    SingleVolumeMaximisationSolution(const std::shared_ptr<WorldSpaceMesh>& itemWorldSpaceMesh,
                                     const std::shared_ptr<WorldSpaceMesh>& containerWorldSpaceMesh);

    [[nodiscard]] const std::shared_ptr<WorldSpaceMesh>& getItemWorldSpaceMesh() const;
    [[nodiscard]] const std::shared_ptr<WorldSpaceMesh>& getContainerWorldSpaceMesh() const;

    [[nodiscard]] bool isFeasible() const override;
    [[nodiscard]] std::shared_ptr<AbstractSolution> clone() const override;
};



#endif //SINGLEVOLUMEMAXIMISATIONPROBLEM_H
