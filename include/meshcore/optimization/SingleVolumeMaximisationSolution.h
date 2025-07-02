//
// Created by Jonas Tollenaere on 02/07/2025.
//

#ifndef SINGLEVOLUMEMAXIMISATIONPROBLEM_H
#define SINGLEVOLUMEMAXIMISATIONPROBLEM_H

#include "AbstractSolution.h"
#include "meshcore/core/WorldSpaceMesh.h"

class SingleVolumeMaximisationSolution: public AbstractSolution {
    std::shared_ptr<WorldSpaceMesh> innerWorldSpaceMesh;
    std::shared_ptr<WorldSpaceMesh> outerWorldSpaceMesh;

    bool isFeasible() const override {

    }
    [[nodiscard]] virtual std::shared_ptr<AbstractSolution> clone() const = 0;
};



#endif //SINGLEVOLUMEMAXIMISATIONPROBLEM_H
