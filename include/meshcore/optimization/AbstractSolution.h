//
// Created by Jonas on 17/09/2022.
//

#ifndef MESHCORE_ABSTRACTSOLUTION_H
#define MESHCORE_ABSTRACTSOLUTION_H

#include <memory>

struct AbstractSolution{
    virtual ~AbstractSolution() = default;
    [[nodiscard]] virtual bool isFeasible() const = 0;
    [[nodiscard]] virtual std::shared_ptr<AbstractSolution> clone() const = 0;
};

#endif //MESHCORE_ABSTRACTSOLUTION_H
