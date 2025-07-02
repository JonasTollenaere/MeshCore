//
// Created by Jonas Tollenaere on 02/07/2025.
//

#ifndef STRIPPACKINGSOLUTION_H
#define STRIPPACKINGSOLUTION_H

#include "AbstractSolution.h"
#include "StripPackingProblem.h"
#include "meshcore/core/WorldSpaceMesh.h"
#include "meshcore/factories/AABBFactory.h"

/*
 * Simple solution class for the strip packing problem.
 * Contains basic reference feasibility check, but should be overridden by more specific solutions.
 */
class StripPackingSolution: public AbstractSolution {

    std::shared_ptr<StripPackingProblem> problem;
    std::vector<std::shared_ptr<WorldSpaceMesh>> items;
    std::vector<std::string> itemNames;
    mutable std::vector<std::optional<AABB>> cachedAABBs;

public:
    explicit StripPackingSolution(const std::shared_ptr<StripPackingProblem>& problem);
    StripPackingSolution(const StripPackingSolution& other);

    [[nodiscard]] const std::vector<std::shared_ptr<WorldSpaceMesh>>& getItems() const;
    [[nodiscard]] const std::shared_ptr<WorldSpaceMesh>& getItem(size_t itemIndex) const;
    [[nodiscard]] const std::string& getItemName(size_t itemIndex) const;
    [[nodiscard]] const AABB& getItemAABB(size_t itemIndex) const;
    [[nodiscard]] const std::shared_ptr<StripPackingProblem>& getProblem() const;
    [[nodiscard]] const Transformation& getItemTransformation(size_t itemIndex) const;

    [[nodiscard]] float computeTotalHeight() const;

    // Inherited from AbstractSolution
    [[nodiscard]] bool isFeasible() const override;
    [[nodiscard]] std::shared_ptr<AbstractSolution> clone() const override;

    // Functions that can be overridden in extended classes
    virtual void setItemTransformation(size_t itemIndex, const Transformation& transformation);

    static std::shared_ptr<StripPackingSolution> fromJson(nlohmann::ordered_json& json);
    static std::shared_ptr<StripPackingSolution> fromJson(std::string& path);
    nlohmann::ordered_json toJson() const;
};



#endif //STRIPPACKINGSOLUTION_H
