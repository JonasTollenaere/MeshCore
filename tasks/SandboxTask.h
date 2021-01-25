//
// Created by Jonas on 4/12/2020.
//

#ifndef MESHCORE_SANDBOXTASK_H
#define MESHCORE_SANDBOXTASK_H

#include "AbstractTask.h"
#include "../core/WorldSpaceMesh.h"

class SandboxTask: public AbstractTask {
private:
    WorldSpaceMesh innerWorldSpaceMesh;
    WorldSpaceMesh outerWorldSpaceMesh;
public:
    SandboxTask();
    void run() override;

    const WorldSpaceMesh& getOuterWorldSpaceMesh() const;
    const WorldSpaceMesh& getInnerWorldSpaceMesh() const;
};

#endif //MESHCORE_SANDBOXTASK_H
