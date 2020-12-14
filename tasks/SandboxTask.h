//
// Created by Jonas on 4/12/2020.
//

#ifndef MESHCORE_SANDBOXTASK_H
#define MESHCORE_SANDBOXTASK_H


#include <RenderWidget.h>
#include "AbstractTask.h"

class SandboxTask: public AbstractTask {

public:
    using AbstractTask::AbstractTask;
    void run() override;
};


#endif //MESHCORE_SANDBOXTASK_H
