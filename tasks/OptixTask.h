//
// Created by Jonas on 17/12/2020.
//

#ifndef MESHCORE_OPTIXTASK_H
#define MESHCORE_OPTIXTASK_H


#include "AbstractTask.h"

class OptixTask: public AbstractTask {
public:
    using AbstractTask::AbstractTask;
    void run() override;
};


#endif //MESHCORE_OPTIXTASK_H
