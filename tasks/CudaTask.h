//
// Created by Jonas on 12/12/2020.
//

#ifndef MESHCORE_CUDATASK_H
#define MESHCORE_CUDATASK_H


#include <RenderWidget.h>
#include "AbstractTask.h"

class CudaTask: public AbstractTask {
public:
    using AbstractTask::AbstractTask;
    void run() override;
};


#endif //MESHCORE_CUDATASK_H
