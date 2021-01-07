//
// Created by Jonas on 7/01/2021.
//

#ifndef MESHCORE_LATENCYBENCHMARKTASK_H
#define MESHCORE_LATENCYBENCHMARKTASK_H


#include "AbstractTask.h"

class LatencyBenchMarkTask: public AbstractTask {
public:
    using AbstractTask::AbstractTask;
    void run() override;
};


#endif //MESHCORE_LATENCYBENCHMARKTASK_H
