//
// Created by jonas on 01.03.21.
//

#ifndef MESHCORE_CONSOLETASKOBSERVER_H
#define MESHCORE_CONSOLETASKOBSERVER_H

#include "AbstractTaskObserver.h"

class ConsoleTaskObserver: public AbstractTaskObserver {

    std::string status;
    float progress;
    void update();

public:
    void notifyStatus(const std::string &newStatus) override;
    void notifySolution(const std::shared_ptr<const AbstractSolution>& solution) override;
    void notifyProgress(float newProgress) override;
    void notifyStarted(const std::string& taskName) override;
    void notifyFinished() override;
};


#endif //MESHCORE_CONSOLETASKOBSERVER_H
