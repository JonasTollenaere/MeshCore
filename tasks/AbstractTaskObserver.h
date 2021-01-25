//
// Created by Jonas on 25/01/2021.
//

#ifndef MESHCORE_ABSTRACTTASKOBSERVER_H
#define MESHCORE_ABSTRACTTASKOBSERVER_H

#include <iostream>

class AbstractTaskObserver {
public:
    virtual void notify() {};

    virtual void notifyProgress(float progress) {}

    virtual void notifyFinished() {}

    virtual void notifyStarted() {}

    virtual void notifyStatus(const std::string& status) {}
};


#endif //MESHCORE_ABSTRACTTASKOBSERVER_H
