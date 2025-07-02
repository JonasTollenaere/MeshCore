//
// Created by Jonas on 25/01/2021.
//

#ifndef MESHCORE_ABSTRACTTASKOBSERVER_H
#define MESHCORE_ABSTRACTTASKOBSERVER_H


#include "meshcore/optimization/AbstractSolution.h"
#include <iostream>

class AbstractTaskObserver {
public:
    virtual void notify() {
        std::cout << "This notify should be overridden!" << std::endl;
    };

    virtual void notifyProgress(float progress) {
        std::cout << "Progress: ";
        printf("%.2f", progress);
        std::cout << " This notifyProgress should be overridden!" << std::endl;
    }

    virtual void notifyFinished() {
        std::cout << "This notifyFinished should be overridden!" << std::endl;
    }

    virtual void notifyStarted() {
        std::cout << "This notifyStarted should be overridden!" << std::endl;
    }

    virtual void notifyStatus(const std::string& status) {
        std::cout << "This notifyStatus should be overridden: " << status << std::endl;
    }

    virtual void notifySolution(const std::shared_ptr<const AbstractSolution>& solution){
        std::cout << "This notifySolution should be overridden!" << std::endl;
    }
};


#endif //MESHCORE_ABSTRACTTASKOBSERVER_H
