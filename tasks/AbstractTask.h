//
// Created by Jonas on 14/12/2020.
//

#ifndef MESHCORE_ABSTRACTTASK_H
#define MESHCORE_ABSTRACTTASK_H

#include <thread>
#include <random>
#include "AbstractTaskObserver.h"

class AbstractTask {
private:
    std::thread* thread;
    std::mt19937 randomEngine;
    std::vector<AbstractTaskObserver *> taskObservers;

public:
    AbstractTask();

    // Basic Task functionality
    virtual void run() = 0;
    void start();
    void finalize();
    void setSeed(unsigned int seed);

protected:
    float getRandomFloat(float maxValue);

public:
    // TaskObserver functionality
    void registerObserver(AbstractTaskObserver* observer);
    void unregisterObserver(AbstractTaskObserver* observer);
    void notifyObserversUpdate() const;
    void notifyObserversProgress(float progress) const;
    void notifyObserversFinished() const;
    void notifyObserversStarted() const;
    void notifyObserversStatus(const std::string& status) const;

};

#endif //MESHCORE_ABSTRACTTASK_H
