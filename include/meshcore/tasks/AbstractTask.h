//
// Created by Jonas on 14/12/2020.
//

#ifndef MESHCORE_ABSTRACTTASK_H
#define MESHCORE_ABSTRACTTASK_H


#include <thread>
#include <atomic>
#include <vector>
#include "AbstractTaskObserver.h"
#include "meshcore/optimization/AbstractSolution.h"

class AbstractTask {

    std::thread* thread;
    std::vector<AbstractTaskObserver *> taskObservers;

protected:
    std::atomic<bool> stopCalled;

public:
    virtual ~AbstractTask() = default;

    AbstractTask();

    virtual void run();
    void start();
    void stop();
    void join();

    void registerObserver(AbstractTaskObserver* observer);
    void unregisterObserver(AbstractTaskObserver* observer);
    void notifyObserversUpdate() const;
    void notifyObserversProgress(float progress) const;
    void notifyObserversStatus(const std::string& status) const;
    void notifyObserversSolution(const std::shared_ptr<const AbstractSolution>& solution) const;
private:
    static void run_static(AbstractTask *task);
    void notifyObserversFinished() const;
    void notifyObserversStarted() const;
};

#endif //MESHCORE_ABSTRACTTASK_H
