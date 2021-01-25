//
// Created by Jonas on 14/12/2020.
//

#ifndef MESHCORE_ABSTRACTTASK_H
#define MESHCORE_ABSTRACTTASK_H


#include <OpenGLRenderWidget.h>
#include <thread>
#include <random>
#include <TaskWidget.h>
#include "AbstractTaskObserver.h"

class AbstractTask {
private:
    std::thread* thread;
    TaskWidget* taskWidget; // TODO tasks should not contain gui elements, instead use observer pattern
    std::mt19937 randomEngine;
    std::vector<AbstractTaskObserver *> taskObservers;

public:
    AbstractTask();
    explicit AbstractTask(TaskWidget *taskWidget);

    virtual void run() = 0;
    void start();
    void finalize();

    void registerObserver(AbstractTaskObserver* observer);
    void unregisterObserver(AbstractTaskObserver* observer);
    void notifyObserversUpdate() const;
    void notifyObserversProgress(float progress) const;
    void notifyObserversFinished() const;
    void notifyObserversStarted() const;
    void notifyObserversStatus(const std::string& status) const;

public:
    void renderMesh(const WorldSpaceMesh& worldSpaceMesh, const Color& color) const;  // TODO tasks should not contain gui elements
    void updateRenderMesh(const WorldSpaceMesh& worldSpaceMesh) const;

    void setSeed(unsigned int seed);
    float getRandomFloat(float maxValue);


};

#endif //MESHCORE_ABSTRACTTASK_H
