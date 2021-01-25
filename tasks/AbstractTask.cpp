//
// Created by Jonas on 14/12/2020.
//

#include "AbstractTask.h"

static void run_static(AbstractTask* task){
    task->run();
}

void AbstractTask::start() {
    if(thread == nullptr){
        thread = new std::thread(run_static, this);
    }
    else std::cout << "Task already started" << std::endl;
}

void AbstractTask::finalize() {
    thread->join();
    delete thread;
}

AbstractTask::AbstractTask():
thread(nullptr),
randomEngine(0)
{}

void AbstractTask::setSeed(unsigned int seed) {
    randomEngine.seed(seed);
}

float AbstractTask::getRandomFloat(float maxValue){
    auto nextFloat = std::uniform_real_distribution<float>(0, maxValue);
    return nextFloat(this->randomEngine);
}

void AbstractTask::notifyObserversUpdate() const {
    for(AbstractTaskObserver* observer: taskObservers){
        observer->notify();
    }
}

void AbstractTask::unregisterObserver(AbstractTaskObserver* observer) {
    auto iterator = taskObservers.begin();
    while(iterator!=taskObservers.end()){
        if(*iterator==observer){
            taskObservers.erase(iterator);
            std::cout << "Unregistered observer" << std::endl;
        }
        iterator++;
    }
}

void AbstractTask::registerObserver(AbstractTaskObserver* observer) {
    taskObservers.emplace_back(observer);
}

void AbstractTask::notifyObserversProgress(float progress) const {
    for(AbstractTaskObserver* observer: taskObservers){
        observer->notifyProgress(progress);
    }
}

void AbstractTask::notifyObserversFinished() const {
    for(AbstractTaskObserver* observer: taskObservers){
        observer->notifyFinished();
    }
}

void AbstractTask::notifyObserversStarted() const {
    for(AbstractTaskObserver* observer: taskObservers){
        observer->notifyStarted();
    }
}

void AbstractTask::notifyObserversStatus(const std::string& status) const {
    for(AbstractTaskObserver* observer: taskObservers){
        observer->notifyStatus(status);
    }
}
